"""
Fine-tune SIDA-7B-description on the competition training data.
Based on train_SIDA_description.py, adapted for:
  - 2-class (real/tampered) instead of 3-class
  - Chinese captions
  - Competition data layout
"""

import argparse
import os
import shutil
import sys
import time
import random
import warnings
from functools import partial

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import deepspeed
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.SIDA_description import SIDAForCausalLM
from model.llava import conversation as conversation_lib
from utils.competition_dataset import collate_fn, CompetitionDataset
from utils.batch_sampler import BatchSampler
from utils.utils import (
    DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
    AverageMeter, ProgressMeter, Summary, dict_to_cuda,
    intersectionAndUnionGPU,
)


def parse_args(args):
    parser = argparse.ArgumentParser(description="Fine-tune SIDA on competition data")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--version", default="./ck/SIDA-7B-description")
    parser.add_argument("--dataset_dir", default="./My_Forgery_Location_Task/dataset/train")
    parser.add_argument("--vision_pretrained", default="./ck/sam_vit_h_4b8939.pth")
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14")
    parser.add_argument("--log_base_dir", default="./runs")
    parser.add_argument("--exp_name", default="SIDA-competition")

    # Training hyperparameters
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--steps_per_epoch", default=200, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--val_ratio", default=0.1, type=float)

    # Model args
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=1024, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj")
    parser.add_argument("--out_dim", default=256, type=int)

    # Loss weights
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=1.0, type=float)
    parser.add_argument("--bce_loss_weight", default=1.0, type=float)
    parser.add_argument("--cls_loss_weight", default=1.0, type=float)
    parser.add_argument("--mask_loss_weight", default=1.0, type=float)

    # Misc
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1")
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--no_eval", action="store_true", default=False)

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    deepspeed.init_distributed()
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # ---- Tokenizer ----
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length,
        padding_side="right", use_fast=False,
    )
    tokenizer.add_tokens("[END]")
    required_tokens = ['[CLS]', '[SEG]', '[END]', DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
    for token in required_tokens:
        assert token in tokenizer.get_vocab(), f"{token} not found in tokenizer"
    args.cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    # ---- Model ----
    torch_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.half}[args.precision]

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "cls_loss_weight": args.cls_loss_weight,
        "mask_loss_weight": args.mask_loss_weight,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "cls_token_idx": args.cls_token_idx,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }

    model = SIDAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    model.get_model().initialize_sida_modules(model.get_model().config)

    # Freeze base model
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False
    for p in model.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # ---- LoRA ----
    if args.lora_r > 0:
        def find_linear_layers(model, targets):
            cls = torch.nn.Linear
            names = set()
            exclude = ["visual_model", "vision_tower", "mm_projector",
                       "text_hidden_fcs", "cls_head", "sida_fc1", "attention_layer"]
            for name, module in model.named_modules():
                if (isinstance(module, cls)
                    and all(x not in name for x in exclude)
                    and any(x in name for x in targets)):
                    names.add(name)
            return sorted(names)

        lora_targets = find_linear_layers(model, args.lora_target_modules.split(","))
        lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=lora_targets, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # Unfreeze key components
    for n, p in model.named_parameters():
        if any(x in n for x in ["lm_head", "mask_decoder", "cls_head",
                                  "text_hidden_fcs", "sida_fc1", "attention_layer",
                                  "embed_tokens"]):
            p.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.local_rank == 0:
        print(f"\nTotal trainable parameters: {total_params:,}")

    # ---- Datasets ----
    args.distributed = torch.cuda.device_count() > 1

    train_dataset = CompetitionDataset(
        base_dir=args.dataset_dir, tokenizer=tokenizer,
        vision_tower=args.vision_tower, precision=args.precision,
        image_size=args.image_size, val_ratio=args.val_ratio, split="train",
    )

    val_dataset = None
    if not args.no_eval and args.val_ratio > 0:
        val_dataset = CompetitionDataset(
            base_dir=args.dataset_dir, tokenizer=tokenizer,
            vision_tower=args.vision_tower, precision=args.precision,
            image_size=args.image_size, val_ratio=args.val_ratio, split="val",
        )

    # ---- DataLoaders ----
    batch_sampler = BatchSampler(
        dataset=train_dataset, batch_size=args.batch_size,
        world_size=torch.cuda.device_count(), rank=args.local_rank,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=batch_sampler,
        num_workers=args.workers, pin_memory=True,
        collate_fn=partial(
            collate_fn, tokenizer=tokenizer, conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank,
            cls_token_idx=args.cls_token_idx,
        ),
    )

    # ---- DeepSpeed ----
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": args.lr, "weight_decay": 0.0, "betas": (args.beta1, args.beta2)},
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0, "warmup_max_lr": args.lr,
                "warmup_num_steps": 50, "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
            "loss_scale": 0, "initial_scale_power": 12,
            "loss_scale_window": 1000, "min_loss_scale": 1, "hysteresis": 2,
        },
        "bf16": {"enabled": args.precision == "bf16"},
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2, "contiguous_gradients": True, "overlap_comm": True,
            "reduce_scatter": True, "reduce_bucket_size": 5e8, "allgather_bucket_size": 5e8,
        },
    }

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=ds_config, training_data=None,
    )

    # ---- Resume ----
    if args.auto_resume and not args.resume:
        resume_path = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume_path):
            args.resume = resume_path
    if args.resume:
        latest_file = os.path.join(args.resume, "latest")
        if os.path.exists(latest_file):
            load_path, _ = model_engine.load_checkpoint(args.resume)
            with open(latest_file, "r") as f:
                ckpt_dir = f.readlines()[0].strip()
            args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
            print(f"Resumed from {args.resume}, starting epoch {args.start_epoch}")
        else:
            print(f"Warning: {latest_file} not found, skipping resume and training from scratch.")
            args.resume = ""

    # ---- Training loop ----
    train_iter = iter(train_loader)
    for epoch in range(args.start_epoch, args.epochs):
        train_iter = train_one_epoch(train_loader, model_engine, epoch, scheduler, writer, train_iter, args)

        # Save checkpoint every epoch
        save_dir = os.path.join(args.log_dir, "ckpt_model")
        if args.local_rank == 0:
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
        if args.distributed:
            torch.distributed.barrier()
        model_engine.save_checkpoint(save_dir)
        if args.local_rank == 0:
            print(f"Checkpoint saved after epoch {epoch + 1}")

    # Final checkpoint
    save_dir = os.path.join(args.log_dir, "final_checkpoint")
    if args.local_rank == 0:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
    if args.distributed:
        torch.distributed.barrier()
    model_engine.save_checkpoint(save_dir)
    if args.local_rank == 0:
        print(f"\nTraining completed. Final checkpoint: {save_dir}")


def train_one_epoch(train_loader, model, epoch, scheduler, writer, train_iter, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    cls_losses = AverageMeter("ClsLoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    progress = ProgressMeter(
        args.steps_per_epoch,
        [batch_time, losses, cls_losses, mask_losses, ce_losses],
        prefix=f"Epoch: [{epoch}]",
    )
    model.train()
    end = time.time()

    for global_step in range(args.steps_per_epoch):
        model.zero_grad()
        for _ in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            input_dict = dict_to_cuda(input_dict)
            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            output_dict = model(**input_dict)
            loss = output_dict["loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            cls_losses.update(output_dict["cls_loss"].item(), input_dict["images"].size(0))
            ce_losses.update(output_dict["ce_loss"].item(), input_dict["images"].size(0))
            if input_dict["cls_labels"][0] == 2:
                mask_losses.update(output_dict["mask_loss"].item(), input_dict["images"].size(0))

            model.backward(loss)
            model.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                losses.all_reduce()
            if args.local_rank == 0:
                progress.display(global_step + 1)
                if writer:
                    writer.add_scalar("train/loss", losses.avg, epoch * args.steps_per_epoch + global_step)
                    writer.add_scalar("train/cls_loss", cls_losses.avg, epoch * args.steps_per_epoch + global_step)
                    writer.add_scalar("train/mask_loss", mask_losses.avg, epoch * args.steps_per_epoch + global_step)
                    writer.add_scalar("train/ce_loss", ce_losses.avg, epoch * args.steps_per_epoch + global_step)
            batch_time.reset()
            losses.reset()
            cls_losses.reset()
            mask_losses.reset()
            ce_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0 and writer:
                writer.add_scalar("train/lr", curr_lr[0], epoch * args.steps_per_epoch + global_step)

    return train_iter


if __name__ == "__main__":
    main(sys.argv[1:])
