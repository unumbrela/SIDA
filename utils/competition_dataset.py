"""
Dataset for the Scene Text Forgery Analysis competition.
Adapts the competition data format to SIDA's expected training interface.

Data layout:
  dataset/train/Black/Image/   (800 tampered images)
  dataset/train/Black/Mask/    (800 binary masks, same stem .png)
  dataset/train/Black/Caption/ (800 .md files with Chinese explanation)
  dataset/train/White/Image/   (200 real images)
  dataset/train/White/Caption/ (200 .md files with Chinese explanation)
"""

import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from .utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN


QUESTION_TEMPLATES = [
    "请判断这张图片是真实拍摄的还是经过伪造篡改的。如果是伪造的，请标注伪造区域并解释原因。",
    "请分析这张场景文本图像的真实性。如果发现伪造痕迹，请定位伪造区域并给出判断依据。",
    "这张图片是否经过篡改？如果是，请标记被篡改的区域，并详细说明伪造的原因。",
    "请鉴定这张图片的真伪。若判定为伪造，请指出伪造位置并阐述理由。",
]


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True,
    local_rank=-1, cls_token_idx=None,
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    cls_labels_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    has_text_description = []

    for (
        image_path, images, images_clip, conversations, masks, label,
        cls_labels, resize, questions, sampled_classes, inference, has_text,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        masks_list.append(masks.float())
        label_list.append(label)
        cls_labels_list.append(torch.tensor(cls_labels))
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)
        has_text_description.append(has_text)

    if use_mm_start_end:
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            conversation_list[i] = conversation_list[i].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    original_input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    original_lengths = [len(ids) for ids in original_input_ids]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        original_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    targets = []
    for i, conversation in enumerate(conversation_list):
        if has_text_description[i]:
            target = input_ids[i].clone()
        else:
            target = torch.full_like(input_ids[i], IGNORE_INDEX)
        targets.append(target)
    targets = torch.stack(targets)

    conv = conversation_lib.conv_templates["llava_v1"].copy()
    sep = conv.sep + conv.roles[1] + ": " if conv_type == "llava_v1" else "[/INST] "

    for idx, (conversation, target, orig_len) in enumerate(zip(conversation_list, targets, original_lengths)):
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                continue
            parts[0] += sep
            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        total_len = orig_len
        if cur_len != total_len:
            print(f"Length mismatch in conversation {idx}: cur_len={cur_len}, total_len={total_len}")
        target[cur_len:] = IGNORE_INDEX

    if not inferences[0]:
        truncate_len = tokenizer.model_max_length - 255
        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "cls_labels": torch.stack(cls_labels_list).view(-1),
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "cls_labels_list": cls_labels_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


class CompetitionDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_dir,       # e.g. ./My_Forgery_Location_Task/dataset/train
        tokenizer,
        vision_tower,
        precision="fp32",
        image_size=1024,
        val_ratio=0.0,  # fraction to hold out for validation
        split="train",  # "train" or "val"
        seed=42,
    ):
        self.base_dir = base_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        # -- Collect tampered (Black) samples --
        black_img_dir = os.path.join(base_dir, "Black", "Image")
        black_mask_dir = os.path.join(base_dir, "Black", "Mask")
        black_cap_dir = os.path.join(base_dir, "Black", "Caption")

        black_images = sorted(glob.glob(os.path.join(black_img_dir, "*")))
        black_images = [p for p in black_images if p.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # -- Collect real (White) samples --
        white_img_dir = os.path.join(base_dir, "White", "Image")
        white_cap_dir = os.path.join(base_dir, "White", "Caption")

        white_images = sorted(glob.glob(os.path.join(white_img_dir, "*")))
        white_images = [p for p in white_images if p.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Build sample list: (image_path, cls_label, mask_path_or_None, caption_path_or_None)
        all_samples = []
        for img_path in black_images:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(black_mask_dir, stem + ".png")
            cap_path = os.path.join(black_cap_dir, stem + ".md")
            if not os.path.exists(mask_path):
                print(f"WARNING: mask not found for {img_path}, skipping")
                continue
            cap = self._read_caption(cap_path)
            all_samples.append((img_path, 2, mask_path, cap))  # cls_label=2 (tampered)

        for img_path in white_images:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            cap_path = os.path.join(white_cap_dir, stem + ".md")
            cap = self._read_caption(cap_path)
            all_samples.append((img_path, 0, None, cap))  # cls_label=0 (real)

        # Split train/val
        rng = random.Random(seed)
        rng.shuffle(all_samples)
        if val_ratio > 0 and split in ("train", "val"):
            n_val = max(1, int(len(all_samples) * val_ratio))
            if split == "val":
                all_samples = all_samples[:n_val]
            else:
                all_samples = all_samples[n_val:]

        self.samples = all_samples

        # Expose cls_labels list for BatchSampler compatibility
        self.cls_labels = [s[1] for s in self.samples]

        n_tampered = sum(1 for s in self.samples if s[1] == 2)
        n_real = sum(1 for s in self.samples if s[1] == 0)
        n_with_cap = sum(1 for s in self.samples if s[3] is not None)
        print(f"\nCompetition Dataset [{split}]:")
        print(f"  Tampered: {n_tampered}, Real: {n_real}, Total: {len(self.samples)}")
        print(f"  With captions: {n_with_cap}")

    @staticmethod
    def _read_caption(path):
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return None

    def __len__(self):
        return len(self.samples)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def _build_response(self, cls_label, caption):
        """Build the target response string."""
        if cls_label == 0:
            base = "[CLS] 这张图片是真实的。"
            if caption:
                return f"{base} {caption} [END]"
            return f"{base} [END]"
        else:  # cls_label == 2 (tampered)
            base = "[CLS] 这张图片经过伪造篡改。"
            if caption:
                return f"{base} [SEG] {caption} [END]"
            return f"{base} [SEG] [END]"

    def __getitem__(self, idx):
        img_path, cls_label, mask_path, caption = self.samples[idx]

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # CLIP processing
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        # SAM processing
        image_sam = self.transform.apply_image(image)
        resize = image_sam.shape[:2]
        image_sam = self.preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())

        # Load mask
        if cls_label == 2 and mask_path and os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_img = self.transform.apply_image(mask_img)
            mask_img = mask_img / 255.0
            mask = torch.from_numpy(mask_img).unsqueeze(0)
        else:
            mask = torch.zeros((1, resize[0], resize[1]))

        # Build conversation
        question = random.choice(QUESTION_TEMPLATES)
        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question}")
        response = self._build_response(cls_label, caption)
        conv.append_message(conv.roles[1], response)
        conversation = conv.get_prompt()

        has_text = caption is not None
        labels = torch.ones(mask.shape[1], mask.shape[2]) * self.ignore_label

        return (
            img_path, image_sam, image_clip, [conversation], mask, labels,
            cls_label, resize, None, None, False, has_text,
        )
