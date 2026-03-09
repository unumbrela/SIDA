"""
Batch inference script for the Scene Text Forgery Analysis competition.
Uses SIDA-7B-description to produce detection, localization, and explanation.
Outputs a CSV file ready for submission.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.SIDA_description import SIDAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def parse_args():
    parser = argparse.ArgumentParser(description="SIDA batch inference for competition")
    parser.add_argument("--version", default="./ck/SIDA-7B-description", type=str,
                        help="Path to SIDA model weights")
    parser.add_argument("--test_dir", default="./My_Forgery_Location_Task/dataset/test", type=str,
                        help="Path to test image directory")
    parser.add_argument("--output_csv", default="./My_Forgery_Location_Task/submission.csv", type=str,
                        help="Path to output CSV file")
    parser.add_argument("--vis_save_path", default="./vis_output_batch", type=str,
                        help="Path to save visualization (optional)")
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=1024, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str)
    parser.add_argument("--save_vis", action="store_true", default=False,
                        help="Whether to save mask visualizations")
    parser.add_argument("--prompt", type=str,
                        default="请判断这张图片是真实拍摄的还是经过伪造篡改的。如果是伪造的，请标注伪造区域并解释原因。",
                        help="Prompt text for the model")
    return parser.parse_args()


def remove_repeated_tags(text):
    """Remove repeated XML-like tag segments from the model output."""
    tag_content_pattern = r'<([^>]+)>([^<]+)'
    matches = re.findall(tag_content_pattern, text)
    if matches:
        first_tag = f"<{matches[0][0]}>"
        parts = text.split(first_tag, 1)
        prefix = parts[0] if len(parts) > 1 else ""
        unique_segments = []
        seen_pairs = set()
        for tag, content in matches:
            key = (tag, content.strip())
            if key not in seen_pairs:
                seen_pairs.add(key)
                unique_segments.append(f"<{tag}>{content}")
        result = prefix + ''.join(unique_segments)
        return result
    return text


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def mask_to_rle(binary_mask: np.ndarray) -> str:
    """Convert a binary mask (H, W) with values {0, 1} to RLE JSON string."""
    mask_fortran = np.asfortranarray(binary_mask.astype(np.uint8))
    rle_dict = mask_utils.encode(mask_fortran)
    if isinstance(rle_dict['counts'], bytes):
        rle_dict['counts'] = rle_dict['counts'].decode('utf-8')
    return json.dumps(rle_dict)


def make_empty_rle(h: int, w: int) -> str:
    """Create an RLE string for an all-zero mask of size (h, w)."""
    empty_mask = np.zeros((h, w), dtype=np.uint8, order='F')
    rle_dict = mask_utils.encode(empty_mask)
    if isinstance(rle_dict['counts'], bytes):
        rle_dict['counts'] = rle_dict['counts'].decode('utf-8')
    return json.dumps(rle_dict)


def extract_explanation(text_output: str) -> str:
    """
    Extract the explanation portion from the SIDA-description model output.
    The model typically outputs something like:
      [CLS] The image is tampered.[SEG] Type: ... Areas: ... Tampered Content: ... Visual Inconsistencies: ...
    We extract everything after [SEG] (or after classification text) as the explanation.
    """
    text = text_output.strip()

    # Remove leading <s> and trailing </s>
    text = re.sub(r'^<s>\s*', '', text)
    text = re.sub(r'\s*</s>$', '', text)
    text = re.sub(r'\s*\[end\]\s*$', '', text, flags=re.IGNORECASE)

    # Try to extract content after [SEG]
    seg_match = re.search(r'\[SEG\]\s*(.*)', text, re.DOTALL)
    if seg_match:
        explanation = seg_match.group(1).strip()
        if explanation:
            return explanation

    # Fallback: extract content after classification sentence
    cls_match = re.search(r'\[CLS\]\s*[^.]*\.\s*(.*)', text, re.DOTALL)
    if cls_match:
        explanation = cls_match.group(1).strip()
        if explanation:
            return explanation

    # Last resort: return the whole cleaned text
    return text


def main():
    args = parse_args()

    if args.save_vis:
        os.makedirs(args.vis_save_path, exist_ok=True)

    # ---- Load model ----
    print(f"Loading model from {args.version} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[END]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        })
    elif args.load_in_8bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                llm_int8_skip_modules=["visual_model"],
                load_in_8bit=True,
            ),
        })

    model = SIDAForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=args.seg_token_idx,
        cls_token_idx=args.cls_token_idx,
        **kwargs,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if torch.cuda.is_available():
        model = model.cuda()

    try:
        model.get_model().initialize_vision_modules(model.get_model().config)
        vision_tower = model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch_dtype)
    except AttributeError:
        print("Vision tower initialization skipped.")

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif args.precision == "fp16":
        model = model.half().cuda()
    else:
        model = model.float().cuda()

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    model.eval()
    print("Model loaded successfully.\n")

    # ---- Build prompt template ----
    prompt_text = args.prompt

    # ---- Collect test images ----
    test_dir = args.test_dir
    image_names = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
    ])
    print(f"Found {len(image_names)} test images in {test_dir}\n")

    # ---- Inference loop ----
    results = []
    for idx, image_name in enumerate(image_names):
        image_path = os.path.join(test_dir, image_name)
        start_time = time.time()

        try:
            # Build conversation prompt
            conv = conversation_lib.conv_templates[args.conv_type].copy()
            conv.messages = []

            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
            if args.use_mm_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt()

            # Load and preprocess image
            image_np = cv2.imread(image_path)
            if image_np is None:
                print(f"[{idx+1}/{len(image_names)}] ERROR: Cannot read {image_name}, skipping.")
                h, w = 1024, 768  # fallback size
                results.append({
                    "image_name": image_name,
                    "label": 0,
                    "location": make_empty_rle(h, w),
                    "explanation": "Unable to read image.",
                })
                continue

            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_h, original_w = image_np.shape[:2]
            original_size_list = [image_np.shape[:2]]

            # CLIP image
            image_clip = (
                clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
                .unsqueeze(0).cuda()
            )
            if args.precision == "bf16":
                image_clip = image_clip.bfloat16()
            elif args.precision == "fp16":
                image_clip = image_clip.half()
            else:
                image_clip = image_clip.float()

            # SAM image
            image = transform.apply_image(image_np)
            resize_list = [image.shape[:2]]
            image = (
                preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
                .unsqueeze(0).cuda()
            )
            if args.precision == "bf16":
                image = image.bfloat16()
            elif args.precision == "fp16":
                image = image.half()
            else:
                image = image.float()

            # Tokenize
            input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            input_ids = input_ids.unsqueeze(0).cuda()

            # Model inference
            with torch.no_grad():
                output_ids, pred_masks = model.evaluate(
                    image_clip,
                    image,
                    input_ids,
                    resize_list,
                    original_size_list,
                    max_new_tokens=512,
                    tokenizer=tokenizer,
                )

            # Decode text output
            output_ids_clean = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
            text_output = tokenizer.decode(output_ids_clean, skip_special_tokens=False)
            text_output = text_output.replace("\n", "").replace("  ", " ")
            text_output = remove_repeated_tags(text_output)

            # ---- Determine label ----
            # Check text output for classification cues (English + Chinese)
            text_lower = text_output.lower()
            has_tampered = any(kw in text_lower for kw in [
                "tampered", "altered", "forged", "manipulated",
            ]) or any(kw in text_output for kw in [
                "伪造", "篡改", "编辑", "合成",
            ])
            has_real = any(kw in text_lower for kw in [
                "real", "authentic", "genuine",
            ]) or any(kw in text_output for kw in [
                "真实",
            ])
            has_synthetic = any(kw in text_lower for kw in [
                "synthetic", "generated", "fake",
            ])
            # Also check if model produced [SEG] token (indicates it found tampered region)
            has_seg = "[SEG]" in text_output

            if has_tampered or has_seg:
                label = 1
                is_tampered = True
            elif has_real and not has_tampered:
                label = 0
                is_tampered = False
            elif has_synthetic:
                label = 1
                is_tampered = False
            else:
                label = 1
                is_tampered = True

            # ---- Build mask RLE ----
            if is_tampered and len(pred_masks) > 0:
                # Merge all predicted masks into one binary mask
                combined_mask = np.zeros((original_h, original_w), dtype=np.uint8)
                for pred_mask in pred_masks:
                    mask_np = pred_mask.detach().cpu().numpy()[0]  # (H, W)
                    binary = (mask_np > 0).astype(np.uint8)
                    combined_mask = np.maximum(combined_mask, binary)
                rle_str = mask_to_rle(combined_mask)

                # Save visualization
                if args.save_vis:
                    vis_mask = combined_mask * 255
                    cv2.imwrite(
                        os.path.join(args.vis_save_path, f"{os.path.splitext(image_name)[0]}_mask.png"),
                        vis_mask,
                    )
            else:
                # No tampered region: output empty mask
                rle_str = make_empty_rle(original_h, original_w)

            # ---- Extract explanation ----
            explanation = extract_explanation(text_output)
            if not explanation or len(explanation) < 5:
                if label == 0:
                    explanation = "该图像为真实拍摄，未发现伪造或篡改痕迹。"
                else:
                    explanation = "该图像存在伪造或篡改痕迹。"

            results.append({
                "image_name": image_name,
                "label": label,
                "location": rle_str,
                "explanation": explanation,
            })

            elapsed = time.time() - start_time
            print(f"[{idx+1}/{len(image_names)}] {image_name} | label={label} | {elapsed:.1f}s | {explanation[:80]}...")

        except Exception as e:
            print(f"[{idx+1}/{len(image_names)}] ERROR on {image_name}: {e}")
            traceback.print_exc()
            # Write a safe fallback row
            try:
                h, w = original_h, original_w
            except NameError:
                h, w = 1024, 768
            results.append({
                "image_name": image_name,
                "label": 0,
                "location": make_empty_rle(h, w),
                "explanation": "Processing error.",
            })

        # Free GPU memory
        torch.cuda.empty_cache()

    # ---- Write CSV ----
    output_csv = args.output_csv
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "label", "location", "explanation"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nDone! Results saved to {output_csv}")
    print(f"Total images: {len(results)}")
    label_counts = {0: 0, 1: 0}
    for r in results:
        label_counts[r["label"]] += 1
    print(f"Label distribution: real={label_counts[0]}, forged={label_counts[1]}")


if __name__ == "__main__":
    main()
