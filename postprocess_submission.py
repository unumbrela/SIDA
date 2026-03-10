"""
Post-process the SIDA inference CSV to fix known issues:
1. Re-determine labels from explanation content (fix all-label-1 bug)
2. Clean [CLS] tokens from explanations
3. Fix garbage/repetitive outputs
4. Set empty mask + proper explanation for real images
5. Provide fallback explanation for garbage outputs
"""

import argparse
import csv
import json
import re

import numpy as np
from pycocotools import mask as mask_utils


def is_garbage_text(text):
    """Detect garbage/repetitive output patterns."""
    # Repeated characters: 00000000, uuuuuuuu, etc.
    if re.search(r'(.)\1{15,}', text):
        return True
    # Repeated words/phrases: "马来西亚马来西亚马来西亚..."
    if re.search(r'(.{2,8})\1{5,}', text):
        return True
    # English gibberish patterns from model hallucination
    if re.search(r'(isms|ursens|ursion|Type Type|B\. B\. B\.)', text):
        return True
    # Repeated [CLS] tokens
    if text.count('[CLS]') >= 3:
        return True
    # Very short or near-empty after cleaning
    cleaned = re.sub(r'\[CLS\]|\[SEG\]|\[END\]', '', text).strip()
    if len(cleaned) < 10:
        return True
    return False


def classify_from_explanation(explanation):
    """
    Determine label from explanation text content.
    Returns: 0 (real), 1 (forged), or None (uncertain)
    """
    # Strong real indicators
    real_phrases = [
        "真实拍摄", "真实的", "未发现伪造", "未发现篡改",
        "未发现数字伪造", "未发现后期篡改", "不存在伪造",
        "真实记录", "真实场景",
    ]
    # Strong forged indicators
    forged_phrases = [
        "伪造", "篡改", "编辑", "合成", "修改", "数字添加",
        "后期", "AI生成", "人工智能生成", "异常", "不一致",
        "不自然", "tampered", "forged", "manipulated",
    ]

    has_real = any(p in explanation for p in real_phrases)
    has_forged = any(p in explanation for p in forged_phrases)

    if has_forged and not has_real:
        return 1
    if has_real and not has_forged:
        return 0
    if has_forged and has_real:
        # Both present - check which comes first / is dominant
        # "这是一张真实的...未发现伪造" -> real
        # "这是一份伪造的...真实" -> forged
        first_real = min((explanation.find(p) for p in real_phrases if p in explanation), default=9999)
        first_forged = min((explanation.find(p) for p in forged_phrases if p in explanation), default=9999)
        if first_real < first_forged:
            return 0
        return 1
    return None  # uncertain


def clean_explanation(text):
    """Clean explanation text: remove special tokens, fix formatting."""
    # Remove special tokens
    text = re.sub(r'\[CLS\]\s*', '', text)
    text = re.sub(r'\[SEG\]\s*', '', text)
    text = re.sub(r'\[END\]\s*', '', text)
    text = re.sub(r'<s>\s*', '', text)
    text = re.sub(r'</s>\s*', '', text)
    # Remove leading classification sentences that leaked through
    text = re.sub(r'^这张图片经过伪造篡改。\s*', '', text)
    text = re.sub(r'^这张图片是真实的。\s*', '', text)
    # Trim whitespace
    text = text.strip()
    return text


def mask_is_empty(rle_str):
    """Check if an RLE mask is all zeros."""
    loc = json.loads(rle_str)
    rle = {'size': loc['size'], 'counts': loc['counts'].encode('utf-8')}
    m = mask_utils.decode(rle)
    return m.sum() == 0


def make_empty_rle(rle_str):
    """Create empty RLE with same size as given RLE."""
    loc = json.loads(rle_str)
    h, w = loc['size']
    empty = np.zeros((h, w), dtype=np.uint8, order='F')
    rle_dict = mask_utils.encode(empty)
    if isinstance(rle_dict['counts'], bytes):
        rle_dict['counts'] = rle_dict['counts'].decode('utf-8')
    return json.dumps(rle_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    with open(args.input_csv, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    print(f"Input: {len(rows)} rows")

    # Stats
    stats = {
        'relabeled_to_real': 0,
        'relabeled_to_forged': 0,
        'garbage_fixed': 0,
        'cls_cleaned': 0,
        'uncertain_kept_forged': 0,
    }

    for row in rows:
        explanation = row['explanation']
        original_label = int(row['label'])

        # Step 1: Clean explanation
        if '[CLS]' in explanation or '[SEG]' in explanation or '[END]' in explanation:
            stats['cls_cleaned'] += 1
        explanation = clean_explanation(explanation)

        # Step 2: Check for garbage
        if is_garbage_text(explanation):
            stats['garbage_fixed'] += 1
            empty = mask_is_empty(row['location'])
            if empty:
                # Likely a real image the model couldn't handle
                row['label'] = 0
                row['explanation'] = "该图像为真实拍摄，未发现伪造或篡改痕迹。"
                row['location'] = make_empty_rle(row['location'])
                stats['relabeled_to_real'] += 1
            else:
                # Has a mask, keep as forged but fix explanation
                row['label'] = 1
                row['explanation'] = "该图像存在伪造篡改痕迹，篡改区域已在掩码中标出。"
            continue

        # Step 3: Re-determine label from explanation
        pred_label = classify_from_explanation(explanation)

        if pred_label == 0:
            row['label'] = 0
            row['location'] = make_empty_rle(row['location'])
            row['explanation'] = explanation
            if original_label != 0:
                stats['relabeled_to_real'] += 1
        elif pred_label == 1:
            row['label'] = 1
            row['explanation'] = explanation
            if original_label != 1:
                stats['relabeled_to_forged'] += 1
        else:
            # Uncertain - use mask as secondary signal
            empty = mask_is_empty(row['location'])
            if empty:
                # No mask + uncertain text -> likely real
                row['label'] = 0
                row['location'] = make_empty_rle(row['location'])
                if not explanation or len(explanation) < 10:
                    row['explanation'] = "该图像为真实拍摄，未发现伪造或篡改痕迹。"
                else:
                    row['explanation'] = explanation
                stats['relabeled_to_real'] += 1
            else:
                # Has mask -> keep as forged
                row['label'] = 1
                row['explanation'] = explanation
                stats['uncertain_kept_forged'] += 1

    # Write output
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "label", "location", "explanation"])
        writer.writeheader()
        writer.writerows(rows)

    # Print stats
    label_dist = {0: 0, 1: 0}
    for r in rows:
        label_dist[int(r['label'])] += 1

    print(f"\nPost-processing stats:")
    print(f"  Relabeled to real: {stats['relabeled_to_real']}")
    print(f"  Relabeled to forged: {stats['relabeled_to_forged']}")
    print(f"  Garbage outputs fixed: {stats['garbage_fixed']}")
    print(f"  [CLS]/[SEG] tokens cleaned: {stats['cls_cleaned']}")
    print(f"  Uncertain kept as forged (has mask): {stats['uncertain_kept_forged']}")
    print(f"\nFinal label distribution: real={label_dist[0]}, forged={label_dist[1]}")
    print(f"Output saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
