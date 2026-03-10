赛题介绍
• 赛题名称
基于多模态大模型的场景文本图像伪造分析

• 出题单位
蚂蚁区块链科技（上海）有限公司

• 赛题背景
GenAI 的飞速发展，在创新生成内容的同时也使得图像伪造变得空前便捷与逼真。金融，医疗，社交等关键行业高度依赖用户提交的文本图像（如合同、票据、公告等）进行真实性校验。黑产借助先进的AI生成工具，伪造验真文件，制造不实言论对社会信任、信息安全构成了严峻挑战。由于这些编辑工具的高保真度，传统依赖于低层次的信号特征(JPEG压缩伪影、噪声分布不一致等)方案，缺乏对图像内容的高层语义理解，难以应对无视觉痕迹的伪造攻击。并且这些方案往往黑盒化，可解释性差。为了进一步促进图像防伪领域的发展，本届赛事特设“基于多模态大模型的场景文本图像伪造分析挑战赛 (Scene Text Forgery Analysis Competition based on MLLM)”，赛事提出了全新的图像伪造分析任务，在图像伪造检测，定位的基础上新增可解释挑战。并将提供特别构建的多模态场景文本篡改图像数据集，覆盖多样化的伪造类型与真实应用场景。我们期待通过这场聚焦“检测—定位—解释”的综合性挑战，推动多模态大模型在AI安全领域的深度应用，加速可解释、高鲁棒性真伪鉴别技术的突破，为构建可信、智能的数字社会基础设施注入关键技术动力。

赛题任务

针对该赛题，参赛者可自行设计方案架构，实现端到端的伪造分析系统。该系统应能接收任意分辨率的场景文本图像作为输入，并同步完成三项核心输出：

（1）伪造检测(Detection)：对输入图像进行“真实”或“伪造”的二分类判断；
（2）伪造定位(Grounding): 预测与输入图像等比例的二值掩码，精确标识所有伪造区域；
（3）可解释(Explanation)：针对找到的每个伪造区域提供归因描述。

本赛题定义的伪造分析任务，将场景文本图像伪造的检测(Detection)，定位(Grounding)，解释(Explanation) 统一在一个任务框架中，考验参赛者在多模态大模型多任务框架设计，AI 安全领域多模态推理等方面的创新能力。

• 任务描述

（1）伪造检测(Detection)：对输入的图像，判断其是否包含任何形式的伪造。输出二分类标签label（0-“真实”, 1-"伪造”）。
（2）伪造定位(Grounding)：如果图像被判断为“伪造”，系统需输出与原图尺寸相同的二值化掩码（Mask），其中伪造区域被标记为前景（白色像素， 像素值 255），真实区域为背景（黑色像素，像素值 0），最终提交结果中将其转换为RLE编码。
（3）可解释(Explanation)：针对定位出的伪造区域，系统需生成一段自然语言文本，详细描述该区域被判定为伪造的原因。解释归因应具体、有逻辑，并结合图像内容，避免过度幻觉。


赛题说明

提交示例
csv文件格式, 以utf-8编码格式保存，详细字段说明如下：

字段	说明
image_name	测试集中的图片名称; 示例: X00016469612.jpg
label	预测的二分类标签; 0 - 真实图像, 1 - 伪造图像.
location	伪造区域二值掩膜MASK，转换为RLE格式的字符串
explanation	可解释归因的描述文本
二值图像掩膜MASK转换成RLE编码示例:

import cv2
import numpy as np
import json
from pycocotools import mask as mask_utils

def mask_file_to_rle(mask_path: str) -> str:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取图像: {mask_path}")
    binary_mask = (mask > 127).astype(np.uint8)
    mask_fortran = np.asfortranarray(binary_mask)
    rle_dict = mask_utils.encode(mask_fortran)
    if isinstance(rle_dict['counts'], bytes):
        rle_dict['counts'] = rle_dict['counts'].decode('utf-8')
    return json.dumps(rle_dict)


评测标准
• 伪造检测(Detection)
采用二分类任务常用的F1分数（F1-Score）作为主要评估指标，综合考察模型的查准率(Precision)与查全率(Recall)，SDetSDet表示最终的F1指标。
• 伪造定位(Grounding)
采用Pixel-level F1-Score作为评测指标。基于GT Mask与预测Mask的像素重合度，计算对应的像素级F1值，SLocSLoc表示最终的F1指标。
• 可解释(Explanation)
采用复合评估，结合大模型及BertScore进行评估。
● 大模型评估(SAutoSAuto)：基于Qwen3-MAX及Rubrics Prompt自动化打分,100分制。
● 语义相似度评估(SSimSSim)：使用BERTScore 指标，评估生成描述与参考答案在语义上的一致性。
● 可解释综合评分：SExp=0.5×SAuto+0.5×SSimSExp=0.5×SAuto+0.5×SSim
赛题最终计算三个任务的加权平均分(S_Fin)作为最终的排名依据，针对可解释指标，在代码复核阶段会安排人类专家对结果进行抽样评估。
SFin=0.45×SDet+0.25×SLoc+0.3×SExpSFin=0.45×SDet+0.25×SLoc+0.3×SExp


利用SIDA 完成以上赛题任务，远程服务器：A6000  48G显存

========== SIDA-13B-description 微调方案 (推荐) ==========

  Step 0: 修复13B模型配置 (首次运行)

  python fix_13b_config.py ./ck/SIDA-13B-description/config.json

  Step 1: 微调训练 (13B)

  先清理之前的checkpoint避免自动resume冲突：
  rm -rf ./runs/SIDA-13B-competition

  deepspeed --master_port=24999 train_competition.py \
      --version="./ck/SIDA-13B-description" \
      --dataset_dir="./My_Forgery_Location_Task/dataset/train" \
      --vision_pretrained="./ck/sam_vit_h_4b8939.pth" \
      --vision-tower="./ck/clip-vit-large-patch14" \
      --exp_name="SIDA-13B-competition" \
      --epochs=5 \
      --steps_per_epoch=200 \
      --batch_size=1 \
      --grad_accumulation_steps=10 \
      --lr=3e-5 \
      --precision=bf16 \
      --val_ratio=0.1 \
      --model_max_length=2048

  Step 2: 转换 DeepSpeed checkpoint + 合并 LoRA

  cd ./runs/SIDA-13B-competition/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
  cd /home/featurize/work/SIDA

  python merge_lora_weights_and_save_hf_model.py \
    --version "./ck/SIDA-13B-description" \
    --weight "./runs/SIDA-13B-competition/pytorch_model.bin" \
    --save_path "./ck/SIDA-13B-competition" \
    --vision-tower "./ck/clip-vit-large-patch14"

  Step 3: 推理

  python batch_inference.py \
    --version "./ck/SIDA-13B-competition" \
    --test_dir "./My_Forgery_Location_Task/dataset/test" \
    --output_csv "./My_Forgery_Location_Task/submission_finetune_SIDA_13B.csv" \
    --precision bf16 \
    --model_max_length 2048 \
    --vision-tower "./ck/clip-vit-large-patch14" \
    --save_vis \
    --vis_save_path "./vis_finetune_SIDA_13B"

========== SIDA-7B-description 微调方案 (备用) ==========

  Step 1: 微调训练 (7B)

  rm -rf ./runs/SIDA-7B-competition

  deepspeed --master_port=24999 train_competition.py \
      --version="./ck/SIDA-7B-description" \
      --dataset_dir="./My_Forgery_Location_Task/dataset/train" \
      --vision_pretrained="./ck/sam_vit_h_4b8939.pth" \
      --vision-tower="./ck/clip-vit-large-patch14" \
      --exp_name="SIDA-7B-competition" \
      --epochs=5 \
      --steps_per_epoch=200 \
      --batch_size=1 \
      --grad_accumulation_steps=10 \
      --lr=5e-5 \
      --precision=bf16 \
      --val_ratio=0.1 \
      --model_max_length=2048

  Step 2: 转换 + 合并

  cd ./runs/SIDA-7B-competition/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
  cd /home/featurize/work/SIDA

  python merge_lora_weights_and_save_hf_model.py \
    --version "./ck/SIDA-7B-description" \
    --weight "./runs/SIDA-7B-competition/pytorch_model.bin" \
    --save_path "./ck/SIDA-7B-competition" \
    --vision-tower "./ck/clip-vit-large-patch14"

  Step 3: 推理

  python batch_inference.py \
    --version "./ck/SIDA-7B-competition" \
    --test_dir "./My_Forgery_Location_Task/dataset/test" \
    --output_csv "./My_Forgery_Location_Task/submission_finetune_SIDA_7B.csv" \
    --precision bf16 \
    --model_max_length 2048 \
    --vision-tower "./ck/clip-vit-large-patch14" \
    --save_vis \
    --vis_save_path "./vis_finetune_SIDA_7B"
