import os
import argparse
from tqdm import tqdm
import time
import logging

import pandas as pd
import torch
from PIL import Image
import numpy as np
from sklearn.metrics import roc_curve, auc
from markllm.evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator

from mark_utils import WatermarkUtils

def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Experiments parameters')
    aa("--json_path", type=str, default="input/")
    aa("--image_dir", type=str, default="input/")
    aa("--model_path", type=str, default="input/")
    aa("--range_num", type=int, default=100)
    aa("--model_name", type=str, default="llava")
    aa("--task_name", type=str, default="AMBER")
    aa("--data_suffix", type=str, default=".jpg")
    aa("--similarity_scheme", type=str, default="cosine")
    aa("--max_tokens", type=int, default=128)
    aa("--min_tokens", type=int, default=40)
    
    return parser


def refresh_logits_processor(model, processor, logits_processor, inputs, image, utils=None, params=None):
    """刷新logits processor用于生成带水印文本"""
    cls_feature = utils.get_cls_feature(model, processor, inputs, image)
    logits_processor.refresh_cls(cls_feature)
    feature = utils.get_feature(model, processor, inputs, image)
    logits_processor.refresh_msg(feature)
    return logits_processor


def refresh_mark_detector(mark_detector, model, processor, inputs, image, utils, params):
    """刷新detector用于检测水印"""
    feature = utils.get_feature(model, processor, inputs, image)
    cls_feature = utils.get_cls_feature(model, processor, inputs, image)
    mark_detector.refresh_logits_processor(input_embeddings=feature, cls_feature=cls_feature)
    return mark_detector


def get_image_data(id: int, utils, params):
    """获取图像数据"""
    image_id = f'{params.task_name}_{id+1}{params.data_suffix}'
    image_path = os.path.join(params.image_dir, image_id)
    image = Image.open(image_path)
    image = image.resize((336, 336), Image.Resampling.LANCZOS)
    
    image = utils.image_watermark(image)
    return image


def calculate_accuracy(y_true, y_scores):
    """计算AUROC等指标"""
    y_scores = np.array(y_scores, dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    tnr = 1 - fpr
    fnr = 1 - tpr

    result = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_auc": roc_auc,
        "tnr": tnr.tolist(),
        "fnr": fnr.tolist(),
        "thresholds": thresholds.tolist()
    }
    return result


def pipeline(utils, params):
    """合并的生成和检测流水线"""
    df = pd.read_json(params.json_path)
    
    model, processor = utils.init_model()
    logits_processor = utils.init_logits_processor(model, processor)
    detector = utils.init_detector(model, processor)
    
    y_true = []
    y_scores = []
    watermark_scores = []
    un_watermark_scores = []
    ro_y_true = []
    ro_y_scores = []
    ro_watermark_scores = []
    ro_un_watermark_scores = []
    
    start_time = time.time()
    
    # 主循环：生成文本并同时进行两种检测
    for i in tqdm(range(0, min(params.range_num, len(df)))):
        row = df.iloc[i]
        if 'image' in row:
            image_id = row['image']
            image_path = os.path.join(params.image_dir, image_id)
            image = Image.open(image_path)
        else:
            image = get_image_data(i, utils, params)
        
        question = row['query']
        inputs = utils.get_inputs(model, processor, image, question)
        
        # 生成不带水印的文本
        unwatermarked_response = utils.generate_responses(model, processor, inputs, None)
        
        # 生成带水印的文本
        refreshed_logits_processor = refresh_logits_processor(
            model, processor, logits_processor, inputs, image, utils, params
        )
        watermarked_response = utils.generate_responses(
            model, processor, inputs, refreshed_logits_processor
        )
        
        # === 正常检测 ===
        detector_normal = refresh_mark_detector(
            detector, model, processor, inputs, image, utils, params
        )
        
        # 检测不带水印的文本
        _, unwatermarked_score = detector_normal.detect_watermark(unwatermarked_response, return_dict=False)
        un_watermark_scores.append(unwatermarked_score)
        y_true.append(0)
        y_scores.append(unwatermarked_score)
        
        # 检测带水印的文本
        _, watermarked_score = detector_normal.detect_watermark(watermarked_response, return_dict=False)
        watermark_scores.append(watermarked_score)
        y_true.append(1)
        y_scores.append(watermarked_score)
        
        # === Random Only检测 ===
        detector_ro = refresh_mark_detector(
            detector, model, processor, inputs, image, utils, params
        )
        detector_ro.set_random_only_process()
        
        # 检测不带水印的文本 (random only)
        _, ro_unwatermarked_score = detector_ro.detect_watermark(unwatermarked_response, return_dict=False)
        ro_un_watermark_scores.append(ro_unwatermarked_score)
        ro_y_true.append(0)
        ro_y_scores.append(ro_unwatermarked_score)
        
        # 检测带水印的文本 (random only)
        _, ro_watermarked_score = detector_ro.detect_watermark(watermarked_response, return_dict=False)
        ro_watermark_scores.append(ro_watermarked_score)
        ro_y_true.append(1)
        ro_y_scores.append(ro_watermarked_score)
        
        torch.cuda.empty_cache()
        logging.info(f"un_water_scores:{un_watermark_scores};water_scores:{watermark_scores}")
        logging.info(f"ro_un_water_scores:{ro_un_watermark_scores};ro_water_scores:{ro_watermark_scores}")
    
    # 计算并输出正常检测结果
    logging.info("\n=== Accuracy Detection Results ===")
    calculator = DynamicThresholdSuccessRateCalculator(labels=['FPR','TPR', 'TNR', 'FNR', 'F1', 'ACC'], rule='best')
    
    algorithm_accuracy_detail = calculate_accuracy(y_true, y_scores)
    detail = calculator.calculate(watermark_scores, un_watermark_scores)
    
    logging.info(f"Algorithm: VLA-Mark")
    logging.info(f"  ROC AUC: {algorithm_accuracy_detail['roc_auc']:.4f}")
    logging.info(f"  F1 Score: {detail['F1']:.4f}")
    logging.info(f"  Accuracy: {detail['ACC']:.4f}")
    
    # 计算并输出random only检测结果
    logging.info("\n=== Random Only Detection Results ===")
    
    ro_algorithm_accuracy_detail = calculate_accuracy(ro_y_true, ro_y_scores)
    ro_detail = calculator.calculate(ro_watermark_scores, ro_un_watermark_scores)
    
    logging.info(f"Algorithm: VLA-Mark (Random Only)")
    logging.info(f"  ROC AUC: {ro_algorithm_accuracy_detail['roc_auc']:.4f}")
    logging.info(f"  F1 Score: {ro_detail['F1']:.4f}")
    logging.info(f"  Accuracy: {ro_detail['ACC']:.4f}")
    
    total_time = time.time() - start_time
    logging.info(f"\nTotal pipeline time: {total_time:.2f}s")
    logging.info(f"Total number of samples processed: {min(params.range_num, len(df))}")


def main(params):
    logging.info(f"CUDA is available: {torch.cuda.is_available()}")
    utils = WatermarkUtils(params)
    pipeline(utils, params)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = get_parser()
    params = parser.parse_args()
    
    main(params)