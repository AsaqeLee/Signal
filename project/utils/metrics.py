import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Any, Union, Optional, List
import logging
import os
from pathlib import Path

def calculate_modulation_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    average: str = 'macro'
) -> Dict[str, Union[float, np.ndarray]]:
    """
    计算调制分类指标
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        average: 多分类指标的平均方式 ('micro', 'macro', 'weighted')
        
    返回:
        dict: 包含各种指标的字典
    """
    # 转换为numpy数组
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    
    # 计算基本指标
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算每个类别的准确率
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # 返回指标字典
    metrics = {
        'mod_accuracy': accuracy,
        'mod_precision': precision,
        'mod_recall': recall,
        'mod_f1': f1,
        'mod_confusion_matrix': cm,
        'mod_per_class_accuracy': per_class_accuracy
    }
    
    return metrics

def calculate_symbol_width_metrics(
    errors: torch.Tensor
) -> Dict[str, float]:
    """
    计算码元宽度估计指标
    
    参数:
        errors: 相对误差序列
        
    返回:
        dict: 包含各种指标的字典
    """
    # 转换为numpy数组
    errors = errors.numpy()
    
    metrics = {
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'median_error': float(np.median(errors)),
        'min_error': float(np.min(errors)),
        'max_error': float(np.max(errors)),
        'error_95th': float(np.percentile(errors, 95))
    }
    
    return metrics

def calculate_symbol_sequence_metrics(
    similarities: torch.Tensor
) -> Dict[str, float]:
    """
    计算码元序列解调指标
    
    参数:
        similarities: 余弦相似度序列
        
    返回:
        dict: 包含各种指标的字典
    """
    # 转换为numpy数组
    similarities = similarities.numpy()
    
    metrics = {
        'mean_similarity': float(np.mean(similarities)),
        'std_similarity': float(np.std(similarities)),
        'median_similarity': float(np.median(similarities)),
        'min_similarity': float(np.min(similarities)),
        'max_similarity': float(np.max(similarities)),
        'similarity_5th': float(np.percentile(similarities, 5))
    }
    
    return metrics

def print_metrics(
    mod_metrics: Optional[Dict[str, Union[float, np.ndarray]]] = None,
    width_metrics: Optional[Dict[str, float]] = None,
    seq_metrics: Optional[Dict[str, float]] = None,
    total_score: Optional[float] = None,
    class_names: Optional[List[str]] = None
) -> None:
    """
    打印评估指标
    
    参数:
        mod_metrics: 调制分类指标
        width_metrics: 码元宽度指标
        seq_metrics: 码元序列指标
        total_score: 总评分
        class_names: 类别名称列表
    """
    print("\n" + "="*50)
    
    if mod_metrics:
        print("\n=== 调制分类指标 ===")
        print(f"准确率: {mod_metrics['mod_accuracy']:.4f}")
        print(f"精确率: {mod_metrics['mod_precision']:.4f}")
        print(f"召回率: {mod_metrics['mod_recall']:.4f}")
        print(f"F1分数: {mod_metrics['mod_f1']:.4f}")
        
        if class_names:
            print("\n--- 每个类别的准确率 ---")
            for i, acc in enumerate(mod_metrics['mod_per_class_accuracy']):
                print(f"{class_names[i]}: {acc:.4f}")
            
            print("\n--- 混淆矩阵 ---")
            cm = mod_metrics['mod_confusion_matrix']
            print("真实类别 (行) vs 预测类别 (列):")
            print("     ", end="")
            for name in class_names:
                print(f"{name:>8}", end="")
            print()
            for i, row in enumerate(cm):
                print(f"{class_names[i]:<5}", end="")
                for val in row:
                    print(f"{val:>8}", end="")
                print()
    
    if width_metrics:
        print("\n=== 码元宽度估计指标 ===")
        print(f"平均相对误差: {width_metrics['mean_error']:.4f}")
        print(f"误差标准差: {width_metrics['std_error']:.4f}")
        print(f"中位相对误差: {width_metrics['median_error']:.4f}")
        print(f"最小相对误差: {width_metrics['min_error']:.4f}")
        print(f"最大相对误差: {width_metrics['max_error']:.4f}")
        print(f"95%分位相对误差: {width_metrics['error_95th']:.4f}")
    
    if seq_metrics:
        print("\n=== 码元序列解调指标 ===")
        print(f"平均相似度: {seq_metrics['mean_similarity']:.4f}")
        print(f"相似度标准差: {seq_metrics['std_similarity']:.4f}")
        print(f"中位相似度: {seq_metrics['median_similarity']:.4f}")
        print(f"最小相似度: {seq_metrics['min_similarity']:.4f}")
        print(f"最大相似度: {seq_metrics['max_similarity']:.4f}")
        print(f"5%分位相似度: {seq_metrics['similarity_5th']:.4f}")
    
    if total_score is not None:
        print("\n=== 总评分 ===")
        print(f"加权总分: {total_score:.2f}")
    
    print("\n" + "="*50) 