import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def calculate_metrics(y_true, y_pred, average='macro'):
    """
    计算分类指标
    
    参数:
        y_true (array-like): 真实标签
        y_pred (array-like): 预测标签
        average (str): 多分类指标的平均方式 ('micro', 'macro', 'weighted')
        
    返回:
        dict: 包含各种指标的字典
    """
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
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'per_class_accuracy': per_class_accuracy
    }
    
    return metrics

def print_metrics(metrics, class_names=None):
    """
    打印评估指标
    
    参数:
        metrics (dict): calculate_metrics返回的指标字典
        class_names (list): 类别名称列表
    """
    print("\n=== 分类指标 ===")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1']:.4f}")
    
    print("\n=== 混淆矩阵 ===")
    cm = metrics['confusion_matrix']
    if class_names:
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
    else:
        print(cm)
    
    print("\n=== 每个类别的准确率 ===")
    for i, acc in enumerate(metrics['per_class_accuracy']):
        class_name = class_names[i] if class_names else f"类别 {i}"
        print(f"{class_name}: {acc:.4f}") 