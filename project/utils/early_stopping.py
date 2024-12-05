import numpy as np
from typing import Dict, Any, Optional, Union
import logging
import os
from pathlib import Path

class EarlyStopping:
    """早停类，用于防止过拟合"""
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0,
        mode: str = 'max',
        monitor: str = 'total_score'
    ) -> None:
        """
        参数:
            patience (int): 在触发早停之前等待的轮数
            min_delta (float): 最小改善阈值
            mode (str): 'min' 用于监控最小化指标(如损失), 'max' 用于监控最大化指标(如分数)
            monitor (str): 监控的指标名称
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor
        self.counter = 0
        self.best_value: Optional[float] = None
        self.early_stop = False
        
        # 记录每个指标的最佳值
        self.best_metrics: Dict[str, float] = {}
        
    def __call__(self, metrics: Union[float, Dict[str, float]]) -> bool:
        """
        检查是否应该触发早停
        
        参数:
            metrics: 当前监控的指标值，可以是单个值或字典
            
        返回:
            bool: 如果应该停止则返回True，否则返回False
        """
        # 如果输入是字典，获取监控的指标值
        if isinstance(metrics, dict):
            if self.monitor not in metrics:
                raise ValueError(f"监控的指标 '{self.monitor}' 不在提供的指标中")
            value = metrics[self.monitor]
            
            # 更新所有指标的最佳值
            for metric_name, metric_value in metrics.items():
                if metric_name not in self.best_metrics or self._is_better(metric_value, self.best_metrics[metric_name]):
                    self.best_metrics[metric_name] = metric_value
        else:
            value = metrics
        
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self._is_better(value, self.best_value):
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False
    
    def _is_better(self, current: float, previous: float) -> bool:
        """
        判断当前值是否比之前的值更好
        
        参数:
            current: 当前值
            previous: 之前的值
            
        返回:
            bool: 如果当前值更好则返回True，否则返回False
        """
        if self.mode == 'min':
            return current < previous - self.min_delta
        else:  # mode == 'max'
            return current > previous + self.min_delta
    
    def get_best_metrics(self) -> Dict[str, float]:
        """获取所有指标的最佳值"""
        return self.best_metrics
    
    def get_best_value(self) -> float:
        """获取监控指标的最佳值"""
        return self.best_value if self.best_value is not None else float('-inf' if self.mode == 'max' else 'inf')
    
    def reset(self) -> None:
        """重置早停状态"""
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_metrics = {}
  