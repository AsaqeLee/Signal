import numpy as np

class EarlyStopping:
    """早停类，用于防止过拟合"""
    def __init__(self, patience=7, min_delta=0, mode='min'):
        """
        参数:
            patience (int): 在触发早停之前等待的轮数
            min_delta (float): 最小改善阈值
            mode (str): 'min' 用于监控最小化指标(如损失), 'max' 用于监控最大化指标(如准确率)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
    def __call__(self, value):
        """
        检查是否应该触发早停
        
        参数:
            value (float): 当前监控的指标值
            
        返回:
            bool: 如果应该停止则返回True，否则返回False
        """
        if self.best_value is None:
            self.best_value = value
            return False
            
        if self.mode == 'min':
            if value < self.best_value - self.min_delta:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if value > self.best_value + self.min_delta:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            return True
            
        return False
        
    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_value = None
        self.early_stop = False 