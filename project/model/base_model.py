import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path
import os
import logging

from project.config import Config

class BaseModel(nn.Module):
    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__()
        self.config = config if config is not None else Config()
    
    def save_model(self, path: str) -> None:
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': self.get_config()
        }, path)
    
    def load_model(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
    
    def get_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            'name': self.__class__.__name__,
            'params': self.config.__dict__
        }
    
    def to_device(self) -> None:
        """将模型移动到指定设备"""
        self.to(self.config.DEVICE)
    
    def _initialize_weights(self) -> None:
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # 使用较小的标准差进行初始化
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias) 