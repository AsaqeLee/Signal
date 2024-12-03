import torch
import torch.nn as nn
from typing import Dict, Any
from project.config import Config

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
    
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