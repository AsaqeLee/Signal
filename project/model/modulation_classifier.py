import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class ModulationClassifier(nn.Module):
    """调制信号分类器"""
    def __init__(self, num_classes: int = 10, feature_dim: int = 256, input_size: int = 2000):
        super().__init__()
        
        # 计算每层卷积后的特征图大小
        l1_size = input_size // 2  # 第一次池化后
        l2_size = l1_size // 2     # 第二次池化后
        l3_size = l2_size // 2     # 第三次池化后
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            # 第一层卷积
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),  # 添加dropout
            
            # 第二层卷积
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),  # 添加dropout
            
            # 第三层卷积
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),  # 添加dropout
            
            # 第四层卷积
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        )
        
        # 调制类型分类
        self.mod_classifier = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        # 码元宽度预测（回归）
        self.width_predictor = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, 1),
            nn.Softplus()  # 确保宽度为正值
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, i_data: torch.Tensor, q_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播
        Args:
            i_data: I路数据 [batch_size, sequence_length]
            q_data: Q路数据 [batch_size, sequence_length]
        Returns:
            包含预测结果的字典
        """
        # 1. 组合IQ数据
        x = torch.stack([i_data, q_data], dim=1)  # [batch_size, 2, sequence_length]
        
        # 2. 提取特征
        features = self.feature_extractor(x)  # [batch_size, 512, 1]
        features = features.squeeze(-1)  # [batch_size, 512]
        
        # 3. 预测
        return {
            'modulation_type': self.mod_classifier(features),  # [batch_size, num_classes]
            'symbol_width': self.width_predictor(features)     # [batch_size, 1]
        }

class ModulationLoss:
    """调制信号分类的损失函数"""
    def __init__(self, config):
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss()
        self.huber_loss = nn.SmoothL1Loss()  # 对异常值更鲁棒
        
    def __call__(self, outputs: Dict[str, torch.Tensor], 
                 targets: Dict[str, Optional[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """计算损失
        Args:
            outputs: 模型输出的预测结果
            targets: 目标值
        Returns:
            包含各项损失的字典
        """
        losses = {}
        
        # 1. 调制类型损失（分类）
        if targets['modulation_type'] is not None:
            losses['mod_loss'] = self.ce_loss(
                outputs['modulation_type'],
                targets['modulation_type']
            )
        
        # 2. 码元宽度损失（回归）
        if targets['symbol_width'] is not None:
            # 使用Huber Loss，对异常值更鲁棒
            losses['width_loss'] = self.huber_loss(
                outputs['symbol_width'].squeeze(),
                targets['symbol_width']
            )
        
        # 3. 计算总损失
        total_loss = (
            self.config.MT_WEIGHT * losses.get('mod_loss', 0) +
            self.config.SW_WEIGHT * losses.get('width_loss', 0)
        )
        
        losses['total_loss'] = total_loss
        return losses 