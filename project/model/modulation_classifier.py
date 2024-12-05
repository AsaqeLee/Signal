import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Callable

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from project.config import Config
from project.model.modules import (
    SEBlock,
    DepthwiseSeparableConv1d,
    MultiScaleModule,
    AttentionPool1d
)
from project.model.base_model import BaseModel

class ResidualBlock(nn.Module):
    """增强版残差块"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        # 第一个卷积分支
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        
        # 第二个卷积���支（不同kernel size）
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels)
        )
        
        # SE注意力
        self.se = SEBlock(out_channels)
        
        # shortcut连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 两个卷积分支
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        
        # 融合两个分支
        out = out1 + out2
        
        # 注意力机制
        out = self.se(out)
        
        # 残差连接
        out += self.shortcut(x)
        out = self.relu(out)
        
        return out

class MultiTaskSignalModel(BaseModel):
    """多任务信号处理模型"""
    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)
        
        # 特征提取主干网络
        self.backbone = nn.Sequential(
            # Stem
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            # 残差层
            self._make_layer(64, 128, 3, stride=1),     # 3个残差块
            self._make_layer(128, 256, 4, stride=2),    # 4个残差块
            self._make_layer(256, 512, 6, stride=2),    # 6个残差块
            self._make_layer(512, 1024, 3, stride=2),   # 3个残差块
            
            # 多尺度特征融合
            MultiScaleModule(1024)
        )
        
        # 调制分类分支
        self.modulation_head = nn.Sequential(
            AttentionPool1d(1024),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, len(self.config.MODULATION_DICT))
        )
        
        # 码元宽度估计分支
        self.symbol_width_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Softplus()  # 确保输出为正数
        )
        
        # 码元序列解调分支
        self.symbol_sequence_head = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 1, kernel_size=1),  # 输出序列
            nn.Tanh()  # 将输出限制在[-1,1]范围
        )
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 提取特征
        features = self.backbone(x)
        
        # 调制分类
        mod_logits = self.modulation_head(features)
        
        # 码元宽度估计
        symbol_width = self.symbol_width_head(features)
        
        # 码元序列解调
        symbol_sequence = self.symbol_sequence_head(features)
        
        return {
            'modulation_type': mod_logits,
            'symbol_width': symbol_width,
            'symbol_sequence': symbol_sequence,
            'features': features
        }
    
    def get_loss_function(self) -> Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], torch.Tensor]:
        """获取多任务损失函数"""
        def criterion(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
            # 调制分类损失 (交叉熵 + 标签平滑)
            mod_targets = self._apply_label_smoothing(
                targets['modulation_type'],
                len(self.config.MODULATION_DICT),
                self.config.LABEL_SMOOTHING
            )
            mod_loss = -(mod_targets * 
                        F.log_softmax(outputs['modulation_type'], dim=1)).sum(dim=1).mean()
            
            # 码元宽度估计损失 (相对误差)
            width_loss = torch.abs(outputs['symbol_width'] - targets['symbol_width']) / targets['symbol_width']
            width_loss = width_loss.mean()
            
            # 码元序列解调损失 (余弦相似度)
            seq_loss = 1 - F.cosine_similarity(
                outputs['symbol_sequence'].squeeze(1),
                targets['symbol_sequence'],
                dim=1
            ).mean()
            
            # 特征正则化损失
            l2_loss = torch.norm(outputs['features'], p=2, dim=1).mean()
            
            # 总损失 (根据评分权重)
            total_loss = (
                0.2 * mod_loss +      # 调制分类 20%
                0.3 * width_loss +    # 码元宽度 30%
                0.5 * seq_loss +      # 码元序列 50%
                0.001 * l2_loss       # 正则化
            )
            
            return total_loss
        
        return criterion
    
    def _apply_label_smoothing(self, targets: torch.Tensor, num_classes: int, smoothing: float = 0.1) -> torch.Tensor:
        """标签平滑"""
        with torch.no_grad():
            targets = targets.reshape(-1)
            targets_one_hot = torch.zeros(
                (targets.size(0), num_classes), 
                dtype=torch.float32,
                device=targets.device
            )
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
            targets_smooth = (1.0 - smoothing) * targets_one_hot + \
                           smoothing / num_classes
        return targets_smooth
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)