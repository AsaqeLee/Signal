import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import logging

from project.config import Config
from .modules import SEBlock, MultiScaleModule, AttentionPool1d

class ConvBlock(nn.Module):
    """卷积块"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.2
    ) -> None:
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batch_norm
        )
        
        self.bn = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.se = SEBlock(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        return self.dropout(x)

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.2
    ) -> None:
        super().__init__()
        
        self.conv1 = ConvBlock(
            channels,
            channels,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate
        )
        
        self.conv2 = ConvBlock(
            channels,
            channels,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate
        )
        
        self.se = SEBlock(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv2(self.conv1(x))
        out = self.se(out)
        return out + identity

class AttentionBlock(nn.Module):
    """注意力块"""
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout_rate: float = 0.2
    ) -> None:
        super().__init__()
        
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            num_heads=num_heads,
            dropout=dropout_rate
        )
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 转换维度: [batch, channels, length] -> [length, batch, channels]
        x = x.transpose(1, 2).transpose(0, 1)
        x = self.norm(x)
        
        # 自注意力
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        
        # 转换回原始维度: [length, batch, channels] -> [batch, channels, length]
        return x.transpose(0, 1).transpose(1, 2)

class DualTaskModel(nn.Module):
    """双任务模型 - 调制分类和码元宽度估计"""
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        # 特征提取主干网络
        self.encoder = nn.ModuleList()
        in_channels = 2  # I/Q两个通道
        
        for channels in config.BACKBONE_CHANNELS:
            self.encoder.append(
                nn.Sequential(
                    ConvBlock(
                        in_channels,
                        channels,
                        kernel_size=7 if in_channels == 2 else 3,
                        stride=2,
                        padding=3 if in_channels == 2 else 1,
                        dropout_rate=config.DROPOUT_RATE
                    ),
                    ResidualBlock(
                        channels,
                        dropout_rate=config.DROPOUT_RATE
                    ),
                    MultiScaleModule(channels),
                    AttentionBlock(
                        channels,
                        dropout_rate=config.DROPOUT_RATE
                    )
                )
            )
            in_channels = channels
        
        # 调制分类头
        self.modulation_classifier = nn.Sequential(
            AttentionPool1d(config.BACKBONE_CHANNELS[-1]),
            nn.Flatten(),
            nn.Linear(config.BACKBONE_CHANNELS[-1], 512),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(512, len(config.MODULATION_DICT))
        )
        
        # 码元宽度估计头
        self.symbol_width_estimator = nn.Sequential(
            AttentionPool1d(config.BACKBONE_CHANNELS[-1]),
            nn.Flatten(),
            nn.Linear(config.BACKBONE_CHANNELS[-1], 256),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(64, 1),
            nn.Softplus()  # 确保输出为正值
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module) -> None:
        """初始化模型权重"""
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            x: 输入信号 [batch_size, 2, sequence_length]
            
        返回:
            modulation_logits: 调制分类logits [batch_size, num_classes]
            symbol_width: 码元宽度预测值 [batch_size, 1]
        """
        # 特征提取
        features = x
        for encoder_block in self.encoder:
            features = encoder_block(features)
        
        # 调制分类
        modulation_logits = self.modulation_classifier(features)
        
        # 码元宽度估计
        symbol_width = self.symbol_width_estimator(features)
        
        return modulation_logits, symbol_width
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        模型预测
        
        参数:
            x: 输入信号 [batch_size, 2, sequence_length]
            
        返回:
            modulation_pred: 调制类型预测 [batch_size]
            symbol_width: 码元宽度预测值 [batch_size, 1]
        """
        modulation_logits, symbol_width = self.forward(x)
        modulation_pred = torch.argmax(modulation_logits, dim=1)
        return modulation_pred, symbol_width