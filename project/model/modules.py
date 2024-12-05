"""模型组件模块"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import logging
import os
from pathlib import Path

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableConv1d(nn.Module):
    """深度可分离卷积"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MultiScaleModule(nn.Module):
    """多尺度特征融合模块"""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(channels, channels // 4, kernel_size=k, padding=k//2)
            for k in [1, 3, 5, 7]
        ])
        self.fuse = nn.Conv1d(channels, channels, 1)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = []
        for branch in self.branches:
            branches.append(branch(x))
        out = torch.cat(branches, dim=1)
        out = self.fuse(out)
        out = self.norm(out)
        out = self.act(out)
        return out + x  # 残差连接

class AttentionPool1d(nn.Module):
    """注意力池化层"""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels // 8, 1),
            nn.BatchNorm1d(channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 8, 1, 1),
            nn.Sigmoid()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(x)
        out = x * weights
        out = self.pool(out)
        return out