import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import logging

class DualTaskLoss(nn.Module):
    """双任务损失函数"""
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.mt_weight = config.MT_WEIGHT
        self.sw_weight = config.SW_WEIGHT
        self.label_smoothing = config.LABEL_SMOOTHING
    
    def forward(
        self,
        modulation_logits: torch.Tensor,
        symbol_width: torch.Tensor,
        modulation_target: torch.Tensor,
        width_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算损失
        
        参数:
            modulation_logits: 调制分类logits [batch_size, num_classes]
            symbol_width: 码元宽度预测值 [batch_size, 1]
            modulation_target: 调制类型标签 [batch_size]
            width_target: 码元宽度标签 [batch_size]
            
        返回:
            total_loss: 总损失
            losses: 各项损失的字典
        """
        # 调制分类损失
        modulation_loss = F.cross_entropy(
            modulation_logits,
            modulation_target,
            label_smoothing=self.label_smoothing
        )
        
        # 码元宽度损失
        width_loss = F.mse_loss(
            symbol_width.squeeze(),
            width_target
        )
        
        # 计算相对误差用于监控
        with torch.no_grad():
            relative_error = torch.abs(symbol_width.squeeze() - width_target) / width_target
            width_accuracy = torch.mean((relative_error <= 0.05).float())
        
        # 总损失
        total_loss = self.mt_weight * modulation_loss + self.sw_weight * width_loss
        
        # 返回总损失和各项损失
        return total_loss, {
            'total_loss': total_loss.item(),
            'modulation_loss': modulation_loss.item(),
            'width_loss': width_loss.item(),
            'width_accuracy': width_accuracy.item(),
            'relative_error': relative_error.mean().item()
        }

class FocalLoss(nn.Module):
    """Focal Loss for 调制分类"""
    def __init__(self, alpha: float = 1, gamma: float = 2) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class AdaptiveWingLoss(nn.Module):
    """Adaptive Wing Loss for 码元宽度回归"""
    def __init__(self, omega: float = 14, theta: float = 0.5, epsilon: float = 1) -> None:
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Adaptive Wing Loss
        
        参数:
            pred: 预测值 [batch_size, 1]
            target: 目标值 [batch_size]
            
        返回:
            loss: 损失值
        """
        y = target.unsqueeze(1) if target.dim() == 1 else target
        delta_y = (pred - y).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.epsilon, 2-self.theta))
        C = self.theta - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, 2-self.theta))
        loss2 = C + self.theta * delta_y2
        
        return (loss1.sum() + loss2.sum()) / (len(delta_y1) + len(delta_y2)) 