import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Union, Optional

class CascadedLoss(nn.Module):
    """级联模型的损失函数"""
    def __init__(
        self,
        mt_weight: float = 0.4,  # 调制分类权重
        sw_weight: float = 0.3,  # 码元宽度权重
        cq_weight: float = 0.3,  # 码元序列权重
        label_smoothing: float = 0.1,
        reduction: str = 'mean'
    ) -> None:
        super().__init__()
        self.mt_weight = mt_weight
        self.sw_weight = sw_weight
        self.cq_weight = cq_weight
        
        # 调制分类损失
        self.mt_criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction=reduction
        )
        
        # 码元宽度损失
        self.sw_criterion = nn.MSELoss(reduction=reduction)
        
        # 码元序列损失
        self.cq_criterion = SymbolSequenceLoss(reduction=reduction)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        计算级联损失
        
        参数:
            outputs: 模型输出字典
                - modulation_type: 调制类型预测 [batch_size, num_classes]
                - symbol_width: 码元宽度预测 [batch_size, 1]
                - symbol_sequence: 码元序列预测 [batch_size, seq_len]
            targets: 目标值字典
                - modulation_type: 调制类型标签 [batch_size]
                - symbol_width: 码元宽度标签 [batch_size]
                - symbol_sequence: 码元序列标签 [batch_size, seq_len]
                
        返回:
            dict: 包含各个任务损失和总损失的字典
        """
        # 1. 调制分类损失
        mt_loss = self.mt_criterion(
            outputs['modulation_type'],
            targets['modulation_type']
        )
        
        # 获取正确的调制类型掩码
        correct_mod = (outputs['modulation_type'].argmax(dim=1) == targets['modulation_type'])
        
        # 2. 码元宽度损失 (只对调制类型正确的样本计算)
        if correct_mod.any():
            sw_loss = self.sw_criterion(
                outputs['symbol_width'][correct_mod].squeeze(1),
                targets['symbol_width'][correct_mod]
            )
        else:
            sw_loss = torch.tensor(0.0, device=mt_loss.device)
        
        # 3. 码元序列损失 (只对调制类型和码元宽度都正确的样本计算)
        if correct_mod.any():
            # 检查码元宽度误差
            width_error = torch.abs(
                outputs['symbol_width'][correct_mod].squeeze(1) - 
                targets['symbol_width'][correct_mod]
            ) / targets['symbol_width'][correct_mod]
            
            correct_width = width_error <= 0.1  # 允许10%的误差
            if correct_width.any():
                correct_mask = torch.zeros_like(correct_mod)
                correct_mask[correct_mod.nonzero().squeeze(1)[correct_width]] = 1
                
                cq_loss = self.cq_criterion(
                    outputs['symbol_sequence'][correct_mask],
                    targets['symbol_sequence'][correct_mask]
                )
            else:
                cq_loss = torch.tensor(0.0, device=mt_loss.device)
        else:
            cq_loss = torch.tensor(0.0, device=mt_loss.device)
        
        # 4. 计算加权总损失
        total_loss = (
            self.mt_weight * mt_loss +
            self.sw_weight * sw_loss +
            self.cq_weight * cq_loss
        )
        
        return {
            'total_loss': total_loss,
            'mt_loss': mt_loss,
            'sw_loss': sw_loss,
            'cq_loss': cq_loss
        }

class SymbolSequenceLoss(nn.Module):
    """码元序列损失函数"""
    def __init__(
        self,
        mse_weight: float = 0.5,
        cosine_weight: float = 0.5,
        reduction: str = 'mean'
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.reduction = reduction
        self.mse_criterion = nn.MSELoss(reduction=reduction)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # MSE损失
        mse_loss = self.mse_criterion(inputs, targets)
        
        # 余弦相似度损失
        cos_sim = F.cosine_similarity(inputs, targets, dim=1)
        cosine_loss = 1 - cos_sim
        
        if self.reduction == 'mean':
            cosine_loss = cosine_loss.mean()
        elif self.reduction == 'sum':
            cosine_loss = cosine_loss.sum()
        
        # 组合损失
        total_loss = (
            self.mse_weight * mse_loss +
            self.cosine_weight * cosine_loss
        )
        
        return total_loss

class PSKSequenceLoss(nn.Module):
    """PSK码元序列损失函数"""
    def __init__(
        self,
        constellation_points: int,
        reduction: str = 'mean'
    ) -> None:
        super().__init__()
        self.constellation_points = constellation_points
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 将预测和目标相位归一化到[0, constellation_points)范围
        pred_phase = (inputs + 1) * self.constellation_points / 2
        target_phase = (targets + 1) * self.constellation_points / 2
        
        # 计算相位差的最小值
        phase_diff = torch.abs(pred_phase - target_phase)
        phase_diff = torch.min(
            phase_diff,
            self.constellation_points - phase_diff
        )
        
        if self.reduction == 'mean':
            return phase_diff.mean()
        elif self.reduction == 'sum':
            return phase_diff.sum()
        else:  # 'none'
            return phase_diff

class QAMSequenceLoss(nn.Module):
    """QAM码元序列损失函数"""
    def __init__(
        self,
        constellation_points: int,
        reduction: str = 'mean'
    ) -> None:
        super().__init__()
        self.constellation_points = constellation_points
        self.reduction = reduction
        self.mse_criterion = nn.MSELoss(reduction=reduction)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 将预测和目标IQ值归一化到合适的范围
        scale = np.sqrt(self.constellation_points) / 2
        pred_iq = (inputs + 1) * scale
        target_iq = (targets + 1) * scale
        
        # 计算IQ平面上的欧氏距离
        return self.mse_criterion(pred_iq, target_iq)

class APSKSequenceLoss(nn.Module):
    """APSK码元序列损失函数"""
    def __init__(
        self,
        constellation_points: int,
        reduction: str = 'mean'
    ) -> None:
        super().__init__()
        self.constellation_points = constellation_points
        self.reduction = reduction
        self.mse_criterion = nn.MSELoss(reduction=reduction)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 将预测和目标幅度/相位值归一化到合适的范围
        scale = np.sqrt(self.constellation_points) / 2
        pred_polar = (inputs + 1) * scale
        target_polar = (targets + 1) * scale
        
        # 分别计算幅度和相位的损失
        amplitude_loss = self.mse_criterion(
            pred_polar[:, 0],
            target_polar[:, 0]
        )
        
        phase_diff = torch.abs(pred_polar[:, 1] - target_polar[:, 1])
        phase_diff = torch.min(
            phase_diff,
            self.constellation_points - phase_diff
        )
        
        if self.reduction == 'mean':
            phase_loss = phase_diff.mean()
        elif self.reduction == 'sum':
            phase_loss = phase_diff.sum()
        else:  # 'none'
            phase_loss = phase_diff
        
        return amplitude_loss + phase_loss

class MSKSequenceLoss(nn.Module):
    """MSK码元序列损失函数"""
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # MSK使用相位差来表示码元
        phase_diff = torch.abs(inputs - targets)
        phase_diff = torch.min(phase_diff, 2 - phase_diff)  # MSK相位差为±π/2
        
        if self.reduction == 'mean':
            return phase_diff.mean()
        elif self.reduction == 'sum':
            return phase_diff.sum()
        else:  # 'none'
            return phase_diff 