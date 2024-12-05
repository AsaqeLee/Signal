import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

from project.config import Config

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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.bn(self.conv(x))))

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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv2(self.conv1(x))
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

class ModulationClassifier(nn.Module):
    """调制分类器"""
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        # 特征提取
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
                    AttentionBlock(
                        channels,
                        dropout_rate=config.DROPOUT_RATE
                    )
                )
            )
            in_channels = channels
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(config.BACKBONE_CHANNELS[-1], 512),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(512, config.get_num_classes())
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 特征提取
        features = x
        for encoder_block in self.encoder:
            features = encoder_block(features)
        
        # 分类
        logits = self.classifier(features)
        
        return logits, features

class SymbolWidthEstimator(nn.Module):
    """码元宽度估计器"""
    def __init__(self, in_channels: int, config: Config) -> None:
        super().__init__()
        
        self.estimator = nn.Sequential(
            nn.Conv1d(in_channels + config.get_num_classes(), 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(64, 1),
            nn.Softplus()  # 确保输出为正值
        )
    
    def forward(self, features: torch.Tensor, mod_type: torch.Tensor) -> torch.Tensor:
        # 将调制类型概率与特征融合
        batch_size = features.size(0)
        # mod_type已经是概率分布,直接使用
        mod_type_expanded = mod_type.unsqueeze(-1).expand(-1, -1, features.size(-1))
        combined_features = torch.cat([features, mod_type_expanded], dim=1)
        
        return self.estimator(combined_features)

class SymbolDemodulator(nn.Module):
    """码元解调器"""
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        # 为每种调制类型创建专门的解调器
        self.demodulators = nn.ModuleDict()
        for mod_type, mod_name in config.MODULATION_DICT.items():
            if mod_name in ['BPSK', 'QPSK', '8PSK']:
                self.demodulators[str(mod_type)] = PSKDemodulator(
                    config.BACKBONE_CHANNELS[-1],
                    2 ** (1 if mod_name == 'BPSK' else 2 if mod_name == 'QPSK' else 3),
                    config.DROPOUT_RATE
                )
            elif 'QAM' in mod_name:
                m = int(mod_name.replace('QAM', ''))
                self.demodulators[str(mod_type)] = QAMDemodulator(
                    config.BACKBONE_CHANNELS[-1],
                    m,
                    config.DROPOUT_RATE
                )
            elif 'APSK' in mod_name:
                m = int(mod_name.replace('APSK', ''))
                self.demodulators[str(mod_type)] = APSKDemodulator(
                    config.BACKBONE_CHANNELS[-1],
                    m,
                    config.DROPOUT_RATE
                )
            elif mod_name == 'MSK':
                self.demodulators[str(mod_type)] = MSKDemodulator(
                    config.BACKBONE_CHANNELS[-1],
                    config.DROPOUT_RATE
                )
    
    def forward(
        self,
        features: torch.Tensor,
        mod_type: torch.Tensor,
        symbol_width: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = features.size(0)
        outputs = []
        
        # 对每个样本使用对应的解调器
        for i in range(batch_size):
            mod_idx = mod_type[i].argmax().item() + 1  # 转换回1-based索引
            demodulator = self.demodulators[str(mod_idx)]
            
            # 获取当前样本的mask
            current_mask = sequence_mask[i] if sequence_mask is not None else None
            
            output = demodulator(
                features[i:i+1],
                symbol_width[i:i+1],
                current_mask
            )
            outputs.append(output)
        
        # 将所有输出填充到相同长度
        max_length = max(output.size(-1) for output in outputs)
        padded_outputs = []
        
        for output in outputs:
            if output.size(-1) < max_length:
                padding = torch.zeros(
                    output.size(0),
                    output.size(1),
                    max_length - output.size(-1),
                    device=output.device,
                    dtype=output.dtype
                )
                output = torch.cat([output, padding], dim=-1)
            padded_outputs.append(output)
        
        return torch.cat(padded_outputs, dim=0)

class PSKDemodulator(nn.Module):
    """PSK解调器"""
    def __init__(self, in_channels: int, constellation_points: int, dropout_rate: float) -> None:
        super().__init__()
        
        self.constellation_points = constellation_points
        self.phase_estimator = nn.Sequential(
            nn.Conv1d(in_channels, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(128, 1, 1),
            nn.Tanh()  # 输出相位,范围[-1,1]
        )
    
    def forward(self, features: torch.Tensor, symbol_width: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 估计相位
        phase = self.phase_estimator(features)
        
        # 将相位映射到[0, M)范围
        phase = (phase + 1) * self.constellation_points / 2
        
        return phase  # [batch, 1, length]

class QAMDemodulator(nn.Module):
    """QAM解调器"""
    def __init__(self, in_channels: int, constellation_points: int, dropout_rate: float) -> None:
        super().__init__()
        
        self.constellation_points = constellation_points
        self.iq_estimator = nn.Sequential(
            nn.Conv1d(in_channels, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(128, 1, 1),  # 输出I/Q分量
            nn.Tanh()  # 输出范围[-1,1]
        )
    
    def forward(self, features: torch.Tensor, symbol_width: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 估计I/Q分量
        iq = self.iq_estimator(features)
        
        # 将I/Q分量映射到[0, sqrt(M))范围
        scale = torch.tensor(np.sqrt(self.constellation_points), device=iq.device, dtype=iq.dtype)
        iq = (iq + 1) * scale / 2
        
        return iq  # [batch, 1, length]

class APSKDemodulator(nn.Module):
    """APSK解调器"""
    def __init__(self, in_channels: int, constellation_points: int, dropout_rate: float) -> None:
        super().__init__()
        
        self.constellation_points = constellation_points
        self.polar_estimator = nn.Sequential(
            nn.Conv1d(in_channels, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(128, 1, 1),  # 输出幅度和相位
            nn.Tanh()  # 输出范围[-1,1]
        )
    
    def forward(self, features: torch.Tensor, symbol_width: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 估计幅度和相位
        polar = self.polar_estimator(features)
        
        # 将输出范围从[-1,1]映射到[0, sqrt(M)]
        scale = torch.tensor(np.sqrt(self.constellation_points), device=polar.device, dtype=polar.dtype)
        polar = (polar + 1) * scale / 2
        
        return polar  # [batch, 1, length]

class MSKDemodulator(nn.Module):
    """MSK解调器"""
    def __init__(self, in_channels: int, dropout_rate: float) -> None:
        super().__init__()
        
        self.phase_estimator = nn.Sequential(
            nn.Conv1d(in_channels, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(128, 1, 1),
            nn.Tanh()  # 输出相位,范围[-1,1]
        )
    
    def forward(self, features: torch.Tensor, symbol_width: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 估计相位
        phase = self.phase_estimator(features)
        
        # 将相位映射到[0,1]范围
        phase = (phase + 1) / 2
        
        return phase  # [batch, 1, length]

class MultiTaskModel(nn.Module):
    """多任务信号处理模型"""
    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__()
        self.config = config if config is not None else Config()
        
        # 调制分类器
        self.modulation_classifier = ModulationClassifier(self.config)
        
        # 码元宽度估计器
        self.symbol_width_estimator = SymbolWidthEstimator(
            self.config.BACKBONE_CHANNELS[-1],
            self.config
        )
        
        # 码元解调器
        self.symbol_demodulator = SymbolDemodulator(self.config)
        
        # 初始化权重
        self._initialize_weights()
    
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
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 获取输入数据
        x = batch['data']
        
        # 调制分类
        mod_logits, features = self.modulation_classifier(x)
        mod_probs = F.softmax(mod_logits, dim=-1)
        
        # 码元宽度估计
        symbol_width = self.symbol_width_estimator(features, mod_probs)
        
        # 码元解调
        sequence_mask = batch.get('targets', {}).get('sequence_mask', None)
        symbol_sequence = self.symbol_demodulator(
            features,
            mod_probs,
            symbol_width,
            sequence_mask
        )
        
        return {
            'modulation_type': mod_logits,
            'symbol_width': symbol_width,
            'symbol_sequence': symbol_sequence
        }
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # 调制分类损失
        mod_loss = F.cross_entropy(
            outputs['modulation_type'],
            targets['modulation_type']
        )
        
        # 码元宽度估计损失
        width_loss = F.mse_loss(
            outputs['symbol_width'].squeeze(),
            targets['symbol_width']
        )
        
        # 码元序列损失 - 只计算有效位置的损失
        sequence_mask = targets.get('sequence_mask', None)
        if sequence_mask is not None:
            # 展平预测和目标
            pred_seq = outputs['symbol_sequence'].squeeze(1)  # 移除通道维度
            true_seq = targets['symbol_sequence']
            
            # 计算每个样本的序列长度
            seq_lengths = sequence_mask.sum(dim=1)
            
            # 只计算有效位置的损失
            sequence_loss = 0
            batch_size = len(seq_lengths)
            for i in range(batch_size):
                valid_length = seq_lengths[i].item()
                # 确保预测序列长度足够
                if pred_seq.size(1) < valid_length:
                    # 如果预测序列太短,只使用可用部分
                    valid_length = pred_seq.size(1)
                sequence_loss += F.mse_loss(
                    pred_seq[i, :valid_length],
                    true_seq[i, :valid_length]
                )
            sequence_loss = sequence_loss / batch_size
        else:
            # 如果没有掩码,使用整个序列
            pred_seq = outputs['symbol_sequence'].squeeze(1)  # 移除通道维度
            true_seq = targets['symbol_sequence']
            # 使用较短的长度
            min_length = min(pred_seq.size(1), true_seq.size(1))
            sequence_loss = F.mse_loss(
                pred_seq[:, :min_length],
                true_seq[:, :min_length]
            )
        
        # 总损失 - 使用配置中的任务权重
        total_loss = (
            self.config.MT_WEIGHT * mod_loss +
            self.config.SW_WEIGHT * width_loss +
            self.config.CQ_WEIGHT * sequence_loss
        )
        
        return {
            'total_loss': total_loss,
            'modulation_loss': mod_loss,
            'width_loss': width_loss,
            'sequence_loss': sequence_loss
        }
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 前向传播
        outputs = self(batch)
        
        # 获取预测结果
        mod_type = outputs['modulation_type'].argmax(dim=1)
        symbol_width = outputs['symbol_width'].squeeze()
        symbol_sequence = outputs['symbol_sequence']
        
        # 如果有sequence_mask,只返回有效长度的序列
        sequence_mask = batch.get('targets', {}).get('sequence_mask', None)
        if sequence_mask is not None:
            seq_lengths = sequence_mask.sum(dim=1)
            valid_sequences = []
            for i in range(len(seq_lengths)):
                valid_length = seq_lengths[i].item()
                valid_sequences.append(symbol_sequence[i, :valid_length])
            symbol_sequence = valid_sequences
        
        return {
            'modulation_type': mod_type,
            'symbol_width': symbol_width,
            'symbol_sequence': symbol_sequence
        }