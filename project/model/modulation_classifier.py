import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from project.config import Config
from project.model.base_model import BaseModel

class SEBlock(nn.Module):
    """Squeeze-and-Excitation块"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableConv1d(nn.Module):
    """深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv1d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = DepthwiseSeparableConv1d(
            out_channels, out_channels, kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ModulationClassifier(BaseModel):
    def __init__(self):
        super().__init__()
        
        # 特征提取
        self.features = nn.Sequential(
            # 第一层卷积块
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # 第二层卷积块
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # 第三层卷积块 (添加残差连接)
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # 注意力层
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 残差连接
        self.shortcut = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256)
        )
        
        # 调制类型分类
        self.modulation_classifier = nn.Sequential(
            nn.Linear(256, self.config.FEATURE_DIM),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT_RATE),
            nn.Linear(self.config.FEATURE_DIM, self.config.NUM_CLASSES)  # 使用NUM_CLASSES
        )
        
        # 码元宽度预测
        self.width_regressor = nn.Sequential(
            nn.Linear(256, self.config.FEATURE_DIM),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT_RATE),
            nn.Linear(self.config.FEATURE_DIM, 1),
            nn.Softplus()  # 确保输出为正值
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:  # 检查bias是否存在
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:  # 检查bias是否存在
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:  # 检查bias是否存在
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 特征提取
        x = self.features(x)  # 输出形状: [batch_size, channels, 1]
        x = x.squeeze(-1)     # 移除最后一个维度，变为 [batch_size, channels]
        
        # 预测
        return {
            'modulation_type': self.modulation_classifier(x),
            'symbol_width': self.width_regressor(x)
        }
    
    def get_loss_function(self):
        """获取损失函数"""
        def criterion(outputs, targets):
            # 调制类型损失（带标签平滑和焦点损失的组合）
            mod_loss = self._focal_loss_with_smoothing(
                outputs['modulation_type'],
                targets['modulation_type'],
                smoothing=0.1,
                gamma=2.0
            )
            
            # 码元宽度损失（相对误差 + Huber损失的组合）
            width_loss = self._width_loss(
                outputs['symbol_width'].squeeze(),
                targets['symbol_width']
            )
            
            # 总损失
            total_loss = (
                self.config.MT_WEIGHT * mod_loss +
                self.config.SW_WEIGHT * width_loss
            )
            
            return total_loss
        
        return criterion
    
    def _focal_loss_with_smoothing(self, pred, target, smoothing=0.1, gamma=2.0):
        """带标签平滑的焦点损失"""
        n_classes = pred.size(1)
        
        # 标签平滑
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - smoothing) + smoothing / n_classes
        
        # 计算概率
        probs = F.softmax(pred, dim=1)
        log_probs = F.log_softmax(pred, dim=1)
        
        # 焦点损失
        pt = (smooth_one_hot * probs).sum(dim=1)
        focal_weight = (1 - pt) ** gamma
        
        loss = (-smooth_one_hot * log_probs).sum(dim=1)
        focal_loss = focal_weight * loss
        
        return focal_loss.mean()
    
    def _width_loss(self, pred, target, beta=0.1):
        """组合码元宽度损失"""
        # 相对误差
        relative_error = torch.abs(pred - target) / target
        
        # Huber损失
        huber_loss = F.smooth_l1_loss(pred, target, beta=beta)
        
        # L1正则化
        l1_reg = torch.abs(pred).mean()
        
        # 组合损失
        return relative_error.mean() + huber_loss + 0.01 * l1_reg