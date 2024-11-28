import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from project.config import Config
from project.model.base_model import BaseModel

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
            nn.Linear(self.config.FEATURE_DIM, len(self.config.MODULATION_DICT))
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 特征提取
        features = self.features(x)
        features = features.squeeze(-1)
        
        # 预测
        return {
            'modulation_type': self.modulation_classifier(features),
            'symbol_width': self.width_regressor(features)
        }
    
    def get_loss_function(self):
        """获取损失函数"""
        def criterion(outputs, targets):
            # 调制类型损失（带标签平滑的交叉熵）
            mod_loss = self._label_smoothing_loss(
                outputs['modulation_type'],
                targets['modulation_type'],
                smoothing=0.1
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
    
    def _label_smoothing_loss(self, pred, target, smoothing=0.1):
        """带标签平滑的交叉熵损失"""
        n_classes = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - smoothing) + smoothing / n_classes
        log_prob = F.log_softmax(pred, dim=1)
        return (-smooth_one_hot * log_prob).sum(dim=1).mean()
    
    def _width_loss(self, pred, target, beta=0.1):
        """组合码元宽度损失"""
        # 相对误差
        relative_error = torch.abs(pred - target) / target
        
        # Huber损失
        huber_loss = F.smooth_l1_loss(pred, target, beta=beta)
        
        # 组合损失
        return relative_error.mean() + huber_loss