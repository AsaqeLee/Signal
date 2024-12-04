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

class BinaryModulationClassifier(BaseModel):
    """二分类调制分类器"""
    def __init__(self, config):
        super().__init__(config)
        
        # 特征提取层
        self.features = nn.Sequential(
            # 第一层卷积
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # 第二层卷积
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # 第三层卷积
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # 第四层卷积
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # 全局平均池化
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(512, self.config.FEATURE_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.DROPOUT_RATE),
            nn.Linear(self.config.FEATURE_DIM, 1)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def forward(self, x):
        # 特征提取
        features = self.features(x)
        features = features.squeeze(-1)
        
        # 分类预测
        logits = self.classifier(features)
        
        return {
            'logits': logits.squeeze(-1)
        }

class SymbolWidthRegressor(BaseModel):
    """码元宽度回归模型"""
    def __init__(self, config):
        super().__init__(config)
        
        # 特征提取
        self.features = nn.Sequential(
            # 第一层卷积块
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.MaxPool1d(2),
            
            # 第二层卷积块
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.MaxPool1d(2),
            
            # 第三层卷积块
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool1d(2),
            
            # 第四层卷积块
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool1d(2),
            
            # 注意力层
            SEBlock(512),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 回归器
        self.regressor = nn.Sequential(
            nn.Linear(512, config.FEATURE_DIM),
            nn.BatchNorm1d(config.FEATURE_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(config.FEATURE_DIM, config.FEATURE_DIM // 2),
            nn.BatchNorm1d(config.FEATURE_DIM // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(config.FEATURE_DIM // 2, 1),
            nn.Softplus()  # 确保输出为正数
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def forward(self, x):
        # 特征提取
        x = self.features(x)
        x = x.squeeze(-1)
        
        # 回归预测
        pred = self.regressor(x)
        
        return {
            'symbol_width': pred.squeeze()
        }

class ModulationClassifierEnsemble(BaseModel):
    """调制分类器集成模型"""
    def __init__(self, config=None):
        super().__init__(config)
        self.classifiers = nn.ModuleDict()
        
        # 为每种调制类型创建一个二分类器
        for mod_type, mod_name in self.config.MODULATION_DICT.items():
            classifier = BinaryModulationClassifier(self.config)
            self.classifiers[mod_name] = classifier
        
        # 初始化温度参数（可训练）
        if self.config.USE_TEMPERATURE_SCALING:
            self.temperature = nn.Parameter(torch.ones(1) * self.config.TEMPERATURE)
    
    def forward(self, x):
        # 收集每个分类器的预测结果
        predictions = {}
        confidences = {}
        raw_outputs = {}
        
        for mod_name, classifier in self.classifiers.items():
            # 获取每个分类器的预测
            outputs = classifier(x)
            logits = outputs['logits']
            
            # 应用温度缩放
            if self.config.USE_TEMPERATURE_SCALING:
                scaled_logits = logits / self.temperature
                probs = torch.sigmoid(scaled_logits)
            else:
                probs = torch.sigmoid(logits)
            
            predictions[mod_name] = (probs > self.config.CONFIDENCE_THRESHOLD)
            confidences[mod_name] = probs
            raw_outputs[mod_name] = logits
        
        # 计算一致性得分
        consistency_scores = {}
        if self.config.USE_CONSISTENCY_SCORE:
            for mod_name in self.config.MODULATION_DICT.values():
                other_scores = []
                for other_name, other_conf in confidences.items():
                    if other_name != mod_name:
                        agreement = 1 - torch.abs(confidences[mod_name] - (1 - other_conf)).mean()
                        other_scores.append(agreement)
                
                if other_scores:
                    consistency_scores[mod_name] = torch.stack(other_scores).mean()
                else:
                    consistency_scores[mod_name] = torch.tensor(0.0).to(x.device)
        
        # 计算最终得分
        final_scores = {}
        for mod_name in self.config.MODULATION_DICT.values():
            base_confidence = confidences[mod_name].mean()
            
            if self.config.USE_CONSISTENCY_SCORE:
                consistency = consistency_scores[mod_name]
                # 结合置信度和一致性得分
                final_scores[mod_name] = base_confidence * (0.7 + 0.3 * consistency)
            else:
                final_scores[mod_name] = base_confidence
        
        # 找到最高得分的预测
        max_score = -float('inf')
        predicted_mod = None
        
        for mod_name, score in final_scores.items():
            if score > max_score:
                max_score = score
                predicted_mod = mod_name
        
        return {
            'predictions': predictions,
            'confidences': confidences,
            'raw_outputs': raw_outputs,
            'consistency_scores': consistency_scores if self.config.USE_CONSISTENCY_SCORE else None,
            'final_scores': final_scores,
            'predicted_mod': predicted_mod,
            'max_confidence': max_score
        }
    
    def train_single_classifier(self, mod_name, data, targets):
        """训练单个分类器"""
        classifier = self.classifiers[mod_name]
        outputs = classifier(data)
        
        # 使用binary_cross_entropy_with_logits
        loss = F.binary_cross_entropy_with_logits(
            outputs['logits'],
            targets['modulation_type']
        )
        
        return loss
    
    def calibrate_temperature(self, val_loader):
        """使用验证集校准温度参数"""
        if not self.config.USE_TEMPERATURE_SCALING:
            return
        
        self.eval()
        nll_criterion = nn.CrossEntropyLoss()
        
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                data = batch['data'].to(self.config.DEVICE)
                labels = batch['modulation_type'].to(self.config.DEVICE)
                
                outputs = self(data)
                # 收集所有分类器的logits
                for mod_name, logits in outputs['raw_outputs'].items():
                    logits_list.append(logits)
                    labels_list.append((labels == self.config.MODULATION_DICT.index(mod_name)).float())
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        # 优化温度参数
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        
        print(f"校准后的温度参数: {self.temperature.item():.3f}")
    
    def save_classifiers(self, path):
        """保存所有分类器"""
        for mod_name, classifier in self.classifiers.items():
            save_path = path / f"{mod_name}_classifier.pth"
            classifier.save_model(str(save_path))
        
        # 保存温度参数
        if self.config.USE_TEMPERATURE_SCALING:
            temp_path = path / "temperature.pth"
            torch.save({'temperature': self.temperature}, str(temp_path))
    
    def load_classifiers(self, path):
        """加载所有分类器"""
        for mod_name, classifier in self.classifiers.items():
            load_path = path / f"{mod_name}_classifier.pth"
            if load_path.exists():
                classifier.load_model(str(load_path))
        
        # 加载温度参数
        if self.config.USE_TEMPERATURE_SCALING:
            temp_path = path / "temperature.pth"
            if temp_path.exists():
                temp_state = torch.load(str(temp_path))
                self.temperature.data = temp_state['temperature']