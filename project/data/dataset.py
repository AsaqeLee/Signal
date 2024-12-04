import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import pickle
import logging
from project.config import Config
import pandas as pd
from torchvision.transforms import Compose

class ModulationDataset(Dataset):
    """调制信号数据集"""
    def __init__(self, data_dir, transform=None, sequence_length=2048):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.sequence_length = sequence_length
        self.samples = []
        
        # 加载所有数据文件
        for mod_type, mod_name in Config.MODULATION_DICT.items():
            mod_dir = self.data_dir / mod_name
            if not mod_dir.exists():
                raise RuntimeError(f"调制类型目录不存在: {mod_dir}")
            
            for file_path in mod_dir.glob("*.csv"):
                self.samples.append({
                    'path': file_path,
                    'modulation_type': mod_type,
                    'mod_name': mod_name
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 读取IQ数据
        df = pd.read_csv(sample['path'], header=None)
        i_data = df[0].values
        q_data = df[1].values
        
        # 统一序列长度
        if len(i_data) > self.sequence_length:
            # 如果序列太长，随机选择一个起点进行裁剪
            start_idx = np.random.randint(0, len(i_data) - self.sequence_length)
            i_data = i_data[start_idx:start_idx + self.sequence_length]
            q_data = q_data[start_idx:start_idx + self.sequence_length]
        elif len(i_data) < self.sequence_length:
            # 如果序列太短，使用循环填充
            i_data = np.resize(i_data, self.sequence_length)
            q_data = np.resize(q_data, self.sequence_length)
        
        # 转换为张量
        iq_data = torch.tensor([i_data, q_data], dtype=torch.float32)
        
        # 应用数据变换
        if self.transform is not None:
            iq_data = self.transform(iq_data)
        
        # 数据归一化
        iq_mean = iq_data.mean(dim=1, keepdim=True)
        iq_std = iq_data.std(dim=1, keepdim=True)
        iq_data = (iq_data - iq_mean) / (iq_std + 1e-6)
        
        # 创建标签张量
        modulation_type = torch.tensor(sample['modulation_type'] - 1, dtype=torch.long)
        
        return {
            'data': iq_data,
            'targets': {
                'modulation_type': modulation_type,
                'mod_name': sample['modulation_type'] - 1  # 直接使用数字索引而不是字符串
            }
        }
    
    @staticmethod
    def get_transforms(config, mode='train'):
        """获取数据变换"""
        transforms = []
        
        if mode == 'train':
            if config.USE_MIXUP:
                transforms.append(Mixup(alpha=config.MIXUP_ALPHA))
            if config.USE_CUTMIX:
                transforms.append(CutMix(alpha=config.CUTMIX_ALPHA))
        
        return Compose(transforms) if transforms else None

class Mixup:
    """Mixup数据增强"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x):
        if self.alpha <= 0:
            return x
        
        # 生成混合权重
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 创建混合数据
        mixed_x = lam * x
        
        # 随机打乱IQ数据
        perm = torch.randperm(2)
        mixed_x = mixed_x + (1 - lam) * x[perm]
        
        return mixed_x

class CutMix:
    """CutMix数据增强"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, x):
        if self.alpha <= 0:
            return x
        
        # 生成混合权重
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 计算裁剪区域
        seq_len = x.size(1)  # IQ数据的序列长度
        cut_len = int(seq_len * (1 - lam))
        cut_start = np.random.randint(0, seq_len - cut_len)
        
        # 创建混合数据
        mixed_x = x.clone()
        
        # 随机打乱IQ数据
        perm = torch.randperm(2)
        mixed_x[:, cut_start:cut_start+cut_len] = x[perm, cut_start:cut_start+cut_len]
        
        return mixed_x

class Compose:
    """组合多个数据变换"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x