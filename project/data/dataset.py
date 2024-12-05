import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Tuple, Optional, List
import os
import re

from project.config import Config
from .augmentations import SignalAugmentor

class MultiTaskSignalDataset(Dataset):
    """多任务信号数据集"""
    def __init__(self, mode: str = 'train', config: Optional[Config] = None) -> None:
        """
        初始化数据集
        
        Args:
            mode: 'train' 或 'val'
            config: 配置对象
        """
        self.config = config if config is not None else Config()
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        
        # 创建增强器
        self.augmentor = SignalAugmentor(self.config)
        
        # 加载数据
        self.data: List[np.ndarray] = []          # IQ数据
        self.mod_labels: List[int] = []           # 调制类型标签
        self.symbol_widths: List[float] = []      # 码元宽度
        self.symbol_sequences: List[np.ndarray] = [] # 码元序列
        self.sample_rates: List[float] = []       # 采样率
        self.samples_per_symbol: List[int] = []   # 每个码元的采样点数
        self._load_data()
    
    def _extract_info_from_filename(self, filename: str) -> Tuple[int, float]:
        """从文件名中提取信息"""
        # 假设文件名格式为: data_XXXXX.csv
        try:
            number = int(re.search(r'data_(\d+)\.csv', filename).group(1))
            return number
        except:
            return 0
    
    def _load_data(self) -> None:
        """加载数据"""
        # 计算训练集和验证集的样本数
        train_samples = int(self.config.SAMPLES_PER_CLASS * 0.8)
        val_samples = self.config.SAMPLES_PER_CLASS - train_samples
        target_samples = train_samples if self.mode == 'train' else val_samples
        
        # 收集所有样本
        for mod_type, mod_name in self.config.MODULATION_DICT.items():
            # 使用配置中的数据目录
            mod_dir = self.config.DATA_DIR / mod_name
            if not mod_dir.exists():
                self.logger.warning(f"未找到{mod_name}的数据目录: {mod_dir}")
                continue
            
            # 获取所有文件并按编号排序
            all_files = list(mod_dir.glob("*.csv"))
            all_files.sort(key=lambda x: self._extract_info_from_filename(x.name))
            
            if len(all_files) < self.config.SAMPLES_PER_CLASS:
                self.logger.warning(
                    f"{mod_name}的样本数量不足: "
                    f"{len(all_files)} < {self.config.SAMPLES_PER_CLASS}"
                )
                continue
            
            # 根据模式选择相应的样本范围
            if self.mode == 'train':
                selected_files = all_files[:train_samples]
            else:  # val
                selected_files = all_files[train_samples:self.config.SAMPLES_PER_CLASS]
            
            # 加载并处理数据
            for file_path in selected_files:
                try:
                    # 读取文件内容
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    if not lines:
                        self.logger.warning(f"空文件: {file_path}")
                        continue
                    
                    # 从第一行获取调制类型和码元宽度
                    first_line = lines[0].strip().split(',')
                    if len(first_line) < 5:
                        self.logger.warning(f"第一行数据不完整: {file_path}")
                        continue
                    
                    # 获取码元宽度 (第5个值)
                    try:
                        symbol_width = float(first_line[4])
                        if symbol_width <= 0:
                            self.logger.warning(f"无效的码元宽度 {symbol_width}: {file_path}")
                            continue
                    except ValueError:
                        self.logger.warning(f"无效的码元宽度格式: {file_path}")
                        continue
                    
                    # 计算每个码元的采样点数
                    samples_per_symbol = max(1, int(self.config.SAMPLING_RATE * symbol_width))
                    
                    # 解析IQ数据和码元序列
                    iq_data = []
                    symbol_seq = []
                    for line in lines[1:]:  # 从第二行开始解析
                        values = [float(x.strip()) for x in line.strip().split(',') if x.strip()]
                        if len(values) >= 2:  # 至少有I和Q数据
                            iq_data.append(values[:2])
                        if len(values) >= 3:  # 有码元序列数据
                            symbol_seq.append(values[2])  # 第3列是码序列
                    
                    # 转换为numpy数组
                    iq_data = np.array(iq_data)
                    if len(iq_data) == 0:
                        self.logger.warning(f"无有效IQ数据: {file_path}")
                        continue
                    
                    # 分离I和Q数据
                    i_data = iq_data[:, 0]
                    q_data = iq_data[:, 1]
                    
                    # 数据预处理
                    i_data, q_data = self._preprocess_data(i_data, q_data)
                    
                    # 处理码元序列
                    symbol_seq = np.array(symbol_seq) if symbol_seq else np.array([])
                    
                    # 确保IQ序列长度一致
                    if len(i_data) > self.config.SEQUENCE_LENGTH:
                        i_data = i_data[:self.config.SEQUENCE_LENGTH]
                        q_data = q_data[:self.config.SEQUENCE_LENGTH]
                    elif len(i_data) < self.config.SEQUENCE_LENGTH:
                        # 填充到指定长度
                        pad_len = self.config.SEQUENCE_LENGTH - len(i_data)
                        i_data = np.pad(i_data, (0, pad_len), mode='constant')
                        q_data = np.pad(q_data, (0, pad_len), mode='constant')
                    
                    # 添加到数据集
                    self.data.append(np.stack([i_data, q_data]))
                    self.mod_labels.append(mod_type - 1)  # 转换为0-based索引
                    self.symbol_widths.append(symbol_width)
                    self.symbol_sequences.append(symbol_seq)
                    self.sample_rates.append(self.config.SAMPLING_RATE)
                    self.samples_per_symbol.append(samples_per_symbol)
                    
                except Exception as e:
                    self.logger.warning(f"加载文件失败 {file_path}: {str(e)}")
        
        # 转换为numpy数组
        if len(self.data) == 0:
            raise RuntimeError("没有加载到任何有效数据")
            
        self.data = np.array(self.data)
        self.mod_labels = np.array(self.mod_labels)
        self.symbol_widths = np.array(self.symbol_widths)
        self.symbol_sequences = np.array(self.symbol_sequences, dtype=object)  # 使用object类型以支持不同长度
        self.sample_rates = np.array(self.sample_rates)
        self.samples_per_symbol = np.array(self.samples_per_symbol)
        
        self.logger.info(f"成功加载了{len(self.data)}个{self.mode}样本")
    
    def _preprocess_data(self, i_data: np.ndarray, q_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """数据预处理"""
        # 归一化
        i_norm = np.sqrt(np.mean(i_data**2))
        q_norm = np.sqrt(np.mean(q_data**2))
        i_data = i_data / (i_norm + 1e-6)
        q_data = q_data / (q_norm + 1e-6)
        
        return i_data, q_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 获取原始数据
        iq_data = self.data[idx]
        i_data, q_data = iq_data[0], iq_data[1]
        symbol_seq = self.symbol_sequences[idx]
        samples_per_symbol = self.samples_per_symbol[idx]
        
        # 训练模式下应用数据增强
        if self.mode == 'train':
            mod_name = self.config.get_modulation_name(self.mod_labels[idx] + 1)  # 转换回1-based索引
            i_data, q_data = self.augmentor.apply_augmentations(
                i_data, q_data,
                mod_name,
                self.symbol_widths[idx]
            )
        
        # 转换为tensor
        data = torch.from_numpy(np.stack([i_data, q_data])).float()
        mod_label = torch.tensor(self.mod_labels[idx]).long()
        symbol_width = torch.tensor(self.symbol_widths[idx]).float()
        
        # 计算实际的symbol序列长度
        actual_symbol_length = len(symbol_seq)
        max_symbol_length = self.config.SEQUENCE_LENGTH // samples_per_symbol
        
        # 创建symbol序列tensor,使用实际长度
        symbol_sequence = torch.from_numpy(symbol_seq[:actual_symbol_length]).float()
        
        # 创建一个有效长度mask
        sequence_mask = torch.zeros(max_symbol_length, dtype=torch.bool)
        sequence_mask[:actual_symbol_length] = True
        
        return {
            'data': data,
            'targets': {
                'modulation_type': mod_label,
                'symbol_width': symbol_width,
                'symbol_sequence': symbol_sequence,
                'sequence_mask': sequence_mask,
                'samples_per_symbol': torch.tensor(samples_per_symbol).long()
            }
        }