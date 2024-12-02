import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
import sys
import os
from scipy import signal
from scipy.interpolate import interp1d

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from project.config import Config

class SignalAugmentor:
    """信号增强器"""
    def __init__(self, config):
        self.config = config
        
    def add_noise(self, i_data: np.ndarray, q_data: np.ndarray,
                 snr_db: float) -> Tuple[np.ndarray, np.ndarray]:
        """添加高斯噪声"""
        # 计算信号功率
        signal_power = np.mean(i_data**2 + q_data**2)
        
        # 计算噪声功率
        noise_power = signal_power / (10**(snr_db/10))
        
        # 生成噪声
        i_noise = np.random.normal(0, np.sqrt(noise_power/2), len(i_data))
        q_noise = np.random.normal(0, np.sqrt(noise_power/2), len(q_data))
        
        return i_data + i_noise, q_data + q_noise
    
    def add_frequency_offset(self, i_data: np.ndarray, q_data: np.ndarray,
                           max_offset: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """添加频率偏移"""
        offset = np.random.uniform(-max_offset, max_offset)
        t = np.arange(len(i_data))
        phase = 2 * np.pi * offset * t
        
        # 应用频率偏移
        i_offset = i_data * np.cos(phase) - q_data * np.sin(phase)
        q_offset = i_data * np.sin(phase) + q_data * np.cos(phase)
        
        return i_offset, q_offset
    
    def add_phase_noise(self, i_data: np.ndarray, q_data: np.ndarray,
                       phase_noise_power: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """添加相位噪声"""
        phase_noise = np.random.normal(0, np.sqrt(phase_noise_power), len(i_data))
        phase = np.cumsum(phase_noise)  # 随机游走相位
        
        # 应用相位噪声
        i_noisy = i_data * np.cos(phase) - q_data * np.sin(phase)
        q_noisy = i_data * np.sin(phase) + q_data * np.cos(phase)
        
        return i_noisy, q_noisy
    
    def add_multipath(self, i_data: np.ndarray, q_data: np.ndarray,
                     max_paths: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """添加多径效应"""
        num_paths = np.random.randint(1, max_paths + 1)
        i_multi = np.zeros_like(i_data)
        q_multi = np.zeros_like(q_data)
        
        for _ in range(num_paths):
            # 随机延迟和衰减
            delay = np.random.randint(1, len(i_data)//10)
            attenuation = np.random.uniform(0.1, 0.5)
            
            # 添加延迟路径
            i_multi[delay:] += attenuation * i_data[:-delay] if delay > 0 else i_data
            q_multi[delay:] += attenuation * q_data[:-delay] if delay > 0 else q_data
        
        return i_data + i_multi, q_data + q_multi
    
    def apply_fading(self, i_data: np.ndarray, q_data: np.ndarray,
                    fading_type: str = 'rayleigh') -> Tuple[np.ndarray, np.ndarray]:
        """应用衰落"""
        if fading_type == 'rayleigh':
            # 瑞利衰落
            h_real = np.random.normal(0, 1/np.sqrt(2), len(i_data))
            h_imag = np.random.normal(0, 1/np.sqrt(2), len(i_data))
        else:  # rician
            # 莱斯衰落
            K = 4  # 莱斯因子
            v = np.sqrt(K/(K+1))  # LOS分量
            sigma = np.sqrt(1/(2*(K+1)))  # 散射分量标准差
            h_real = v + np.random.normal(0, sigma, len(i_data))
            h_imag = np.random.normal(0, sigma, len(i_data))
        
        # 应用衰落
        i_faded = i_data * h_real - q_data * h_imag
        q_faded = i_data * h_imag + q_data * h_real
        
        return i_faded, q_faded
    
    def apply_timing_offset(self, i_data: np.ndarray, q_data: np.ndarray,
                          max_offset: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """应用定时偏移"""
        # 生成原始时间点
        t = np.arange(len(i_data))
        
        # 生成新的时间点（带偏移）
        offset = np.random.uniform(-max_offset, max_offset)
        t_new = t + offset
        
        # 插值
        i_interp = interp1d(t, i_data, kind='cubic', bounds_error=False, fill_value=0)
        q_interp = interp1d(t, q_data, kind='cubic', bounds_error=False, fill_value=0)
        
        return i_interp(t_new), q_interp(t_new)

class ModulationDataset(Dataset):
    """调制信号数据集"""
    def __init__(self, mode: str = 'train', max_seq_len: Optional[int] = None):
        self.config = Config()
        self.mode = mode
        self.samples = []
        self.max_seq_len = max_seq_len
        self.augmentor = SignalAugmentor(self.config)
        
        # 设置日志
        self._setup_logging()
        
        # 加载数据
        self._load_data()
        
        # 如果没有指定最大序列长度，计算一个合适的值
        if self.max_seq_len is None:
            self.max_seq_len = self._calculate_max_seq_len()
        
        self.logger.info(f"数据集初始化完成: 模式={mode}, 样本数={len(self.samples)}, 最大序列长度={self.max_seq_len}")
    
    def _setup_logging(self):
        """设置日志"""
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.mode}")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
    
    def _calculate_max_seq_len(self, sample_size: int = 1000) -> int:
        """计算合适的最大序列长度"""
        lengths = []
        sampled_indices = np.random.choice(len(self.samples), 
                                         min(sample_size, len(self.samples)), 
                                         replace=False)
        
        for idx in sampled_indices:
            try:
                df = pd.read_csv(self.samples[idx]['file_path'], header=None)
                lengths.append(len(df))
            except Exception as e:
                self.logger.warning(f"计算序列长度时出错: {str(e)}")
                continue
        
        if not lengths:
            return 1000  # 默认值
        
        # 使用95%分位数作为最大长度
        max_len = int(np.percentile(lengths, 95))
        return max_len
    
    def _load_data(self):
        """加载数据"""
        try:
            # 计算训练集和验证集的样本数
            train_samples = int(self.config.SAMPLES_PER_CLASS * 0.8)
            val_samples = self.config.SAMPLES_PER_CLASS - train_samples
            target_samples = train_samples if self.mode == 'train' else val_samples
            
            # 收集所有样本
            for mod_type, mod_name in self.config.MODULATION_DICT.items():
                mod_dir = Path('train_data_true') / mod_name
                if not mod_dir.exists():
                    raise RuntimeError(f"未找到{mod_name}的数据目录: {mod_dir}")
                
                # 获取所有文件
                all_files = list(mod_dir.glob("*.csv"))
                if len(all_files) < self.config.SAMPLES_PER_CLASS:
                    raise RuntimeError(
                        f"{mod_name}的样本数量不足: "
                        f"{len(all_files)} < {self.config.SAMPLES_PER_CLASS}"
                    )
                
                # 根据模式选择相应的样本范围
                if self.mode == 'train':
                    selected_files = all_files[:train_samples]
                else:  # val
                    selected_files = all_files[train_samples:self.config.SAMPLES_PER_CLASS]
                
                # 验证并添加样本
                valid_files = []
                for file_path in selected_files:
                    if self._validate_file(file_path):
                        valid_files.append({
                            'file_path': str(file_path),
                            'modulation_type': mod_type - 1  # 转换为0-based索引
                        })
                
                self.samples.extend(valid_files)
                self.logger.info(f"{mod_name}: 选择{len(selected_files)}个文件, "
                               f"有效{len(valid_files)}个")
            
            # 随机打乱
            np.random.shuffle(self.samples)
            
        except Exception as e:
            self.logger.error(f"加载数据时出错: {str(e)}")
            raise
    
    def _validate_file(self, file_path: Path) -> bool:
        """验证单个数据文件"""
        try:
            df = pd.read_csv(file_path, header=None)
            
            # 基本检查
            if df.shape[1] < 5:
                self.logger.warning(f"{file_path}: 列数不足")
                return False
            
            # 检查IQ数据
            i_data = df.iloc[:, 0].values
            q_data = df.iloc[:, 1].values
            if len(i_data) != len(q_data):
                self.logger.warning(f"{file_path}: IQ数据长度不匹配")
                return False
            
            # 检查数值有效性
            if np.any(np.isnan(i_data)) or np.any(np.isnan(q_data)):
                self.logger.warning(f"{file_path}: 包含NaN值")
                return False
            
            if np.any(np.isinf(i_data)) or np.any(np.isinf(q_data)):
                self.logger.warning(f"{file_path}: 包含Inf值")
                return False
            
            # 检查码元宽度
            symbol_width = float(df.iloc[0, 4])
            if symbol_width <= 0 or np.isnan(symbol_width):
                self.logger.warning(f"{file_path}: 无效的码元宽度")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"验证文件{file_path}时出错: {str(e)}")
            return False
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx):
        try:
            # 加载数据
            sample = self.samples[idx]
            df = pd.read_csv(sample['file_path'], header=None)
            
            # 获取IQ数据
            i_data = df.iloc[:, 0].values.copy()  # 使用copy()确保连续内存
            q_data = df.iloc[:, 1].values.copy()  # 使用copy()确保连续内存
            
            # 获取码元宽度
            symbol_width = float(df.iloc[0, 4])
            
            # 预处理信号
            i_data, q_data = self._preprocess_signal_pair(i_data, q_data)
            
            # 如果是训练模式，应用数据增强
            if self.mode == 'train':
                i_data, q_data = self._apply_augmentations(i_data, q_data)
            
            # 确保数据是连续的
            i_data = np.ascontiguousarray(i_data)
            q_data = np.ascontiguousarray(q_data)
            
            # 组合为2xN的张量
            data = torch.from_numpy(np.stack([i_data, q_data])).float()
            
            return {
                'data': data,
                'targets': {
                    'modulation_type': torch.tensor(sample['modulation_type'], dtype=torch.long),
                    'symbol_width': torch.tensor(symbol_width, dtype=torch.float32)
                }
            }
            
        except Exception as e:
            self.logger.error(f"加载样本{sample['file_path']}时出错: {str(e)}")
            # 返回空样本
            return self._get_empty_sample(sample['modulation_type'])
    
    def _preprocess_signal_pair(self, i_data: np.ndarray, q_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预处理IQ信号对"""
        # 去直流
        i_data = i_data - np.mean(i_data)
        q_data = q_data - np.mean(q_data)
        
        # 功率归一化
        power = np.mean(i_data**2 + q_data**2)
        if power > 0:
            i_data = i_data / np.sqrt(power)
            q_data = q_data / np.sqrt(power)
        
        # 中值滤波去除脉冲噪声
        i_data = signal.medfilt(i_data, kernel_size=3)
        q_data = signal.medfilt(q_data, kernel_size=3)
        
        # 带通滤波
        nyq = 0.5 * self.config.SAMPLE_RATE
        low = self.config.FILTER_LOW_FREQ / nyq
        high = self.config.FILTER_HIGH_FREQ / nyq
        b, a = signal.butter(4, [low, high], btype='band')
        i_data = signal.filtfilt(b, a, i_data)
        q_data = signal.filtfilt(b, a, q_data)
        
        # 调整长度
        i_data = self._adjust_length(i_data)
        q_data = self._adjust_length(q_data)
        
        return i_data, q_data
    
    def _apply_augmentations(self, i_data: np.ndarray, q_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用数据增强"""
        # 随机选择要应用的增强方法
        augmentations = []
        
        # 添加噪声 (80%概率)
        if np.random.random() < 0.8:
            # 使用随机SNR值，范围从-10dB到20dB
            snr_db = np.random.uniform(-10, 20)
            augmentations.append(lambda i, q: self.augmentor.add_noise(i, q, snr_db))
        
        # 频率偏移 (60%概率)
        if np.random.random() < 0.6:
            augmentations.append(lambda i, q: self.augmentor.add_frequency_offset(i, q))
        
        # 相位噪声 (40%概率)
        if np.random.random() < 0.4:
            augmentations.append(lambda i, q: self.augmentor.add_phase_noise(i, q))
        
        # 多径效应 (30%概率)
        if np.random.random() < 0.3:
            augmentations.append(lambda i, q: self.augmentor.add_multipath(i, q))
        
        # 衰落 (50%概率)
        if np.random.random() < 0.5:
            fading_type = np.random.choice(['rayleigh', 'rician'])
            augmentations.append(lambda i, q: self.augmentor.apply_fading(i, q, fading_type))
        
        # 定时偏移 (40%概率)
        if np.random.random() < 0.4:
            augmentations.append(lambda i, q: self.augmentor.apply_timing_offset(i, q))
        
        # 随机打乱增强顺序并应用
        np.random.shuffle(augmentations)
        for aug_func in augmentations:
            i_data, q_data = aug_func(i_data, q_data)
        
        return i_data, q_data
    
    def _adjust_length(self, signal: np.ndarray) -> np.ndarray:
        """调整信号长度"""
        if len(signal) > self.max_seq_len:
            # 随机选择一个起始点
            start = np.random.randint(0, len(signal) - self.max_seq_len + 1)
            return signal[start:start + self.max_seq_len]
        elif len(signal) < self.max_seq_len:
            # 使用反射填充
            return np.pad(signal, 
                         (0, self.max_seq_len - len(signal)),
                         mode='reflect')
        return signal
    
    def _get_empty_sample(self, modulation_type: int) -> Dict[str, torch.Tensor]:
        """生成空样本"""
        return {
            'data': torch.zeros((2, self.max_seq_len), dtype=torch.float32),
            'targets': {
                'modulation_type': torch.tensor(modulation_type, dtype=torch.long),
                'symbol_width': torch.tensor(0.0, dtype=torch.float32)
            }
        }
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """获取样本信息"""
        sample = self.samples[idx]
        return {
            'file_path': sample['file_path'],
            'modulation_type': self.config.MODULATION_DICT[sample['modulation_type'] + 1],
            'index': idx
        }
