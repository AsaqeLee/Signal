import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from typing import Dict, Any, Tuple, Optional, List, Union
import logging
import os
from pathlib import Path

from project.config import Config

class SignalAugmentor:
    """信号增强器"""
    def __init__(self, config: Config) -> None:
        self.config = config
    
    def add_noise(self, i_data: np.ndarray, q_data: np.ndarray, snr_db: float) -> Tuple[np.ndarray, np.ndarray]:
        """添加高斯噪声"""
        # 计算信号功率
        signal_power = np.mean(i_data**2 + q_data**2)
        
        # 计算噪声功率
        noise_power = signal_power / (10**(snr_db/10))
        
        # 生成噪声
        i_noise = np.random.normal(0, np.sqrt(noise_power/2), i_data.shape)
        q_noise = np.random.normal(0, np.sqrt(noise_power/2), q_data.shape)
        
        return i_data + i_noise, q_data + q_noise
    
    def add_frequency_offset(self, i_data: np.ndarray, q_data: np.ndarray, max_offset: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """添加频率偏移"""
        # 生成随机频率偏移
        freq_offset = np.random.uniform(-max_offset, max_offset)
        
        # 生成时间序列
        t = np.arange(len(i_data))
        phase = 2 * np.pi * freq_offset * t
        
        # 应用频率偏移
        i_shifted = i_data * np.cos(phase) - q_data * np.sin(phase)
        q_shifted = i_data * np.sin(phase) + q_data * np.cos(phase)
        
        return i_shifted, q_shifted
    
    def add_phase_noise(self, i_data: np.ndarray, q_data: np.ndarray, std: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """添加相位噪声"""
        # 生成随机相位噪声
        phase_noise = np.random.normal(0, std, len(i_data))
        
        # 应用相位噪声
        phase = np.exp(1j * phase_noise)
        complex_data = (i_data + 1j * q_data) * phase
        
        return np.real(complex_data), np.imag(complex_data)
    
    def add_multipath(self, i_data: np.ndarray, q_data: np.ndarray, 
                     num_paths: int = 3, max_delay: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """添加多径效应"""
        # 生成多径参数
        delays = np.random.randint(1, max_delay, num_paths)
        amplitudes = np.random.uniform(0.1, 0.3, num_paths)
        
        # 初始化输出
        i_out = np.copy(i_data)
        q_out = np.copy(q_data)
        
        # 添加多径分量
        for delay, amplitude in zip(delays, amplitudes):
            i_out[delay:] += amplitude * i_data[:-delay]
            q_out[delay:] += amplitude * q_data[:-delay]
        
        return i_out, q_out
    
    def apply_fading(self, i_data: np.ndarray, q_data: np.ndarray, 
                    fading_type: str = 'rayleigh') -> Tuple[np.ndarray, np.ndarray]:
        """应用衰落"""
        if fading_type == 'rayleigh':
            # 瑞利衰落
            h_real = np.random.normal(0, 1/np.sqrt(2))
            h_imag = np.random.normal(0, 1/np.sqrt(2))
        else:  # Rician
            # 莱斯衰落
            K = 4  # 莱斯因子
            v = np.sqrt(K/(K+1))  # LOS分量
            sigma = np.sqrt(1/(2*(K+1)))  # 散射分量标准差
            h_real = v + np.random.normal(0, sigma)
            h_imag = np.random.normal(0, sigma)
        
        # 应用衰落
        h = complex(h_real, h_imag)
        complex_data = (i_data + 1j * q_data) * h
        
        return np.real(complex_data), np.imag(complex_data)
    
    def apply_timing_offset(self, i_data: np.ndarray, q_data: np.ndarray, 
                          max_offset: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """应用定时偏移"""
        # 生成随机定时偏移
        offset = int(len(i_data) * np.random.uniform(-max_offset, max_offset))
        
        if offset > 0:
            i_shifted = np.pad(i_data[:-offset], (offset, 0), mode='constant')
            q_shifted = np.pad(q_data[:-offset], (offset, 0), mode='constant')
        else:
            i_shifted = np.pad(i_data[-offset:], (0, -offset), mode='constant')
            q_shifted = np.pad(q_data[-offset:], (0, -offset), mode='constant')
        
        return i_shifted, q_shifted
    
    def apply_spectral_augment(self, i_data: np.ndarray, q_data: np.ndarray,
                             num_masks: int = 2, mask_width: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """应用频谱增强"""
        # 计算FFT
        i_freq = np.fft.fft(i_data)
        q_freq = np.fft.fft(q_data)
        
        # 应用频率屏蔽
        freq_len = len(i_freq)
        for _ in range(num_masks):
            f0 = np.random.randint(0, freq_len - mask_width)
            i_freq[f0:f0+mask_width] = 0
            q_freq[f0:f0+mask_width] = 0
        
        # 反变换回时域
        i_masked = np.real(np.fft.ifft(i_freq))
        q_masked = np.real(np.fft.ifft(q_freq))
        
        return i_masked, q_masked
    
    def apply_random_erasing(self, i_data: np.ndarray, q_data: np.ndarray,
                           p: float = 0.5, max_len: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """应用随机擦除"""
        if np.random.random() > p:
            return i_data, q_data
        
        # 生成擦除区域
        seq_len = len(i_data)
        erase_len = int(seq_len * np.random.uniform(0.01, max_len))
        erase_start = np.random.randint(0, seq_len - erase_len)
        
        # 应用擦除
        i_erased = np.copy(i_data)
        q_erased = np.copy(q_data)
        
        i_erased[erase_start:erase_start+erase_len] = 0
        q_erased[erase_start:erase_start+erase_len] = 0
        
        return i_erased, q_erased
    
    def apply_augmentations(self, i_data: np.ndarray, q_data: np.ndarray, mod_type: str, symbol_width: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用所有增强
        
        参数:
            i_data: I路数据
            q_data: Q路数据
            mod_type: 调制类型
            symbol_width: 码元宽度
        """
        augmentations = []
        
        # 基础信号增强
        if np.random.random() < 0.9:
            # 根据调制类型调整SNR范围
            if mod_type in ['BPSK', 'QPSK']:
                snr_range = (-10, 20)  # 抗噪性能好的调制方式
            elif mod_type in ['8PSK', 'MSK']:
                snr_range = (-5, 20)
            else:  # QAM和APSK
                snr_range = (0, 25)  # 高阶调制需要更好的信噪比
            
            snr_db = np.random.uniform(*snr_range)
            augmentations.append(lambda i, q: self.add_noise(i, q, snr_db))
        
        # 频率域增强
        if np.random.random() < 0.7:
            # 根据码元宽度调整最大频偏
            max_offset = min(0.1, 1 / (symbol_width * self.config.SAMPLING_RATE))
            augmentations.append(lambda i, q: self.add_frequency_offset(i, q, max_offset))
        
        # 相位增强
        if np.random.random() < 0.6:
            # 根据调制类型调整相位噪声强度
            if 'PSK' in mod_type or mod_type == 'MSK':
                phase_std = 0.05  # PSK对相位敏感
            else:
                phase_std = 0.1
            augmentations.append(lambda i, q: self.add_phase_noise(i, q, phase_std))
        
        # 信道效应
        if np.random.random() < 0.5:
            # 根据码元宽度调整最大时延
            max_delay = min(
                10,
                int(symbol_width * self.config.SAMPLING_RATE * 0.1)  # 最大时延不超过码元宽度的10%
            )
            augmentations.append(lambda i, q: self.add_multipath(i, q, max_delay=max_delay))
        
        # 时域增强
        if np.random.random() < 0.6:
            # 根据码元宽度调整最大时偏
            max_offset = min(0.1, symbol_width / 10)  # 最大时偏不超过码元宽度的10%
            augmentations.append(lambda i, q: self.apply_timing_offset(i, q, max_offset))
        
        # 频谱增强
        if np.random.random() < 0.4:
            # 根据码元宽度调整掩码宽度
            mask_width = min(
                50,
                int(self.config.SAMPLING_RATE * symbol_width * 0.1)  # 掩码宽度不超过码元带宽的10%
            )
            augmentations.append(lambda i, q: self.apply_spectral_augment(i, q, mask_width=mask_width))
        
        # 随机擦除
        if np.random.random() < 0.3:
            # 根据码元宽度调整最大擦除长度
            max_len = min(0.1, symbol_width * 2)  # 最大擦除长度不超过2个码元
            augmentations.append(lambda i, q: self.apply_random_erasing(i, q, max_len=max_len))
        
        # 随机组合增强
        np.random.shuffle(augmentations)
        for aug_func in augmentations:
            i_data, q_data = aug_func(i_data, q_data)
        
        return i_data, q_data