import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import pickle
import logging
from project.config import Config
import pandas as pd

class ModulationDataset(Dataset):
    def __init__(self, transform=None):
        self.config = Config()
        self.transform = transform
        self.logger = logging.getLogger(__name__)
        
        # 加载数据
        self.data = []
        self.labels = []
        self.symbol_widths = []
        
        self._load_data()
        
    def _load_data(self):
        """加载和预处理数据"""
        try:
            total_files = 0
            valid_files = 0
            
            for mod_type, mod_name in self.config.MODULATION_DICT.items():
                mod_dir = self.config.DATA_DIR / mod_name
                if not mod_dir.exists():
                    self.logger.warning(f"目录不存在: {mod_dir}")
                    continue
                
                files = list(mod_dir.glob("*.csv"))
                total_files += len(files)
                self.logger.info(f"在 {mod_name} 目录中找到 {len(files)} 个文件")
                
                for file_path in files:
                    try:
                        # 读取CSV文件，不使用header
                        df = pd.read_csv(file_path, header=None, dtype=float)
                        
                        # 基本验证
                        if df.shape[1] < 5:
                            self.logger.warning(f"文件列数不足: {file_path}")
                            continue
                            
                        if df.empty:
                            self.logger.warning(f"文件为空: {file_path}")
                            continue
                        
                        # 获取IQ数据
                        i_data = df.iloc[:, 0].values
                        q_data = df.iloc[:, 1].values
                        
                        # 检查数据有效性
                        if np.any(np.isnan(i_data)) or np.any(np.isnan(q_data)):
                            self.logger.warning(f"文件包含NaN值: {file_path}")
                            continue
                            
                        if np.any(np.isinf(i_data)) or np.any(np.isinf(q_data)):
                            self.logger.warning(f"文件包含Inf值: {file_path}")
                            continue
                        
                        signal = np.stack([i_data, q_data], axis=0)
                        
                        # 获取标签信息 - 使用第一行的值
                        try:
                            file_mod_type = int(df.iloc[0, 3])  # 调制类型
                            symbol_width = float(df.iloc[0, 4])  # 码元宽度
                            
                            # 验证调制类型是否匹配
                            if file_mod_type != mod_type:
                                self.logger.warning(f"调制类型不匹配 {file_path}: 期望 {mod_type}, 实际 {file_mod_type}")
                                continue
                                
                        except (ValueError, TypeError) as e:
                            self.logger.warning(f"标签转换失败 {file_path}: {str(e)}")
                            continue
                        
                        # 验证码元宽度
                        if symbol_width <= 0:
                            self.logger.warning(f"无效的码元宽度 {symbol_width}: {file_path}")
                            continue
                        
                        # 确保信号长度一致
                        if signal.shape[1] > self.config.SIGNAL_LENGTH:
                            signal = signal[:, :self.config.SIGNAL_LENGTH]
                        elif signal.shape[1] < self.config.SIGNAL_LENGTH:
                            pad_width = ((0, 0), (0, self.config.SIGNAL_LENGTH - signal.shape[1]))
                            signal = np.pad(signal, pad_width, mode='constant')
                        
                        self.data.append(signal)
                        self.labels.append(mod_type)  # 使用原始调制类型索引
                        self.symbol_widths.append(symbol_width)
                        valid_files += 1
                        
                    except Exception as e:
                        self.logger.error(f"加载文件失败 {file_path}: {str(e)}")
                        continue
            
            if valid_files == 0:
                raise RuntimeError("没有找到有效的数据文件")
            
            self.data = np.array(self.data)
            self.labels = np.array(self.labels)
            self.symbol_widths = np.array(self.symbol_widths)
            
            self.logger.info(f"\n=== 数据加载统计 ===")
            self.logger.info(f"总文件数: {total_files}")
            self.logger.info(f"有效文件数: {valid_files}")
            self.logger.info(f"有效率: {valid_files/total_files*100:.2f}%\n")
            
            for mod_type, mod_name in self.config.MODULATION_DICT.items():
                count = np.sum(self.labels == mod_type)
                self.logger.info(f"{mod_name}: {count} 个样本")
                
        except Exception as e:
            self.logger.error(f"加载数据集失败: {str(e)}")
            raise
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        signal = torch.from_numpy(self.data[idx]).float()
        label = self.labels[idx]
        symbol_width = self.symbol_widths[idx]
        
        if self.transform:
            signal = self.transform(signal)
        
        return {
            'data': signal,
            'modulation_type': label,
            'symbol_width': symbol_width
        } 