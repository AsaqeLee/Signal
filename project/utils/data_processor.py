import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple
import torch
import logging

class SignalProcessor:
    """信号处理类"""
    def __init__(self):
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def process_file(self, file_path: str) -> Dict[str, Union[np.ndarray, float]]:
        """处理单个数据文件
        Args:
            file_path: 数据文件路径
        Returns:
            处理后的数据字典
        """
        try:
            # 1. 读取数据
            df = pd.read_csv(file_path, header=None)
            if df.shape[1] < 5:
                raise ValueError(f"数据列数不足: {df.shape[1]} < 5")
            
            # 2. 提取数据
            i_data = df.iloc[:, 0].values
            q_data = df.iloc[:, 1].values
            code_sequence = pd.to_numeric(df.iloc[:, 2], errors='coerce').values
            mod_type = int(df.iloc[0, 3])
            symbol_width = float(df.iloc[0, 4])
            
            # 3. 数据验证
            if len(i_data) != len(q_data):
                raise ValueError(f"IQ数据长度不匹配: {len(i_data)} != {len(q_data)}")
            
            # 4. 处理码序列
            valid_code = code_sequence[~np.isnan(code_sequence)]
            if len(valid_code) == 0:
                raise ValueError("没有有效的码序列数据")
            
            # 5. 检查码元宽度
            if symbol_width <= 0 or np.isnan(symbol_width):
                raise ValueError(f"无效的码元宽度: {symbol_width}")
            
            # 6. 检查序列长度匹配
            points_per_symbol = int(symbol_width * 20)  # 每微秒20个采样点
            expected_iq_len = len(valid_code) * points_per_symbol
            actual_iq_len = len(i_data)
            
            # 7. 调整数据长度，保持码元完整性
            if actual_iq_len != expected_iq_len:
                # 计算完整码元数
                complete_symbols = min(
                    len(valid_code),
                    actual_iq_len // points_per_symbol
                )
                
                # 只保留完整的码元
                valid_code = valid_code[:complete_symbols]
                i_data = i_data[:complete_symbols * points_per_symbol]
                q_data = q_data[:complete_symbols * points_per_symbol]
                
                self.logger.warning(
                    f"调整序列长度以保持码元完整性:\n"
                    f"  - 原始IQ长度: {actual_iq_len}\n"
                    f"  - 期望IQ长度: {expected_iq_len}\n"
                    f"  - 调整后长度: {len(i_data)}\n"
                    f"  - 完整码元数: {complete_symbols}\n"
                    f"  - 每码元采样点数: {points_per_symbol}"
                )
            
            # 8. 数据归一化
            i_data = self._normalize(i_data)
            q_data = self._normalize(q_data)
            
            # 9. 验证最终数据
            final_iq_len = len(i_data)
            final_code_len = len(valid_code)
            final_points_per_symbol = final_iq_len / final_code_len
            
            if not np.isclose(final_points_per_symbol, points_per_symbol, rtol=1e-5):
                raise ValueError(
                    f"码元采样点数不匹配:\n"
                    f"  - 期望值: {points_per_symbol}\n"
                    f"  - 实际值: {final_points_per_symbol}"
                )
            
            return {
                'i_data': i_data.astype(np.float32),
                'q_data': q_data.astype(np.float32),
                'code_sequence': valid_code.astype(np.float32),
                'symbol_width': symbol_width
            }
            
        except Exception as e:
            self.logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
            raise
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """归一化数据
        Args:
            data: 输入数据
        Returns:
            归一化后的数据
        """
        if np.all(data == 0):
            return data
        return (data - np.mean(data)) / (np.std(data) + 1e-8)
    
    def validate_data(self, file_path: str) -> Tuple[bool, str]:
        """验证数据文件
        Args:
            file_path: 数据文件路径
        Returns:
            (是否有效, 错误信息)
        """
        try:
            # 处理数据
            result = self.process_file(file_path)
            
            # 验证IQ数据
            if np.any(np.isnan(result['i_data'])) or np.any(np.isnan(result['q_data'])):
                return False, "IQ数据包含NaN值"
            
            if np.any(np.isinf(result['i_data'])) or np.any(np.isinf(result['q_data'])):
                return False, "IQ数据包含Inf值"
            
            # 验证码序列
            if len(result['code_sequence']) == 0:
                return False, "没有有效的码序列"
            
            # 验证码元宽度
            if result['symbol_width'] <= 0:
                return False, f"无效的码元宽度: {result['symbol_width']}"
            
            return True, "数据有效"
            
        except Exception as e:
            return False, str(e)
