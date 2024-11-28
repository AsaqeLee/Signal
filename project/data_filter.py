import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

from config import Config

def validate_data_file(file_path: str) -> bool:
    """验证单个数据文件
    Args:
        file_path: 数据文件路径
    Returns:
        数据是否有效
    """
    try:
        # 读取数据
        df = pd.read_csv(file_path, header=None)
        
        # 1. 检查基本数据
        if df.shape[1] < 5:  # 至少需要5列
            return False
            
        # 2. 获取数据
        i_data = df.iloc[:, 0].values
        q_data = df.iloc[:, 1].values
        code_sequence = pd.to_numeric(df.iloc[:, 2], errors='coerce').values
        try:
            mod_type = int(df.iloc[0, 3])
            symbol_width = float(df.iloc[0, 4])
        except (ValueError, TypeError):
            return False
            
        # 3. 检查IQ数据
        if len(i_data) != len(q_data):
            return False
            
        # 4. 检查码序列
        valid_code = code_sequence[~np.isnan(code_sequence)]
        if len(valid_code) == 0:
            return False
            
        # 5. 检查码元宽度
        if symbol_width <= 0 or np.isnan(symbol_width):
            return False
            
        # 6. 检查序列长度匹配
        points_per_symbol = int(symbol_width * 20e6 / 1e6)  # 每微秒20个采样点
        expected_iq_len = len(valid_code) * points_per_symbol
        if len(i_data) != expected_iq_len:
            return False
            
        # 7. 检查数值范围
        if np.any(np.isinf(i_data)) or np.any(np.isinf(q_data)):
            return False
        if np.any(np.isnan(i_data)) or np.any(np.isnan(q_data)):
            return False
            
        return True
        
    except Exception as e:
        print(f"Error validating file {file_path}: {str(e)}")
        return False

def filter_dataset(src_dir: str = 'train_data', 
                  dst_dir: str = 'train_data_true',
                  config: Config = None):
    """筛选有效数据集
    Args:
        src_dir: 源数据目录
        dst_dir: 目标数据目录
        config: 配置对象
    """
    if config is None:
        config = Config()
        
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 创建目标目录
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    stats = {mod_type: {
        'total': 0,
        'valid': 0
    } for mod_type in config.MODULATION_DICT.keys()}
    
    # 遍历所有调制类型
    for mod_type, mod_name in config.MODULATION_DICT.items():
        logger.info(f"\n处理 {mod_name} 数据...")
        
        # 创建目标子目录
        src_mod_dir = Path(src_dir) / mod_name
        dst_mod_dir = dst_path / mod_name
        dst_mod_dir.mkdir(exist_ok=True)
        
        if not src_mod_dir.exists():
            logger.warning(f"目录不存在: {src_mod_dir}")
            continue
            
        # 遍历该调制类型的所有文件
        for file_path in tqdm(list(src_mod_dir.glob("*.csv")), desc=f"验证{mod_name}"):
            stats[mod_type]['total'] += 1
            
            # 验证数据
            if validate_data_file(str(file_path)):
                # 复制有效文件
                dst_file = dst_mod_dir / file_path.name
                shutil.copy2(file_path, dst_file)
                stats[mod_type]['valid'] += 1
    
    # 输出统计信息
    logger.info("\n=== 数据筛选结果 ===")
    min_valid = float('inf')
    for mod_type, mod_stats in stats.items():
        valid_ratio = mod_stats['valid'] / mod_stats['total'] * 100
        logger.info(f"{config.MODULATION_DICT[mod_type]}:")
        logger.info(f"  - 总文件数: {mod_stats['total']}")
        logger.info(f"  - 有效文件数: {mod_stats['valid']}")
        logger.info(f"  - 有效率: {valid_ratio:.2f}%")
        min_valid = min(min_valid, mod_stats['valid'])
    
    logger.info(f"\n最小有效样本数: {min_valid}")
    logger.info("建议使用这个数量作为每种调制类型的训练样本数，以保持数据平衡")

if __name__ == '__main__':
    filter_dataset() 