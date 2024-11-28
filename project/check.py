import pandas as pd
import numpy as np
import os

def check_csv_file(file_path: str):
    """查看CSV文件的内容
    Args:
        file_path: CSV文件路径
    """
    print(f"\n=== 查看文件: {file_path} ===")
    
    # 读取CSV文件
    df = pd.read_csv(file_path, header=None)
    
    # 显示基本信息
    print("\n1. 文件基本信息:")
    print(f"总行数: {df.shape[0]}")
    print(f"总列数: {df.shape[1]}")
    
    # 分析每列的有效数据
    print("\n2. 列数据分析:")
    mod_type = None
    for col in range(df.shape[1]):
        non_nan = df[col].notna().sum()
        print(f"第{col+1}列 - 有效值数量: {non_nan}/{df.shape[0]} ({non_nan/df.shape[0]*100:.2f}%)")
        if col == 0:
            print(f"    I路数据范围: [{df[0].min():.4f}, {df[0].max():.4f}]")
        elif col == 1:
            print(f"    Q路数据范围: [{df[1].min():.4f}, {df[1].max():.4f}]")
        elif col == 2 and non_nan > 0:
            valid_codes = df[2].dropna()
            unique_codes = sorted(set(valid_codes))
            print(f"    码序列值: {unique_codes}")
            print(f"    码序列基数: {len(unique_codes)}")  # 调制阶数
            if mod_type:
                print(f"    理论码序列范围: 0-{2**int(np.log2(mod_type))-1}")
        elif col == 3 and non_nan > 0:
            mod_type = int(df[3].dropna().iloc[0])
            mod_name = {
                1: 'BPSK (2相位)',
                2: 'QPSK (4相位)',
                3: '8PSK (8相位)',
                4: 'MSK',
                5: '8QAM (8星座点)',
                6: '16QAM (16星座点)',
                7: '32QAM (32星座点)',
                8: '8APSK (8星座点)',
                9: '16APSK (16星座点)',
                10: '32APSK (32星座点)'
            }.get(mod_type, '未知')
            print(f"    调制类型: {mod_type} - {mod_name}")
        elif col == 4 and non_nan > 0:
            symbol_width = df[4].dropna().iloc[0]
            print(f"    码元宽度: {symbol_width}微秒")
    
    # 计算IQ序列和码序列的关系
    valid_codes = df[2].dropna()
    if len(valid_codes) > 0:
        print(f"\n3. 序列长度分析:")
        print(f"IQ序列长度: {len(df)}")
        print(f"码序列长度: {len(valid_codes)}")
        
        # 获取码元宽度
        symbol_width = df[4].dropna().iloc[0] if df.shape[1] > 4 and df[4].notna().any() else None
        
        if symbol_width is not None:
            # 计算每个码元的采样点数（采样率20MHz = 每微秒20个点）
            points_per_symbol = symbol_width * 20
            print(f"码元宽度: {symbol_width}微秒")
            print(f"每码元采样点数: {points_per_symbol}")
            expected_length = len(valid_codes) * points_per_symbol
            print(f"期望IQ长度: {expected_length}")
            
            if abs(len(df) - expected_length) < 1:  # 使用浮点数比较
                print("√ 符合采样率和码元宽度的要求")
                if mod_type:
                    bits_per_symbol = int(np.log2(len(unique_codes)))
                    total_bits = len(valid_codes) * bits_per_symbol
                    print(f"每码元比特数: {bits_per_symbol}")
                    print(f"总比特数: {total_bits}")
            else:
                print(f"× 序列长度不符合要求 (差值: {len(df) - expected_length})")
        else:
            print("无法验证序列长度（缺少码元宽度信息）")
    
    # 显示数据预览
    print("\n4. 数据预览:")
    print("\n前5行:")
    print("     I路        Q路     码序列    调制类型   码元宽度")
    print("-" * 60)
    for idx in range(min(5, len(df))):
        row = df.iloc[idx]
        i_data = f"{row[0]:10.4f}"
        q_data = f"{row[1]:10.4f}"
        code = f"{row[2]:8.0f}" if not np.isnan(row[2]) else "   NaN  "
        mod_type = f"{row[3]:8.0f}" if df.shape[1] > 3 and not np.isnan(row[3]) else "   NaN  "
        symbol_width = f"{row[4]:8.4f}" if df.shape[1] > 4 and not np.isnan(row[4]) else "   NaN  "
        print(f"{i_data} {q_data} {code} {mod_type} {symbol_width}")
    
    if len(df) > 5:
        print("\n...")
        print("\n最后5行:")
        for idx in range(max(0, len(df)-5), len(df)):
            row = df.iloc[idx]
            i_data = f"{row[0]:10.4f}"
            q_data = f"{row[1]:10.4f}"
            code = f"{row[2]:8.0f}" if not np.isnan(row[2]) else "   NaN  "
            mod_type = f"{row[3]:8.0f}" if df.shape[1] > 3 and not np.isnan(row[3]) else "   NaN  "
            symbol_width = f"{row[4]:8.4f}" if df.shape[1] > 4 and not np.isnan(row[4]) else "   NaN  "
            print(f"{i_data} {q_data} {code} {mod_type} {symbol_width}")

if __name__ == "__main__":
    # 让用户选择要查看的文件
    print("\n请输入要查看的文件路径（例如：train_data/BPSK/data_9995.csv）：")
    file_path = input().strip()
    
    if os.path.exists(file_path):
        check_csv_file(file_path)
    else:
        print(f"错误：文件 '{file_path}' 不存在") 