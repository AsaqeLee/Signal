import torch

class Config:
    # 数据参数
    SAMPLING_RATE = 20e6  # 20MHz，即每微秒20个采样点
    SAMPLES_PER_CLASS = 16200  # 每个调制类型的样本数
    
    # 模型参数
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动选择设备
    BATCH_SIZE = 64  # 减小批次大小，避免内存问题
    
    # 调制类型映射
    MODULATION_DICT = {
        1: 'BPSK',
        2: 'QPSK',
        3: '8PSK',
        4: 'MSK',
        5: '8QAM',
        6: '16QAM',
        7: '32QAM',
        8: '8APSK',
        9: '16APSK',
        10: '32APSK'
    }
    
    # 调制类型对应的码序列值范围
    MODULATION_VALUES = {
        1: 2,   # BPSK: 0-1
        2: 4,   # QPSK: 0-3
        3: 8,   # 8PSK: 0-7
        4: 2,   # MSK: 0-1
        5: 8,   # 8QAM: 0-7
        6: 16,  # 16QAM: 0-15
        7: 32,  # 32QAM: 0-31
        8: 8,   # 8APSK: 0-7
        9: 16,  # 16APSK: 0-15
        10: 32  # 32APSK: 0-31
    }
    
    # 评分参数
    MT_WEIGHT = 0.2  # 调制类型权重
    SW_WEIGHT = 0.3  # 码元宽度权重
    CQ_WEIGHT = 0.5  # 码序列权重
    
    # 训练参数
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_RATIO = 0.2
    
    # 资源限制
    MAX_WORKERS = min(8, torch.get_num_threads())  # CPU核心数，不超过系统核心数
    MAX_MEMORY = 8  # GB，降低内存使用
    
    # 模型参数
    FEATURE_DIM = 256
    DROPOUT_RATE = 0.5
    
    def __init__(self):
        # 打印重要配置信息
        print("\n=== 配置信息 ===")
        print(f"运行设备: {self.DEVICE}")
        print(f"批次大小: {self.BATCH_SIZE}")
        print(f"每类样本数: {self.SAMPLES_PER_CLASS}")
        print(f"CPU工作进程: {self.MAX_WORKERS}")
        print(f"采样率: {self.SAMPLING_RATE/1e6:.1f}MHz")
        print("===============\n")
