import torch
import os
import math
import json
from datetime import datetime
from pathlib import Path

class Config:
    # 资源限制
    MAX_WORKERS = 2  # 限制使用2个CPU核心
    MAX_MEMORY = 8  # 限制内存使用为8GB
    
    # 数据参数
    SAMPLING_RATE = 20e6  # 20MHz，即每微秒20个采样点
    SAMPLES_PER_CLASS = 16200  # 每个调制类型的样本数
    
    # 模型参数
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 16  # 批次大小
    
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
    
    # 评分参数
    MT_WEIGHT = 0.4  # 调制类型权重
    SW_WEIGHT = 0.6  # 码元宽度权重
    
    # 训练参数
    NUM_EPOCHS = 1  # 每次运行一个epoch
    LEARNING_RATE = 0.001
    MIN_LEARNING_RATE = 1e-6  # 最小学习率
    VALIDATION_RATIO = 0.2
    GRADIENT_ACCUMULATION_STEPS = 8  # 梯度累积步数
    
    # 优化器参数
    OPTIMIZER = 'adam'  # 可选: 'adam', 'adamw', 'sgd'
    WEIGHT_DECAY = 1e-5  # L2正则化
    MOMENTUM = 0.9  # SGD动量
    BETA1 = 0.9  # Adam/AdamW的beta1参数
    BETA2 = 0.999  # Adam/AdamW的beta2参数
    EPS = 1e-8  # 数值稳定性参数
    
    # 学习率调度参数
    LR_SCHEDULER = 'cosine'  # 可选: 'cosine', 'step', 'linear', 'exponential'
    LR_DECAY_RATE = 0.1  # 学习率衰减率
    LR_STEP_SIZE = 30  # 学习率衰减步长
    WARMUP_EPOCHS = 5  # 预热训练轮数
    WARMUP_METHOD = 'linear'  # 可选: 'linear', 'exponential'
    
    # 模型参数
    FEATURE_DIM = 128  # 特征维度
    DROPOUT_RATE = 0.5  # Dropout比率
    LABEL_SMOOTHING = 0.1  # 标签平滑参数
    
    # 数据增强参数
    USE_MIXUP = True  # 是否使用Mixup
    MIXUP_ALPHA = 0.2  # Mixup的alpha参数
    USE_CUTMIX = True  # 是否使用CutMix
    CUTMIX_ALPHA = 1.0  # CutMix的alpha参数
    
    # 训练稳定性参数
    GRADIENT_CLIP_VAL = 1.0  # 梯度裁剪阈值
    AMP_ENABLED = True  # 是否启用自动混合精度
    
    # 内存管理参数
    PREFETCH_FACTOR = 2  # 预加载因子
    PERSISTENT_WORKERS = True  # 保持工作进程存活
    PIN_MEMORY = False  # 禁用内存固定
    
    # 检查点和状态保存
    CHECKPOINT_DIR = Path("checkpoints")
    STATE_FILE = CHECKPOINT_DIR / "training_state.json"
    LAST_CHECKPOINT = CHECKPOINT_DIR / "last_checkpoint.pth"
    BEST_CHECKPOINT = CHECKPOINT_DIR / "best_checkpoint.pth"
    
    # 日志参数
    LOG_DIR = Path("logs")
    LOG_INTERVAL = 10  # 每隔多少步记录一次日志
    SAVE_INTERVAL = 100  # 每隔多少步保存一次模型
    
    def __init__(self):
        # 设置torch线程数
        torch.set_num_threads(self.MAX_WORKERS)
        
        # 设置CUDA内存分配器
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.7)
            torch.cuda.empty_cache()
        
        # 创建必要的目录
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)
        
        # 加载训练状态
        self.training_state = self.load_training_state()
        
        # 打印重要配置信息
        print("\n=== 配置信息 ===")
        print(f"运行设备: {self.DEVICE}")
        print(f"批次大小: {self.BATCH_SIZE}")
        print(f"梯度累积步数: {self.GRADIENT_ACCUMULATION_STEPS}")
        print(f"有效批次大小: {self.BATCH_SIZE * self.GRADIENT_ACCUMULATION_STEPS}")
        print(f"当前epoch: {self.training_state['current_epoch']}")
        print(f"最佳验证分数: {self.training_state['best_val_score']:.4f}")
        print(f"CPU核心数: {self.MAX_WORKERS}")
        print(f"内存限制: {self.MAX_MEMORY}GB")
        print(f"是否使用AMP: {self.AMP_ENABLED}")
        print(f"是否使用数据增强: Mixup={self.USE_MIXUP}, CutMix={self.USE_CUTMIX}")
        print("===============\n")
    
    def load_training_state(self):
        """加载训练状态"""
        default_state = {
            'current_epoch': 0,
            'best_val_score': float('-inf'),
            'total_epochs_run': 0,
            'last_lr': self.LEARNING_RATE,
            'training_time': 0,
            'last_run_date': None,
            'early_stop_counter': 0,
            'best_epoch': 0
        }
        
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, 'r') as f:
                    state = json.load(f)
                print(f"加载训练状态：当前第{state['current_epoch']}轮")
                return state
            except Exception as e:
                print(f"加载状态文件失败: {e}")
                return default_state
        return default_state
    
    def save_training_state(self, **kwargs):
        """保存训练状态"""
        state = self.training_state.copy()
        state.update(kwargs)
        state['last_run_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)
        print(f"保存训练状态：当前第{state['current_epoch']}轮")
    
    def get_optimizer(self, model_parameters):
        """获取优化器"""
        if self.OPTIMIZER == 'adam':
            return torch.optim.Adam(
                model_parameters,
                lr=self.LEARNING_RATE,
                betas=(self.BETA1, self.BETA2),
                eps=self.EPS,
                weight_decay=self.WEIGHT_DECAY
            )
        elif self.OPTIMIZER == 'adamw':
            return torch.optim.AdamW(
                model_parameters,
                lr=self.LEARNING_RATE,
                betas=(self.BETA1, self.BETA2),
                eps=self.EPS,
                weight_decay=self.WEIGHT_DECAY
            )
        elif self.OPTIMIZER == 'sgd':
            return torch.optim.SGD(
                model_parameters,
                lr=self.LEARNING_RATE,
                momentum=self.MOMENTUM,
                weight_decay=self.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"不支持的优化器类型: {self.OPTIMIZER}")
    
    def get_lr_scheduler(self, optimizer):
        """获取学习率调度器"""
        if self.LR_SCHEDULER == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=100,  # 假设总共训练100轮
                eta_min=self.MIN_LEARNING_RATE,
                last_epoch=self.training_state['current_epoch'] - 1
            )
        elif self.LR_SCHEDULER == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.LR_STEP_SIZE,
                gamma=self.LR_DECAY_RATE
            )
        elif self.LR_SCHEDULER == 'linear':
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=100
            )
        elif self.LR_SCHEDULER == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.LR_DECAY_RATE
            )
        return None
    
    def should_stop_early(self, current_score):
        """检查是否应该早停"""
        if current_score > self.training_state['best_val_score']:
            self.training_state['early_stop_counter'] = 0
            self.training_state['best_val_score'] = current_score
            self.training_state['best_epoch'] = self.training_state['current_epoch']
            return False
        
        self.training_state['early_stop_counter'] += 1
        if self.training_state['early_stop_counter'] >= 15:  # 15轮没有改善就停止
            print(f"早停：{self.training_state['early_stop_counter']}轮未改善")
            return True
        
        return False
