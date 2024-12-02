import torch
import os
import math
import json
from datetime import datetime
from pathlib import Path
import torch.distributed as dist
import time
import shutil

class Config:
    """配置类"""
    
    # 移除资源限制
    MAX_WORKERS = os.cpu_count()  # 使用所有CPU核心
    MAX_MEMORY = None  # 不限制内存使用
    
    # 数据参数
    SAMPLE_RATE = 20e6  # 20MHz，即每微秒20个采样点
    SAMPLES_PER_CLASS = 16200  # 每个调制类型的样本数
    FILTER_LOW_FREQ = 0.002  # 归一化后的低频截止频率 (20kHz/10MHz)
    FILTER_HIGH_FREQ = 0.98  # 归一化后的高频截止频率 (略小于奈奎斯特频率)
    FILTER_ORDER = 8  # 滤波器阶数
    
    # 信号质量参数
    USE_SNR_FILTER = False  # 是否使用SNR过滤，设为False表示接受所有信噪比
    SNR_UNKNOWN = -999  # 用于表示未知或不需要考虑SNR的情况
    
    # 模型参数 - 优化GPU使用
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 512  # 增大批次大小以充分利用GPU
    
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
    
    # 模型相关参数
    NUM_CLASSES = len(MODULATION_DICT)  # 移到MODULATION_DICT定义之后
    
    # 评分参数
    MT_WEIGHT = 0.4  # 调制类型权重
    SW_WEIGHT = 0.6  # 码元宽度权重
    
    # 训练参数 - 优化训练过程
    NUM_EPOCHS = 1  # 每次运行一个epoch
    MAX_EPOCHS = 100  # 最大训练轮数
    TARGET_ACCURACY = 0.95  # 目标准确率
    LEARNING_RATE = 0.001
    MAX_LR = 3e-4  # 最大学习率,用于OneCycleLR调度器
    MIN_LEARNING_RATE = 1e-6
    VALIDATION_RATIO = 0.2
    GRADIENT_ACCUMULATION_STEPS = 4  # 减少梯度累步数因为有更大的batch size
    
    # 早停参数
    EARLY_STOPPING_PATIENCE = 10  # 早停耐心值
    EARLY_STOPPING_MIN_DELTA = 0.001  # 最小改善阈值
    EARLY_STOPPING_MODE = 'max'  # 监控模式: 'min' 用于损失, 'max' 用于准确率
    
    # 优化器参数
    OPTIMIZER = 'adamw'  # 使用AdamW优化器
    WEIGHT_DECAY = 1e-5
    MOMENTUM = 0.9
    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 1e-8
    
    # 学习率调度参数
    LR_SCHEDULER = 'cosine'
    LR_DECAY_RATE = 0.1
    LR_STEP_SIZE = 30
    WARMUP_EPOCHS = 5
    WARMUP_METHOD = 'linear'
    
    # 模型参数
    FEATURE_DIM = 256  # 增大特征维度
    DROPOUT_RATE = 0.3  # 减小dropout率
    LABEL_SMOOTHING = 0.1
    
    # 数据增强参数
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2
    USE_CUTMIX = True
    CUTMIX_ALPHA = 1.0
    
    # GPU优化参数
    AMP_ENABLED = True  # 启用自动混合精度
    PIN_MEMORY = True  # 启用内存固定
    GRADIENT_CLIP_VAL = 1.0
    
    # 数据加载优化
    PREFETCH_FACTOR = 4  # 增大预加载因子
    PERSISTENT_WORKERS = True  # 保持工作进程存活
    NUM_WORKERS = os.cpu_count() * 2  # 使用更多的数据加载线程
    
    # 日志和检查点参数
    LOG_INTERVAL = 10  # 每10个batch打印一次日志
    SAVE_INTERVAL = 100  # 每100个batch保存一次检查点
    EARLY_STOP_PATIENCE = 15  # 15轮无改善就早停
    
    # 实验追踪参数
    USE_WANDB = False  # 默认不使用wandb
    WANDB_PROJECT = "modulation_classification"  # wandb目名称
    WANDB_ENTITY = None  # wandb实体名称
    WANDB_NAME = None  # 实验名称
    WANDB_NOTES = None  # 实验备注
    WANDB_TAGS = ["modulation", "deep-learning"]  # 实验标签
    
    # 分布式训练参数
    DISTRIBUTED = False  # 默认不使用分布式训练
    WORLD_SIZE = 1  # 默认只使用1个进程
    RANK = 0  # 默认进程排名
    
    # 文件路径
    PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录
    DATA_DIR = PROJECT_ROOT / 'train_data_true'  # 数据目录
    OUTPUT_DIR = PROJECT_ROOT / 'output'  # 输出目录
    CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'  # 检查点目录
    LOG_DIR = OUTPUT_DIR / 'logs'  # 日志目录
    STATE_FILE = CHECKPOINT_DIR / "training_state.json"  # 训练状态文件
    LAST_CHECKPOINT = CHECKPOINT_DIR / "last_checkpoint.pth"  # 最新检查点
    BEST_CHECKPOINT = CHECKPOINT_DIR / "best_checkpoint.pth"  # 最佳检查点
    
    def __init__(self):
        # 检查数据目录
        if not self.DATA_DIR.exists():
            raise RuntimeError(f"数据目录不存在: {self.DATA_DIR}")
        
        # 创建必要的目录（使用parents=True确保父目录也被创建）
        self.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        self.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        self.LOG_DIR.mkdir(exist_ok=True, parents=True)
        
        # 备份并清理旧的检查点
        self._backup_and_clean_checkpoints()
        
        # 检查CUDA可用性
        self.CUDA_AVAILABLE = torch.cuda.is_available()
        if not self.CUDA_AVAILABLE and self.DEVICE == 'cuda':
            self.DEVICE = 'cpu'
            self.AMP_ENABLED = False
            print("警告: CUDA不可用,切换到CPU模式")
        
        # 不限制torch线程数
        if torch.cuda.is_available():
            # 不限制GPU内存使用
            torch.cuda.empty_cache()
        
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
        print(f"是否使用AMP: {self.AMP_ENABLED}")
        print(f"是否使用数据增强: Mixup={self.USE_MIXUP}, CutMix={self.USE_CUTMIX}")
        print("===============\n")
    
    def _backup_and_clean_checkpoints(self):
        """备份并清理旧的检查点文件"""
        timestamp = int(time.time())
        
        # 备份检查点文件
        for checkpoint_file in [self.LAST_CHECKPOINT, self.BEST_CHECKPOINT]:
            if checkpoint_file.exists():
                backup_file = checkpoint_file.parent / f"{checkpoint_file.stem}_backup_{timestamp}{checkpoint_file.suffix}"
                shutil.copy2(checkpoint_file, backup_file)
                checkpoint_file.unlink()  # 删除原文件
                print(f"已备份检查点到: {backup_file}")
        
        # 清理训练状态文件
        if self.STATE_FILE.exists():
            backup_file = self.STATE_FILE.parent / f"training_state_backup_{timestamp}.json"
            shutil.copy2(self.STATE_FILE, backup_file)
            self.STATE_FILE.unlink()  # 删除原文件
            print(f"已备份训练状态到: {backup_file}")
        
        print("已清理所有旧的检查点文件")
    
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
            'best_epoch': 0,
            'no_improvement_count': 0
        }
        
        # 如果存在旧的状态文件，备份它
        if self.STATE_FILE.exists():
            backup_file = self.STATE_FILE.parent / f"training_state_backup_{int(time.time())}.json"
            shutil.copy2(self.STATE_FILE, backup_file)
            print(f"已备份旧的训练状态到: {backup_file}")
        
        # 返回新的默认状态
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
                weight_decay=self.WEIGHT_DECAY,
                nesterov=True
            )
        else:
            raise ValueError(f"不支持的优化器类型: {self.OPTIMIZER}")
    
    def get_lr_scheduler(self, optimizer):
        """获取学习率调度器"""
        if self.LR_SCHEDULER == 'one_cycle':
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.MAX_LR,
                epochs=self.MAX_EPOCHS,
                steps_per_epoch=1000,  # 根据实际情况调整
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1000.0
            )
        elif self.LR_SCHEDULER == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.MAX_EPOCHS,
                eta_min=self.MIN_LEARNING_RATE,
                last_epoch=self.training_state['current_epoch'] - 1
            )
        elif self.LR_SCHEDULER == 'cosine_warm':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,  # 第一次重启的epoch数
                T_mult=2,  # 每次重启后周期乘数
                eta_min=self.MIN_LEARNING_RATE
            )
        elif self.LR_SCHEDULER == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.LR_STEP_SIZE,
                gamma=self.LR_DECAY_RATE
            )
        elif self.LR_SCHEDULER == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.1,
                patience=5,
                verbose=True,
                min_lr=self.MIN_LEARNING_RATE
            )
        return None
    
    def should_stop_early(self, current_score):
        """检查是否应该早停"""
        if current_score > self.training_state['best_val_score']:
            self.training_state['no_improvement_count'] = 0
            self.training_state['best_val_score'] = current_score
            self.training_state['best_epoch'] = self.training_state['current_epoch']
            return False
        
        self.training_state['no_improvement_count'] += 1
        if self.training_state['no_improvement_count'] >= self.EARLY_STOP_PATIENCE:
            print(f"早停：{self.training_state['no_improvement_count']}轮未改善")
            return True
        
        return False
