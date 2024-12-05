import os
import torch
import torch.optim as optim
from pathlib import Path
import json
import logging
from typing import Dict, Any, List, Union

class Config:
    """配置类"""
    
    # 资源参数
    MAX_WORKERS: int = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PIN_MEMORY = True
    
    # 数据参数
    SAMPLING_RATE: float = 20e6  # 20MHz采样率
    SAMPLES_PER_CLASS: int = 16200
    SEQUENCE_LENGTH: int = 1024
    MIN_SYMBOL_WIDTH: float = 1e-6  # 最小码元宽度(1us)
    MAX_SYMBOL_WIDTH: float = 1e-4  # 最大码元宽度(100us)
    
    # 模型参数
    BACKBONE_CHANNELS: List[int] = [64, 128, 256, 512, 1024]  # 主干网络通道数
    FEATURE_DIM: int = 1024
    DROPOUT_RATE: float = 0.2
    USE_TEMPERATURE_SCALING: bool = True
    TEMPERATURE: float = 1.0
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # 训练参数
    BATCH_SIZE: int = 128
    GRADIENT_ACCUMULATION_STEPS: int = 2
    MAX_EPOCHS: int = 300
    
    # 任务权重
    MT_WEIGHT: float = 0.5
    SW_WEIGHT: float = 0.3
    CQ_WEIGHT: float = 0.2
    
    # 评分阈值
    SW_THRESHOLDS: Dict[str, float] = {
        'perfect': 0.05,  # ERk <= 0.05 得满分
        'acceptable': 0.2  # 0.05 < ERk <= 0.2 按比例得分
    }
    
    CQ_THRESHOLDS: Dict[str, float] = {
        'perfect': 0.95,  # CSi >= 0.95 得满分
        'acceptable': 0.7  # 0.7 <= CSi < 0.95 按比例得分
    }
    
    # 优化器参数
    OPTIMIZER: str = 'adamw'
    LEARNING_RATE: float = 5e-4
    MIN_LEARNING_RATE: float = 1e-6
    WEIGHT_DECAY: float = 0.01
    MOMENTUM = 0.9
    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 1e-8
    
    # 学习率调度参数
    LR_SCHEDULER = 'one_cycle'
    WARMUP_PCT = 0.1
    DIV_FACTOR = 10.0
    FINAL_DIV_FACTOR = 1e3
    
    # 正则化参数
    LABEL_SMOOTHING: float = 0.1
    GRADIENT_CLIP_VAL: float = 0.5
    L2_REG_WEIGHT: float = 0.005
    
    # 数据增强参数
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2
    USE_CUTMIX = False
    CUTMIX_ALPHA = 1.0
    USE_SPECAUGMENT = True
    USE_RANDOM_ERASING = False
    
    # 早停参数
    EARLY_STOPPING_PATIENCE: int = 30
    EARLY_STOPPING_MIN_DELTA: float = 1e-4
    EARLY_STOPPING_METRIC: str = 'mt_score'
    
    # GPU优化参数
    AMP_ENABLED: bool = True
    
    # 调制类型映射
    MODULATION_DICT: Dict[int, str] = {
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
    
    # 调制类型参数
    MODULATION_PARAMS: Dict[str, Dict[str, Any]] = {
        'BPSK': {'constellation_points': 2, 'bits_per_symbol': 1},
        'QPSK': {'constellation_points': 4, 'bits_per_symbol': 2},
        '8PSK': {'constellation_points': 8, 'bits_per_symbol': 3},
        'MSK': {'constellation_points': 2, 'bits_per_symbol': 1},
        '8QAM': {'constellation_points': 8, 'bits_per_symbol': 3},
        '16QAM': {'constellation_points': 16, 'bits_per_symbol': 4},
        '32QAM': {'constellation_points': 32, 'bits_per_symbol': 5},
        '8APSK': {'constellation_points': 8, 'bits_per_symbol': 3},
        '16APSK': {'constellation_points': 16, 'bits_per_symbol': 4},
        '32APSK': {'constellation_points': 32, 'bits_per_symbol': 5}
    }
    
    # 日志和检查点参数
    LOG_INTERVAL: int = 10
    SAVE_INTERVAL: int = 100
    
    # wandb配置
    USE_WANDB: bool = True
    WANDB_PROJECT: str = "signal-cascade-model"
    WANDB_ENTITY: Union[str, None] = None
    WANDB_NAME: Union[str, None] = None
    WANDB_TAGS: List[str] = ['cascade-model', 'signal-processing']
    WANDB_NOTES: str = "级联信号处理模型训练"
    
    # 训练技巧
    USE_SCHEDULER: bool = True
    
    def __init__(self) -> None:
        """初始化配置"""
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # 设置路径
        self.WORKING_DIR = Path(os.getcwd()).resolve()
        self.OUTPUT_DIR = self.WORKING_DIR / 'output'
        self.CHECKPOINT_DIR = self.OUTPUT_DIR / 'checkpoints'
        self.LOG_DIR = self.OUTPUT_DIR / 'logs'
        self.DATA_DIR = self.WORKING_DIR / 'train_data_true'
        
        # 创建必要的目录
        self.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)
        
        # 检查数据目录
        if not self.DATA_DIR.exists():
            raise RuntimeError(f"数据目录不存在: {self.DATA_DIR}")
        
        # 检查CUDA可用性
        self.CUDA_AVAILABLE = torch.cuda.is_available()
        if not self.CUDA_AVAILABLE:
            logging.warning("CUDA不可用,切换到CPU模式")
            self.DEVICE = torch.device('cpu')
            self.AMP_ENABLED = False
        else:
            self.DEVICE = torch.device('cuda')
            
        # 打印置信息
        logging.info("配置初始化完成:")
        logging.info(f"工作目录: {self.WORKING_DIR}")
        logging.info(f"数据目录: {self.DATA_DIR}")
        logging.info(f"输出目录: {self.OUTPUT_DIR}")
        logging.info(f"设备: {self.DEVICE}")
        logging.info(f"批次大小: {self.BATCH_SIZE}")
        logging.info(f"学习率: {self.LEARNING_RATE}")
        logging.info(f"优化器: {self.OPTIMIZER}")
        logging.info(f"是否使用AMP: {self.AMP_ENABLED}")
        logging.info(f"是否使用调度器: {self.USE_SCHEDULER}")
        logging.info(f"任务权重: MT={self.MT_WEIGHT}, SW={self.SW_WEIGHT}, CQ={self.CQ_WEIGHT}")
    
    def get_optimizer(self, parameters) -> optim.Optimizer:
        """获取优化器"""
        if self.OPTIMIZER.lower() == 'adamw':
            return optim.AdamW(
                parameters,
                lr=self.LEARNING_RATE,
                weight_decay=self.WEIGHT_DECAY,
                betas=(self.BETA1, self.BETA2),
                eps=self.EPS
            )
        elif self.OPTIMIZER.lower() == 'adam':
            return optim.Adam(
                parameters,
                lr=self.LEARNING_RATE,
                weight_decay=self.WEIGHT_DECAY,
                betas=(self.BETA1, self.BETA2),
                eps=self.EPS
            )
        elif self.OPTIMIZER.lower() == 'sgd':
            return optim.SGD(
                parameters,
                lr=self.LEARNING_RATE,
                momentum=self.MOMENTUM,
                weight_decay=self.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"不支持的优化器: {self.OPTIMIZER}")
    
    def get_scheduler(self, optimizer) -> optim.lr_scheduler._LRScheduler:
        """获取学习率调度器"""
        if not self.USE_SCHEDULER:
            return None
            
        # 计算每个epoch的步数
        steps_per_epoch = self.SAMPLES_PER_CLASS * len(self.MODULATION_DICT) // self.BATCH_SIZE
        
        if self.LR_SCHEDULER.lower() == 'one_cycle':
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.LEARNING_RATE,
                epochs=self.MAX_EPOCHS,
                steps_per_epoch=steps_per_epoch,
                pct_start=self.WARMUP_PCT,
                div_factor=self.DIV_FACTOR,
                final_div_factor=self.FINAL_DIV_FACTOR,
                anneal_strategy='cos'
            )
        elif self.LR_SCHEDULER.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.MAX_EPOCHS,
                eta_min=self.MIN_LEARNING_RATE
            )
        elif self.LR_SCHEDULER.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.1,
                patience=10,
                min_lr=self.MIN_LEARNING_RATE
            )
        else:
            raise ValueError(f"不支持的学习率调度器: {self.LR_SCHEDULER}")
    
    def save_config(self, path: str) -> None:
        """保存配置到文件"""
        config_dict = {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
        
        # 将Path对象转换为字符串
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            elif isinstance(value, torch.device):
                config_dict[key] = str(value)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def load_config(self, path: str) -> None:
        """从文件加载配置"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # 恢复Path对象和device
        for key, value in config_dict.items():
            if key.endswith('_DIR'):
                config_dict[key] = Path(value)
            elif key == 'DEVICE':
                config_dict[key] = torch.device(value)
        
        self.__dict__.update(config_dict)
    
    def get_modulation_name(self, mod_type: int) -> str:
        """获取调制类型名称"""
        return self.MODULATION_DICT.get(mod_type, 'Unknown')
    
    def get_modulation_type(self, mod_name: str) -> int:
        """获取调制类型编号"""
        for mod_type, name in self.MODULATION_DICT.items():
            if name == mod_name:
                return mod_type
        return -1
    
    def get_num_classes(self) -> int:
        """获取类别数量"""
        return len(self.MODULATION_DICT)
    
    def get_class_names(self) -> List[str]:
        """获取所有类别名称"""
        return list(self.MODULATION_DICT.values())
    
    def get_class_weights(self) -> torch.Tensor:
        """获取类别权重"""
        weights = torch.ones(self.get_num_classes())
        return weights.to(self.DEVICE)
    
    def get_training_steps(self) -> int:
        """获取总训练步数"""
        steps_per_epoch = self.SAMPLES_PER_CLASS * len(self.MODULATION_DICT) // self.BATCH_SIZE
        return steps_per_epoch * self.MAX_EPOCHS
    
    def get_modulation_params(self, mod_type: Union[int, str]) -> Dict[str, Any]:
        """获取调制类型参数"""
        if isinstance(mod_type, int):
            mod_name = self.get_modulation_name(mod_type)
        else:
            mod_name = mod_type
        return self.MODULATION_PARAMS.get(mod_name, {})
