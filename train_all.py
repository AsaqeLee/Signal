import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import gc
import sys
import logging
import signal
import json

# 全局停止标志
stop_flag = False

def signal_handler(signum, frame):
    """处理中断信号"""
    global stop_flag
    logging.info("\n收到停止信号,将在当前epoch训练完成后停止...")
    stop_flag = True

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class Config:
    """配置类"""
    
    # 资源限制 - 在Kaggle环境中不限制
    MAX_WORKERS = os.cpu_count()  # 使用所有可用CPU核心
    MAX_MEMORY = None  # 不限制内存使用
    
    # 数据参数
    SAMPLING_RATE = 20e6  # 20MHz，即每微秒20个采样点
    SAMPLES_PER_CLASS = 16200  # 每个调制类型的样本数
    
    # 模型参数
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 128  # 增大批次大小以充分利用GPU
    
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
    MAX_EPOCHS = 100  # 最大训练轮数
    TARGET_ACCURACY = 0.95  # 目标准确率
    LEARNING_RATE = 0.001
    MIN_LEARNING_RATE = 1e-6
    VALIDATION_RATIO = 0.2
    GRADIENT_ACCUMULATION_STEPS = 8
    
    # 优化器参数
    OPTIMIZER = 'adam'
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
    FEATURE_DIM = 128
    DROPOUT_RATE = 0.5
    LABEL_SMOOTHING = 0.1
    
    # 数据增强参数
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2
    USE_CUTMIX = True
    CUTMIX_ALPHA = 1.0
    
    # GPU优化参数
    AMP_ENABLED = True
    PIN_MEMORY = True
    GRADIENT_CLIP_VAL = 1.0
    
    # 文件路径
    INPUT_DIR = Path('/kaggle/input/train-data-true')  # 只读数据目录
    WORKING_DIR = Path('/kaggle/working')  # 可写工作目录
    OUTPUT_DIR = WORKING_DIR / 'output'  # 输出目录
    CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'  # 检查点目录
    LOG_DIR = OUTPUT_DIR / 'logs'  # 日志目录
    STATE_FILE = CHECKPOINT_DIR / "training_state.json"  # 训练状态文件
    LAST_CHECKPOINT = CHECKPOINT_DIR / "last_checkpoint.pth"  # 最新检查点
    BEST_CHECKPOINT = CHECKPOINT_DIR / "best_checkpoint.pth"  # 最佳检查点
    DATA_DIR = INPUT_DIR  # 添加这一行，使用INPUT_DIR作为数据目录
    
    # 日志参数
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 100
    
    def __init__(self):
        # 检查输入目录
        if not self.INPUT_DIR.exists():
            raise RuntimeError(f"数据目录不存在: {self.INPUT_DIR}")
            
        # 创建工作目录（只在/kaggle/working下创建）
        self.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)
        
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
                T_max=self.MAX_EPOCHS,
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
                total_iters=self.MAX_EPOCHS
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

class BaseModel(nn.Module):
    """模型基类"""
    def __init__(self):
        super().__init__()
        self.config = Config()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        raise NotImplementedError
    
    def get_loss_function(self):
        """获取损失函数"""
        raise NotImplementedError

class ModulationClassifier(BaseModel):
    """调制分类器"""
    def __init__(self):
        super().__init__()
        
        # 特征提取
        self.features = nn.Sequential(
            # 第一层卷积块
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # 第二层卷积块
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # 第三层卷积块 (添加残差连接)
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # 注意力层
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 残差连接
        self.shortcut = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256)
        )
        
        # 调制类型分类
        self.modulation_classifier = nn.Sequential(
            nn.Linear(256, self.config.FEATURE_DIM),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT_RATE),
            nn.Linear(self.config.FEATURE_DIM, len(self.config.MODULATION_DICT))
        )
        
        # 码元宽度预测
        self.width_regressor = nn.Sequential(
            nn.Linear(256, self.config.FEATURE_DIM),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT_RATE),
            nn.Linear(self.config.FEATURE_DIM, 1),
            nn.Softplus()  # 确保输出为正值
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 特征提取
        features = self.features(x)
        features = features.squeeze(-1)
        
        # 预测
        return {
            'modulation_type': self.modulation_classifier(features),
            'symbol_width': self.width_regressor(features)
        }
    
    def get_loss_function(self):
        """获取损失函数"""
        def criterion(outputs, targets):
            # 调制类型损失（带标签平滑的交叉熵）
            mod_loss = self._label_smoothing_loss(
                outputs['modulation_type'],
                targets['modulation_type'],
                smoothing=0.1
            )
            
            # 码元宽度损失（相对误差 + Huber损失的组合）
            width_loss = self._width_loss(
                outputs['symbol_width'].squeeze(),
                targets['symbol_width']
            )
            
            # 总损失
            total_loss = (
                self.config.MT_WEIGHT * mod_loss +
                self.config.SW_WEIGHT * width_loss
            )
            
            return total_loss
        
        return criterion
    
    def _label_smoothing_loss(self, pred, target, smoothing=0.1):
        """带标签平滑的交叉熵损失"""
        n_classes = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - smoothing) + smoothing / n_classes
        log_prob = F.log_softmax(pred, dim=1)
        return (-smooth_one_hot * log_prob).sum(dim=1).mean()
    
    def _width_loss(self, pred, target, beta=0.1):
        """组合码元宽度损失"""
        # 相对误差
        relative_error = torch.abs(pred - target) / target
        
        # Huber损失
        huber_loss = F.smooth_l1_loss(pred, target, beta=beta)
        
        # 组合损失
        return relative_error.mean() + huber_loss

class ModulationDataset(Dataset):
    """调制信号数据集"""
    def __init__(self, mode='train'):
        self.config = Config()
        self.mode = mode
        self.data_dir = self.config.DATA_DIR
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        all_data = []
        all_labels = []
        
        # 遍历每个调制类型文件夹
        for mod_type, mod_name in self.config.MODULATION_DICT.items():
            mod_dir = self.data_dir / mod_name
            if not mod_dir.exists():
                logging.warning(f"调制类型文件夹不存在: {mod_dir}")
                continue
                
            # 读取该调制类型下的所有数据文件
            files = list(mod_dir.glob('*'))  # 获取所有文件
            logging.info(f"找到{len(files)}个文件在{mod_name}文件夹中")
            
            for file in files:
                try:
                    # 加载数据
                    data = np.load(file)  # 假设数据是numpy格式
                    
                    # 添加到列表
                    all_data.append(data)
                    all_labels.append({
                        'mod_type': mod_type,
                        'symbol_width': 1.0  # 需要根据实际情况设置
                    })
                except Exception as e:
                    logging.warning(f"加载文件失败 {file}: {str(e)}")
        
        if not all_data:
            raise RuntimeError("未找到任何有效的数据文件")
            
        # 转换为numpy数组
        self.data = np.array(all_data)
        self.labels = all_labels
        
        # 训练/验证集划分
        total_samples = len(self.data)
        indices = np.random.permutation(total_samples)
        split_idx = int(total_samples * 0.8)
        
        if self.mode == 'val':
            val_indices = indices[split_idx:]
            self.data = self.data[val_indices]
            self.labels = [self.labels[i] for i in val_indices]
        else:  # 训练模式
            train_indices = indices[:split_idx]
            self.data = self.data[train_indices]
            self.labels = [self.labels[i] for i in train_indices]
        
        logging.info(f"加载{self.mode}数据: {len(self.data)}个样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx]).float()
        label = self.labels[idx]
        
        return {
            'data': data,
            'targets': {
                'modulation_type': torch.tensor(label['mod_type']).long(),
                'symbol_width': torch.tensor(label['symbol_width']).float()
            }
        }

def setup_logging(config):
    """设置日志系统"""
    # 创建日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = config.LOG_DIR / f"training_{timestamp}.log"
    
    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console_handler)
    
    return logger

def train_one_epoch(model, train_loader, criterion, optimizer, config, epoch, scaler=None):
    """训练一个epoch"""
    global stop_flag
    model.train()
    total_loss = 0
    start_time = time.time()
    
    # 获取当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    logging.info(f"\n当前学习率: {current_lr:.6f}")
    
    try:
        for batch_idx, batch in enumerate(train_loader):
            if stop_flag:
                break
                
            data = batch['data'].to(config.DEVICE)
            targets = {k: v.to(config.DEVICE) for k, v in batch['targets'].items()}
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 使用自动混合精度
            if config.AMP_ENABLED and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                
                # 缩放损失并反向传播
                scaler.scale(loss).backward()
                
                # 梯度累积
                if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                    # 梯度裁剪
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VAL)
                    
                    # 优化器步进
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # 常规训练
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # 梯度累积
                if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VAL)
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # 打印进度
            if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                logging.info(f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f} "
                          f"Time: {time.time()-start_time:.2f}s")
        
    except Exception as e:
        logging.error(f"训练过程发生错误: {str(e)}")
        raise
    finally:
        # 确保清理资源
        if stop_flag:
            # 关闭数据加载器
            train_loader._iterator = None
            
    avg_loss = total_loss / (batch_idx + 1)  # 使用实际处理的批次数
    epoch_time = time.time() - start_time
    
    return avg_loss, epoch_time

def validate(model, val_loader, criterion, config):
    """验证模型"""
    model.eval()
    total_loss = 0
    correct_mod = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            data = batch['data'].to(config.DEVICE)
            targets = {k: v.to(config.DEVICE) for k, v in batch['targets'].items()}
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # 计算调制类型准确率
            pred_mod = torch.argmax(outputs['modulation_type'], dim=1)
            correct_mod += (pred_mod == targets['modulation_type']).sum().item()
            total += data.size(0)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct_mod / total
    
    return avg_loss, accuracy

def save_checkpoint(model, optimizer, scheduler, config, train_loss, val_loss, val_accuracy):
    """保存检查点"""
    checkpoint = {
        'epoch': config.training_state['current_epoch'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'config': config.__dict__
    }
    
    # 保存最新检查点
    torch.save(checkpoint, config.LAST_CHECKPOINT)
    
    # 如果是最佳模型，保存最佳检查点
    if val_accuracy > config.training_state['best_val_score']:
        torch.save(checkpoint, config.BEST_CHECKPOINT)
        logging.info(f"保存最佳模型，准确率: {val_accuracy:.4f}")

def cleanup():
    """清理资源"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def train_one_round():
    """运行一轮训练"""
    global stop_flag
    train_loader = None
    val_loader = None
    
    try:
        # 加载配置
        config = Config()
        
        # 获取当前epoch和最大轮数
        current_epoch = config.training_state['current_epoch']
        max_epochs = config.MAX_EPOCHS
        
        # 检查是否达到最大轮数
        if current_epoch >= max_epochs:
            logging.info(f"\n已达到最大训练轮数 {max_epochs},训练结束")
            return True
            
        # 设置日志
        logger = setup_logging(config)
        
        # 创建数据加载器
        train_dataset = ModulationDataset(mode='train')
        val_dataset = ModulationDataset(mode='val')
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.MAX_WORKERS,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=False  # 禁用持久化worker
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.MAX_WORKERS,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=False  # 禁用持久化worker
        )
        
        # 创建或加载模型
        model = ModulationClassifier()
        model.to(config.DEVICE)
        
        # 创建优化器和损失函数
        optimizer = config.get_optimizer(model.parameters())
        criterion = model.get_loss_function()
        
        # 如果存在检查点，加载模型状态
        if config.LAST_CHECKPOINT.exists():
            logging.info(f"加载检查点: {config.LAST_CHECKPOINT}")
            checkpoint = torch.load(config.LAST_CHECKPOINT, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 设置初始学习率
            for param_group in optimizer.param_groups:
                param_group['initial_lr'] = config.LEARNING_RATE
        
        # 创建学习率调度器
        scheduler = config.get_lr_scheduler(optimizer)
        if scheduler and config.LAST_CHECKPOINT.exists():
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 创建AMP缩放器
        if config.AMP_ENABLED and torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
            config.AMP_ENABLED = False  # 禁用AMP
        
        # 训练一个epoch
        logging.info(f"\n开始第 {current_epoch + 1} 轮训练...")
        train_loss, epoch_time = train_one_epoch(
            model, train_loader, criterion, optimizer, config, 
            current_epoch + 1, scaler
        )
        
        # 验证
        val_loss, val_accuracy = validate(model, val_loader, criterion, config)
        logging.info(f"\n验证结果 - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        # 更新学习率
        if scheduler:
            scheduler.step()
        
        # 保存检查点
        save_checkpoint(model, optimizer, scheduler, config, train_loss, val_loss, val_accuracy)
        
        # 更新训练状态
        config.save_training_state(
            current_epoch=current_epoch + 1,
            best_val_score=max(val_accuracy, config.training_state['best_val_score']),
            total_epochs_run=config.training_state['total_epochs_run'] + 1,
            last_lr=optimizer.param_groups[0]['lr'],
            training_time=config.training_state['training_time'] + epoch_time
        )
        
        # 打印训练信息
        logging.info(f"\n第 {current_epoch + 1}/{max_epochs} 轮训练完成:")
        logging.info(f"训练损失: {train_loss:.4f}")
        logging.info(f"验证损失: {val_loss:.4f}")
        logging.info(f"验证准确率: {val_accuracy:.4f}")
        logging.info(f"耗时: {timedelta(seconds=int(epoch_time))}")
        logging.info(f"总训练时间: {timedelta(seconds=int(config.training_state['training_time']))}")
        
        # 检查是否达到目标准确率
        if val_accuracy >= config.TARGET_ACCURACY:
            logging.info(f"\n已达到目标准确率 {config.TARGET_ACCURACY},训练结束")
            return True
            
        # 检查是否收到停止信号
        if stop_flag:
            logging.info("\n收到停止信号,训练已暂停")
            return True
        
        # 检查是否应该早停
        if config.should_stop_early(val_accuracy):
            logging.info("\n达到早停条件,训练结束")
            return True
            
        return False

    except Exception as e:
        logging.error(f"训练过程中出错: {str(e)}", exc_info=True)
        raise
    finally:
        # 清理资源
        if train_loader:
            train_loader._iterator = None
        if val_loader:
            val_loader._iterator = None
        cleanup()

if __name__ == '__main__':
    config = Config()
    logging.info(f"\n开始训练,最大训练轮数: {config.MAX_EPOCHS}")
    logging.info(f"目标准确率: {config.TARGET_ACCURACY}")
    
    try:
        while True:
            if stop_flag:
                break
            stopped = train_one_round()
            if stopped:
                break
    except KeyboardInterrupt:
        logging.info("\n收到键盘中断,正在优雅停止...")
        stop_flag = True
    finally:
        logging.info("\n训练结束")