import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Dict
import sys
import os
import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project.model.modulation_classifier import ModulationClassifier, ModulationLoss
from project.utils.data_processor import SignalProcessor
from project.config import Config

class ModulationDataset(Dataset):
    """调制信号数据集"""
    def __init__(self, data_dir: str, split: str = 'train', samples_per_class: int = 16200):
        self.samples = []
        self.processor = SignalProcessor()
        
        # 计算训练集和验证集的样本数
        train_samples = int(samples_per_class * 0.8)  # 80%用于训练
        val_samples = samples_per_class - train_samples  # 20%用于验证
        target_samples = train_samples if split == 'train' else val_samples
        
        # 统计每种调制方式的样本数
        mod_samples = {mod_type: [] for mod_type in Config.MODULATION_DICT.keys()}
        
        # 第一次遍历：计算最大序列长度
        self.max_seq_len = 0
        print("\n计算最大序列长度...")
        for mod_type, mod_name in Config.MODULATION_DICT.items():
            mod_dir = Path(data_dir) / mod_name
            if not mod_dir.exists():
                raise RuntimeError(f"未找到{mod_name}的数据目录: {mod_dir}")
            
            all_files = list(mod_dir.glob("*.csv"))
            if len(all_files) < target_samples:
                raise RuntimeError(f"{mod_name}的样本数量不足: {len(all_files)} < {target_samples}")
            
            # 随机选择一些文件计算最大长度
            sample_files = np.random.choice(all_files, min(100, len(all_files)), replace=False)
            for file_path in tqdm(sample_files, desc=f"处理{mod_name}"):
                try:
                    df = pd.read_csv(str(file_path), header=None)
                    symbol_width = float(df.iloc[0, 4])
                    code_sequence = pd.to_numeric(df.iloc[:, 2], errors='coerce').values
                    valid_code = code_sequence[~np.isnan(code_sequence)]
                    points_per_symbol = int(symbol_width * 20)  # 每微秒20个采样点
                    seq_len = len(valid_code) * points_per_symbol
                    self.max_seq_len = max(self.max_seq_len, seq_len)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
        
        print(f"最大序列长度: {self.max_seq_len}")
        
        # 第二次遍历：收集样本
        for mod_type, mod_name in Config.MODULATION_DICT.items():
            mod_dir = Path(data_dir) / mod_name
            all_files = list(mod_dir.glob("*.csv"))
            np.random.shuffle(all_files)
            
            # 根据split选择相应的样本范围
            if split == 'train':
                selected_files = all_files[:train_samples]
            else:  # val
                selected_files = all_files[train_samples:samples_per_class]
            
            # 添加到样本列表
            for file_path in selected_files:
                mod_samples[mod_type].append({
                    'file_path': str(file_path),
                    'modulation_type': mod_type - 1  # 转换为0-based索引
                })
        
        # 合并所有样本
        for mod_type, samples in mod_samples.items():
            self.samples.extend(samples)
            
        # 随机打乱
        np.random.shuffle(self.samples)
        
        print(f"\n[{split}] 数据集统计:")
        print(f"每种调制方式使用 {target_samples} 个样本")
        print(f"总样本数: {len(self.samples)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def pad_sequence(self, data: np.ndarray) -> np.ndarray:
        """填充序列到固定长度"""
        if len(data) > self.max_seq_len:
            return data[:self.max_seq_len]
        elif len(data) < self.max_seq_len:
            pad_width = self.max_seq_len - len(data)
            return np.pad(data, (0, pad_width), mode='constant', constant_values=0)
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        try:
            # 处理数据
            result = self.processor.process_file(str(Path(sample['file_path'])))
            
            # 填充序列
            i_data = self.pad_sequence(result['i_data'])
            q_data = self.pad_sequence(result['q_data'])
            
            # 转换为tensor
            return {
                'i_data': torch.tensor(i_data, dtype=torch.float32),
                'q_data': torch.tensor(q_data, dtype=torch.float32),
                'modulation_type': torch.tensor(sample['modulation_type'], dtype=torch.long),
                'symbol_width': torch.tensor(result['symbol_width'], dtype=torch.float32)
            }
            
        except Exception as e:
            print(f"Error loading sample {sample['file_path']}: {str(e)}")
            # 返回一个空的样本，使用最大序列长度
            return {
                'i_data': torch.zeros(self.max_seq_len, dtype=torch.float32),
                'q_data': torch.zeros(self.max_seq_len, dtype=torch.float32),
                'modulation_type': torch.tensor(sample['modulation_type'], dtype=torch.long),
                'symbol_width': torch.tensor(0.0, dtype=torch.float32)
            }

def calculate_width_accuracy(pred_width: torch.Tensor, true_width: torch.Tensor, 
                           relative_threshold: float = 0.05) -> torch.Tensor:
    """计算码元宽度预测的准确率
    Args:
        pred_width: 预测的码元宽度
        true_width: 真实的码元宽度
        relative_threshold: 相对误差阈值（默认5%）
    Returns:
        准确率
    """
    # 计算相对误差
    relative_error = torch.abs(pred_width - true_width) / true_width
    # 统计相对误差小于阈值的样本比例
    accuracy = (relative_error <= relative_threshold).float().mean()
    return accuracy

def train_modulation(
    data_dir: str = 'train_data_true',
    config: Config = None,
    device: str = None,
    checkpoint_path: str = None
) -> ModulationClassifier:
    """训练调制信号分类器"""
    # 1. 初始化配置
    if config is None:
        config = Config()
    if device is None:
        device = config.DEVICE
        
    # 2. 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 创建检查点目录
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. 创建数据加载器
    try:
        train_dataset = ModulationDataset(
            data_dir, 
            split='train',
            samples_per_class=config.SAMPLES_PER_CLASS
        )
        val_dataset = ModulationDataset(
            data_dir,
            split='val',
            samples_per_class=config.SAMPLES_PER_CLASS
        )
    except Exception as e:
        logger.error(f"Error creating datasets: {str(e)}")
        raise
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=config.MAX_WORKERS,
        drop_last=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.MAX_WORKERS,
        drop_last=True,
        pin_memory=True
    )
    
    # 4. 创建模型和优化器
    model = ModulationClassifier(
        num_classes=len(config.MODULATION_DICT),
        feature_dim=config.FEATURE_DIM,
        input_size=train_dataset.max_seq_len
    ).to(device)
    
    criterion = ModulationLoss(config)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5,
        verbose=True
    )
    
    # 5. 加载检查点（如果存在）
    start_epoch = 0
    best_val_acc = 0.0
    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['best_acc']
            logger.info(f"Resuming from epoch {start_epoch} with best accuracy: {best_val_acc:.4f}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            logger.info("Starting training from scratch")
    
    # 6. 训练循环
    try:
        for epoch in range(start_epoch, config.NUM_EPOCHS):
            # 训练阶段
            model.train()
            train_stats = {
                'loss': 0.0,
                'mod_acc': 0.0,
                'width_acc': 0.0,
                'width_error': 0.0,
                'samples': 0
            }
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}'):
                try:
                    # 准备数据
                    i_data = batch['i_data'].to(device)
                    q_data = batch['q_data'].to(device)
                    
                    targets = {
                        'modulation_type': batch['modulation_type'].to(device),
                        'symbol_width': batch['symbol_width'].to(device)
                    }
                    
                    # 前向传播
                    outputs = model(i_data, q_data)
                    
                    # 计算损失
                    loss_dict = criterion(outputs, targets)
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss_dict['total_loss'].backward()
                    optimizer.step()
                    
                    # 计算准确率
                    with torch.no_grad():
                        # 调制类型准确率
                        mod_pred = torch.argmax(outputs['modulation_type'], dim=1)
                        mod_acc = (mod_pred == targets['modulation_type']).float().mean()
                        
                        # 码元宽度准确率和误差
                        width_acc = calculate_width_accuracy(
                            outputs['symbol_width'].squeeze(),
                            targets['symbol_width']
                        )
                        width_error = torch.abs(outputs['symbol_width'].squeeze() - targets['symbol_width']) / targets['symbol_width']
                        width_error = width_error.mean()
                    
                    # 更新统计信息
                    batch_size = i_data.size(0)
                    train_stats['loss'] += loss_dict['total_loss'].item() * batch_size
                    train_stats['mod_acc'] += mod_acc.item() * batch_size
                    train_stats['width_acc'] += width_acc.item() * batch_size
                    train_stats['width_error'] += width_error.item() * batch_size
                    train_stats['samples'] += batch_size
                    
                except Exception as e:
                    logger.error(f"Error in training batch: {str(e)}")
                    continue
            
            # 计算训练阶段平均值
            if train_stats['samples'] > 0:
                for key in ['loss', 'mod_acc', 'width_acc', 'width_error']:
                    train_stats[key] /= train_stats['samples']
            
            # 验证阶段
            model.eval()
            val_stats = {
                'loss': 0.0,
                'mod_acc': 0.0,
                'width_acc': 0.0,
                'width_error': 0.0,
                'samples': 0
            }
            
            width_errors = []  # 收集所有码元宽度误差
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='Validation'):
                    try:
                        # 准备数据
                        i_data = batch['i_data'].to(device)
                        q_data = batch['q_data'].to(device)
                        
                        targets = {
                            'modulation_type': batch['modulation_type'].to(device),
                            'symbol_width': batch['symbol_width'].to(device)
                        }
                        
                        # 前向传播
                        outputs = model(i_data, q_data)
                        
                        # 计算损失
                        loss_dict = criterion(outputs, targets)
                        
                        # 计算准确率
                        mod_pred = torch.argmax(outputs['modulation_type'], dim=1)
                        mod_acc = (mod_pred == targets['modulation_type']).float().mean()
                        
                        # 码元宽度准确率和误差
                        width_acc = calculate_width_accuracy(
                            outputs['symbol_width'].squeeze(),
                            targets['symbol_width']
                        )
                        width_error = torch.abs(outputs['symbol_width'].squeeze() - targets['symbol_width']) / targets['symbol_width']
                        width_errors.extend(width_error.cpu().numpy())
                        width_error = width_error.mean()
                        
                        # 更新统计信息
                        batch_size = i_data.size(0)
                        val_stats['loss'] += loss_dict['total_loss'].item() * batch_size
                        val_stats['mod_acc'] += mod_acc.item() * batch_size
                        val_stats['width_acc'] += width_acc.item() * batch_size
                        val_stats['width_error'] += width_error.item() * batch_size
                        val_stats['samples'] += batch_size
                        
                    except Exception as e:
                        logger.error(f"Error in validation batch: {str(e)}")
                        continue
            
            # 计算验证阶段平均值
            if val_stats['samples'] > 0:
                for key in ['loss', 'mod_acc', 'width_acc', 'width_error']:
                    val_stats[key] /= val_stats['samples']
                
                # 更新学习率
                scheduler.step(val_stats['loss'])
                
                # 计算总体准确率（调整权重）
                val_total_acc = (
                    0.4 * val_stats['mod_acc'] +  # 增加调制类型权重
                    0.6 * val_stats['width_acc']  # 增加码元宽度权重
                )
                
                # 保存检查点
                try:
                    # 保存最佳模型
                    if val_total_acc > best_val_acc:
                        best_val_acc = val_total_acc
                        best_model_path = checkpoint_dir / 'best_model.pth'
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'best_acc': best_val_acc,
                            'config': config.__dict__
                        }, str(best_model_path))
                        logger.info(f"Saved new best model with accuracy: {best_val_acc:.4f}")
                    
                    # 保存最新检查点
                    latest_checkpoint_path = checkpoint_dir / 'latest_checkpoint.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_acc': best_val_acc,
                        'config': config.__dict__
                    }, str(latest_checkpoint_path))
                except Exception as e:
                    logger.error(f"Error saving checkpoint: {str(e)}")
                    continue
                
                # 计算码元宽度误差的统计信息
                width_errors = np.array(width_errors)
                error_stats = {
                    'mean': np.mean(width_errors),
                    'std': np.std(width_errors),
                    'median': np.median(width_errors),
                    'p95': np.percentile(width_errors, 95)
                }
                
                # 输出日志
                logger.info(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}:")
                logger.info("Training:")
                logger.info(f"  Loss: {train_stats['loss']:.4f}")
                logger.info(f"  Modulation Acc: {train_stats['mod_acc']:.4f}")
                logger.info(f"  Width Acc (±5%): {train_stats['width_acc']:.4f}")
                logger.info(f"  Width Relative Error: {train_stats['width_error']:.4f}")
                logger.info("\nValidation:")
                logger.info(f"  Loss: {val_stats['loss']:.4f}")
                logger.info(f"  Modulation Acc: {val_stats['mod_acc']:.4f}")
                logger.info(f"  Width Acc (±5%): {val_stats['width_acc']:.4f}")
                logger.info(f"  Width Error Statistics:")
                logger.info(f"    - Mean: {error_stats['mean']:.4f}")
                logger.info(f"    - Std: {error_stats['std']:.4f}")
                logger.info(f"    - Median: {error_stats['median']:.4f}")
                logger.info(f"    - 95th percentile: {error_stats['p95']:.4f}")
                logger.info(f"  Total Acc: {val_total_acc:.4f}")
                logger.info(f"  Best Acc: {best_val_acc:.4f}")
            
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        # 保存中断时的检查点
        try:
            interrupt_checkpoint_path = checkpoint_dir / 'interrupt_checkpoint.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_val_acc,
                'config': config.__dict__
            }, str(interrupt_checkpoint_path))
            logger.info("Saved interrupt checkpoint")
        except Exception as e:
            logger.error(f"Error saving interrupt checkpoint: {str(e)}")
    
    return model

if __name__ == '__main__':
    config = Config()
    try:
        # 检查是否存在中断检查点
        checkpoint_dir = Path('checkpoints')
        checkpoint_path = None
        
        if (checkpoint_dir / 'interrupt_checkpoint.pth').exists():
            checkpoint_path = str(checkpoint_dir / 'interrupt_checkpoint.pth')
        elif (checkpoint_dir / 'latest_checkpoint.pth').exists():
            checkpoint_path = str(checkpoint_dir / 'latest_checkpoint.pth')
            
        train_modulation(
            data_dir='train_data_true',
            config=config,
            checkpoint_path=checkpoint_path
        )
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise