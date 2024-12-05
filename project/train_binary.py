import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import signal
from torch.cuda.amp import GradScaler
import wandb
from datetime import datetime

from project.config import Config
from project.model.modulation_classifier import BinaryModulationClassifier, SymbolWidthRegressor
from project.utils.data_processor import BinaryModulationDataset, SymbolWidthDataset

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

def setup_logging(config):
    """设置日志系统"""
    # 创建日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path('logs') / f"training_{timestamp}.log"
    log_file.parent.mkdir(exist_ok=True)
    
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

def train_binary_classifier(config, mod_type):
    """训练单个二分类模型"""
    logging.info(f"\n开始训练 {mod_type} 分类器...")
    
    # 创建数据集
    train_dataset = BinaryModulationDataset(config.DATA_DIR, config, mod_type, mode='train')
    val_dataset = BinaryModulationDataset(config.DATA_DIR, config, mod_type, mode='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # 创建模型
    model = BinaryModulationClassifier(config).to(config.DEVICE)
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss()
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # 创建梯度缩放器
    scaler = GradScaler() if config.AMP_ENABLED else None
    
    # 训练循环
    best_val_acc = 0.0
    for epoch in range(config.NUM_EPOCHS):
        if stop_flag:
            break
        
        # 训练一个epoch
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        start_time = time.time()
        
        logging.info(f"\nEpoch {epoch} 训练开始...")
        
        for batch_idx, batch in enumerate(train_loader):
            data = batch['data'].to(config.DEVICE)
            targets = batch['targets']['is_target'].float().to(config.DEVICE)
            
            optimizer.zero_grad()
            
            if config.AMP_ENABLED and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
                    loss = criterion(outputs['is_target'], targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(data)
                loss = criterion(outputs['is_target'], targets)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            pred = (torch.sigmoid(outputs['is_target']) > 0.5).float()
            train_correct += (pred == targets).sum().item()
            train_total += targets.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                           f"Loss: {loss.item():.4f} "
                           f"Acc: {train_correct/train_total:.4f} "
                           f"LR: {current_lr:.6f}")
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        logging.info(f"\nEpoch {epoch} 验证开始...")
        
        with torch.no_grad():
            for batch in val_loader:
                data = batch['data'].to(config.DEVICE)
                targets = batch['targets']['is_target'].float().to(config.DEVICE)
                
                outputs = model(data)
                loss = criterion(outputs['is_target'], targets)
                
                val_loss += loss.item()
                pred = (torch.sigmoid(outputs['is_target']) > 0.5).float()
                val_correct += (pred == targets).sum().item()
                val_total += targets.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        epoch_time = time.time() - start_time
        
        # 更新学习率
        scheduler.step(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_model_{mod_type}.pth")
            logging.info(f"保存新的最佳模型，验证准确率: {val_acc:.4f}")
        
        logging.info(f"\nEpoch {epoch} 完成 (用时: {epoch_time:.2f}s):")
        logging.info(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        logging.info(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        logging.info(f"当前最佳验证准确率: {best_val_acc:.4f}")
    
    logging.info(f"\n{mod_type} 分类器训练完成，最佳验证准确率: {best_val_acc:.4f}")
    return best_val_acc

def train_width_regressor(config):
    """训练码元宽度回归模型"""
    logger = logging.getLogger()
    logger.info("\n开始训练码元宽度回归模型...")
    
    # 创建数据集
    train_dataset = SymbolWidthDataset(config.DATA_DIR, config, mode='train')
    val_dataset = SymbolWidthDataset(config.DATA_DIR, config, mode='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # 创建模型
    model = SymbolWidthRegressor(config).to(config.DEVICE)
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # 创建梯度缩放器
    scaler = GradScaler() if config.AMP_ENABLED else None
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(config.NUM_EPOCHS):
        if stop_flag:
            break
        
        # 训练一个epoch
        model.train()
        train_loss = 0
        start_time = time.time()
        
        logger.info(f"\nEpoch {epoch} 训练开始...")
        
        for batch_idx, batch in enumerate(train_loader):
            data = batch['data'].to(config.DEVICE)
            targets = batch['targets']['symbol_width'].to(config.DEVICE)
            
            optimizer.zero_grad()
            
            if config.AMP_ENABLED and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
                    loss = criterion(outputs['symbol_width'], targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(data)
                loss = criterion(outputs['symbol_width'], targets)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                           f"Loss: {loss.item():.4f} "
                           f"LR: {current_lr:.6f}")
        
        # 验证
        model.eval()
        val_loss = 0
        
        logger.info(f"\nEpoch {epoch} 验证开始...")
        
        with torch.no_grad():
            for batch in val_loader:
                data = batch['data'].to(config.DEVICE)
                targets = batch['targets']['symbol_width'].to(config.DEVICE)
                
                outputs = model(data)
                loss = criterion(outputs['symbol_width'], targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        epoch_time = time.time() - start_time
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_width.pth")
            logger.info(f"保存新的最佳模型，验证损失: {val_loss:.4f}")
        
        logger.info(f"\nEpoch {epoch} 完成 (用时: {epoch_time:.2f}s):")
        logger.info(f"训练损失: {train_loss:.4f}")
        logger.info(f"验证损失: {val_loss:.4f}")
        logger.info(f"当前最佳验证损失: {best_val_loss:.4f}")
    
    logger.info(f"\n码元宽度回归模型训练完成，最佳验证损失: {best_val_loss:.4f}")
    return best_val_loss

def main():
    """主训练函数"""
    config = Config()
    
    # 设置日志
    logger = setup_logging(config)
    logger.info("=== 开始训练 ===")
    
    # 训练10个二分类模型
    results = {}
    for mod_type in config.MODULATION_DICT.values():
        logger.info(f"\n开始训练 {mod_type} 分类器")
        best_acc = train_binary_classifier(config, mod_type)
        results[mod_type] = best_acc
        logger.info(f"{mod_type} 分类器最佳准确率: {best_acc:.4f}")
    
    # 训练码元宽度回归模型
    logger.info("\n开始训练码元宽度回归模型")
    best_loss = train_width_regressor(config)
    logger.info(f"码元宽度回归模型最佳损失: {best_loss:.4f}")
    
    # 打印所有结果
    logger.info("\n=== 训练结果汇总 ===")
    for mod_type, acc in results.items():
        logger.info(f"{mod_type}: {acc:.4f}")
    logger.info(f"码元宽度回归损失: {best_loss:.4f}")

if __name__ == "__main__":
    main() 