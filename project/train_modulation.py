import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime, timedelta
import gc
import sys
import logging
import signal

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project.config import Config
from project.model.modulation_classifier import ModulationClassifier
from project.utils.data_processor import ModulationDataset

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
            scaler = torch.amp.GradScaler('cuda')
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