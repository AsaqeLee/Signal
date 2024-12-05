import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
import wandb
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, List, Union, Tuple, Optional

from project.config import Config
from project.data.dataset import MultiTaskSignalDataset
from project.model.multi_task_model import MultiTaskModel
from project.utils.metrics import calculate_modulation_metrics, calculate_symbol_width_metrics, calculate_symbol_sequence_metrics
from project.utils.early_stopping import EarlyStopping

def calculate_scores(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    config: Config
) -> Dict[str, float]:
    """计算各个任务的评分"""
    scores = {}
    details = {}
    
    # 1. 调制分类评分 (MT_score)
    mod_pred = outputs['modulation_type'].argmax(dim=1)
    mod_correct = (mod_pred == targets['modulation_type'])
    mod_accuracy = mod_correct.float().mean().item() * 100  # 转换为百分比
    scores['mt_score'] = mod_accuracy
    
    # 计算每种调制类型的准确率
    details['mod_type_accuracy'] = {}
    for mod_type, mod_name in config.MODULATION_DICT.items():
        mod_type_mask = targets['modulation_type'] == (mod_type - 1)  # 转换为0-based索引
        if mod_type_mask.any():
            type_correct = mod_correct[mod_type_mask]
            type_accuracy = type_correct.float().mean().item() * 100
            details['mod_type_accuracy'][mod_name] = type_accuracy
    
    # 2. 码元宽度评分 (SW_score)
    width_error = torch.abs(
        outputs['symbol_width'].squeeze(1) -
        targets['symbol_width']
    ) / targets['symbol_width']
    
    width_correct = width_error <= config.SW_THRESHOLDS['acceptable']
    width_accuracy = width_correct.float().mean().item() * 100
    
    # 记录码元宽度的预测值和实际值
    details['symbol_width'] = {
        'predictions': outputs['symbol_width'].squeeze(1).detach().cpu().numpy().tolist(),
        'targets': targets['symbol_width'].detach().cpu().numpy().tolist(),
        'errors': width_error.detach().cpu().numpy().tolist()
    }
    
    width_scores = torch.zeros_like(width_error)
    width_scores[width_error <= config.SW_THRESHOLDS['perfect']] = 100
    mask = (width_error > config.SW_THRESHOLDS['perfect']) & (width_error <= config.SW_THRESHOLDS['acceptable'])
    width_scores[mask] = (1 - (width_error[mask] - config.SW_THRESHOLDS['perfect']) / 
                        (config.SW_THRESHOLDS['acceptable'] - config.SW_THRESHOLDS['perfect'])) * 100
    scores['sw_score'] = width_scores.mean().item()
    
    # 3. 码元序列评分 (CQ_score)
    # 只有当调制类型和码元宽度的准确率都超过80%时才计算序列相似度
    if mod_accuracy >= 80 and width_accuracy >= 80:
        # 获取同时满足调制类型和码元宽度正确的样本
        valid_mask = mod_correct & width_correct
        
        if valid_mask.any():
            # 获取预测和目标序列
            pred_seq = outputs['symbol_sequence'].squeeze(1)[valid_mask]  # [N, L]
            true_seq = targets['symbol_sequence'][valid_mask]  # [N, L]
            
            # 确保序列长度一致
            min_length = min(pred_seq.size(1), true_seq.size(1))
            pred_seq = pred_seq[:, :min_length]
            true_seq = true_seq[:, :min_length]
            
            # 计算余弦似度
            cos_sim = F.cosine_similarity(pred_seq, true_seq, dim=1).abs()
            
            # 计算分数
            seq_scores = torch.zeros_like(cos_sim)
            seq_scores[cos_sim >= config.CQ_THRESHOLDS['perfect']] = 100
            mask = (cos_sim >= config.CQ_THRESHOLDS['acceptable']) & (cos_sim < config.CQ_THRESHOLDS['perfect'])
            seq_scores[mask] = ((cos_sim[mask] - config.CQ_THRESHOLDS['acceptable']) / 
                              (config.CQ_THRESHOLDS['perfect'] - config.CQ_THRESHOLDS['acceptable'])) * 100
            scores['cq_score'] = seq_scores.mean().item()
            
            # 记录序列相似度
            details['sequence_similarity'] = cos_sim.detach().cpu().numpy().tolist()
        else:
            scores['cq_score'] = 0.0
    else:
        scores['cq_score'] = 0.0
    
    # 4. 计算总分
    # 根据准确率阈值调整权重
    if mod_accuracy >= 80 and width_accuracy >= 80:
        # 当准确率达标时,增加序列预测的权重
        scores['total_score'] = (
            0.3 * scores['mt_score'] +    # 保持调制分类的权重
            0.3 * scores['sw_score'] +    # 保持码元宽度的权重
            0.4 * scores['cq_score']      # 增加序列预测的权重
        )
    else:
        # 当确率未达标时,只关注调制分类和码元宽度
        scores['total_score'] = (
            0.5 * scores['mt_score'] +    # 增加调制分类的权重
            0.5 * scores['sw_score']      # 增加码元宽度的权重
        )
    
    return scores, details

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: callable,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    config: Config,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler = None
) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()
    total_loss = 0
    score_sums = {'mt_score': 0, 'sw_score': 0, 'cq_score': 0, 'total_score': 0}
    num_batches = len(train_loader)
    
    # 使用tqdm显示进度
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # 获取数据和标签
        data = batch['data'].to(config.DEVICE)
        targets = {k: v.to(config.DEVICE) for k, v in batch['targets'].items()}
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        if config.AMP_ENABLED and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model({'data': data, 'targets': targets})
                loss_dict = criterion(outputs, targets)
                loss = loss_dict['total_loss']
                
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度累积
            if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VAL)
                
                # 优化器步进
                scaler.step(optimizer)
                scaler.update()
                
                # 学习率调度
                if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
        else:
            # 常规训练
            outputs = model({'data': data, 'targets': targets})
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['total_loss']
            loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VAL)
                optimizer.step()
                optimizer.zero_grad()
                
                # 习率调度
                if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
        
        # 计算各项评分
        scores, details = calculate_scores(outputs, targets, config)
        for k, v in scores.items():
            score_sums[k] += v
        
        total_loss += loss.item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mt_score': f'{scores["mt_score"]:.1f}',
            'sw_score': f'{scores["sw_score"]:.1f}',
            'cq_score': f'{scores["cq_score"]:.1f}',
            'total': f'{scores["total_score"]:.1f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # 记录到wandb和日志
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            # 记录基本指标
            log_dict = {
                'train_batch_loss': loss.item(),
                'train_batch_mt_loss': loss_dict['modulation_loss'].item(),
                'train_batch_sw_loss': loss_dict['width_loss'].item(),
                'train_batch_cq_loss': loss_dict['sequence_loss'].item(),
                'train_batch_mt_score': scores['mt_score'],
                'train_batch_sw_score': scores['sw_score'],
                'train_batch_cq_score': scores['cq_score'],
                'train_batch_total_score': scores['total_score'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            # 记录每种调制类型的准确率
            for mod_name, accuracy in details['mod_type_accuracy'].items():
                log_dict[f'train_batch_accuracy_{mod_name}'] = accuracy
            
            # 记录码元宽度统计信息
            width_errors = np.array(details['symbol_width']['errors'])
            log_dict.update({
                'train_batch_width_error_mean': width_errors.mean(),
                'train_batch_width_error_std': width_errors.std(),
                'train_batch_width_error_max': width_errors.max(),
                'train_batch_width_error_min': width_errors.min()
            })
            
            if config.USE_WANDB:
                wandb.log(log_dict)
            
            # 输出详细日志
            logging.info(f"\nBatch {batch_idx + 1}/{num_batches}")
            logging.info("调制类型准确率:")
            for mod_name, accuracy in details['mod_type_accuracy'].items():
                logging.info(f"  {mod_name}: {accuracy:.2f}%")
            logging.info(f"码元宽度误差统计:")
            logging.info(f"  平均误差: {width_errors.mean():.4f}")
            logging.info(f"  最大误差: {width_errors.max():.4f}")
            logging.info(f"  最小误差: {width_errors.min():.4f}")
            if 'sequence_similarity' in details:
                sim = np.array(details['sequence_similarity'])
                logging.info(f"序列相似度统计:")
                logging.info(f"  平均相似度: {sim.mean():.4f}")
                logging.info(f"  最大相似度: {sim.max():.4f}")
                logging.info(f"  最小相似度: {sim.min():.4f}")
    
    # 计算平均值
    avg_loss = total_loss / num_batches
    avg_scores = {k: v / num_batches for k, v in score_sums.items()}
    
    return {'loss': avg_loss, **avg_scores}

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: callable,
    config: Config
) -> Dict[str, float]:
    """验证模型性能"""
    model.eval()
    total_loss = 0
    score_sums = {'mt_score': 0, 'sw_score': 0, 'cq_score': 0, 'total_score': 0}
    num_batches = len(val_loader)
    
    # 用于计算详细指标
    all_mod_preds = []
    all_mod_targets = []
    all_width_errors = []
    all_seq_sims = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            data = batch['data'].to(config.DEVICE)
            targets = {k: v.to(config.DEVICE) for k, v in batch['targets'].items()}
            
            outputs = model({'data': data, 'targets': targets})
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['total_loss']
            
            # 计算各项评分
            scores, details = calculate_scores(outputs, targets, config)
            for k, v in scores.items():
                score_sums[k] += v
            
            total_loss += loss.item()
            
            # 收集详指标数据
            all_mod_preds.append(outputs['modulation_type'].argmax(dim=1).cpu())
            all_mod_targets.append(targets['modulation_type'].cpu())
            
            # 只收集调制类型正确的样本的码元宽度误差
            mod_correct = (outputs['modulation_type'].argmax(dim=1) == targets['modulation_type'])
            if mod_correct.any():
                width_error = torch.abs(
                    outputs['symbol_width'][mod_correct].squeeze(1) -
                    targets['symbol_width'][mod_correct]
                ) / targets['symbol_width'][mod_correct]
                all_width_errors.append(width_error.cpu())
                
                # 只收集调制类型和码元宽度都正确的样本的序列相似度
                width_correct = width_error <= config.SW_THRESHOLDS['acceptable']
                if width_correct.any():
                    valid_mask = mod_correct.clone()
                    valid_mask[mod_correct] = width_correct
                    if valid_mask.any():
                        # 获取序列掩码
                        sequence_mask = targets.get('sequence_mask', None)
                        if sequence_mask is not None:
                            # 只计算有效位置的相似度
                            valid_lengths = sequence_mask[valid_mask].sum(dim=1)
                            for i, length in enumerate(valid_lengths):
                                # 获取当前样本的预测和目标序列
                                pred_seq = outputs['symbol_sequence'][valid_mask][i]
                                true_seq = targets['symbol_sequence'][valid_mask][i]
                                
                                # 确保长度一致
                                length = int(length.item())
                                if pred_seq.size(-1) < length:
                                    length = pred_seq.size(-1)
                                if true_seq.size(-1) < length:
                                    length = true_seq.size(-1)
                                
                                # 确保序列维度正确
                                if len(pred_seq.shape) == 1:
                                    pred_seq = pred_seq.unsqueeze(0)
                                if len(true_seq.shape) == 1:
                                    true_seq = true_seq.unsqueeze(0)
                                
                                # 计算相似度
                                try:
                                    cos_sim = F.cosine_similarity(
                                        pred_seq[:, :length],
                                        true_seq[:, :length],
                                        dim=1
                                    ).abs()
                                    all_seq_sims.append(cos_sim.cpu())
                                except RuntimeError as e:
                                    logging.warning(f"计算相似度时出错: {str(e)}")
                                    logging.warning(f"pred_seq shape: {pred_seq.shape}")
                                    logging.warning(f"true_seq shape: {true_seq.shape}")
                                    logging.warning(f"length: {length}")
                                    continue
                        else:
                            # 如果没有掩码,使用较短序列的长度
                            pred_seq = outputs['symbol_sequence'][valid_mask]
                            true_seq = targets['symbol_sequence'][valid_mask]
                            
                            # 确保维度正确
                            if len(pred_seq.shape) == 2:
                                pred_seq = pred_seq.unsqueeze(1)
                            if len(true_seq.shape) == 2:
                                true_seq = true_seq.unsqueeze(1)
                            
                            # 使用最短的序列长度
                            min_length = min(pred_seq.size(-1), true_seq.size(-1))
                            
                            try:
                                cos_sim = F.cosine_similarity(
                                    pred_seq[:, :min_length],
                                    true_seq[:, :min_length],
                                    dim=1
                                ).abs()
                                all_seq_sims.append(cos_sim.cpu())
                            except RuntimeError as e:
                                logging.warning(f"计算相似度时出错: {str(e)}")
                                logging.warning(f"pred_seq shape: {pred_seq.shape}")
                                logging.warning(f"true_seq shape: {true_seq.shape}")
                                logging.warning(f"min_length: {min_length}")
                                continue
    
    # 计算平均值
    avg_loss = total_loss / num_batches
    avg_scores = {k: v / num_batches for k, v in score_sums.items()}
    
    # 计算详细指标
    all_mod_preds = torch.cat(all_mod_preds)
    all_mod_targets = torch.cat(all_mod_targets)
    mod_metrics = calculate_modulation_metrics(all_mod_preds, all_mod_targets)
    
    if all_width_errors:
        all_width_errors = torch.cat(all_width_errors)
        width_metrics = calculate_symbol_width_metrics(all_width_errors)
    else:
        width_metrics = {'mean_error': float('inf'), 'std_error': float('inf')}
    
    if all_seq_sims:
        all_seq_sims = torch.cat(all_seq_sims)
        seq_metrics = calculate_symbol_sequence_metrics(all_seq_sims)
    else:
        seq_metrics = {'mean_similarity': 0.0, 'std_similarity': 0.0}
    
    # 合并所有指标
    metrics = {
        'loss': avg_loss,
        **avg_scores,
        **mod_metrics,
        **width_metrics,
        **seq_metrics
    }
    
    return metrics

def custom_collate_fn(batch):
    """自定义的批处理函数,处理不同长度的序列"""
    # 取批次中的最大序列长度
    max_seq_length = max(len(item['targets']['symbol_sequence']) for item in batch)
    
    # 准备批次数据
    batch_data = []
    batch_targets = {
        'modulation_type': [],
        'symbol_width': [],
        'symbol_sequence': [],
        'sequence_mask': []
    }
    
    # 处理每个样本
    for item in batch:
        # 数据部分直接添加
        batch_data.append(item['data'])
        
        # 目标部分
        batch_targets['modulation_type'].append(item['targets']['modulation_type'])
        batch_targets['symbol_width'].append(item['targets']['symbol_width'])
        
        # 处理序列数据
        seq = item['targets']['symbol_sequence']
        seq_len = len(seq)
        
        # 创建填后的序列
        padded_seq = torch.zeros(max_seq_length, dtype=seq.dtype, device=seq.device)
        padded_seq[:seq_len] = seq
        batch_targets['symbol_sequence'].append(padded_seq)
        
        # 创建掩码
        mask = torch.zeros(max_seq_length, dtype=torch.bool, device=seq.device)
        mask[:seq_len] = True
        batch_targets['sequence_mask'].append(mask)
    
    # 堆叠所有张量
    batch_data = torch.stack(batch_data, dim=0)
    batch_targets['modulation_type'] = torch.stack(batch_targets['modulation_type'], dim=0)
    batch_targets['symbol_width'] = torch.stack(batch_targets['symbol_width'], dim=0)
    batch_targets['symbol_sequence'] = torch.stack(batch_targets['symbol_sequence'], dim=0)
    batch_targets['sequence_mask'] = torch.stack(batch_targets['sequence_mask'], dim=0)
    
    return {'data': batch_data, 'targets': batch_targets}

def train(config: Config) -> None:
    """训练主函数"""
    # 设置日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = config.LOG_DIR / f'training_{timestamp}.log'
    
    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 记录训练开始
    logging.info("="*50)
    logging.info("训练开始")
    logging.info("配置信息：")
    logging.info(f"工作目录: {config.WORKING_DIR}")
    logging.info(f"数据目录: {config.DATA_DIR}")
    logging.info(f"输出目录: {config.OUTPUT_DIR}")
    logging.info(f"设备: {config.DEVICE}")
    logging.info(f"批次大小: {config.BATCH_SIZE}")
    logging.info(f"学习率: {config.LEARNING_RATE}")
    logging.info(f"优化器: {config.OPTIMIZER}")
    logging.info(f"是否使用AMP: {config.AMP_ENABLED}")
    logging.info(f"是否使用调度器: {config.USE_SCHEDULER}")
    logging.info(f"任务权重: MT={config.MT_WEIGHT}, SW={config.SW_WEIGHT}, CQ={config.CQ_WEIGHT}")
    logging.info("="*50)
    
    # 初始化wandb
    if config.USE_WANDB:
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name=config.WANDB_NAME or f"cascade_model_{timestamp}",
            tags=config.WANDB_TAGS,
            notes=config.WANDB_NOTES,
            config=config.__dict__
        )
    
    # 创建数据加载器
    train_dataset = MultiTaskSignalDataset(mode='train', config=config)
    val_dataset = MultiTaskSignalDataset(mode='val', config=config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.MAX_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=custom_collate_fn  # 使用自定义的collate函数
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.MAX_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=custom_collate_fn  # 使用自定义的collate函数
    )
    
    # 创建模型
    model = MultiTaskModel(config)
    model.to(config.DEVICE)
    
    # 获取损失函数和优化器
    criterion = model.compute_loss  # 使用新的损失计算方法
    optimizer = config.get_optimizer(model.parameters())
    scheduler = config.get_scheduler(optimizer)
    
    # 创建早停对象
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        mode='max',
        monitor=config.EARLY_STOPPING_METRIC
    )
    
    # 创建AMP scaler
    scaler = torch.cuda.amp.GradScaler() if config.AMP_ENABLED else None
    
    # 训练循环
    best_total_score = 0
    for epoch in range(config.MAX_EPOCHS):
        logging.info(f'\nEpoch {epoch+1}/{config.MAX_EPOCHS}')
        
        # 训练一个epoch
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scheduler, config, epoch+1, scaler
        )
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, config)
        
        # 记录训练信息
        logging.info(
            f"训练损失: {train_metrics['loss']:.4f}, "
            f"训练总���: {train_metrics['total_score']:.2f}\n"
            f"验证损失: {val_metrics['loss']:.4f}, "
            f"验证总分: {val_metrics['total_score']:.2f}\n"
            f"验证分项分数: MT={val_metrics['mt_score']:.1f}, "
            f"SW={val_metrics['sw_score']:.1f}, "
            f"CQ={val_metrics['cq_score']:.1f}\n"
            f"学习率: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # 记录到wandb
        if config.USE_WANDB:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_mt_score': train_metrics['mt_score'],
                'train_sw_score': train_metrics['sw_score'],
                'train_cq_score': train_metrics['cq_score'],
                'train_total_score': train_metrics['total_score'],
                'val_loss': val_metrics['loss'],
                'val_mt_score': val_metrics['mt_score'],
                'val_sw_score': val_metrics['sw_score'],
                'val_cq_score': val_metrics['cq_score'],
                'val_total_score': val_metrics['total_score'],
                'learning_rate': optimizer.param_groups[0]['lr'],
                **{f'val_{k}': v for k, v in val_metrics.items() 
                   if k not in ['loss', 'mt_score', 'sw_score', 'cq_score', 'total_score']}
            })
        
        # 保存最佳模型
        if val_metrics['total_score'] > best_total_score:
            best_total_score = val_metrics['total_score']
            model_path = config.OUTPUT_DIR / f'best_model_{timestamp}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_total_score': best_total_score,
                'config': config.__dict__,
                'val_metrics': val_metrics
            }, model_path)
            logging.info(f"保存最佳模型 - 验证总分: {best_total_score:.2f}")
        
        # 更新学习率
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics[config.EARLY_STOPPING_METRIC])
            else:
                scheduler.step()
        
        # 早停检查
        if early_stopping(val_metrics):
            logging.info(f"触发早停，共训练{epoch + 1}个epoch")
            break
    
    # 保存最终模型
    final_model_path = config.OUTPUT_DIR / f'final_model_{timestamp}.pth'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_total_score': best_total_score,
        'config': config.__dict__,
        'val_metrics': val_metrics
    }, final_model_path)
    
    logging.info("训练完成")
    logging.info(f"最佳验证总分: {best_total_score:.2f}")
    logging.info("="*50)
    
    if config.USE_WANDB:
        wandb.finish()

if __name__ == '__main__':
    config = Config()
    train(config) 