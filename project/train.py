import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm
import wandb
import os
from datetime import datetime
import traceback
import sys
import torch.nn.functional as F

from project.config import Config
from project.data.dataset import ModulationDataset
from project.model.modulation_classifier import ModulationClassifierEnsemble
from project.utils.metrics import calculate_metrics
from project.utils.early_stopping import EarlyStopping

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

class ModulationTrainer:
    """调制分类器训练器"""
    def __init__(self):
        self.config = Config()
        self.device = self.config.DEVICE
        self.logger = setup_logging(self.config)
        
        # 初始化模型
        self.model = ModulationClassifierEnsemble()
        self.model.to(self.device)
        
        # 准备数据
        self.train_loader = None
        self.val_loader = None
        
        # 记录训练状态
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.early_stopping = EarlyStopping(
            patience=self.config.EARLY_STOPPING_PATIENCE,
            min_delta=self.config.EARLY_STOPPING_MIN_DELTA,
            mode=self.config.EARLY_STOPPING_MODE
        )
    
    def prepare_data(self):
        """准备数据加载器"""
        # 创建数据集
        dataset = ModulationDataset(
            data_dir=self.config.DATA_DIR,
            transform=ModulationDataset.get_transforms(self.config, mode='train'),
            sequence_length=self.config.SEQUENCE_LENGTH
        )
        
        # 划分训练集和验证集
        train_size = int((1 - self.config.VALIDATION_RATIO) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
        )
        
        self.logger.info(f"数据集大小: {len(dataset)}")
        self.logger.info(f"训练集大小: {len(train_dataset)}")
        self.logger.info(f"验证集大小: {len(val_dataset)}")
        self.logger.info(f"批次大小: {self.config.BATCH_SIZE}")
        self.logger.info(f"序列长度: {self.config.SEQUENCE_LENGTH}")
    
    def train_classifier(self, mod_name, train_loader, val_loader):
        """训练单个分类器"""
        classifier = self.model.classifiers[mod_name]
        optimizer = self.config.get_optimizer(classifier.parameters())
        scheduler = self.config.get_lr_scheduler(optimizer)
        scaler = torch.cuda.amp.GradScaler() if self.config.AMP_ENABLED else None
        
        best_val_acc = 0.0
        early_stopping = EarlyStopping(
            patience=self.config.EARLY_STOPPING_PATIENCE,
            min_delta=self.config.EARLY_STOPPING_MIN_DELTA,
            mode=self.config.EARLY_STOPPING_MODE
        )
        
        # 获取当前调制类型的索引
        mod_idx = list(self.config.MODULATION_DICT.values()).index(mod_name)
        
        for epoch in range(self.config.NUM_EPOCHS):
            # 训练阶段
            classifier.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, batch in enumerate(train_loader):
                data = batch['data'].to(self.device)
                targets = batch['targets']
                target_labels = (targets['modulation_type'].to(self.device) == mod_idx).float()
                
                optimizer.zero_grad()
                
                if self.config.AMP_ENABLED:
                    with torch.cuda.amp.autocast():
                        loss = self.model.train_single_classifier(mod_name, data, {'modulation_type': target_labels})
                        scaler.scale(loss).backward()
                        if self.config.GRADIENT_CLIP_VAL > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                classifier.parameters(),
                                self.config.GRADIENT_CLIP_VAL
                            )
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    loss = self.model.train_single_classifier(mod_name, data, {'modulation_type': target_labels})
                    loss.backward()
                    if self.config.GRADIENT_CLIP_VAL > 0:
                        torch.nn.utils.clip_grad_norm_(
                            classifier.parameters(),
                            self.config.GRADIENT_CLIP_VAL
                        )
                    optimizer.step()
                
                train_loss += loss.item()
                
                # 计算准确率
                outputs = classifier(data)
                pred = (torch.sigmoid(outputs['logits']) > 0.5).float()
                correct += (pred == target_labels).sum().item()
                total += target_labels.size(0)
                
                if batch_idx % self.config.LOG_INTERVAL == 0:
                    self.logger.info(
                        f"{mod_name} Epoch: {epoch}, Batch: {batch_idx}, "
                        f"Loss: {loss.item():.4f}, "
                        f"Acc: {100.*correct/total:.2f}%"
                    )
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = correct / total
            
            # 验证阶段
            classifier.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    data = batch['data'].to(self.device)
                    targets = batch['targets']
                    target_labels = (targets['modulation_type'].to(self.device) == mod_idx).float()
                    
                    outputs = classifier(data)
                    pred = (torch.sigmoid(outputs['logits']) > 0.5).float()
                    
                    loss = F.binary_cross_entropy_with_logits(outputs['logits'], target_labels)
                    val_loss += loss.item()
                    
                    correct += (pred == target_labels).sum().item()
                    total += target_labels.size(0)
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            
            # 更新学习率
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
            
            # 记录到wandb
            if self.config.USE_WANDB:
                wandb.log({
                    f"{mod_name}/train_loss": avg_train_loss,
                    f"{mod_name}/train_acc": train_acc,
                    f"{mod_name}/val_loss": avg_val_loss,
                    f"{mod_name}/val_acc": val_acc,
                    f"{mod_name}/learning_rate": optimizer.param_groups[0]['lr']
                })
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = self.config.CHECKPOINT_DIR / mod_name / "best_model.pth"
                classifier.save_model(str(save_path))
                
                # 如果达到95%准确率，提前结束训练
                if val_acc >= 0.95:
                    self.logger.info(f"{mod_name} 达到目标准确率95%，提前结束训练")
                    break
            
            # 早停检查
            if early_stopping(val_acc):
                self.logger.info(f"{mod_name} 触发早停，在第{epoch}轮")
                break
        
        # 加载最佳模型
        best_model_path = self.config.CHECKPOINT_DIR / mod_name / "best_model.pth"
        classifier.load_model(str(best_model_path))
        
        return best_val_acc
    
    def predict_ensemble(self, data):
        """集成预测函数"""
        # 收集所有分类器的预测结果
        predictions = {}
        confidences = {}
        raw_outputs = {}
        
        # 获取每个分类器的原始输出和置信度
        for mod_name, classifier in self.model.classifiers.items():
            classifier.eval()
            with torch.no_grad():
                outputs = classifier(data)
                logits = outputs['modulation_type'].squeeze()
                
                # 使用温度缩放进行置信度校准
                temperature = 2.0  # 可调的温度参数
                scaled_logits = logits / temperature
                probs = torch.sigmoid(scaled_logits)
                
                raw_outputs[mod_name] = logits.cpu()
                confidences[mod_name] = probs.cpu()
                predictions[mod_name] = (probs > self.config.CONFIDENCE_THRESHOLD).float().cpu()
        
        # 计算每个类别的综合得分
        scores = {}
        for mod_name in self.config.MODULATION_DICT.values():
            # 当前分类器的置信度得分
            base_confidence = confidences[mod_name].mean().item()
            
            # 计算与其他分类器的一致性得分
            consistency_scores = []
            for other_name, other_conf in confidences.items():
                if other_name != mod_name:
                    # 计算当前分类器和其他分类器预测的一致性
                    agreement = 1 - abs(confidences[mod_name] - (1 - other_conf)).mean().item()
                    consistency_scores.append(agreement)
            
            # 计算一致性得分
            consistency_score = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
            
            # 综合得分：结合置信度和一致性
            scores[mod_name] = base_confidence * (0.7 + 0.3 * consistency_score)
        
        # 使用加权投票和置信度阈值的混合策略
        if self.config.ENSEMBLE_MODE == 'hybrid':
            # 初始化投票计数
            votes = {mod_name: 0.0 for mod_name in self.config.MODULATION_DICT.values()}
            
            # 根据置信度进加权投票
            for mod_name, pred in predictions.items():
                confidence = confidences[mod_name].mean().item()
                if confidence > self.config.MIN_VOTE_CONFIDENCE:
                    # 投票权重 = 基础权重 * 置信度
                    votes[mod_name] += confidence
            
            # 结合投票结果和综合得分
            final_scores = {}
            for mod_name in self.config.MODULATION_DICT.values():
                vote_weight = votes[mod_name] / (sum(votes.values()) + 1e-6)
                score_weight = scores[mod_name] / (sum(scores.values()) + 1e-6)
                final_scores[mod_name] = 0.4 * vote_weight + 0.6 * score_weight
            
            # 选择最终预测
            final_prediction = max(final_scores.items(), key=lambda x: x[1])[0]
            confidence = final_scores[final_prediction]
        else:
            # 使用原有的最大置信度策略
            final_prediction = max(scores.items(), key=lambda x: x[1])[0]
            confidence = scores[final_prediction]
        
        return final_prediction, confidence
    
    def evaluate_ensemble(self, data_loader):
        """评估集成模型的性能"""
        total = 0
        correct = 0
        confusion_mat = np.zeros((len(self.config.MODULATION_DICT), len(self.config.MODULATION_DICT)))
        
        # 用于计算每个类别的性能
        class_metrics = {mod_name: {
            'correct': 0,
            'total': 0,
            'confidences': [],
            'consistency_scores': [],
            'false_positives': 0,
            'false_negatives': 0
        } for mod_name in self.config.MODULATION_DICT.values()}
        
        for batch in tqdm(data_loader, desc="Evaluating ensemble"):
            data = batch['data'].to(self.device)
            true_labels = batch['modulation_type'].cpu()
            
            for i in range(len(data)):
                # 获取预测结果和详细指标
                pred_mod, confidence, details = self.predict_ensemble_with_details(data[i:i+1])
                true_mod = self.config.MODULATION_DICT[true_labels[i].item()]
                
                # 更新基本统计
                total += 1
                if pred_mod == true_mod:
                    correct += 1
                    class_metrics[true_mod]['correct'] += 1
                else:
                    # 记录错误分类
                    class_metrics[pred_mod]['false_positives'] += 1
                    class_metrics[true_mod]['false_negatives'] += 1
                
                # 更新类别统计
                class_metrics[true_mod]['total'] += 1
                class_metrics[pred_mod]['confidences'].append(confidence)
                if 'consistency_score' in details:
                    class_metrics[pred_mod]['consistency_scores'].append(details['consistency_score'])
                
                # 更新混淆矩阵
                pred_idx = list(self.config.MODULATION_DICT.values()).index(pred_mod)
                true_idx = list(self.config.MODULATION_DICT.values()).index(true_mod)
                confusion_mat[true_idx][pred_idx] += 1
        
        # 计算总体准确率
        accuracy = correct / total
        
        # 计算每个类别的详细指标
        class_performance = {}
        for mod_name, metrics in class_metrics.items():
            if metrics['total'] > 0:
                precision = metrics['correct'] / (metrics['correct'] + metrics['false_positives']) if (metrics['correct'] + metrics['false_positives']) > 0 else 0
                recall = metrics['correct'] / metrics['total']
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                class_performance[mod_name] = {
                    'accuracy': metrics['correct'] / metrics['total'],
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'avg_confidence': np.mean(metrics['confidences']) if metrics['confidences'] else 0,
                    'avg_consistency': np.mean(metrics['consistency_scores']) if metrics['consistency_scores'] else 0,
                    'sample_count': metrics['total']
                }
        
        # 记录评估结果
        self.logger.info("\n=== 集成模型评估结果 ===")
        self.logger.info(f"总体准确率: {accuracy:.4f}")
        self.logger.info("\n各类别性能:")
        for mod_name, perf in class_performance.items():
            self.logger.info(
                f"{mod_name}: "
                f"准确率={perf['accuracy']:.4f}, "
                f"精确率={perf['precision']:.4f}, "
                f"召回率={perf['recall']:.4f}, "
                f"F1分数={perf['f1_score']:.4f}, "
                f"平均置信度={perf['avg_confidence']:.4f}, "
                f"平均一致性={perf['avg_consistency']:.4f}, "
                f"样本数={perf['sample_count']}"
            )
        
        if self.config.USE_WANDB:
            wandb.log({
                "ensemble_accuracy": accuracy,
                "class_performance": class_performance,
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=np.argmax(confusion_mat, axis=1),
                    preds=np.argmax(confusion_mat, axis=0),
                    class_names=list(self.config.MODULATION_DICT.values())
                )
            })
        
        return accuracy, class_performance, confusion_mat
    
    def predict_ensemble_with_details(self, data):
        """带详细信息的集成预测函数"""
        pred_mod, confidence = self.predict_ensemble(data)
        
        # 收集预测详情
        details = {
            'raw_confidences': {},
            'consistency_scores': {},
            'vote_weights': {},
            'final_scores': {}
        }
        
        # 获取每个分类器的预测结果
        predictions = {}
        confidences = {}
        for mod_name, classifier in self.model.classifiers.items():
            classifier.eval()
            with torch.no_grad():
                outputs = classifier(data)
                logits = outputs['modulation_type'].squeeze()
                
                if self.config.USE_TEMPERATURE_SCALING:
                    scaled_logits = logits / self.config.TEMPERATURE
                    probs = torch.sigmoid(scaled_logits)
                else:
                    probs = torch.sigmoid(logits)
                
                predictions[mod_name] = (probs > self.config.CONFIDENCE_THRESHOLD).float().cpu()
                confidences[mod_name] = probs.cpu()
                details['raw_confidences'][mod_name] = probs.mean().item()
        
        # 计算一致性得分
        if self.config.USE_CONSISTENCY_SCORE:
            for mod_name in self.config.MODULATION_DICT.values():
                consistency_scores = []
                for other_name, other_conf in confidences.items():
                    if other_name != mod_name:
                        agreement = 1 - abs(confidences[mod_name] - (1 - other_conf)).mean().item()
                        consistency_scores.append(agreement)
                
                avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
                details['consistency_scores'][mod_name] = avg_consistency
        
        # 计算投票权重
        votes = {mod_name: 0.0 for mod_name in self.config.MODULATION_DICT.values()}
        for mod_name, pred in predictions.items():
            confidence = confidences[mod_name].mean().item()
            if confidence > self.config.MIN_VOTE_CONFIDENCE:
                votes[mod_name] += confidence
        
        total_votes = sum(votes.values()) + 1e-6
        for mod_name in votes:
            details['vote_weights'][mod_name] = votes[mod_name] / total_votes
        
        # 记录最终得分
        details['final_scores'] = {
            mod_name: confidence for mod_name, confidence in zip(
                self.config.MODULATION_DICT.values(),
                confidences.values()
            )
        }
        
        return pred_mod, confidence, details
    
    def train(self):
        """训练模型"""
        self.prepare_data()
        
        # 初始化wandb（如果启用）
        if self.config.USE_WANDB:
            try:
                wandb.init(
                    project=self.config.WANDB_PROJECT,
                    entity=self.config.WANDB_ENTITY,
                    config=self.config.__dict__,
                    name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            except Exception as e:
                self.logger.warning(f"wandb初始化失败: {str(e)}")
                self.config.USE_WANDB = False
        
        # 训练每个分类器
        for mod_name in self.config.MODULATION_DICT.values():
            self.logger.info(f"\n训练分类器: {mod_name}")
            best_acc = self.train_classifier(mod_name, self.train_loader, self.val_loader)
            self.logger.info(f"{mod_name} 最佳验证准确率: {best_acc:.4f}")
        
        # 校准温度参数
        if self.config.USE_TEMPERATURE_SCALING:
            self.logger.info("\n开始校准温度参数...")
            self.model.calibrate_temperature(self.val_loader)
        
        # 评估集成模型
        self.logger.info("\n开始评估集成模型...")
        accuracy, class_performance, confusion_mat = self.evaluate_ensemble(self.val_loader)
        
        # 保存最终模型
        save_path = self.config.CHECKPOINT_DIR / "final_model.pth"
        self.model.save_classifiers(save_path.parent)
        
        # 关闭wandb
        if self.config.USE_WANDB:
            wandb.finish()
        
        return accuracy, class_performance

def main():
    """主函数"""
    try:
        trainer = ModulationTrainer()
        accuracy, class_performance = trainer.train()
        
        print("\n=== 训练完成 ===")
        print(f"总体准确率: {accuracy:.4f}")
        print("\n各类别性能:")
        for mod_name, perf in class_performance.items():
            print(
                f"{mod_name}: "
                f"准确率={perf['accuracy']:.4f}, "
                f"精确率={perf['precision']:.4f}, "
                f"召回率={perf['recall']:.4f}, "
                f"F1分数={perf['f1_score']:.4f}"
            )
    
    except Exception as e:
        print(f"\n训练发生错误: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 