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

from project.config import Config
from project.data.dataset import ModulationDataset
from project.model.modulation_classifier import ModulationClassifierEnsemble
from project.utils.metrics import calculate_metrics
from project.utils.early_stopping import EarlyStopping

class ModulationTrainer:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ModulationClassifierEnsemble().to(self.device)
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志系统"""
        # 清除现有的处理器
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # 配置根日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.LOG_FILE),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 输出初始配置信息
        self.logger.info("\n=== 配置信息 ===")
        self.logger.info(f"运行设备: {self.config.DEVICE}")
        self.logger.info(f"批次大小: {self.config.BATCH_SIZE}")
        self.logger.info(f"梯度累积步数: {self.config.GRADIENT_ACCUMULATION_STEPS}")
        self.logger.info(f"有效批次大小: {self.config.BATCH_SIZE * self.config.GRADIENT_ACCUMULATION_STEPS}")
        self.logger.info(f"当前epoch: {self.config.training_state.get('current_epoch', 0)}")
        self.logger.info(f"最佳验证分数: {self.config.training_state.get('best_val_score', float('-inf'))}")
        self.logger.info(f"CPU核心数: {os.cpu_count()}")
        self.logger.info(f"是否使用AMP: {self.config.AMP_ENABLED}")
        self.logger.info(f"是否使用数据增强: Mixup={self.config.USE_MIXUP}, CutMix={self.config.USE_CUTMIX}")
        self.logger.info("===============\n")
        
    def prepare_data(self):
        """准备数据加载器"""
        dataset = ModulationDataset()
        val_size = int(len(dataset) * self.config.VALIDATION_RATIO)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY,
            prefetch_factor=self.config.PREFETCH_FACTOR
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY,
            prefetch_factor=self.config.PREFETCH_FACTOR
        )
        
    def train_classifier(self, mod_name, train_loader, val_loader):
        """训练单个分类器"""
        classifier = self.model.classifiers[mod_name]
        optimizer = optim.AdamW(
            classifier.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.MAX_LR,
            epochs=self.config.NUM_EPOCHS,
            steps_per_epoch=len(train_loader)
        )
        
        early_stopping = EarlyStopping(
            patience=self.config.EARLY_STOPPING_PATIENCE,
            min_delta=self.config.EARLY_STOPPING_MIN_DELTA,
            mode=self.config.EARLY_STOPPING_MODE
        )
        
        paths = self.config.get_classifier_paths(mod_name)
        best_val_acc = 0.0
        
        # 获取当前调制类型的索引
        mod_type = next(key for key, value in self.config.MODULATION_DICT.items() if value == mod_name)
        
        for epoch in range(self.config.NUM_EPOCHS):
            # 训练阶段
            classifier.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training {mod_name}")):
                data = batch['data'].to(self.device)
                # 创建二分类标签：1表示当前调制类型，0表示其他类型
                targets = (batch['modulation_type'] == mod_type).float().to(self.device)
                
                optimizer.zero_grad()
                outputs = classifier(data)
                loss = nn.BCEWithLogitsLoss()(outputs['modulation_type'].squeeze(), targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), self.config.GRADIENT_CLIP_VAL)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                pred = (outputs['modulation_type'].squeeze() > 0.5).float()
                train_correct += (pred == targets).sum().item()
                train_total += targets.size(0)
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # 验证阶段
            classifier.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validating {mod_name}"):
                    data = batch['data'].to(self.device)
                    targets = (batch['modulation_type'] == mod_type).float().to(self.device)
                    
                    outputs = classifier(data)
                    loss = nn.BCEWithLogitsLoss()(outputs['modulation_type'].squeeze(), targets)
                    
                    val_loss += loss.item()
                    pred = (outputs['modulation_type'].squeeze() > 0.5).float()
                    val_correct += (pred == targets).sum().item()
                    val_total += targets.size(0)
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            # 记录训练状态
            self.logger.info(
                f"{mod_name} Epoch {epoch+1}/{self.config.NUM_EPOCHS} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            if self.config.USE_WANDB:
                wandb.log({
                    f"{mod_name}/train_loss": train_loss,
                    f"{mod_name}/train_acc": train_acc,
                    f"{mod_name}/val_loss": val_loss,
                    f"{mod_name}/val_acc": val_acc,
                    f"{mod_name}/learning_rate": scheduler.get_last_lr()[0]
                })
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(classifier.state_dict(), paths['best'])
            
            # 保存最新模型和训练状态
            torch.save(classifier.state_dict(), paths['last'])
            training_state = {
                'epoch': epoch + 1,
                'best_val_acc': best_val_acc,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }
            with open(paths['state'], 'w') as f:
                json.dump(training_state, f, indent=4)
            
            # 早停检查
            if early_stopping(val_acc):
                self.logger.info(f"{mod_name} Early stopping triggered")
                break
        
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
                probs = torch.sigmoid(logits)
                raw_outputs[mod_name] = logits
                confidences[mod_name] = probs
                predictions[mod_name] = (probs > self.config.CONFIDENCE_THRESHOLD).float()
        
        # 计算每个类别的综合得分
        scores = {}
        for mod_name in self.config.MODULATION_DICT.values():
            # 基础得分：当前分类器的置信度
            base_score = confidences[mod_name].mean()
            
            # 其他分类器的否定得分
            other_scores = []
            for other_name, other_conf in confidences.items():
                if other_name != mod_name:
                    # 其他分类器预测为负的置信度（1 - prob）
                    other_scores.append(1 - other_conf.mean())
            
            # 综合得分：当前分类器的置信度 * 其他分类器的平均否定得分
            if other_scores:
                other_score_mean = sum(other_scores) / len(other_scores)
                scores[mod_name] = base_score * other_score_mean
            else:
                scores[mod_name] = base_score
        
        # 根据配置的集成模式选择最终预测
        if self.config.ENSEMBLE_MODE == 'max_confidence':
            # 选择综合得分最高的类别
            final_prediction = max(scores.items(), key=lambda x: x[1])[0]
            confidence = scores[final_prediction]
        else:  # voting
            # 计算加权投票
            vote_scores = {}
            for mod_name, pred in predictions.items():
                if pred.item() == 1:  # 如果分类器投票支持
                    vote_scores[mod_name] = scores[mod_name]
            
            if vote_scores:
                # 在有投票的类别中选择得分最高的
                final_prediction = max(vote_scores.items(), key=lambda x: x[1])[0]
                confidence = vote_scores[final_prediction]
            else:
                # 如果没有分类器达到阈值，选择原始得分最高的
                final_prediction = max(scores.items(), key=lambda x: x[1])[0]
                confidence = scores[final_prediction]
        
        return final_prediction, confidence
    
    def evaluate_ensemble(self, data_loader):
        """评估集成模型的性能"""
        total = 0
        correct = 0
        confusion_mat = np.zeros((len(self.config.MODULATION_DICT), len(self.config.MODULATION_DICT)))
        
        # 用于计算每个类别的性能
        class_correct = {mod_name: 0 for mod_name in self.config.MODULATION_DICT.values()}
        class_total = {mod_name: 0 for mod_name in self.config.MODULATION_DICT.values()}
        
        # 收集每个类别的置信度分布
        class_confidences = {mod_name: [] for mod_name in self.config.MODULATION_DICT.values()}
        
        for batch in tqdm(data_loader, desc="Evaluating ensemble"):
            data = batch['data'].to(self.device)
            true_labels = batch['modulation_type']
            
            for i in range(len(data)):
                pred_mod, confidence = self.predict_ensemble(data[i:i+1])
                true_mod = self.config.MODULATION_DICT[true_labels[i].item()]
                
                # 更新统计信息
                total += 1
                if pred_mod == true_mod:
                    correct += 1
                    class_correct[true_mod] += 1
                class_total[true_mod] += 1
                class_confidences[pred_mod].append(confidence)
                
                # 更新混淆矩阵
                pred_idx = list(self.config.MODULATION_DICT.values()).index(pred_mod)
                true_idx = list(self.config.MODULATION_DICT.values()).index(true_mod)
                confusion_mat[true_idx][pred_idx] += 1
        
        # 计算总体准确率
        accuracy = correct / total
        
        # 计算每个类别的准确率和平均置信度
        class_accuracies = {}
        class_avg_confidences = {}
        for mod_name in self.config.MODULATION_DICT.values():
            class_accuracies[mod_name] = class_correct[mod_name]/class_total[mod_name]
            if class_confidences[mod_name]:
                class_avg_confidences[mod_name] = np.mean(class_confidences[mod_name])
            else:
                class_avg_confidences[mod_name] = 0.0
        
        # 记录评估结果
        self.logger.info("\n=== 集成模型评估结果 ===")
        self.logger.info(f"总体准确率: {accuracy:.4f}")
        self.logger.info("\n各类别性能:")
        for mod_name in self.config.MODULATION_DICT.values():
            self.logger.info(
                f"{mod_name}: "
                f"准确率={class_accuracies[mod_name]:.4f}, "
                f"平均置信度={class_avg_confidences[mod_name]:.4f}, "
                f"样本数={class_total[mod_name]}"
            )
        
        if self.config.USE_WANDB:
            wandb.log({
                "ensemble_accuracy": accuracy,
                "class_accuracies": class_accuracies,
                "class_confidences": class_avg_confidences,
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=np.argmax(confusion_mat, axis=1),
                    preds=np.argmax(confusion_mat, axis=0),
                    class_names=list(self.config.MODULATION_DICT.values())
                )
            })
        
        return accuracy, class_accuracies, confusion_mat
    
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
        
        # 评估集成模型
        self.logger.info("\n开始评估集成模型...")
        accuracy, class_accuracies, confusion_mat = self.evaluate_ensemble(self.val_loader)
        
        # 关闭wandb
        if self.config.USE_WANDB:
            wandb.finish()

def main():
    try:
        trainer = ModulationTrainer()
        trainer.train()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练发生错误: {str(e)}")
        raise
    finally:
        # 清理资源
        if 'trainer' in locals():
            del trainer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 