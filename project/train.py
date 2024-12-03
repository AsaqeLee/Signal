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
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True
        )
        
    def train_classifier(self, mod_name, train_loader, val_loader):
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
        
        for epoch in range(self.config.NUM_EPOCHS):
            # 训练阶段
            classifier.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training {mod_name}")):
                data = batch['data'].to(self.device)
                targets = {
                    'modulation_type': (batch['modulation_type'] == 
                        self.config.MODULATION_DICT.index(mod_name)).float().to(self.device)
                }
                
                optimizer.zero_grad()
                outputs = classifier(data)
                loss = classifier.train_single_classifier(data, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), self.config.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                pred = (outputs['modulation_type'] > 0.5).float()
                train_correct += (pred == targets['modulation_type']).sum().item()
                train_total += targets['modulation_type'].size(0)
            
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
                    targets = {
                        'modulation_type': (batch['modulation_type'] == 
                            self.config.MODULATION_DICT.index(mod_name)).float().to(self.device)
                    }
                    
                    outputs = classifier(data)
                    loss = classifier.train_single_classifier(data, targets)
                    
                    val_loss += loss.item()
                    pred = (outputs['modulation_type'] > 0.5).float()
                    val_correct += (pred == targets['modulation_type']).sum().item()
                    val_total += targets['modulation_type'].size(0)
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            # 记录训练状态
            self.logger.info(
                f"{mod_name} Epoch {epoch+1}/{self.config.NUM_EPOCHS} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            if wandb.run is not None:
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
    
    def train(self):
        self.prepare_data()
        
        if wandb.run is None:
            wandb.init(
                project="modulation-classification",
                config=self.config.__dict__,
                name="ensemble-training"
            )
        
        best_accuracies = {}
        
        for mod_name in self.config.MODULATION_DICT.values():
            self.logger.info(f"\nTraining classifier for {mod_name}")
            best_acc = self.train_classifier(mod_name, self.train_loader, self.val_loader)
            best_accuracies[mod_name] = best_acc
        
        # 记录最终结果
        self.logger.info("\nTraining completed. Best validation accuracies:")
        for mod_name, acc in best_accuracies.items():
            self.logger.info(f"{mod_name}: {acc:.4f}")
        
        if wandb.run is not None:
            wandb.log({"best_accuracies": best_accuracies})
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