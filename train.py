import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
# from torch import compile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import os
import warnings

# 屏蔽RDKit的deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*GetValence.*")
warnings.filterwarnings("ignore", message=".*please use GetValence.*")
warnings.filterwarnings("ignore", message=".*DEPRECATION WARNING.*")

# 设置RDKit日志级别
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from echo import ECHO
from rdkit import Chem
from utils.atom_extractor import load_smiles

class SMILESDataset(Dataset):
    """SMILES数据集类，适配ECHO模型的输入格式"""
    def __init__(self, data_indices):
        self.data_indices = data_indices

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        data_idx = self.data_indices[idx]
        try:
            smiles, label = load_smiles(data_idx)
            return smiles, float(label)
        except Exception as e:
            # 返回一个默认值或重新抛出异常
            raise RuntimeError(f"处理索引 {data_idx} 时出错: {str(e)}")

def custom_collate(batch):
    """自定义批处理函数，适配ECHO模型"""
    # 过滤掉None值
    valid_batch = [(smiles, label) for smiles, label in batch if smiles is not None]
    
    if not valid_batch:
        return [], torch.tensor([])
    
    smiles_list, label_list = zip(*valid_batch)
    label_batch = torch.tensor(label_list, dtype=torch.float32)
    return list(smiles_list), label_batch

class ECHOTrainer:
    """ECHO模型训练器"""
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )  # 移除了 verbose=True 参数
        
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []  # 存储验证集AUC
    
    def train_epoch(self, dataloader):
        """优化的训练epoch"""
        self.model.train()
        total_loss = 0
        num_samples = 0
        
        for smiles_batch, label_batch in tqdm(dataloader, desc="训练中"):
            if len(smiles_batch) == 0:
                continue
            
            # 重置梯度
            self.optimizer.zero_grad()
            
            try:
                # 批处理forward
                label_batch = label_batch.to(self.device)
                outputs = self.model(smiles_batch)  # 直接传入整个batch
                
                # 确保输出维度正确
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(1)
                if label_batch.dim() == 1:
                    label_batch = label_batch.unsqueeze(1)
                
                # 计算损失
                loss = self.criterion(outputs, label_batch)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item() * len(smiles_batch)
                num_samples += len(smiles_batch)
                
            except Exception as e:
                import traceback
                print(f"批处理出错: {str(e)}")
                traceback.print_exc()
                continue
        
        avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate_epoch(self, dataloader):
        """验证一个epoch - 修复批处理逻辑，只计算ROC AUC"""
        self.model.eval()
        total_loss = 0
        all_probabilities = []  # 存储预测概率
        all_labels = []
        num_samples = 0
        
        with torch.no_grad():
            for smiles_batch, label_batch in tqdm(dataloader, desc="验证中"):
                if len(smiles_batch) == 0:
                    continue
                    
                batch_outputs = []
                batch_targets = []
                
                label_batch = label_batch.to(self.device)
                
                for i, smiles in enumerate(smiles_batch):
                    try:
                        output = self.model(smiles)
                        
                        if output.dim() == 0:
                            output = output.unsqueeze(0)
                        
                        batch_outputs.append(output)
                        batch_targets.append(label_batch[i:i+1])
                        
                        # 计算预测概率
                        prob = torch.sigmoid(output)
                        all_probabilities.append(prob.cpu().item())
                        all_labels.append(label_batch[i].cpu().item())
                        
                    except Exception as e:
                        print(f"验证样本出错: {smiles[:50]}..., 错误: {str(e)}")
                        continue
                
                if batch_outputs:
                    batch_output_tensor = torch.cat(batch_outputs, dim=0)
                    batch_target_tensor = torch.cat(batch_targets, dim=0)
                    
                    loss = self.criterion(batch_output_tensor, batch_target_tensor)
                    total_loss += loss.item() * len(batch_outputs)
                    num_samples += len(batch_outputs)

        avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
        
        # 计算ROC AUC
        auc = 0.0
        if all_labels and len(set(all_labels)) > 1:  # 确保有两个类别
            try:
                auc = roc_auc_score(all_labels, all_probabilities)
            except ValueError as e:
                print(f"计算AUC时出错: {str(e)}")
                auc = 0.0
        
        self.val_losses.append(avg_loss)
        self.val_aucs.append(auc)
        
        # 记录学习率调整前的值
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # 更新学习率
        self.scheduler.step(avg_loss)
        
        # 检查学习率是否发生变化
        new_lr = self.optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"adjusting lr: {current_lr:.6f} -> {new_lr:.6f}")
        else:
            print(f"current lr: {current_lr:.6f}")
        
        return avg_loss, auc

def evaluate_model(model, dataloader, device):
    """评估模型，只计算ROC AUC"""
    model.eval()
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for smiles_batch, label_batch in tqdm(dataloader, desc="评估中"):
            if len(smiles_batch) == 0:
                continue
                
            label_batch = label_batch.to(device)
            
            for i, smiles in enumerate(smiles_batch):
                try:
                    output = model(smiles)
                    
                    if output.dim() == 0:
                        output = output.unsqueeze(0)
                    
                    # 计算预测概率
                    prob = torch.sigmoid(output)
                    all_probabilities.append(prob.cpu().item())
                    all_labels.append(label_batch[i].cpu().item())
                    
                except Exception as e:
                    print(f"评估样本出错: {smiles[:50]}..., 错误: {str(e)}")
                    continue
    
    if not all_labels or len(set(all_labels)) <= 1:
        return None
    
    try:
        auc = roc_auc_score(all_labels, all_probabilities)
        return {'auc': auc}
    except ValueError as e:
        print(f"计算AUC时出错: {str(e)}")
        return None

def plot_training_curves(trainer, save_path='training_curves.png'):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(trainer.train_losses, label='Traing loss', color='blue')
    plt.plot(trainer.val_losses, label='Val loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Val loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制训练损失详情
    plt.subplot(1, 3, 2)
    plt.plot(trainer.train_losses, label='Traing loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train loss details')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制验证AUC
    plt.subplot(1, 3, 3)
    plt.plot(trainer.val_aucs, label='Val AUC', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Val ROC AUC') 
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_path}")
    plt.show()

def main():
    """主训练流程"""
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 首先检查实际数据大小
    import pandas as pd
    file_path = os.path.join('data', 'hiv_cleaned.csv')
    df = pd.read_csv(file_path)
    actual_num_samples = len(df)
    print(f"数据文件中实际样本数: {actual_num_samples}")
    
    # 使用实际的样本数量
    num_samples = actual_num_samples
    all_indices = list(range(num_samples))
    
    # 分割数据集
    train_val_indices, test_indices = train_test_split(
        all_indices, test_size=0.1, random_state=42
    )
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=0.20, random_state=42
    )
    
    print(f"训练集: {len(train_indices)} 样本")
    print(f"验证集: {len(val_indices)} 样本")
    print(f"测试集: {len(test_indices)} 样本")
    
    # 创建数据集和数据加载器
    train_dataset = SMILESDataset(train_indices)
    val_dataset = SMILESDataset(val_indices)
    test_dataset = SMILESDataset(test_indices)
    
    batch_size = 32  # 增加批量大小，充分利用GPU内存
    # 添加多进程数据加载
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=custom_collate,
        num_workers=4,  # 多进程加载数据
        pin_memory=True,  # 加速CPU到GPU数据传输
        persistent_workers=True  # 保持worker进程
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=custom_collate,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=custom_collate,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 创建模型
    print("创建ECHO模型...")
    model = ECHO(
        atom_in_channels=74,
        fg_in_channels=167,
        bond_channels=4,
        hidden_channels=128,
        out_channels=1,
        heads=4,
        dropout=0.1
    )
    # model = compile(model, mode="default") wryyyyyyyyyy!!!!!
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {total_params:,}")
    
    # 创建训练器
    trainer = ECHOTrainer(model, device, learning_rate=0.001)
    
    # 训练模型
    print("开始训练...")
    num_epochs = 50
    best_val_auc = 0.0  # 改为监控AUC
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        # 训练
        train_loss = trainer.train_epoch(train_dataloader)
        print(f"训练损失: {train_loss:.4f}")
        
        # 验证
        val_loss, val_auc = trainer.validate_epoch(val_dataloader)
        print(f"验证损失: {val_loss:.4f}, 验证AUC: {val_auc:.4f}")
        
        # 早停检查 - 改为监控AUC（越大越好）
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_echo_model.pth')
            print("✓ 保存最佳模型")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"早停触发 (patience={patience})")
            break
    
    # 加载最佳模型进行最终评估
    print("\n最终评估...")
    model.load_state_dict(torch.load('best_echo_model.pth'))
    
    # 评估测试集
    test_metrics = evaluate_model(model, test_dataloader, device)
    
    if test_metrics:
        print("\n测试集评估结果:")
        print(f"ROC AUC: {test_metrics['auc']:.4f}")
    
    # 绘制训练曲线
    plot_training_curves(trainer)
    
    # 保存最终模型
    torch.save(model.state_dict(), 'final_echo_model.pth')
    print("训练完成！模型已保存。")

if __name__ == "__main__":
    main()