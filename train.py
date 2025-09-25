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

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*GetValence.*")
warnings.filterwarnings("ignore", message=".*please use GetValence.*")
warnings.filterwarnings("ignore", message=".*DEPRECATION WARNING.*")

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from echo import ECHO
from rdkit import Chem
from utils.atom_extractor import load_smiles

class SMILESDataset(Dataset):
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
            raise RuntimeError(f"index {data_idx} error: {str(e)}")

def custom_collate(batch):
    valid_batch = [(smiles, label) for smiles, label in batch if smiles is not None]
    
    if not valid_batch:
        return [], torch.tensor([])
    
    smiles_list, label_list = zip(*valid_batch)
    label_batch = torch.tensor(label_list, dtype=torch.float32)
    return list(smiles_list), label_batch

class ECHOTrainer:
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        num_samples = 0
        
        for smiles_batch, label_batch in tqdm(dataloader, desc="training"):
            if len(smiles_batch) == 0:
                continue
            
            self.optimizer.zero_grad()
            
            try:
                label_batch = label_batch.to(self.device)
                outputs = self.model(smiles_batch)    
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(1)
                if label_batch.dim() == 1:
                    label_batch = label_batch.unsqueeze(1)
                
                loss = self.criterion(outputs, label_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item() * len(smiles_batch)
                num_samples += len(smiles_batch)
                
            except Exception as e:
                import traceback
                print(f"batch error: {str(e)}")
                traceback.print_exc()
                continue
        
        avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_probabilities = []
        all_labels = []
        num_samples = 0
        
        with torch.no_grad():
            for smiles_batch, label_batch in tqdm(dataloader, desc="val-ing"):
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
                        
                        prob = torch.sigmoid(output)
                        all_probabilities.append(prob.cpu().item())
                        all_labels.append(label_batch[i].cpu().item())
                        
                    except Exception as e:
                        print(f"val error: {smiles[:50]}: {str(e)}")
                        continue
                
                if batch_outputs:
                    batch_output_tensor = torch.cat(batch_outputs, dim=0)
                    batch_target_tensor = torch.cat(batch_targets, dim=0)
                    
                    loss = self.criterion(batch_output_tensor, batch_target_tensor)
                    total_loss += loss.item() * len(batch_outputs)
                    num_samples += len(batch_outputs)

        avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
        
        auc = 0.0
        auc = roc_auc_score(all_labels, all_probabilities)
        
        self.val_losses.append(avg_loss)
        self.val_aucs.append(auc)
        
        current_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step(avg_loss)
        
        new_lr = self.optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"adjusting lr: {current_lr:.6f} -> {new_lr:.6f}")
        else:
            print(f"current lr: {current_lr:.6f}")
        
        return avg_loss, auc

def evaluate_model(model, dataloader, device):
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
                    
                    prob = torch.sigmoid(output)
                    all_probabilities.append(prob.cpu().item())
                    all_labels.append(label_batch[i].cpu().item())
                    
                except Exception as e:
                    print(f"test error, index: {smiles[:50]}, error: {str(e)}")
                    continue
    
    if not all_labels or len(set(all_labels)) <= 1:
        return None
    
    try:
        auc = roc_auc_score(all_labels, all_probabilities)
        return {'auc': auc}
    except ValueError as e:
        print(f"calculate error: {str(e)}")
        return None

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    import pandas as pd
    file_path = os.path.join('data', 'hiv_cleaned.csv')
    df = pd.read_csv(file_path)
    actual_num_samples = len(df)
    num_samples = actual_num_samples
    all_indices = list(range(num_samples))
    
    train_val_indices, test_indices = train_test_split(
        all_indices, test_size=0.1, random_state=42
    )
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=0.20, random_state=42
    )
    
    train_dataset = SMILESDataset(train_indices)
    val_dataset = SMILESDataset(val_indices)
    test_dataset = SMILESDataset(test_indices)
    
    batch_size = 32
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=custom_collate,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
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
    trainer = ECHOTrainer(model, device, learning_rate=0.001)
    
    num_epochs = 50
    best_val_auc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        train_loss = trainer.train_epoch(train_dataloader)
        print(f"train loss: {train_loss:.4f}")
        
        val_loss, val_auc = trainer.validate_epoch(val_dataloader)
        print(f"val loss: {val_loss:.4f}, val ROCAUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    model.load_state_dict(torch.load('best_echo_model.pth'))
    
    test_metrics = evaluate_model(model, test_dataloader, device)
    
    if test_metrics:
        print(f"ROC AUC: {test_metrics['auc']:.4f}")
    
if __name__ == "__main__":
    main()