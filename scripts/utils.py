import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from pathlib import Path

def create_optimizer(model, config):
    
    optimizer = optim.Adam(model.parameters(), lr=config.CLASSIFIER_LR, weight_decay=1e-4)
    return optimizer

def create_scheduler(optimizer, num_training_steps, num_warmup_steps=0):
    from transformers import get_linear_schedule_with_warmup
    
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    total_mae = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Обучение")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        targets = batch['target'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, images=images)
        
        loss = criterion(outputs.squeeze(), targets.squeeze())
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
            
        if hasattr(train_loader.dataset, 'calorie_scaler') and train_loader.dataset.calorie_scaler is not None:
            scaler = train_loader.dataset.calorie_scaler
            pred_array = outputs.cpu().detach().numpy().reshape(-1, 1)
            target_array = targets.cpu().detach().numpy().reshape(-1, 1)
            pred_denorm = torch.tensor(scaler.inverse_transform(pred_array).flatten(), dtype=torch.float32)
            target_denorm = torch.tensor(scaler.inverse_transform(target_array).flatten(), dtype=torch.float32)
            mae = torch.mean(torch.abs(pred_denorm - target_denorm))
        else:
            mae = torch.mean(torch.abs(outputs - targets))
            
        total_mae += mae.item()
        num_batches += 1
        
        progress_bar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'MAE': f"{mae.item():.2f}"
        })
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    
    return avg_loss, avg_mae


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_mae = 0
    total_rmse = 0
    num_batches = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Валидация")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, images=images)
            
            loss = criterion(outputs.squeeze(), targets.squeeze())
            
            if hasattr(val_loader.dataset, 'calorie_scaler') and val_loader.dataset.calorie_scaler is not None:
                scaler = val_loader.dataset.calorie_scaler
                pred_array = outputs.cpu().numpy().reshape(-1, 1)
                target_array = targets.cpu().numpy().reshape(-1, 1)
                pred_denorm = torch.tensor(scaler.inverse_transform(pred_array).flatten(), dtype=torch.float32)
                target_denorm = torch.tensor(scaler.inverse_transform(target_array).flatten(), dtype=torch.float32)
                mae = torch.mean(torch.abs(pred_denorm - target_denorm))
                rmse = torch.sqrt(torch.mean((pred_denorm - target_denorm) ** 2))
                
                all_predictions.extend(pred_denorm.numpy())
                all_targets.extend(target_denorm.numpy())
            else:
                mae = torch.mean(torch.abs(outputs - targets))
                rmse = torch.sqrt(torch.mean((outputs - targets) ** 2))
                
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            
            total_loss += loss.item()
            total_mae += mae.item()
            total_rmse += rmse.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'MAE': f"{mae.item():.2f}",
                'RMSE': f"{rmse.item():.2f}"
            })
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    avg_rmse = total_rmse / num_batches
    
    # R2 score
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    ss_res = np.sum((all_targets - all_predictions) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return avg_loss, avg_mae, avg_rmse, r2_score, all_predictions, all_targets


def train(model, train_loader, val_loader, config):
    
    print(f"Устройство: {config.device}")
    print(f"Размер train: {len(train_loader.dataset)}")
    print(f"Размер val: {len(val_loader.dataset)}")
    
    criterion = nn.L1Loss()
    optimizer = create_optimizer(model, config)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=3
    )
    
    best_val_mae = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'train_mae': [],
        'val_loss': [], 'val_mae': [], 'val_rmse': [], 'val_r2': []
    }
    
    for epoch in range(config.EPOCHS):
        print(f"\nЭПОХА {epoch+1}/{config.EPOCHS}")
        
        train_loss, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, config.device
        )
        val_loss, val_mae, val_rmse, val_r2, _, _ = validate_epoch(
            model, val_loader, criterion, config.device
        )
        
        scheduler.step(val_mae)
        
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)
        
        print(f"Train - Loss: {train_loss:.4f}, MAE: {train_mae:.2f}")
        print(f"Val - Loss: {val_loss:.4f}, MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'config': config
            }, config.SAVE_PATH)
            
            print(f"Сохранена лучшая модель (MAE: {val_mae:.2f})")
            
            if val_mae <= 50:
                print(f"Достигнута целевая метрика MAE <= 50 kCal")
                break
        else:
            patience_counter += 1
            print(f"Нет улучшения ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})")
            
            if patience_counter >= 5:
                print(f"Early stopping на эпохе {epoch+1}")
                print(f"Лучший MAE: {best_val_mae:.2f} kCal")
                break
    
    print(f"\nОбучение завершено")
    print(f"Лучший MAE: {best_val_mae:.2f}")
    
    return history


def validate(model, val_loader, config):
    print(f"\nФинальная валидация")
    
    criterion = nn.L1Loss()
    val_loss, val_mae, val_rmse, val_r2, predictions, targets = validate_epoch(
        model, val_loader, criterion, config.device
    )
    
    print(f"Финальные результаты:")
    print(f"MAE: {val_mae:.2f} ккал")
    print(f"RMSE: {val_rmse:.2f} ккал")
    print(f"R2: {val_r2:.4f}")
    
    if val_mae < 50:
        print(f"Достигнута целевая метрика MAE < 50 ккал")
    else:
        print(f"Текущий MAE: {val_mae:.2f} ккал (цель: < 50 ккал)")
    
    return val_loss, val_mae, val_rmse, val_r2


def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # MAE
    axes[0, 1].plot(history['train_mae'], label='Train MAE')
    axes[0, 1].plot(history['val_mae'], label='Val MAE')
    axes[0, 1].set_title('MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (kCal)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # RMSE
    axes[1, 0].plot(history['val_rmse'], label='Val RMSE', color='orange')
    axes[1, 0].set_title('RMSE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE (kCal)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # R2
    axes[1, 1].plot(history['val_r2'], label='Val R2', color='green')
    axes[1, 1].set_title('R2 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('R2')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print(f"График сохранен")
