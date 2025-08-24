import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    progress_bar = tqdm(train_loader, desc="Обучение")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        targets = batch['target'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, images).squeeze()
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_predictions.extend(outputs.cpu().detach().numpy())
        all_targets.extend(targets.cpu().detach().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    
    if hasattr(train_loader.dataset, 'calorie_scaler') and train_loader.dataset.calorie_scaler is not None:
        scaler = train_loader.dataset.calorie_scaler
        pred_array = np.array(all_predictions).reshape(-1, 1)
        target_array = np.array(all_targets).reshape(-1, 1)
        pred_denorm = scaler.inverse_transform(pred_array).flatten()
        target_denorm = scaler.inverse_transform(target_array).flatten()
        mae = np.mean(np.abs(pred_denorm - target_denorm))
    else:
        mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))
    
    return avg_loss, mae

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Валидация")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(input_ids, attention_mask, images).squeeze()
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(val_loader)
    
    if hasattr(val_loader.dataset, 'calorie_scaler') and val_loader.dataset.calorie_scaler is not None:
        scaler = val_loader.dataset.calorie_scaler
        pred_array = np.array(all_predictions).reshape(-1, 1)
        target_array = np.array(all_targets).reshape(-1, 1)
        pred_denorm = scaler.inverse_transform(pred_array).flatten()
        target_denorm = scaler.inverse_transform(target_array).flatten()
        
        mae = np.mean(np.abs(pred_denorm - target_denorm))
        rmse = np.sqrt(np.mean((pred_denorm - target_denorm) ** 2))
        r2 = r2_score(target_denorm, pred_denorm)
    else:
        mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))
        rmse = np.sqrt(np.mean((np.array(all_predictions) - np.array(all_targets)) ** 2))
        r2 = r2_score(np.array(all_targets), np.array(all_predictions))
    
    return avg_loss, mae, rmse, r2

def create_optimizer(model, config):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    return optimizer

def train(model, train_loader, val_loader, config):
    device = config.DEVICE
    model = model.to(device)
    
    criterion = nn.L1Loss()
    optimizer = create_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=3
    )
    
    history = {
        'train_loss': [], 'train_mae': [],
        'val_loss': [], 'val_mae': [], 'val_rmse': [], 'val_r2': []
    }
    
    best_val_mae = float('inf')
    patience_counter = 0
    
    print(f"Начинаем обучение на {config.NUM_EPOCHS} эпох...")
    print(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nЭпоха {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 50)
        
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        
        val_loss, val_mae, val_rmse, val_r2 = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step(val_mae)
        
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)
        
        print(f"Train - Loss: {train_loss:.4f}, MAE: {train_mae:.2f} ккал")
        print(f"Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.2f} ккал, RMSE: {val_rmse:.2f} ккал, R²: {val_r2:.4f}")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'config': config
            }, 'best_model.pth')
            print(f"Новая лучшая модель! MAE: {val_mae:.2f} ккал")
        else:
            patience_counter += 1
            print(f"Early stopping: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
            
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping сработал на эпохе {epoch + 1}")
                break
        
        print(f"Текущий learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    
    print(f"\nОбучение завершено!")
    print(f"Лучший MAE: {best_val_mae:.2f} ккал")
    
    return history

def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Validation', color='red')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    axes[0, 1].plot(history['train_mae'], label='Train', color='blue')
    axes[0, 1].plot(history['val_mae'], label='Validation', color='red')
    axes[0, 1].set_title('MAE (ккал)')
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('MAE (ккал)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSE
    axes[1, 0].plot(history['val_rmse'], label='Validation', color='red')
    axes[1, 0].set_title('RMSE (ккал)')
    axes[1, 0].set_xlabel('Эпоха')
    axes[1, 0].set_ylabel('RMSE (ккал)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # R2
    axes[1, 1].plot(history['val_r2'], label='Validation', color='red')
    axes[1, 1].set_title('R2 Score')
    axes[1, 1].set_xlabel('Эпоха')
    axes[1, 1].set_ylabel('R2')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_loader, device, scaler=None):
    model.eval()
    all_predictions = []
    all_targets = []
    all_dish_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Тестирование"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            dish_ids = batch['dish_id']
            
            outputs = model(input_ids, attention_mask, images).squeeze()
            
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_dish_ids.extend(dish_ids)
    
    if scaler is not None:
        pred_array = np.array(all_predictions).reshape(-1, 1)
        target_array = np.array(all_targets).reshape(-1, 1)
        pred_denorm = scaler.inverse_transform(pred_array).flatten()
        target_denorm = scaler.inverse_transform(target_array).flatten()
    else:
        pred_denorm = np.array(all_predictions)
        target_denorm = np.array(all_targets)
    
    mae = np.mean(np.abs(pred_denorm - target_denorm))
    rmse = np.sqrt(np.mean((pred_denorm - target_denorm) ** 2))
    r2 = r2_score(target_denorm, pred_denorm)
    
    errors = pred_denorm - target_denorm
    abs_errors = np.abs(errors)
    
    print(f"MAE: {mae:.2f} ккал")
    print(f"RMSE: {rmse:.2f} ккал")
    print(f"R2: {r2:.4f}")
    print(f"Средняя абсолютная ошибка: {np.mean(abs_errors):.2f} ккал")
    print(f"Медианная абсолютная ошибка: {np.median(abs_errors):.2f} ккал")
    print(f"Максимальная ошибка: {np.max(abs_errors):.2f} ккал")
    
    print(f"\nАнализ ошибок по диапазонам калорийности")
    ranges = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]
    
    for low, high in ranges:
        mask = (target_denorm >= low) & (target_denorm < high)
        if np.sum(mask) > 0:
            range_mae = np.mean(abs_errors[mask])
            range_count = np.sum(mask)
            print(f"Калории {low}-{high}: MAE = {range_mae:.2f} ккал (n={range_count})")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': pred_denorm,
        'targets': target_denorm,
        'errors': errors,
        'dish_ids': all_dish_ids
    }
