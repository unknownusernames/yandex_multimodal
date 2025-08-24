import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import MinMaxScaler
import timm

class MultimodalDataset(Dataset):
    def __init__(self, dish_df, ingredients_df, images_dir, tokenizer, split='train', config=None):
        self.dish_df = dish_df
        self.ingredients_df = ingredients_df
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.split = split
        self.config = config
        self.image_size = config.IMAGE_SIZE if config else 224
        
        if config and hasattr(config, 'IMAGE_MODEL_NAME'):
            try:
                temp_model = timm.create_model(config.IMAGE_MODEL_NAME, pretrained=False, num_classes=0)
                if hasattr(temp_model, 'default_cfg'):
                    self.imagenet_mean = temp_model.default_cfg.get('mean', [0.485, 0.456, 0.406])
                    self.imagenet_std = temp_model.default_cfg.get('std', [0.229, 0.224, 0.225])
                else:
                    self.imagenet_mean = [0.485, 0.456, 0.406]
                    self.imagenet_std = [0.229, 0.224, 0.225]
                del temp_model
            except:
                self.imagenet_mean = [0.485, 0.456, 0.406]
                self.imagenet_std = [0.229, 0.224, 0.225]
        else:
            self.imagenet_mean = [0.485, 0.456, 0.406]
            self.imagenet_std = [0.229, 0.224, 0.225]
        
        self.ingredients_dict = dict(zip(ingredients_df['id'], ingredients_df['ingr']))
        
        self.calorie_scaler = MinMaxScaler()
        if split == 'train':
            calories_array = self.dish_df['total_calories'].values.reshape(-1, 1)
            self.calorie_scaler.fit(calories_array)
            print(f"Min: {self.calorie_scaler.data_min_[0]:.2f}, Max: {self.calorie_scaler.data_max_[0]:.2f}")
    
    def __len__(self):
        return len(self.dish_df)
    
    def __getitem__(self, idx):
        row = self.dish_df.iloc[idx]
        
        dish_id = str(row['dish_id'])
        image_path = self.images_dir / dish_id / 'rgb.png'
        
        if image_path.exists():
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        else:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        if self.split == 'train':
            transform = self.get_transforms(is_training=True)
        else:
            transform = self.get_transforms(is_training=False)
        
        image = transform(image=image)['image']
        
        ingredients_str = row['ingredients']
        if pd.isna(ingredients_str):
            ingredients_text = "no ingredients"
        else:
            ingredient_ids = ingredients_str.split(';')
            ingredients_text = '; '.join([self.ingredients_dict.get(ingr_id, ingr_id) for ingr_id in ingredient_ids])
        
        encoding = self.tokenizer(
            ingredients_text,
            truncation=True,
            padding='max_length',
            max_length=self.config.MAX_SEQ_LENGTH if self.config else 128,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        target = torch.tensor(row['total_calories'], dtype=torch.float32)
        if hasattr(self, 'calorie_scaler') and self.calorie_scaler is not None:
            target = torch.tensor(
                self.calorie_scaler.transform([[target.item()]])[0][0], 
                dtype=torch.float32
            )
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': image,
            'target': target,
            'dish_id': row['dish_id']
        }
    
    def get_transforms(self, is_training=False):
        if is_training and self.config:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.ColorJitter(
                    brightness=self.config.COLOR_JITTER_BRIGHTNESS,
                    contrast=self.config.COLOR_JITTER_CONTRAST,
                    saturation=self.config.COLOR_JITTER_SATURATION
                ),
                A.HorizontalFlip(p=self.config.HORIZONTAL_FLIP_P),
                A.Normalize(
                    mean=self.imagenet_mean,
                    std=self.imagenet_std
                ),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(
                    mean=self.imagenet_mean,
                    std=self.imagenet_std
                ),
                ToTensorV2(),
            ])

def create_dataloaders(config):
    data_dir = Path("data")
    dish_df = pd.read_csv(data_dir / "dish.csv")
    ingredients_df = pd.read_csv(data_dir / "ingredients.csv")
    images_dir = data_dir / "images"
    
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    
    train_df = dish_df[dish_df['split'] == 'train'].reset_index(drop=True)
    val_df = dish_df[dish_df['split'] == 'test'].reset_index(drop=True)
    
    train_dataset = MultimodalDataset(train_df, ingredients_df, images_dir, tokenizer, 'train', config)
    
    val_dataset = MultimodalDataset(val_df, ingredients_df, images_dir, tokenizer, 'test', config)
    val_dataset.calorie_scaler = train_dataset.calorie_scaler
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Созданы DataLoader:")
    print(f"Тренировочный: {len(train_loader)} батчей, {len(train_dataset)} samples")
    print(f"Валидационный: {len(val_loader)} батчей, {len(val_dataset)} samples")
    
    return train_loader, val_loader, train_dataset.calorie_scaler


if __name__ == "__main__":
    from config import Config
    
    config = Config()
    
    train_loader, val_loader, scaler = create_dataloaders(config)
    
    sample = next(iter(train_loader))
    print(f"Ключи: {sample.keys()}")
    print(f"Размер изображения: {sample['image'].shape}")
    print(f"Размер input_ids: {sample['input_ids'].shape}")
    print(f"Калории: {sample['target']}")
    print(f"Ингредиенты: {sample['input_ids']}")

