import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer

class MultimodalDataset(Dataset):
    def __init__(self, dish_df_path, ingredients_df_path, images_dir, 
                 split='train', max_calories=750, max_mass=600, image_size=224, config=None):
        self.dish_df = pd.read_csv(dish_df_path)
        self.ingredients_df = pd.read_csv(ingredients_df_path)
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.config = config
        
        self.ingredients_dict = dict(zip(self.ingredients_df['id'], self.ingredients_df['ingr']))
        
        print(f"Исходный размер датасета: {len(self.dish_df)}")
        
        if split == 'train':
            self.dish_df = self.dish_df[self.dish_df['split'] == 'train']
        else:
            self.dish_df = self.dish_df[self.dish_df['split'] == 'test']
        
        print(f"После фильтрации по split '{split}': {len(self.dish_df)}")
        
        self.dish_df = self.dish_df[self.dish_df['total_calories'] <= max_calories]
        print(f"После фильтрации по калорийности (≤{max_calories} ккал): {len(self.dish_df)}")
        
        self.dish_df = self.dish_df[self.dish_df['total_mass'] <= max_mass]
        print(f"После фильтрации по массе (≤{max_mass} г): {len(self.dish_df)}")
        
        print(f"Размер {split} датасета: {len(self.dish_df)}")
        
        self.calorie_scaler = MinMaxScaler()
        if split == 'train':
            calories_array = self.dish_df['total_calories'].values.reshape(-1, 1)
            self.calorie_scaler.fit(calories_array)
            print(f"MinMaxScaler обучен на тренировочных данных")
            print(f"Min: {self.calorie_scaler.data_min_[0]:.2f}, Max: {self.calorie_scaler.data_max_[0]:.2f}")
        
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
        self.transforms = self.get_transforms(split == 'train')
    
    def __len__(self):
        return len(self.dish_df)
    
    def __getitem__(self, idx):
        row = self.dish_df.iloc[idx]
        
        ingredients_text = self.get_ingredients_text(row['ingredients'])
        
        encoding = self.tokenizer(
            ingredients_text,
            truncation=True,
            padding='max_length',
            max_length=self.config.MAX_SEQ_LENGTH if self.config else 128,
            return_tensors='pt'
        )
        
        image = self.load_image(row['dish_id'])
        
        if self.transforms:
            image = self.transforms(image=image)['image']
        
        target = torch.tensor(row['total_calories'], dtype=torch.float32)
        if hasattr(self, 'calorie_scaler') and self.calorie_scaler is not None:
            target = torch.tensor(
                self.calorie_scaler.transform([[target.item()]])[0][0], 
                dtype=torch.float32
            )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'image': image,
            'target': target,
            'dish_id': row['dish_id']
        }
    
    def get_ingredients_text(self, ingredients_str):
        if pd.isna(ingredients_str) or ingredients_str == '':
            return "no ingredients"
        
        ingredient_ids = ingredients_str.split(';')
        ingredient_names = []
        
        for ingr_id in ingredient_ids:
            try:
                if ingr_id.startswith('ingr_'):
                    numeric_id = int(ingr_id.replace('ingr_', ''))
                    if numeric_id in self.ingredients_dict:
                        ingredient_names.append(self.ingredients_dict[numeric_id])
                    else:
                        ingredient_names.append(f"unknown_ingredient_{numeric_id}")
                else:
                    ingredient_names.append(f"invalid_id_{ingr_id}")
            except ValueError:
                ingredient_names.append(f"invalid_id_{ingr_id}")
        
        if not ingredient_names:
            return "no ingredients"
        
        return ", ".join(ingredient_names)
    
    def load_image(self, dish_id):
        image_path = self.images_dir / f"{dish_id}" / "rgb.png"
        
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        return image
    
    def get_calorie_scaler(self):
        return self.calorie_scaler
    
    def get_transforms(self, is_training):
    if is_training and self.config:
        return A.Compose([
            A.SmallestMaxSize(max_size=self.config.IMAGE_SIZE, p=1.0),
            A.RandomCrop(height=self.config.IMAGE_SIZE, width=self.config.IMAGE_SIZE, p=1.0),
            A.SquareSymmetry(p=1.0),
            A.Affine(scale=(0.8, 1.2),
                     rotate=(-15, 15),
                     translate_percent=(-0.1, 0.1),
                     shear=(-10, 10),
                     fill=0,
                     p=0.8),
            A.ColorJitter(brightness=self.config.COLOR_JITTER_BRIGHTNESS,
                          contrast=self.config.COLOR_JITTER_CONTRAST,
                          saturation=self.config.COLOR_JITTER_SATURATION,
                          hue=0.1,
                          p=0.7),
            A.HorizontalFlip(p=self.config.HORIZONTAL_FLIP_P),
            A.Normalize(
                mean=self.config.IMAGENET_MEAN,
                std=self.config.IMAGENET_STD
            ),
            ToTensorV2(),
        ])
        else:
            return A.Compose([
                A.SmallestMaxSize(max_size=224, p=1.0),
                A.CenterCrop(height=224, width=224, p=1.0),
                A.Normalize(
                    mean=self.config.IMAGENET_MEAN if self.config else [0.485, 0.456, 0.406], 
                    std=self.config.IMAGENET_STD if self.config else [0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    images = torch.stack([item['image'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    dish_ids = [item['dish_id'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'image': images,
        'target': targets,
        'dish_id': dish_ids
    }

def create_dataloaders(config):
    train_dataset = MultimodalDataset(
        dish_df_path=config.DISH_PATH,
        ingredients_df_path=config.INGREDIENTS_PATH,
        images_dir=config.IMAGE_PATH,
        split='train',
        max_calories=config.MAX_CALORIES,
        max_mass=config.MAX_MASS,
        image_size=config.IMAGE_SIZE,
        config=config
    )
    
    val_dataset = MultimodalDataset(
        dish_df_path=config.DISH_PATH,
        ingredients_df_path=config.INGREDIENTS_PATH,
        images_dir=config.IMAGE_PATH,
        split='test',
        max_calories=config.MAX_CALORIES,
        max_mass=config.MAX_MASS,
        image_size=config.IMAGE_SIZE,
        config=config
    )
    
    val_dataset.calorie_scaler = train_dataset.calorie_scaler
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"Train: {len(train_loader)} батчей по {config.BATCH_SIZE}")
    print(f"Val: {len(val_loader)} батчей по {config.BATCH_SIZE}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    from config import Config
    
    config = Config()
    
    train_loader, val_loader = create_dataloaders(config)
    
    sample = next(iter(train_loader))
    print(f"Ключи: {sample.keys()}")
    print(f"Размер изображения: {sample['image'].shape}")
    print(f"Размер input_ids: {sample['input_ids'].shape}")
    print(f"Калории: {sample['target']}")
    print(f"Ингредиенты: {sample['input_ids']}")