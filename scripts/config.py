import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR


class Config:
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Модели
    TEXT_MODEL_NAME = "roberta-base"
    IMAGE_MODEL_NAME = "efficientnet_b0"  # resnet101
    
    IMAGE_SIZE = 224
    MAX_SEQ_LENGTH = 128
    BATCH_SIZE = 128 #256
    
    CNN_OUTPUT_DIM = 256  #2048
    TEXT_FEATURE_DIM = 256  #768
    HIDDEN_DIM = 512  #4096
    
    REGRESSOR_DROPOUT = 0.2  #0.5
    
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 5
    
    UNFREEZE_BACKBONE = True
    UNFREEZE_TEXT = True
    
    FUSION_TYPE = "concatenation"
    
    COLOR_JITTER_BRIGHTNESS = 0.1  #0.2
    COLOR_JITTER_CONTRAST = 0.1    #0.2
    COLOR_JITTER_SATURATION = 0.1  #0.2
    HORIZONTAL_FLIP_P = 0.5
    
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def print_config(self):
        print(f"Устройство: {self.DEVICE}")
        print(f"Текстовая модель: {self.TEXT_MODEL_NAME}")
        print(f"Изображения модель: {self.IMAGE_MODEL_NAME}")
        print(f"Размер изображения: {self.IMAGE_SIZE}")
        print(f"Максимальная длина текста: {self.MAX_SEQ_LENGTH}")
        print(f"Размер батча: {self.BATCH_SIZE}")
        print(f"Размерность CNN: {self.CNN_OUTPUT_DIM}")
        print(f"Размерность текста: {self.TEXT_FEATURE_DIM}")
        print(f"Скрытый слой: {self.HIDDEN_DIM}")
        print(f"Dropout: {self.REGRESSOR_DROPOUT}")
        print(f"Learning rate: {self.LEARNING_RATE}")
        print(f"Weight decay: {self.WEIGHT_DECAY}")
        print(f"Количество эпох: {self.NUM_EPOCHS}")
        print(f"Early stopping patience: {self.EARLY_STOPPING_PATIENCE}")
        print(f"Разморозка backbone: {self.UNFREEZE_BACKBONE}")
        print(f"Разморозка текста: {self.UNFREEZE_TEXT}")
        print(f"Тип слияния: {self.FUSION_TYPE}")
        print(f"ColorJitter brightness: {self.COLOR_JITTER_BRIGHTNESS}")
        print(f"ColorJitter contrast: {self.COLOR_JITTER_CONTRAST}")
        print(f"ColorJitter saturation: {self.COLOR_JITTER_SATURATION}")
        print(f"Horizontal flip: {self.HORIZONTAL_FLIP_P}")
        print(f"ImageNet mean: {self.IMAGENET_MEAN}")
        print(f"ImageNet std: {self.IMAGENET_STD}")

config = Config()