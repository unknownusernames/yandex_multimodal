import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR


class Config:
    
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    TEXT_MODEL_NAME = "roberta-base"
    IMAGE_MODEL_NAME = "resnet101"
    
    # фул разморозка
    UNFREEZE_BACKBONE = True
    UNFREEZE_TEXT = True
    
    TEXT_MODEL_UNFREEZE = "encoder.layer.11"
    IMAGE_MODEL_UNFREEZE = "layer4"
    
    # Learning rates для разных частей модели
    TEXT_LR = 1e-4
    IMAGE_LR = 1e-4
    CLASSIFIER_LR = 1e-3
    
    BATCH_SIZE = 64 #32, 64, 128
    GRADIENT_ACCUMULATION_STEPS = 1
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    
    TEXT_LR = 1e-4
    IMAGE_LR = 1e-4
    CLASSIFIER_LR = 1e-3
    WEIGHT_DECAY = 0.01
    GRADIENT_CLIP = 1.0
    
    # регрессор
    REGRESSOR_DROPOUT = 0.2 #0.1 0.3
    HIDDEN_DIM = 256 # 256, 512, 1024, 2048
    FUSION_TYPE = "concatenation"
    
    CNN_OUTPUT_DIM = 2048
    TEXT_FEATURE_DIM = 768
    
    IMAGE_SIZE = 224
    MAX_SEQ_LENGTH = 128
    MAX_CALORIES = 750
    MAX_MASS = 600
    
    COLOR_JITTER_BRIGHTNESS = 0.2
    COLOR_JITTER_CONTRAST = 0.2
    COLOR_JITTER_SATURATION = 0.2
    HORIZONTAL_FLIP_P = 0.5
    
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    NORMALIZE_TARGETS = True
    
    DATA_DIR = "data"
    INGREDIENTS_PATH = f"{DATA_DIR}/ingredients.csv"
    DISH_PATH = f"{DATA_DIR}/dish.csv"
    IMAGE_PATH = f"{DATA_DIR}/images"
    
    SAVE_PATH = "best_model.pth"
    LOG_DIR = "logs"
    
    USE_SCHEDULER = True
    SCHEDULER_TYPE = "ReduceLROnPlateau"
    SCHEDULER_PARAMS = {
        "mode": "min",
        "factor": 0.7,
        "patience": 3,
        "verbose": True
    }

    
    MONITOR_METRIC = "val_mae"
    MONITOR_MODE = "min"
    
    def __init__(self):
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.SEED)
            
        self._auto_configure()
    
    def _auto_configure(self):
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if gpu_memory < 8:
                self.BATCH_SIZE = 16
                self.GRADIENT_ACCUMULATION_STEPS = 4
                self.HIDDEN_DIM = 512
            elif gpu_memory < 16:
                self.BATCH_SIZE = 32
                self.GRADIENT_ACCUMULATION_STEPS = 2
            else:
                self.BATCH_SIZE = 256
                self.GRADIENT_ACCUMULATION_STEPS = 1
    
    def print_config(self):
        print(f"Девайс: {self.DEVICE}")
        print(f"Текст: {self.TEXT_MODEL_NAME}")
        print(f"Изображения: {self.IMAGE_MODEL_NAME}")
        print(f"Разморозка: текст - {self.TEXT_MODEL_UNFREEZE}, изображения - {self.IMAGE_MODEL_UNFREEZE}")
        print(f"Размер батча: {self.BATCH_SIZE} (накопление градиентов: {self.GRADIENT_ACCUMULATION_STEPS})")
        print(f"Learning rates: текст={self.TEXT_LR}, изображения={self.IMAGE_LR}, классификатор={self.CLASSIFIER_LR}")
        print(f"Размер скрытого слоя: {self.HIDDEN_DIM}")
        print(f"Размер CNN: {self.CNN_OUTPUT_DIM}")
        print(f"Размер текст: {self.TEXT_FEATURE_DIM}")
        print(f"Dropout регрессора: {self.REGRESSOR_DROPOUT}")
        print(f"Тип слияния: {self.FUSION_TYPE}")
        print(f"Нормализация таргетов: {self.NORMALIZE_TARGETS}")
        print(f"Шедулер: {self.SCHEDULER_TYPE}")
        print(f"Early Stopping Patience: {self.EARLY_STOPPING_PATIENCE}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
    
    def get_optimizer_params(self, model):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_params = [
            {
                "params": [p for n, p in model.text_model.named_parameters() 
                          if p.requires_grad and any(nd in n for nd in no_decay)],
                "lr": self.TEXT_LR,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.text_model.named_parameters() 
                          if p.requires_grad and not any(nd in n for nd in no_decay)],
                "lr": self.TEXT_LR,
                "weight_decay": self.WEIGHT_DECAY,
            },
            {
                "params": [p for n, p in model.image_model.named_parameters() 
                          if p.requires_grad and any(nd in n for nd in no_decay)],
                "lr": self.IMAGE_LR,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.image_model.named_parameters() 
                          if p.requires_grad and not any(nd in n for nd in no_decay)],
                "lr": self.IMAGE_LR,
                "weight_decay": self.WEIGHT_DECAY,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if not n.startswith("text_model") and 
                          not n.startswith("image_model") and
                          any(nd in n for nd in no_decay)],
                "lr": self.CLASSIFIER_LR,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if not n.startswith("text_model") and 
                          not n.startswith("image_model") and
                          not any(nd in n for nd in no_decay)],
                "lr": self.CLASSIFIER_LR,
                "weight_decay": self.WEIGHT_DECAY,
            },
        ]
        return optimizer_params
    
    def get_scheduler(self, optimizer, train_loader):
        if self.SCHEDULER_TYPE == "ReduceLROnPlateau":
            return ReduceLROnPlateau(
                optimizer, 
                mode=self.MONITOR_MODE,
                factor=self.SCHEDULER_PARAMS["factor"],
                patience=self.SCHEDULER_PARAMS["patience"],
                verbose=self.SCHEDULER_PARAMS["verbose"]
            )
        elif self.SCHEDULER_TYPE == "CosineAnnealing":
            steps_per_epoch = len(train_loader)
            total_steps = steps_per_epoch * self.EPOCHS
            return CosineAnnealingLR(optimizer, T_max=total_steps)
        else:
            return None
    
    @property
    def device(self):
        return self.DEVICE


if __name__ == "__main__":
    config = Config()
    config.print_config()