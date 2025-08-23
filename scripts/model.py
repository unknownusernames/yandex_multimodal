import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import timm


class CNN(nn.Module):
    
    def __init__(self, output_dim=2048):
        super().__init__()
        
        self.backbone = timm.create_model('resnet101', pretrained=True, num_classes=0)
        self.fc = nn.Linear(2048, output_dim)
        
        # Размораживаем backbone
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


class MultimodalModel(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        for param in self.text_model.parameters():
            param.requires_grad = True
        
        self.image_model = CNN(output_dim=config.CNN_OUTPUT_DIM)
        
        fusion_dim = config.CNN_OUTPUT_DIM + config.TEXT_FEATURE_DIM
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.REGRESSOR_DROPOUT),
            nn.Linear(config.HIDDEN_DIM, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, input_ids, attention_mask, images):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state[:, 0, :]
        
        image_features = self.image_model(images)
        
        fused_features = torch.cat([image_features, text_features], dim=1)
        
        calories = self.regressor(fused_features)
        
        return calories
    
    def get_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def print_model_info(self):
        print(f"Мультимодальная модель:")
        print(f"CV: ResNet101 (разморожен)")
        print(f"NLP: roberta")
        print(f"Текст и изображение конкатенируются")
        print(f"Регрессор: 2 FC слоя (4096) + 50% Dropout")
        
        total_params, trainable_params = self.get_trainable_parameters()
        print(f"Параметры модели: {total_params:,} всего, {trainable_params:,} обучаемых")


if __name__ == "__main__":
    from config import Config
    
    config = Config()
    
    model = MultimodalModel(config)
    
    batch_size = 2
    input_ids = torch.randint(0, 1000, (batch_size, 128))
    attention_mask = torch.ones(batch_size, 128)
    images = torch.randn(batch_size, 3, 224, 224)
    
    print(f"Входные размеры:")
    print(f"input_ids: {input_ids.shape}")
    print(f"attention_mask: {attention_mask.shape}")
    print(f"images: {images.shape}")
    
    with torch.no_grad():
        output = model(input_ids, attention_mask, images)
        print(f"Выход: {output.shape}")
        print(f"Значения: {output.squeeze()}")
    
    model.print_model_info()
    
    total_params, trainable_params = model.get_trainable_parameters()
    print(f"После настройки: {total_params:,} всего, {trainable_params:,} обучаемых")