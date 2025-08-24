import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import timm

class CNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        
        backbone_features = self.backbone.num_features
        
        self.projection = nn.Sequential(
            nn.Linear(backbone_features, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Полностью размораживаем backbone
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        features = self.backbone(x)
        projected_features = self.projection(features)
        return projected_features

class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
            
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        
        self.text_projection = nn.Sequential(
            nn.Linear(768, config.TEXT_FEATURE_DIM),  # 768 -> 256
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
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
        for module in self.regressor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask, images):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state[:, 0, :]
        text_features = self.text_projection(text_features)
        
        image_features = self.image_model(images)
        
        fused_features = torch.cat([image_features, text_features], dim=1)
        
        calories = self.regressor(fused_features)
        
        return calories
    
    def print_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Общее количество параметров: {total_params:,}")
        print(f"Обучаемых параметров: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"Архитектура: EfficientNet-B0 + RoBERTa-base + Concatenation")
        print(f"Размерность CNN: {self.config.CNN_OUTPUT_DIM}")
        print(f"Размерность текста: {self.config.TEXT_FEATURE_DIM}")
        print(f"Скрытый слой: {self.config.HIDDEN_DIM}")
        print(f"Dropout: {self.config.REGRESSOR_DROPOUT}")
        print(f"Тип слияния: {self.config.FUSION_TYPE}")
        print(f"Полная разморозка: Backbone={self.config.UNFREEZE_BACKBONE}, Text={self.config.UNFREEZE_TEXT}")


if __name__ == "__main__":
    from config import Config
    
    config = Config()
    
    model = MultimodalModel(config)
    
    batch_size = 2
    input_ids = torch.randint(0, 1000, (batch_size, 128))
    attention_mask = torch.ones(batch_size, 128)
    images = torch.randn(batch_size, 3, 224, 224)
    
    print(f"input_ids: {input_ids.shape}")
    print(f"attention_mask: {attention_mask.shape}")
    print(f"images: {images.shape}")
    
    with torch.no_grad():
        output = model(input_ids, attention_mask, images)
    
    model.print_model_info()
    
    total_params, trainable_params = model.get_trainable_parameters()
    print(f"После настройки: {total_params:,} всего, {trainable_params:,} обучаемых")