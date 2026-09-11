# -*- coding: utf-8 -*-
"""
Network Architecture Implementations

This module contains various multimodal network architectures that can use
different fusion methods, demonstrating AECF as a drop-in replacement.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any
from .fusion_layers import (
    FusionInterface, ConcatenationFusion, AECFFusion, 
    AttentionFusion, BilinearFusion, TransformerFusion
)

class BaseMultimodalArchitecture(nn.Module):
    """Base class for all architectures with configurable fusion."""
    
    def __init__(self, image_dim: int, text_dim: int, num_classes: int, 
                 fusion_method: str = 'concat'):
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        
        # These will be implemented by subclasses
        self.image_encoder = None
        self.text_encoder = None
        self.fusion_layer = None
        self.classifier = None
    
    def create_fusion_layer(self, fusion_method: str, input_dims: List[int], 
                          output_dim: int) -> FusionInterface:
        """Factory method to create fusion layers."""
        if fusion_method == 'concat':
            return ConcatenationFusion(input_dims, output_dim)
        elif fusion_method == 'aecf':
            return AECFFusion(input_dims, output_dim)
        elif fusion_method == 'attention':
            return AttentionFusion(input_dims, output_dim)
        elif fusion_method == 'bilinear':
            return BilinearFusion(input_dims, output_dim)
        elif fusion_method == 'transformer':
            return TransformerFusion(input_dims, output_dim)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        """Forward pass - implemented by subclasses."""
        raise NotImplementedError

class SimpleMLPArchitecture(BaseMultimodalArchitecture):
    """Simple MLP-based architecture."""
    
    def __init__(self, image_dim: int, text_dim: int, num_classes: int, 
                 fusion_method: str = 'concat'):
        super().__init__(image_dim, text_dim, num_classes, fusion_method)
        
        # Simple projections
        hidden_dim = 256
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Configurable fusion
        self.fusion_layer = self.create_fusion_layer(
            fusion_method, [hidden_dim, hidden_dim], hidden_dim
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        img_feat = self.image_encoder(batch['image'])
        txt_feat = self.text_encoder(batch['text'])
        
        # Pass original features to AECF for missing data detection
        if isinstance(self.fusion_layer, AECFFusion):
            fused = self.fusion_layer([img_feat, txt_feat], [batch['image'], batch['text']])
        else:
            fused = self.fusion_layer([img_feat, txt_feat])
        
        logits = self.classifier(fused)
        
        # Return additional info for AECF
        if isinstance(self.fusion_layer, AECFFusion):
            return logits, self.fusion_layer.last_fusion_info
        return logits

class DeepMLPArchitecture(BaseMultimodalArchitecture):
    """Deeper MLP with residual connections."""
    
    def __init__(self, image_dim: int, text_dim: int, num_classes: int, 
                 fusion_method: str = 'concat'):
        super().__init__(image_dim, text_dim, num_classes, fusion_method)
        
        hidden_dim = 512
        
        # Deeper encoders
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.fusion_layer = self.create_fusion_layer(
            fusion_method, [hidden_dim, hidden_dim], hidden_dim
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        img_feat = self.image_encoder(batch['image'])
        txt_feat = self.text_encoder(batch['text'])
        
        # Pass original features to AECF for missing data detection
        if isinstance(self.fusion_layer, AECFFusion):
            fused = self.fusion_layer([img_feat, txt_feat], [batch['image'], batch['text']])
        else:
            fused = self.fusion_layer([img_feat, txt_feat])
        
        logits = self.classifier(fused)
        
        if isinstance(self.fusion_layer, AECFFusion):
            return logits, self.fusion_layer.last_fusion_info
        return logits

class CNNTextArchitecture(BaseMultimodalArchitecture):
    """CNN-based feature processing."""
    
    def __init__(self, image_dim: int, text_dim: int, num_classes: int, 
                 fusion_method: str = 'concat'):
        super().__init__(image_dim, text_dim, num_classes, fusion_method)
        
        hidden_dim = 384
        
        # "CNN-like" processing using 1D convolutions
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, 1024),  # Expand first
            nn.Unflatten(-1, (64, 16)),  # Reshape for conv
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(128 * 8, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 1024),
            nn.Unflatten(-1, (64, 16)),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(128 * 8, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.fusion_layer = self.create_fusion_layer(
            fusion_method, [hidden_dim, hidden_dim], hidden_dim
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        img_feat = self.image_encoder(batch['image'])
        txt_feat = self.text_encoder(batch['text'])
        
        # Pass original features to AECF for missing data detection
        if isinstance(self.fusion_layer, AECFFusion):
            fused = self.fusion_layer([img_feat, txt_feat], [batch['image'], batch['text']])
        else:
            fused = self.fusion_layer([img_feat, txt_feat])
        
        logits = self.classifier(fused)
        
        if isinstance(self.fusion_layer, AECFFusion):
            return logits, self.fusion_layer.last_fusion_info
        return logits

class MultiScaleArchitecture(BaseMultimodalArchitecture):
    """Multi-scale feature processing."""
    
    def __init__(self, image_dim: int, text_dim: int, num_classes: int, 
                 fusion_method: str = 'concat'):
        super().__init__(image_dim, text_dim, num_classes, fusion_method)
        
        base_dim = 256
        
        # Multi-scale processing for each modality
        self.image_scales = nn.ModuleList([
            nn.Sequential(nn.Linear(image_dim, base_dim), nn.ReLU()),
            nn.Sequential(nn.Linear(image_dim, base_dim * 2), nn.ReLU(), 
                         nn.Linear(base_dim * 2, base_dim)),
            nn.Sequential(nn.Linear(image_dim, base_dim * 4), nn.ReLU(),
                         nn.Linear(base_dim * 4, base_dim * 2), nn.ReLU(),
                         nn.Linear(base_dim * 2, base_dim))
        ])
        
        self.text_scales = nn.ModuleList([
            nn.Sequential(nn.Linear(text_dim, base_dim), nn.ReLU()),
            nn.Sequential(nn.Linear(text_dim, base_dim * 2), nn.ReLU(),
                         nn.Linear(base_dim * 2, base_dim)),
            nn.Sequential(nn.Linear(text_dim, base_dim * 4), nn.ReLU(),
                         nn.Linear(base_dim * 4, base_dim * 2), nn.ReLU(),
                         nn.Linear(base_dim * 2, base_dim))
        ])
        
        # Aggregate multi-scale features
        self.img_aggregator = nn.Linear(base_dim * 3, base_dim)
        self.txt_aggregator = nn.Linear(base_dim * 3, base_dim)
        
        self.fusion_layer = self.create_fusion_layer(
            fusion_method, [base_dim, base_dim], base_dim
        )
        
        self.classifier = nn.Linear(base_dim, num_classes)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        # Multi-scale processing
        img_features = [scale(batch['image']) for scale in self.image_scales]
        txt_features = [scale(batch['text']) for scale in self.text_scales]
        
        # Aggregate scales
        img_feat = self.img_aggregator(torch.cat(img_features, dim=-1))
        txt_feat = self.txt_aggregator(torch.cat(txt_features, dim=-1))
        
        # Pass original features to AECF for missing data detection
        if isinstance(self.fusion_layer, AECFFusion):
            fused = self.fusion_layer([img_feat, txt_feat], [batch['image'], batch['text']])
        else:
            fused = self.fusion_layer([img_feat, txt_feat])
        
        logits = self.classifier(fused)
        
        if isinstance(self.fusion_layer, AECFFusion):
            return logits, self.fusion_layer.last_fusion_info
        return logits

class ResNetLikeArchitecture(BaseMultimodalArchitecture):
    """ResNet-inspired architecture with skip connections."""
    
    def __init__(self, image_dim: int, text_dim: int, num_classes: int, 
                 fusion_method: str = 'concat'):
        super().__init__(image_dim, text_dim, num_classes, fusion_method)
        
        hidden_dim = 512
        
        # ResNet-like blocks
        self.image_input = nn.Linear(image_dim, hidden_dim)
        self.image_blocks = nn.ModuleList([
            self._make_resnet_block(hidden_dim) for _ in range(3)
        ])
        
        self.text_input = nn.Linear(text_dim, hidden_dim)
        self.text_blocks = nn.ModuleList([
            self._make_resnet_block(hidden_dim) for _ in range(3)
        ])
        
        self.fusion_layer = self.create_fusion_layer(
            fusion_method, [hidden_dim, hidden_dim], hidden_dim
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes)
        )
    
    def _make_resnet_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        # Process with residual connections
        img_feat = self.image_input(batch['image'])
        for block in self.image_blocks:
            img_feat = img_feat + block(img_feat)  # Skip connection
        
        txt_feat = self.text_input(batch['text'])
        for block in self.text_blocks:
            txt_feat = txt_feat + block(txt_feat)  # Skip connection
        
        # Pass original features to AECF for missing data detection
        if isinstance(self.fusion_layer, AECFFusion):
            fused = self.fusion_layer([img_feat, txt_feat], [batch['image'], batch['text']])
        else:
            fused = self.fusion_layer([img_feat, txt_feat])
        
        logits = self.classifier(fused)
        
        if isinstance(self.fusion_layer, AECFFusion):
            return logits, self.fusion_layer.last_fusion_info
        return logits