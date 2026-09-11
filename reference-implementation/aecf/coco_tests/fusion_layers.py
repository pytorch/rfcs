# -*- coding: utf-8 -*-
"""
Fusion Layer Implementations

This module contains different fusion methods including the AECF fusion
and baseline fusion approaches for multimodal data integration.
"""

import torch
import torch.nn as nn
from typing import List, Any
from AECFLayer import MultimodalAttentionPool, CurriculumMasking

class FusionInterface(nn.Module):
    """Abstract interface that all fusion methods must implement."""
    
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

class ConcatenationFusion(FusionInterface):
    """Simple concatenation baseline."""
    
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__(input_dims, output_dim)
        total_dim = sum(input_dims)
        self.projection = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        concatenated = torch.cat(modalities, dim=-1)
        return self.projection(concatenated)

class AECFFusion(FusionInterface):
    """AECF-based fusion - the drop-in replacement."""
    
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__(input_dims, output_dim)
        
        # Ensure all modalities have same dimension for attention
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])
        
        # AECF components
        self.curriculum_masking = CurriculumMasking(
            base_mask_prob=0.15,
            entropy_target=0.7,
            min_active=1
        )
        
        self.attention_pool = MultimodalAttentionPool(
            embed_dim=output_dim,
            num_heads=8,
            dropout=0.1,
            curriculum_masking=self.curriculum_masking,
            batch_first=True
        )
        
        # Learnable fusion query
        self.fusion_query = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)
        
        # Store info for analysis
        self.last_fusion_info = {}
    
    def forward(self, modalities: List[torch.Tensor], original_modalities: List[torch.Tensor] = None) -> torch.Tensor:
        batch_size = modalities[0].size(0)
        
        # If original modalities provided, detect missing data for masking
        key_padding_mask = None
        if original_modalities is not None and len(original_modalities) == 2:
            # Detect missing modalities based on original input (before projection)
            img_present = torch.norm(original_modalities[0], dim=1) > 1e-6  # [batch_size]
            txt_present = torch.norm(original_modalities[1], dim=1) > 1e-6   # [batch_size]
            
            # Create attention mask (True = should be ignored in attention)
            key_padding_mask = torch.stack([~img_present, ~txt_present], dim=1)  # [batch, 2]
        
        # Project all modalities to same dimension
        projected = [proj(mod) for proj, mod in zip(self.projections, modalities)]
        
        # Stack for attention: [batch, num_modalities, output_dim]
        stacked = torch.stack(projected, dim=1)
        
        # Create query for each sample
        query = self.fusion_query.expand(batch_size, -1, -1)
        
        # Apply AECF attention with proper masking
        fused, info = self.attention_pool(
            query=query,
            key=stacked,
            value=stacked,
            key_padding_mask=key_padding_mask,  # Properly mask missing modalities
            return_info=True
        )
        
        # Store info for analysis
        self.last_fusion_info = info
        
        return fused.squeeze(1)  # [batch, output_dim]

class AttentionFusion(FusionInterface):
    """Standard attention fusion without curriculum learning."""
    
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__(input_dims, output_dim)
        
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])
        
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.fusion_query = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        batch_size = modalities[0].size(0)
        
        projected = [proj(mod) for proj, mod in zip(self.projections, modalities)]
        stacked = torch.stack(projected, dim=1)
        
        query = self.fusion_query.expand(batch_size, -1, -1)
        
        fused, _ = self.attention(query, stacked, stacked)
        return fused.squeeze(1)

class BilinearFusion(FusionInterface):
    """Bilinear fusion for two modalities."""
    
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__(input_dims, output_dim)
        assert len(input_dims) == 2, "Bilinear fusion requires exactly 2 modalities"
        
        self.proj1 = nn.Linear(input_dims[0], output_dim)
        self.proj2 = nn.Linear(input_dims[1], output_dim)
        self.bilinear = nn.Bilinear(output_dim, output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        x1 = self.proj1(modalities[0])
        x2 = self.proj2(modalities[1])
        fused = self.bilinear(x1, x2)
        return self.norm(fused)

class TransformerFusion(FusionInterface):
    """Transformer-based fusion."""
    
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__(input_dims, output_dim)
        
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=8,
            dim_feedforward=output_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls_token = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        batch_size = modalities[0].size(0)
        
        projected = [proj(mod) for proj, mod in zip(self.projections, modalities)]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        stacked = torch.cat([cls_tokens] + [x.unsqueeze(1) for x in projected], dim=1)
        
        # Apply transformer
        output = self.transformer(stacked)
        
        # Return CLS token representation
        return output[:, 0]