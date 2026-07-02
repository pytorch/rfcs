# -*- coding: utf-8 -*-
"""
Legacy Model Implementations

This module contains the original single-architecture models for backward
compatibility with existing test code.
"""

import torch
import torch.nn as nn
from AECFLayer import MultimodalAttentionPool, CurriculumMasking

class MultimodalClassifier(nn.Module):
    """Original unified multimodal classifier for backward compatibility."""
    
    def __init__(self, image_dim=512, text_dim=512, num_classes=80, fusion_type='baseline'):
        super().__init__()
        self.fusion_type = fusion_type
        
        # Shared feature projections
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusion layers based on type
        if fusion_type == 'baseline':
            self.fusion = nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        elif fusion_type == 'aecf':
            # Use proper AECF components
            self.curriculum_masking = CurriculumMasking(
                base_mask_prob=0.1,  # Conservative masking
                entropy_target=0.7,  # Target 70% of max entropy
                min_active=1
            )
            
            self.attention_pool = MultimodalAttentionPool(
                embed_dim=256,
                num_heads=8,
                dropout=0.1,
                curriculum_masking=self.curriculum_masking,
                batch_first=True
            )
            
            # Learnable fusion query
            self.fusion_query = nn.Parameter(torch.randn(1, 1, 256) * 0.02)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        # Shared classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, batch):
        img_feat = self.image_proj(batch['image'])
        txt_feat = self.text_proj(batch['text'])
        
        if self.fusion_type == 'baseline':
            # Simple concatenation fusion - vulnerable to missing data
            fused = self.fusion(torch.cat([img_feat, txt_feat], dim=-1))
            logits = self.classifier(fused)
            return logits
            
        elif self.fusion_type == 'aecf':
            # AECF attention-based fusion with proper missing data handling
            batch_size = img_feat.size(0)
            
            # Detect missing modalities based on input (before projection)
            img_present = torch.norm(batch['image'], dim=1) > 1e-6  # [batch_size]
            txt_present = torch.norm(batch['text'], dim=1) > 1e-6   # [batch_size]
            
            # Stack modalities for attention
            modalities = torch.stack([img_feat, txt_feat], dim=1)  # [batch, 2, 256]
            
            # Create attention mask (True = should be ignored in attention)
            key_padding_mask = torch.stack([~img_present, ~txt_present], dim=1)  # [batch, 2]
            
            # Create fusion query for each sample in batch
            query = self.fusion_query.expand(batch_size, -1, -1)  # [batch, 1, 256]
            
            # Apply multimodal attention with proper masking
            fused, info = self.attention_pool(
                query=query,
                key=modalities,
                value=modalities,
                key_padding_mask=key_padding_mask,  # Properly mask missing modalities
                return_info=True
            )
            
            # Extract the single fused representation
            fused = fused.squeeze(1)  # [batch, 256]
            
            # Classify
            logits = self.classifier(fused)
            
            # Process info for training
            fusion_info = {}
            if 'entropy' in info:
                fusion_info['entropy'] = info['entropy']
            if 'mask_rate' in info:
                fusion_info['masking_rate'] = info['mask_rate']
            if 'attention_weights' in info:
                fusion_info['attention_weights'] = info['attention_weights']
                
            # Compute entropy loss if we have entropy info
            if 'entropy' in info:
                entropy_loss = self.curriculum_masking.entropy_loss(info['entropy'])
                fusion_info['entropy_loss'] = entropy_loss
            
            return logits, fusion_info