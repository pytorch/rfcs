"""
AECF (Attention Entropy Curriculum Filtering) Package

A PyTorch implementation of modality masking using entropy-driven curriculum learning
for multimodal attention mechanisms.
"""

from .AECFLayer import (
    CurriculumMasking,
    MultimodalAttentionPool,
    multimodal_attention_pool,
    create_fusion_pool,
)

__version__ = "0.1.0"
__all__ = [
    "CurriculumMasking",
    "MultimodalAttentionPool", 
    "multimodal_attention_pool",
    "create_fusion_pool",
]
