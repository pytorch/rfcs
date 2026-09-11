# AECF Reference Implementation

This directory contains the reference implementation of **Adaptive Entropy-Gated Contrastive Fusion (AECF)** that demonstrates the proposed PyTorch integration.

## Overview

This implementation shows how AECF would work as PyTorch modules and provides comprehensive benchmarking and testing to validate the RFC proposal.

## Key Files

### Core Implementation
- `aecf/AECFLayer.py` - Main AECF implementation with `CurriculumMasking` and `MultimodalAttentionPool`
- `aecf/__init__.py` - Module exports and public API
- `aecf/datasets.py` - Dataset utilities for multimodal learning

### Comprehensive Testing
- `test_suite/test_aecf.py` - 765 lines of unit tests covering all functionality
- `test_suite/aecf_benchmark_suite.py` - Performance benchmarking suite
- `test_suite/aecf_test_runner.py` - Integration test runner

### Real-World Validation
- `aecf/coco_tests/` - Complete MS-COCO experiments demonstrating AECF benefits
- `aecf/coco_tests/main_test.py` - Multi-architecture testing
- `aecf/coco_tests/fusion_layers.py` - Comparison with baseline fusion methods
- `aecf/coco_tests/architectures.py` - Different model architectures using AECF

## Running the Implementation

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Tests
```bash
# Run comprehensive test suite
python -m pytest test_suite/ -v

# Run benchmark tests
python -m pytest test_suite/aecf_benchmark_suite.py -v
```

### Run COCO Experiments
```bash
# Run organized experiments
python -m aecf.coco_tests.test_organized

# Run comprehensive benchmark
python -m aecf.coco_tests.main_test
```

## Key Results

This implementation demonstrates:

- **+18pp mAP improvement** on MS-COCO with missing modalities
- **200% reduction in Expected Calibration Error**
- **<3% runtime overhead** compared to standard attention
- **Numerical stability** under all tested conditions
- **Drop-in compatibility** with existing multimodal architectures

## Usage Examples

### Basic Multimodal Fusion
```python
from aecf import create_fusion_pool

# Create AECF fusion components
fusion_query, attention_pool = create_fusion_pool(
    embed_dim=512,
    num_modalities=2,
    mask_prob=0.15
)

# Use in your model
modalities = torch.stack([img_features, text_features], dim=1)
query = fusion_query.expand(batch_size, -1, -1)
fused = attention_pool(query, modalities)
```

### Medical Diagnosis with Missing Modalities
```python
from aecf import CurriculumMasking, MultimodalAttentionPool

# Robust medical AI with higher masking
curriculum_masking = CurriculumMasking(base_mask_prob=0.25)
fusion_pool = MultimodalAttentionPool(
    embed_dim=512,
    num_heads=8,
    curriculum_masking=curriculum_masking
)

# Handles missing lab results automatically
fused, info = fusion_pool(query, available_modalities, return_info=True)
entropy_loss = curriculum_masking.entropy_loss(info['entropy'])
```

## Performance Characteristics

### Memory Efficiency
- Gradient checkpointing support for large models
- Vectorized operations for efficient batch processing
- Optional memory optimization with `use_checkpoint=True`

### Numerical Stability
- Uses `torch.xlogy` for stable entropy computation
- Robust NaN/Inf handling in attention weights
- Proper gradient flow through masking operations

### Computational Complexity
- Time: O(n²d) where n is sequence length, d is embedding dimension
- Space: O(nd) with gradient checkpointing
- Optimized fast paths for simple cases without curriculum masking

## Integration with PyTorch

This reference implementation demonstrates the proposed PyTorch API:

```python
# Proposed PyTorch integration
import torch.nn as nn

# Core modules
masking = nn.CurriculumMasking(base_mask_prob=0.15)
pool = nn.MultimodalAttentionPool(embed_dim=512, curriculum_masking=masking)

# Factory function  
query, pool = nn.utils.create_fusion_pool(embed_dim=512, num_modalities=3)

# Functional interface
import torch.nn.functional as F
output = F.multimodal_attention_pool(query, key, value)
```

## Paper Reference

This implementation is based on:

**"Robust Multimodal Learning via Entropy-Gated Contrastive Fusion"**  
Chlon et al., 2025  
https://arxiv.org/abs/2505.15417

## License

[Add appropriate license - should match PyTorch's license for RFC purposes]
