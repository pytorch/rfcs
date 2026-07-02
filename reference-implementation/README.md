# AECF: Adaptive Entropy-Gated Contrastive Fusion

Real-world multimodal systems routinely face missing-input scenarios, and in reality, robots lose audio in a factory or a clinical record omits lab tests at inference time. Standard fusion layers either preserve robustness or calibration but never both. We introduce Adaptive Entropy-Gated Contrastive Fusion (AECF), a single light-weight layer that (i) adapts its entropy coefficient per instance, (ii) enforces monotone calibration across all modality subsets, and (iii) drives a curriculum mask directly from training-time entropy.

📄 **Paper**: [Adaptive Entropy-Gated Contrastive Fusion](https://arxiv.org/abs/2505.15417)

## 🔥 Key Features

- **Adaptive Entropy Control**: Dynamically adjusts entropy coefficients per instance for optimal fusion
- **Robust Missing Modality Handling**: Maintains performance when modalities are missing at inference
- **Curriculum Learning**: Progressive masking based on attention entropy for improved training
- **Drop-in Replacement**: Compatible with any attention-based multimodal architecture
- **Calibrated Predictions**: Ensures well-calibrated confidence scores across modality subsets
- **PyTorch Optimized**: Efficient implementation with gradient checkpointing and numerical stability

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/aecf.git
cd aecf
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from aecf import CurriculumMasking, MultimodalAttentionPool, create_fusion_pool

# Option 1: Simple factory function (recommended)
fusion_query, attention_pool = create_fusion_pool(
    embed_dim=512,
    num_modalities=3,
    mask_prob=0.15
)

# Forward pass
batch_size = 32
modalities = torch.randn(batch_size, 3, 512)  # [batch, modalities, features]
expanded_query = fusion_query.expand(batch_size, -1, -1)
fused_features = attention_pool(expanded_query, modalities)  # [batch, 1, 512]

# Option 2: Manual setup for custom configurations
curriculum_masking = CurriculumMasking(
    base_mask_prob=0.15,
    entropy_target=0.7,
    min_active=1
)

attention_pool = MultimodalAttentionPool(
    embed_dim=512,
    num_heads=8,
    curriculum_masking=curriculum_masking
)

# Get training info including entropy for loss computation
output, info = attention_pool(query, key, value, return_info=True)
entropy_loss = curriculum_masking.entropy_loss(info['entropy'])
```

## 🏗️ Architecture Overview

AECF consists of three main components:

### 1. CurriculumMasking
Applies entropy-driven adaptive masking to attention weights with curriculum learning:

```python
masking = CurriculumMasking(
    base_mask_prob=0.15,    # Base probability for masking attention weights
    entropy_target=0.7,     # Target entropy as fraction of maximum
    min_active=1           # Minimum number of active attention weights
)

# During training, applies progressive masking
masked_weights, info = masking(attention_weights)
entropy_loss = masking.entropy_loss(info['entropy'])
```

**Key Features:**
- Entropy-based adaptive masking probability
- Ensures minimum number of active modalities
- Curriculum learning that reduces masking as model learns
- Numerical stability with proper NaN/Inf handling

### 2. MultimodalAttentionPool
Attention-based pooling with optional curriculum masking:

```python
pool = MultimodalAttentionPool(
    embed_dim=512,
    num_heads=8,
    dropout=0.1,
    curriculum_masking=masking,  # Optional
    batch_first=True
)

# Standard usage
output = pool(query, key, value)

# With gradient checkpointing for memory efficiency
output = pool(query, key, value, use_checkpoint=True)

# Get detailed information
output, info = pool(query, key, value, return_info=True)
```

### 3. Functional Interface
For simple cases without learnable parameters:

```python
from aecf import multimodal_attention_pool

# Fast path for simple attention
output = multimodal_attention_pool(query, modalities)

# With curriculum masking
output = multimodal_attention_pool(
    query, modalities,
    curriculum_masking=masking,
    training=True
)
```

## 📊 Integration Examples

### Vision-Language Model

```python
import torch
import torch.nn as nn
from aecf import create_fusion_pool

class VisionLanguageModel(nn.Module):
    def __init__(self, img_dim=2048, txt_dim=768, hidden_dim=512, num_classes=1000):
        super().__init__()
        
        # Modality projections
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)
        
        # AECF fusion layer
        self.fusion_query, self.fusion_pool = create_fusion_pool(
            embed_dim=hidden_dim,
            num_modalities=2,
            mask_prob=0.15
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, image_feats, text_feats, return_info=False):
        # Project modalities to common space
        img_proj = self.img_proj(image_feats)  # [batch, hidden_dim]
        txt_proj = self.txt_proj(text_feats)   # [batch, hidden_dim]
        
        # Stack modalities
        modalities = torch.stack([img_proj, txt_proj], dim=1)  # [batch, 2, hidden_dim]
        
        # Expand fusion query for batch
        batch_size = modalities.size(0)
        query = self.fusion_query.expand(batch_size, -1, -1)
        
        # Apply AECF fusion
        if return_info:
            fused, info = self.fusion_pool(query, modalities, return_info=True)
            return self.classifier(fused.squeeze(1)), info
        else:
            fused = self.fusion_pool(query, modalities)
            return self.classifier(fused.squeeze(1))

# Usage
model = VisionLanguageModel()
img_feats = torch.randn(32, 2048)
txt_feats = torch.randn(32, 768)

# Training with entropy regularization
logits, info = model(img_feats, txt_feats, return_info=True)
entropy_loss = model.fusion_pool.curriculum_masking.entropy_loss(info['entropy'])
total_loss = F.cross_entropy(logits, labels) + 0.01 * entropy_loss
```

### Multi-Modal Medical Diagnosis

```python
class MedicalDiagnosisModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Modality encoders
        self.image_encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.lab_encoder = nn.Sequential(
            nn.Linear(50, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.clinical_encoder = nn.Sequential(
            nn.Linear(200, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # AECF fusion with higher masking for robustness
        self.fusion_query, self.fusion_pool = create_fusion_pool(
            embed_dim=512,
            num_modalities=3,
            mask_prob=0.25,  # Higher masking for medical robustness
            num_heads=8
        )
        
        self.classifier = nn.Linear(512, 10)  # 10 disease classes
        
    def forward(self, image=None, lab=None, clinical=None):
        modalities = []
        
        # Handle missing modalities gracefully
        if image is not None:
            modalities.append(self.image_encoder(image))
        if lab is not None:
            modalities.append(self.lab_encoder(lab))
        if clinical is not None:
            modalities.append(self.clinical_encoder(clinical))
            
        if not modalities:
            raise ValueError("At least one modality must be provided")
        
        # Stack available modalities
        modality_tensor = torch.stack(modalities, dim=1)
        batch_size = modality_tensor.size(0)
        
        query = self.fusion_query.expand(batch_size, -1, -1)
        fused = self.fusion_pool(query, modality_tensor)
        
        return self.classifier(fused.squeeze(1))
```

## 🧪 Testing and Validation

### Running Tests

```bash
# Run comprehensive test suite
python -m pytest test_suite/ -v

# Run specific component tests
python -m pytest test_suite/test_aecf.py::TestCurriculumMasking -v

# Run benchmark tests
python -m pytest test_suite/aecf_benchmark_suite.py -v
```

### Running COCO Experiments

```bash
# Download COCO features (if not present)
cd aecf/coco_tests/coco_features/
# Place your CLIP features: train_60k_clip_feats.pt, val_5k_clip_feats.pt, test_5k_clip_feats.pt

# Run comprehensive benchmark
python -m aecf.coco_tests.main_test

# Run organized experiments
python -m aecf.coco_tests.test_organized
```

### Performance Validation

```python
import torch
from aecf import CurriculumMasking

# Test entropy computation
masking = CurriculumMasking()
weights = torch.softmax(torch.randn(100, 10), dim=-1)
masked_weights, info = masking(weights)

print(f"Original entropy: {info['entropy'].mean():.3f}")
print(f"Mask rate: {info['mask_rate'].mean():.3f}")
print(f"Target entropy: {info['target_entropy'].mean():.3f}")

# Validate numerical stability
extreme_weights = torch.tensor([[1.0, 0.0, 0.0], [0.33, 0.33, 0.34]])
masked, _ = masking(extreme_weights)
assert torch.isfinite(masked).all(), "Should handle extreme distributions"
```

## 📈 Performance Characteristics

### Memory Efficiency
- **Gradient Checkpointing**: Reduces memory usage for large models
- **Vectorized Operations**: Efficient batch processing
- **Minimal Parameters**: Only learnable fusion query (optional)

### Computational Complexity
- **Time**: O(n²d) where n is sequence length, d is embedding dimension
- **Space**: O(nd) with gradient checkpointing
- **Fast Paths**: Optimized single-head attention without curriculum masking

### Numerical Stability
- **Entropy Computation**: Uses `torch.xlogy` for stable x*log(x) computation  
- **NaN/Inf Handling**: Robust handling of degenerate attention weights
- **Gradient Flow**: Proper gradient preservation through masking operations

## 🔧 Advanced Configuration

### Custom Curriculum Schedules

```python
class CustomCurriculumMasking(CurriculumMasking):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step_count = 0
        
    def forward(self, weights):
        # Reduce masking over training steps
        self.base_mask_prob = max(0.05, 0.2 * (0.99 ** self.step_count))
        self.step_count += 1
        return super().forward(weights)
```

### Multi-Scale Fusion

```python
class MultiScaleFusion(nn.Module):
    def __init__(self, dims=[256, 512, 1024]):
        super().__init__()
        self.fusion_layers = nn.ModuleList([
            create_fusion_pool(dim, num_modalities=2)[1] 
            for dim in dims
        ])
        
    def forward(self, multi_scale_features):
        fused_scales = []
        for features, fusion_layer in zip(multi_scale_features, self.fusion_layers):
            query = torch.randn(features.size(0), 1, features.size(-1), device=features.device)
            fused = fusion_layer(query, features)
            fused_scales.append(fused)
        return torch.cat(fused_scales, dim=-1)
```

## 📚 API Reference

### CurriculumMasking

```python
CurriculumMasking(
    base_mask_prob: float = 0.15,      # Base masking probability (0, 1]
    entropy_target: float = 0.7,       # Target entropy as fraction of max (0, 1]  
    min_active: int = 1                # Minimum active elements >= 1
)
```

**Methods:**
- `forward(weights)` → `(masked_weights, info_dict)`
- `entropy_loss(entropy)` → `scalar_loss`
- `compute_entropy(weights)` → `entropy_tensor`

### MultimodalAttentionPool

```python
MultimodalAttentionPool(
    embed_dim: int,                               # Embedding dimension
    num_heads: int = 1,                          # Number of attention heads
    dropout: float = 0.0,                        # Dropout probability [0, 1]
    bias: bool = True,                           # Add bias to projections
    curriculum_masking: CurriculumMasking = None, # Optional masking module
    batch_first: bool = True,                    # Batch-first tensor format
    device: torch.device = None,                 # Device for parameters
    dtype: torch.dtype = None                    # Parameter dtype
)
```

**Methods:**
- `forward(query, key, value=None, ...)` → `output` or `(output, info)`

### Factory Functions

```python
create_fusion_pool(
    embed_dim: int,          # Feature dimension
    num_modalities: int,     # Number of input modalities  
    mask_prob: float = 0.15, # Base masking probability
    **kwargs                 # Additional arguments to MultimodalAttentionPool
) → (fusion_query, attention_pool)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/your-username/aecf.git
cd aecf
pip install -r requirements.txt
pip install -e .  # Install in development mode

# Run tests
python -m pytest test_suite/ -v

# Run style checks
flake8 aecf/
black aecf/
```

## 📄 Citation

```bibtex
@article{aecf2024,
  title={Adaptive Entropy-Gated Contrastive Fusion for Robust Multimodal Learning},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:2505.15417},
  year={2024}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♀️ Support

- **Issues**: [GitHub Issues](https://github.com/your-username/aecf/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/aecf/discussions)
- **Email**: your.email@university.edu

---

## 🔍 Troubleshooting

### Common Issues

**Q: Getting NaN losses during training?**
A: Ensure your input features are properly normalized and not containing NaN/Inf values. AECF includes robust handling, but extreme input distributions can still cause issues.

```python
# Normalize features before fusion
features = F.normalize(features, p=2, dim=-1)
```

**Q: Memory issues with large sequences?**
A: Use gradient checkpointing and consider reducing batch size:

```python
output = pool(query, key, value, use_checkpoint=True)
```

**Q: Poor performance with missing modalities?**
A: Increase the `mask_prob` parameter to train with more aggressive masking:

```python
masking = CurriculumMasking(base_mask_prob=0.3)  # Higher masking
```

**Q: Want to disable curriculum learning?**
A: Set `curriculum_masking=None` or use the functional interface:

```python
pool = MultimodalAttentionPool(embed_dim=512, curriculum_masking=None)
```
