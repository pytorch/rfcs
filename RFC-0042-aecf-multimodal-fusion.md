# RFC-0042: Adaptive Entropy-Gated Contrastive Fusion (AECF) for Robust Multimodal Learning

**Authors:**
* @lchlon
* @maggiechlon  
* @marcantonio-awada

## **Summary**

We propose adding **Adaptive Entropy-Gated Contrastive Fusion (AECF)** to PyTorch as a standard multimodal fusion layer in `torch.nn`. AECF is a single lightweight attention-based layer that addresses a critical gap in multimodal deep learning: maintaining both robustness and calibration when input modalities are missing at inference time.

Key contributions:
- **Adaptive entropy control**: Dynamically adjusts entropy coefficients per instance for optimal fusion
- **Curriculum masking**: Progressive training strategy that improves robustness to missing modalities
- **Drop-in replacement**: Compatible with any attention-based multimodal architecture
- **Calibrated predictions**: Ensures well-calibrated confidence scores across all modality subsets

AECF demonstrates +18pp mAP improvement on missing-input scenarios while reducing Expected Calibration Error (ECE) by up to 200%, with only 1% runtime overhead.

> **📁 Reference Implementation**: A complete working implementation with comprehensive tests and benchmarks is included in the `reference-implementation/` directory of this RFC. See [`REFERENCE_README.md`](reference-implementation/REFERENCE_README.md) for details.

## **Motivation**

### Real-World Problem
Multimodal systems in production routinely face missing-input scenarios:
- **Robotics**: Audio sensors fail in noisy factory environments
- **Healthcare**: Clinical records miss lab test results at inference time  
- **Autonomous vehicles**: Camera sensors become occluded by weather
- **Content moderation**: Text or image data may be corrupted or incomplete

### Current Limitations
Existing fusion approaches in PyTorch fall into two categories, both with significant limitations:

1. **Concatenation-based fusion** (`torch.cat` + `nn.Linear`):
   - Simple but brittle to missing inputs
   - No principled way to handle variable modality availability
   - Poor calibration under distribution shift

2. **Attention-based fusion** (`nn.MultiheadAttention`):
   - Better than concatenation but still lacks robustness mechanisms
   - No built-in curriculum learning for missing-modality training
   - Attention weights often poorly calibrated

### Impact on PyTorch Ecosystem
This feature would provide PyTorch users with a **robust, production-ready multimodal fusion layer** that:
- Works as a drop-in replacement for existing fusion approaches
- Provides built-in robustness to missing modalities without architectural changes
- Maintains calibrated predictions across different modality subsets
- Enables curriculum learning through entropy-driven masking

The implementation would benefit researchers and practitioners working on:
- Vision-language models (CLIP, BLIP variants)
- Medical AI with multimodal inputs 
- Robotics and embodied AI
- Content understanding and moderation
- Any multimodal deep learning application

## **Proposed Implementation**

### Core Components

#### 1. CurriculumMasking Module
```python
class CurriculumMasking(nn.Module):
    """Entropy-driven curriculum masking for attention weights."""
    
    def __init__(
        self, 
        base_mask_prob: float = 0.15,
        entropy_target: float = 0.7, 
        min_active: int = 1
    ):
        """
        Args:
            base_mask_prob: Base probability for masking attention weights
            entropy_target: Target entropy as fraction of maximum entropy
            min_active: Minimum number of active attention weights
        """
        
    def forward(self, weights: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply adaptive entropy-based masking to attention weights."""
        
    def entropy_loss(self, entropy: torch.Tensor) -> torch.Tensor:
        """Compute entropy regularization loss."""
```

#### 2. MultimodalAttentionPool Module  
```python
class MultimodalAttentionPool(nn.Module):
    """Attention pooling with optional curriculum masking."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        curriculum_masking: Optional[CurriculumMasking] = None,
        **kwargs
    ):
        """
        Args:
            embed_dim: Embedding dimension 
            num_heads: Number of attention heads
            dropout: Dropout probability
            curriculum_masking: Optional curriculum masking module
        """
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: Optional[torch.Tensor] = None,
        return_info: bool = False,
        use_checkpoint: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Multimodal attention pooling with optional curriculum masking."""
```

#### 3. Factory Function
```python
def create_fusion_pool(
    embed_dim: int,
    num_modalities: int, 
    mask_prob: float = 0.15,
    **kwargs
) -> Tuple[nn.Parameter, MultimodalAttentionPool]:
    """Factory function for creating multimodal fusion components."""
```

### Integration with PyTorch

#### Location in PyTorch
- **Primary location**: `torch.nn.MultimodalAttentionPool` and `torch.nn.CurriculumMasking`
- **Functional interface**: `torch.nn.functional.multimodal_attention_pool`
- **Factory utilities**: `torch.nn.utils.create_fusion_pool`

#### Usage Examples

**Basic Vision-Language Fusion:**
```python
import torch
import torch.nn as nn

class VisionLanguageModel(nn.Module):
    def __init__(self, img_dim=2048, txt_dim=768, hidden_dim=512, num_classes=1000):
        super().__init__()
        
        # Modality projections
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)
        
        # AECF fusion layer
        self.fusion_query, self.fusion_pool = nn.utils.create_fusion_pool(
            embed_dim=hidden_dim,
            num_modalities=2,
            mask_prob=0.15
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, image_feats, text_feats):
        # Project modalities
        img_proj = self.img_proj(image_feats)
        txt_proj = self.txt_proj(text_feats)
        
        # Stack modalities: [batch, num_modalities, hidden_dim]
        modalities = torch.stack([img_proj, txt_proj], dim=1)
        
        # Expand fusion query for batch
        query = self.fusion_query.expand(modalities.size(0), -1, -1)
        
        # Apply AECF fusion
        fused, info = self.fusion_pool(query, modalities, return_info=True)
        
        # Extract entropy loss for training
        entropy_loss = self.fusion_pool.curriculum_masking.entropy_loss(info['entropy'])
        
        return self.classifier(fused.squeeze(1)), entropy_loss
```

**Medical Multimodal Diagnosis:**
```python
class MedicalDiagnosisModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Modality encoders
        self.image_encoder = nn.Linear(1024, 512) 
        self.lab_encoder = nn.Linear(50, 512)
        self.clinical_encoder = nn.Linear(200, 512)
        
        # AECF fusion with higher masking for medical robustness
        self.fusion_query, self.fusion_pool = nn.utils.create_fusion_pool(
            embed_dim=512,
            num_modalities=3,
            mask_prob=0.25,  # More aggressive masking for robustness
            num_heads=8
        )
        
        self.classifier = nn.Linear(512, 10)
        
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
        
        # AECF handles variable number of modalities
        modality_tensor = torch.stack(modalities, dim=1)
        query = self.fusion_query.expand(modality_tensor.size(0), -1, -1)
        fused = self.fusion_pool(query, modality_tensor)
        
        return self.classifier(fused.squeeze(1))
```

### Technical Details

#### Entropy-Based Adaptive Masking
The core innovation is computing adaptive masking probability based on attention entropy:

```python
def compute_adaptive_mask_prob(self, attention_weights):
    # Compute Shannon entropy
    entropy = -torch.xlogy(attention_weights, attention_weights).sum(dim=-1)
    
    # Normalize by maximum possible entropy
    max_entropy = math.log(attention_weights.size(-1))
    norm_entropy = (entropy / max_entropy).clamp(0.0, 1.0)
    
    # Higher entropy → less masking (curriculum learning)
    adaptive_prob = self.base_mask_prob * (1.0 - norm_entropy)
    return adaptive_prob
```

#### Numerical Stability
- Uses `torch.xlogy` for stable entropy computation
- Proper handling of NaN/Inf values in attention weights
- Gradient checkpointing support for memory efficiency
- Vectorized operations for performance

#### Curriculum Learning
The masking probability decreases as attention becomes more structured (lower entropy), implementing curriculum learning where:
1. **Early training**: High masking forces robustness learning
2. **Later training**: Lower masking allows fine-tuning on complete inputs

## **Metrics**

### Performance Metrics
1. **Robustness**: mAP/accuracy under missing modality scenarios (0%, 20%, 40%, 60% missing)
2. **Calibration**: Expected Calibration Error (ECE) across modality subsets
3. **Runtime**: Overhead compared to standard attention (target: <5%)
4. **Memory**: Peak memory usage with/without gradient checkpointing

### Success Criteria
- **+10pp mAP improvement** on missing-input scenarios vs. standard attention
- **ECE reduction of 50%+** compared to baseline fusion methods
- **<3% runtime overhead** in production settings
- **Drop-in compatibility** with existing multimodal architectures

### Benchmarking Plan
- **Vision-Language**: MS-COCO, Flickr30K with simulated missing modalities
- **Medical**: MIMIC-III multimodal patient data
- **Audio-Visual**: VGGSound, AudioSet with missing audio/video
- **Robotics**: Embodied AI tasks with sensor dropout

## **Drawbacks**

### Implementation Complexity
- **Moderate complexity**: More sophisticated than simple concatenation, but manageable
- **Additional hyperparameters**: `base_mask_prob`, `entropy_target`, `min_active` require tuning
- **Training overhead**: Entropy loss computation adds minor computational cost

### API Surface Expansion  
- **New modules**: Adds `CurriculumMasking` and `MultimodalAttentionPool` to `torch.nn`
- **Functional interface**: New function in `torch.nn.functional`
- **Utility functions**: Factory function in `torch.nn.utils`

### Backward Compatibility
- **No breaking changes**: All additions are new modules/functions
- **Optional dependencies**: Works with existing PyTorch installations
- **Migration path**: Clear upgrade path from existing fusion approaches

### Maintenance Burden
- **Specialized knowledge**: Requires understanding of multimodal learning and entropy-based curriculum
- **Testing complexity**: Need comprehensive tests for missing modality scenarios
- **Documentation**: Requires detailed examples and best practices

## **Alternatives**

### Alternative 1: External Package
**Approach**: Keep AECF as a separate pip-installable package
- **Pros**: Faster iteration, no PyTorch maintenance burden
- **Cons**: Fragmented ecosystem, harder discovery, potential compatibility issues

### Alternative 2: TorchVision Integration
**Approach**: Add to `torchvision.models` as multimodal model components
- **Pros**: Natural fit for vision-language models
- **Cons**: Limits usage to vision domain, less general purpose

### Alternative 3: Contrib Module
**Approach**: Add to `torch.contrib` or similar experimental namespace
- **Pros**: Lower commitment, easier to iterate on API
- **Cons**: Signals experimental status, may reduce adoption

### Alternative 4: Do Nothing
**Impact of not implementing**:
- PyTorch users continue using suboptimal fusion approaches
- Fragmented ecosystem of multimodal fusion implementations
- Missing opportunity to establish PyTorch as leader in multimodal AI
- Continued poor robustness and calibration in production multimodal systems

## **Prior Art**

### Academic Literature
1. **"Robust Multimodal Learning via Entropy-Gated Contrastive Fusion"** (Chlon et al., 2025)
   - Original AECF paper showing +18pp mAP improvement
   - Demonstrates superior calibration properties
   - Extensive evaluation on AV-MNIST and MS-COCO

2. **Multimodal Deep Learning** (Ngiam et al., 2011)
   - Early work on multimodal fusion
   - Showed benefits of robustness to missing inputs

3. **Attention mechanisms** (Bahdanau et al., 2015; Vaswani et al., 2017)
   - Foundation for attention-based fusion
   - AECF builds on these established mechanisms

### Existing Implementations

#### In Other Frameworks
- **HuggingFace Transformers**: Some multimodal models but no general fusion layer
- **TensorFlow**: `tf.keras.layers.MultiHeadAttention` but no curriculum masking
- **JAX/Flax**: Research implementations but no standardized API

#### Lessons Learned
1. **Importance of robustness**: Production systems frequently face missing inputs
2. **Calibration matters**: Overconfident predictions are dangerous in high-stakes domains
3. **Ease of use**: Complex research techniques need simple APIs for adoption
4. **Performance**: Even small runtime overheads matter at scale

### Comparison with Existing PyTorch Features

| Feature | Current PyTorch | AECF Addition |
|---------|----------------|----------------|
| Basic fusion | `torch.cat` + `nn.Linear` | ✓ Maintains compatibility |
| Attention fusion | `nn.MultiheadAttention` | ✓ Adds curriculum masking |
| Missing input handling | Manual masking | ✓ Automatic robustness |
| Calibration | No built-in support | ✓ Entropy-based calibration |
| Curriculum learning | Manual implementation | ✓ Built-in adaptive curriculum |

## **Reference Implementation**

A complete working implementation is provided in the `reference-implementation/` directory, demonstrating:

### Comprehensive Testing
- **765 lines of unit tests** covering all functionality (`test_suite/test_aecf.py`)
- **Performance benchmarking suite** with memory and speed profiling  
- **Integration tests** with real multimodal architectures
- **Numerical stability validation** under edge cases (NaN/Inf handling)

### Real-World Validation  
- **MS-COCO experiments** showing +18pp mAP improvement with missing modalities
- **Multiple architecture comparisons** (MLP, Transformer, CNN-based)
- **Medical AI validation** with missing clinical data
- **Robustness testing** across different missing modality rates (20%, 50%, 80%)

### Production-Ready Features
- **Gradient checkpointing** for memory efficiency
- **Mixed precision training** compatibility
- **CUDA optimization** with vectorized operations
- **Batch processing** optimizations

### Key Performance Results
```python
# Benchmark Results (from reference implementation)
Missing Rate | Standard Attention | AECF Improvement
0% (complete) | 100% (baseline)   | 100% (maintained)
20% missing   | 85%               | +12pp → 97%
50% missing   | 62%               | +18pp → 80%  
80% missing   | 23%               | +25pp → 48%

Runtime Overhead: <3%
Memory Overhead: <5% (without checkpointing)
```

The reference implementation can be run immediately:
```bash
cd reference-implementation/
pip install -r requirements.txt  
python -m pytest test_suite/ -v
python -m aecf.coco_tests.test_organized
```

## **How We Teach This**

### Naming and Terminology
- **"Multimodal Attention Pool"**: Clear, descriptive name following PyTorch conventions
- **"Curriculum Masking"**: Established terminology from curriculum learning literature  
- **"Adaptive Entropy-Gated"**: Descriptive of the core mechanism
- **"Fusion"**: Standard term in multimodal learning

### Documentation Structure

#### 1. Tutorial: "Multimodal Learning with AECF"
```
tutorials/
├── multimodal_fusion_basics.py        # Basic concepts and usage
├── vision_language_example.py         # Complete VL model example  
├── missing_modality_robustness.py     # Handling missing inputs
└── advanced_curriculum_learning.py    # Custom curriculum strategies
```

#### 2. API Documentation
- **Module documentation**: Complete docstrings with mathematical formulations
- **Parameter guides**: When to tune `base_mask_prob`, `entropy_target`, etc.
- **Performance tips**: Gradient checkpointing, batching best practices

#### 3. Examples Repository
```python
# examples/multimodal/
├── medical_diagnosis.py       # Healthcare multimodal example
├── robotics_sensor_fusion.py  # Robotics with sensor dropout
├── content_moderation.py      # Text + image content analysis
└── audio_visual_learning.py   # Audio-video multimodal tasks
```

### Teaching Progression

#### Beginner Level
1. **Start with motivation**: Why robustness matters in production
2. **Simple example**: Two-modality fusion with clear benefits
3. **Drop-in replacement**: Show how to upgrade existing code

#### Intermediate Level  
1. **Entropy concepts**: Explain adaptive masking mechanism
2. **Curriculum learning**: How masking probability changes during training
3. **Custom configurations**: Tuning hyperparameters for specific domains

#### Advanced Level
1. **Mathematical foundations**: Entropy-based adaptive coefficients
2. **Custom curriculum strategies**: Extending `CurriculumMasking`
3. **Performance optimization**: Memory and compute best practices

### Integration with Existing Docs
- **Add to multimodal learning section** in PyTorch tutorials
- **Cross-reference** with attention mechanism documentation
- **Include in model zoo examples** for vision-language models
- **Add performance benchmarks** to PyTorch performance documentation

## **Unresolved Questions**

### Design Questions (RFC Process)
1. **API surface**: Should factory functions be in `torch.nn.utils` or separate module?
2. **Default parameters**: What default values for `base_mask_prob` and `entropy_target` work across domains?
3. **Integration depth**: Should this integrate with existing `MultiheadAttention` or remain separate?
4. **Naming**: Is `MultimodalAttentionPool` the best name, or should it be more generic?

### Implementation Questions (Development Process)  
1. **CUDA kernels**: Would custom CUDA kernels for entropy computation provide significant speedup?
2. **Mixed precision**: How should AECF interact with automatic mixed precision training?
3. **Distributed training**: Any special considerations for distributed multimodal training?
4. **Mobile deployment**: Can the implementation be optimized for mobile/edge deployment?

### Validation Questions (Before Stabilization)
1. **Generalization**: Does AECF work well across different types of modalities (beyond vision-language)?
2. **Scale**: How does performance scale to models with 10+ modalities?
3. **Domain transfer**: Do hyperparameters transfer across different application domains?
4. **Long-term stability**: Are the curriculum learning dynamics stable over very long training runs?

### Future Scope (Out of RFC)
1. **Hierarchical multimodal fusion**: Extending to nested/hierarchical modality structures
2. **Dynamic modality weighting**: Learning importance weights for different modalities
3. **Adversarial robustness**: Extending curriculum masking to adversarial scenarios
4. **AutoML integration**: Automatic hyperparameter tuning for AECF parameters

## Resolution

*[To be filled during RFC review process]*

### Level of Support
*[To be determined]*

### Additional Context  
*[To be added based on community feedback]*

### Next Steps
*[To be defined after acceptance]*

#### Tracking Issue
*[GitHub issue URL to be added]*

#### Implementation Timeline
*[Proposed timeline for implementation phases]*
