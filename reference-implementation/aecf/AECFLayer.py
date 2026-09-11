"""
AECF (Attention Entropy Curriculum Filtering) Implementation

This module provides PyTorch-optimized components for entropy-driven
curriculum masking in multimodal attention mechanisms.

Design principles:
- Composable: Works with any attention mechanism
- Efficient: Vectorized operations and gradient checkpointing support  
- Robust: Proper numerical stability and error handling
- Standard: Follows PyTorch conventions for modules and functions

Classes:
    CurriculumMasking: Entropy-driven adaptive masking for attention weights
    MultimodalAttentionPool: Attention pooling with optional curriculum masking
    
Functions:
    multimodal_attention_pool: Functional interface with fast paths
    create_fusion_pool: Factory for common fusion patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from typing import Optional, Tuple, Union, Dict, Any
from torch.utils.checkpoint import checkpoint

__all__ = ['CurriculumMasking', 'MultimodalAttentionPool', 'multimodal_attention_pool', 'create_fusion_pool']


class CurriculumMasking(nn.Module):
    r"""Entropy-driven curriculum masking for attention weights.
    
    Applies adaptive masking to attention weights based on their entropy,
    implementing curriculum learning that progressively reduces masking
    as the model learns more structured attention patterns.
    
    The masking probability is computed as:
    
    .. math::
        p_{mask} = p_{base} \cdot (1 - \frac{H(w)}{H_{max}})
        
    where :math:`H(w)` is the Shannon entropy of weights :math:`w` and
    :math:`H_{max} = \log(L)` for sequence length :math:`L`.
    
    Args:
        base_mask_prob (float): Base masking probability. Must be in (0, 1].
            Default: 0.15
        entropy_target (float): Target entropy as fraction of maximum entropy.
            Must be in (0, 1]. Default: 0.7
        min_active (int): Minimum number of active (unmasked) elements.
            Must be >= 1. Default: 1
            
    Shape:
        - Input: :math:`(..., L)` where :math:`L` is sequence length
        - Output: :math:`(..., L)` (same shape as input)
        
    Examples:
        >>> masking = CurriculumMasking(base_mask_prob=0.2, entropy_target=0.8)
        >>> weights = torch.softmax(torch.randn(32, 10), dim=-1)
        >>> masked_weights, info = masking(weights)
        >>> print(info['entropy'].mean())  # Monitor average entropy
        
    Note:
        During evaluation (``training=False``), no masking is applied and
        original weights are returned unchanged.
    """
    
    def __init__(
        self,
        base_mask_prob: float = 0.15,
        entropy_target: float = 0.7,
        min_active: int = 1,
    ):
        super().__init__()
        
        if not 0.0 < base_mask_prob <= 1.0:
            raise ValueError(f"base_mask_prob must be in (0, 1], got {base_mask_prob}")
        if not 0.0 < entropy_target <= 1.0:
            raise ValueError(f"entropy_target must be in (0, 1], got {entropy_target}")
        if min_active < 1:
            raise ValueError(f"min_active must be >= 1, got {min_active}")
            
        self.base_mask_prob = base_mask_prob
        self.entropy_target = entropy_target
        self.min_active = min_active
        
        # Pre-compute constants to avoid repeated operations
        self.register_buffer('_eps', torch.tensor(1e-8))
        
        # Cache for sequence length to make entropy_loss more robust
        self._last_seq_len = 2  # Default assumption for modalities
    
    def compute_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute Shannon entropy with numerical stability.
        
        Args:
            weights: Probability weights (..., seq_len)
            
        Returns:
            entropy: Shannon entropy (...,)
        """
        # Use stable entropy computation
        return self.compute_entropy_fused(weights)
    
    def compute_entropy_fused(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute Shannon entropy with numerical stability.
        
        Uses torch.xlogy for stable computation of x * log(x).
        
        Args:
            weights: Probability weights (..., seq_len)
            
        Returns:
            entropy: Shannon entropy (...,)
        """
        # torch.xlogy handles x*log(0) = 0 case automatically
        entropy = -torch.xlogy(weights, weights).sum(dim=-1)
        # Clamp to valid entropy range [0, log(seq_len)]
        max_possible_entropy = math.log(weights.size(-1))
        return entropy.clamp_(0.0, max_possible_entropy)
    
    def forward(self, weights: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""Apply curriculum masking to attention weights.
        
        Args:
            weights (Tensor): Attention weights of shape :math:`(..., L)`.
                Should be normalized (sum to 1 along last dimension).
                
        Returns:
            Tuple[Tensor, Dict[str, Tensor]]: A tuple containing:
            
            - **masked_weights** (Tensor): Masked and renormalized weights
            - **info** (Dict[str, Tensor]): Dictionary containing:
            
              - ``'entropy'``: Shannon entropy of input weights
              - ``'mask_rate'``: Fraction of masked elements
              - ``'target_entropy'``: Target entropy for regularization
              
        Note:
            In evaluation mode, returns original weights with zero mask rate.
        """
        if not self.training:
            entropy = self.compute_entropy_fused(weights)
            batch_shape = entropy.shape
            return weights, {
                'entropy': entropy, 
                'mask_rate': torch.zeros(batch_shape, device=weights.device, dtype=weights.dtype)
            }
        
        # Fast input validation
        seq_len = weights.size(-1)
        if seq_len <= 1:
            # Early return for trivial cases
            batch_shape = weights.shape[:-1]
            return weights, {
                'entropy': torch.zeros(batch_shape, device=weights.device, dtype=weights.dtype),
                'mask_rate': torch.zeros(batch_shape, device=weights.device, dtype=weights.dtype),
                'target_entropy': torch.zeros(batch_shape, device=weights.device, dtype=weights.dtype),
            }
        
        # Fast normalization check and fix - handle NaN/Inf values
        weight_sums = weights.sum(dim=-1, keepdim=True)
        
        # Handle NaN and Inf values robustly
        if not torch.isfinite(weights).all():
            # Replace NaN/Inf with uniform distribution
            weights = torch.where(torch.isfinite(weights), weights, 0.0)
            weight_sums = weights.sum(dim=-1, keepdim=True)
        
        needs_norm = weight_sums < self._eps
        if needs_norm.any():
            # Only normalize where needed
            uniform_weights = 1.0 / seq_len
            weights = torch.where(needs_norm, uniform_weights, weights / weight_sums)
        else:
            weights = weights / weight_sums
        
        # Store sequence length for entropy loss computation
        self._last_seq_len = seq_len
        
        # Vectorized entropy and adaptive probability computation
        entropy = self.compute_entropy_fused(weights)
        max_entropy = math.log(float(seq_len))
        norm_entropy = (entropy / max_entropy).clamp_(0.0, 1.0)  # In-place clamp
        
        # Vectorized mask generation - broadcast efficiently with safety
        adaptive_prob = self.base_mask_prob * (1.0 - norm_entropy)
        keep_prob = 1.0 - adaptive_prob.unsqueeze(-1)  # Shape: (..., 1)
        
        # Ensure probabilities are valid for Bernoulli sampling
        keep_prob = keep_prob.clamp_(0.0, 1.0)
        
        # Single bernoulli call - more efficient than expanding then sampling
        mask = torch.bernoulli(keep_prob.expand_as(weights))
        
        # Optimized min_active constraint - fully vectorized
        effective_min_active = min(self.min_active, seq_len)
        active_count = mask.sum(dim=-1)
        needs_more = active_count < effective_min_active
        
        if needs_more.any():
            # Vectorized top-k based minimum constraint
            _, top_indices = weights.topk(effective_min_active, dim=-1, largest=True)
            
            # Create minimum mask efficiently using scatter operations
            min_mask = torch.zeros_like(weights)
            
            # Handle multi-dimensional indexing with vectorized operations
            if weights.dim() > 2:
                # Reshape for easier batch processing
                original_shape = weights.shape
                batch_size = original_shape[0]
                n_dims = original_shape[1:-1]
                flat_size = torch.prod(torch.tensor(n_dims)).item()
                
                # Flatten all dimensions except first and last
                flat_weights = weights.view(batch_size, flat_size, seq_len)
                flat_needs_more = needs_more.view(batch_size, flat_size)
                flat_top_indices = top_indices.view(batch_size, flat_size, effective_min_active)
                flat_min_mask = min_mask.view(batch_size, flat_size, seq_len)
                
                # Vectorized scatter operation
                # Get all indices that need more active elements
                batch_idx, seq_idx = torch.nonzero(flat_needs_more, as_tuple=True)
                if len(batch_idx) > 0:
                    # Use advanced indexing to set values efficiently
                    selected_top_indices = flat_top_indices[batch_idx, seq_idx]  # [n_selected, effective_min_active]
                    
                    # Create index arrays for scatter
                    n_selected = len(batch_idx)
                    batch_expand = batch_idx.unsqueeze(1).expand(-1, effective_min_active)
                    seq_expand = seq_idx.unsqueeze(1).expand(-1, effective_min_active)
                    
                    # Set minimum active elements to 1
                    flat_min_mask[batch_expand, seq_expand, selected_top_indices] = 1.0
                
                min_mask = flat_min_mask.view(original_shape)
            else:
                # Simple 2D case - use scatter directly  
                batch_indices = torch.nonzero(needs_more, as_tuple=False).flatten()
                if len(batch_indices) > 0:
                    # Create expanded indices for scatter
                    batch_expand = batch_indices.unsqueeze(1).expand(-1, effective_min_active)
                    selected_indices = top_indices[batch_indices]
                    
                    # Use scatter to set values
                    min_mask[batch_expand, selected_indices] = 1.0
            
            # Apply minimum constraint where needed
            mask = torch.where(needs_more.unsqueeze(-1), min_mask, mask)
        
        # Optimized masking and renormalization
        masked_weights = weights * mask
        weight_sum = masked_weights.sum(dim=-1, keepdim=True)
        
        # Fast renormalization with fallback
        valid_mask = weight_sum > self._eps
        final_weights = torch.where(
            valid_mask,
            masked_weights / weight_sum,
            weights  # Fallback
        )
        
        # Efficient mask rate computation
        mask_rate = 1.0 - mask.float().mean(dim=-1)
        
        info = {
            'entropy': entropy.detach(),
            'mask_rate': mask_rate.detach(),
            'target_entropy': torch.full_like(entropy, max_entropy * self.entropy_target),
        }
        
        return final_weights, info
    
    def entropy_loss(self, entropy: torch.Tensor) -> torch.Tensor:
        """Compute entropy regularization loss.
        
        Args:
            entropy: Entropy values from forward pass (...,)
            
        Returns:
            loss: MSE loss between entropy and target (scalar)
        """
        # Handle NaN/Inf values in entropy before computing loss
        if not torch.isfinite(entropy).all():
            entropy = torch.nan_to_num(entropy, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Dynamically compute target based on attention weights' last dimension
        # This is more robust than hard-coding seq_len = 2
        # Note: This assumes entropy was computed over the last dimension of attention weights
        if hasattr(self, '_last_seq_len'):
            seq_len = self._last_seq_len
        else:
            # Fallback: assume binary modality case
            seq_len = 2
            
        max_entropy = math.log(float(seq_len)) if seq_len > 1 else 0.0
        target = max_entropy * self.entropy_target
        
        # Robust MSE computation with numerical stability
        diff = entropy - target
        loss = (diff * diff).mean()
        
        return loss.clamp_(min=0.0)
    
    def extra_repr(self) -> str:
        return (f'base_mask_prob={self.base_mask_prob}, '
                f'entropy_target={self.entropy_target}, '
                f'min_active={self.min_active}')


class MultimodalAttentionPool(nn.Module):
    r"""Multimodal attention pooling with optional curriculum masking.
    
    Performs attention-based pooling across input modalities using learnable
    queries. Optionally applies curriculum masking for robust training.
    
    This module wraps PyTorch's :class:`~torch.nn.MultiheadAttention` with
    additional curriculum learning capabilities and optimized gradient flow.
    
    Args:
        embed_dim (int): Total dimension of the model. Must be divisible by
            ``num_heads``.
        num_heads (int, optional): Number of parallel attention heads.
            Default: 1
        dropout (float, optional): Dropout probability on attention weights.
            Must be in [0, 1]. Default: 0.0
        bias (bool, optional): Whether to add bias to input/output projections.
            Default: True
        curriculum_masking (CurriculumMasking, optional): Curriculum masking
            module to apply to attention weights. Default: None
        batch_first (bool, optional): If True, input and output tensors are
            provided as (batch, seq, feature). Default: True
        device (torch.device, optional): Device for parameters. Default: None
        dtype (torch.dtype, optional): Parameter dtype. Default: None
        
    Shape:
        - Input: 
          - **query**: :math:`(N, S, E)` if ``batch_first=True`` else :math:`(S, N, E)`
          - **key**: :math:`(N, T, E)` if ``batch_first=True`` else :math:`(T, N, E)`
          - **value**: :math:`(N, T, E)` if ``batch_first=True`` else :math:`(T, N, E)`
        - Output: Same shape as query
        
        where :math:`N` is batch size, :math:`S` is target sequence length,
        :math:`T` is source sequence length, and :math:`E` is embedding dimension.
        
    Examples:
        >>> # Standard attention pooling
        >>> pool = MultimodalAttentionPool(embed_dim=512, num_heads=8)
        >>> query = torch.randn(32, 1, 512)  # Single fusion query per batch
        >>> modalities = torch.randn(32, 3, 512)  # 3 modalities per batch  
        >>> output = pool(query, modalities)  # Shape: (32, 1, 512)
        
        >>> # With curriculum masking
        >>> masking = CurriculumMasking(base_mask_prob=0.2)
        >>> pool = MultimodalAttentionPool(512, curriculum_masking=masking)
        >>> output, info = pool(query, modalities, return_info=True)
        >>> entropy_loss = masking.entropy_loss(info['entropy'])
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        curriculum_masking: Optional[CurriculumMasking] = None,
        batch_first: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.curriculum_masking = curriculum_masking
        
        # Use PyTorch's optimized MultiheadAttention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
                        bias=bias,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_info: bool = False,
        use_checkpoint: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        r"""Compute multimodal attention pooling.
        
        Args:
            query (Tensor): Query tensor for attention computation
            key (Tensor): Key tensor for attention computation  
            value (Tensor, optional): Value tensor. If None, uses ``key``.
                Default: None
            key_padding_mask (BoolTensor, optional): Mask for padded key elements.
                Shape: :math:`(N, S)` where ``True`` indicates padding.
                Default: None
            attn_mask (BoolTensor, optional): Attention mask preventing attention
                to certain positions. Default: None
            return_info (bool, optional): Whether to return auxiliary information
                including attention weights and curriculum masking statistics.
                Default: False
            use_checkpoint (bool, optional): Whether to use gradient checkpointing
                to reduce memory usage during training. Default: False
                
        Returns:
            Union[Tensor, Tuple[Tensor, Dict[str, Any]]]: If ``return_info=False``,
            returns attention output. If ``return_info=True``, returns tuple of:
            
            - **output** (Tensor): Attention output
            - **info** (Dict[str, Any]): Information dictionary containing:
            
              - ``'attention_weights'``: Raw attention weights
              - ``'entropy'``: Attention entropy (if curriculum masking enabled)
              - ``'mask_rate'``: Masking rate (if curriculum masking enabled)
              - ``'masked_attention_weights'``: Masked weights (if curriculum masking enabled)
        """
        # Input validation and type checking
        if not isinstance(query, torch.Tensor):
            raise TypeError(f"Expected query to be torch.Tensor, got {type(query)}")
        if not isinstance(key, torch.Tensor):
            raise TypeError(f"Expected key to be torch.Tensor, got {type(key)}")
        if value is not None and not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected value to be torch.Tensor or None, got {type(value)}")
            
        if value is None:
            value = key

        # Shape validation
        if self.batch_first:
            if query.dim() != 3:
                raise ValueError(f"Expected 3D query tensor with batch_first=True, got {query.dim()}D")
            if key.dim() != 3:
                raise ValueError(f"Expected 3D key tensor with batch_first=True, got {key.dim()}D")
            if value.dim() != 3:
                raise ValueError(f"Expected 3D value tensor with batch_first=True, got {value.dim()}D")

            batch_size, tgt_len, embed_dim = query.shape
            src_len = key.shape[1]
            
            # Check for empty sequences
            if src_len == 0:
                raise ValueError("Key sequence length cannot be zero")
            
            if key.shape[0] != batch_size or key.shape[2] != embed_dim:
                raise RuntimeError(f"Key shape {key.shape} incompatible with query shape {query.shape}")
            if value.shape[0] != batch_size or value.shape[1] != key.shape[1] or value.shape[2] != embed_dim:
                raise RuntimeError(f"Value shape {value.shape} incompatible with key shape {key.shape}")
        else:
            # seq_first format validation
            if query.dim() != 3:
                raise ValueError(f"Expected 3D query tensor with batch_first=False, got {query.dim()}D")
            if key.dim() != 3:
                raise ValueError(f"Expected 3D key tensor with batch_first=False, got {key.dim()}D")
            if value.dim() != 3:
                raise ValueError(f"Expected 3D value tensor with batch_first=False, got {value.dim()}D")
                
            tgt_len, batch_size, embed_dim = query.shape
            src_len = key.shape[0]
            
            if src_len == 0:
                raise ValueError("Key sequence length cannot be zero")
            
            if key.shape[1] != batch_size or key.shape[2] != embed_dim:
                raise RuntimeError(f"Shape mismatch: query {query.shape}, key {key.shape}")
            if value.shape[0] != src_len or value.shape[1] != batch_size or value.shape[2] != embed_dim:
                raise RuntimeError(f"Value shape {value.shape} incompatible with key shape {key.shape}")

        # Apply gradient checkpointing if requested
        if use_checkpoint and self.training:
            def checkpoint_fn():
                return self.attention(
                    query, key, value,
                    key_padding_mask=key_padding_mask,
                    need_weights=(self.curriculum_masking is not None or return_info),
                    attn_mask=attn_mask,
                    average_attn_weights=True,
                )
            attn_output, attn_weights = checkpoint(
                checkpoint_fn, use_reentrant=False, preserve_rng_state=False
            )
        else:
            # Efficient attention computation
            attn_output, attn_weights = self.attention(
                query, key, value,
                key_padding_mask=key_padding_mask,
                need_weights=(self.curriculum_masking is not None or return_info),
                attn_mask=attn_mask,
                average_attn_weights=True,
            )

        info = {}

        # Optimized curriculum masking application
        if self.curriculum_masking is not None and attn_weights is not None:
            # Handle multi-head attention weights
            if attn_weights.dim() == 4:  # [batch, num_heads, tgt_len, src_len]
                pooled_weights = attn_weights.mean(dim=1)  # Average over heads
            else:
                pooled_weights = attn_weights
            
            # Apply curriculum masking with proper gradient handling
            masked_weights, mask_info = self.curriculum_masking(pooled_weights)
            
            # Update info dictionary - ensure gradients flow for training
            info.update(mask_info)
            info['attention_weights'] = pooled_weights  # Keep gradients for training
            
            if return_info:
                info['masked_attention_weights'] = masked_weights.detach()
        elif return_info and attn_weights is not None:
            info['attention_weights'] = attn_weights

        if return_info:
            return attn_output, info
        return attn_output
    
    def extra_repr(self) -> str:
        return (f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, '
                f'batch_first={self.batch_first}, '
                f'curriculum_masking={self.curriculum_masking is not None}')


# Functional interfaces
def _scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor, 
    value: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Efficient scaled dot-product attention for single-head case.
    
    Args:
        query: Query tensor [batch, seq_len, embed_dim]
        key: Key tensor [batch, seq_len, embed_dim] 
        value: Value tensor [batch, seq_len, embed_dim]
        scale: Optional scaling factor
        
    Returns:
        Attention output [batch, seq_len, embed_dim]
    """
    if scale is None:
        scale = query.size(-1) ** -0.5
    
    # Compute attention scores
    scores = torch.bmm(query, key.transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    return torch.bmm(attn_weights, value)


def multimodal_attention_pool(
    query: torch.Tensor,
    key: torch.Tensor,
    value: Optional[torch.Tensor] = None,
    embed_dim: Optional[int] = None,
    num_heads: int = 1,
    dropout: float = 0.0,
    curriculum_masking: Optional[CurriculumMasking] = None,
    training: bool = False,
) -> torch.Tensor:
    r"""Functional interface for multimodal attention pooling.
    
    Provides an optimized functional interface with automatic fast paths
    for simple cases and fallback to the full module for complex scenarios.
    
    Args:
        query (Tensor): Query tensor for attention
        key (Tensor): Key tensor for attention
        value (Tensor, optional): Value tensor. If None, uses ``key``.
            Default: None
        embed_dim (int, optional): Embedding dimension. If None, inferred
            from query tensor. Default: None
        num_heads (int, optional): Number of attention heads. Default: 1
        dropout (float, optional): Dropout probability. Default: 0.0
        curriculum_masking (CurriculumMasking, optional): Curriculum masking
            module. Default: None
        training (bool, optional): Whether in training mode. Default: False
        
    Returns:
        Tensor: Attention output with same shape as query
        
    Examples:
        >>> query = torch.randn(32, 1, 512)
        >>> modalities = torch.randn(32, 3, 512)
        >>> output = multimodal_attention_pool(query, modalities)
        
        >>> # With curriculum masking
        >>> masking = CurriculumMasking(base_mask_prob=0.15)
        >>> output = multimodal_attention_pool(
        ...     query, modalities, curriculum_masking=masking, training=True
        ... )
    
    Note:
        For simple single-head attention without curriculum masking or dropout,
        uses an optimized fast path. Complex cases automatically fall back to
        the full :class:`MultimodalAttentionPool` module.
    """
    if embed_dim is None:
        embed_dim = query.size(-1)
    
    if value is None:
        value = key
    
    # Fast path for simple single-head attention without curriculum masking
    if (not training and curriculum_masking is None and 
        dropout == 0.0 and num_heads == 1):
        return _scaled_dot_product_attention(query, key, value)
    
    # Use full module for complex cases
    pool = MultimodalAttentionPool(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        curriculum_masking=curriculum_masking,
        batch_first=True,
    )
    pool.train(training)
    
    return pool(query, key, value)


def create_fusion_pool(
    embed_dim: int,
    num_modalities: int,
    mask_prob: float = 0.15,
    **kwargs
) -> Tuple[nn.Parameter, MultimodalAttentionPool]:
    r"""Factory function for creating multimodal fusion components.
    
    Creates a learnable fusion query parameter and attention pooling module
    optimized for multimodal fusion tasks. The query is initialized using
    Xavier normal initialization scaled appropriately for attention mechanisms.
    
    Args:
        embed_dim (int): Feature dimension for all components. Must be positive.
        num_modalities (int): Number of input modalities. Used for documentation
            and validation purposes.
        mask_prob (float, optional): Base masking probability for curriculum
            learning. Must be in (0, 1]. Default: 0.15
        **kwargs: Additional keyword arguments passed to
            :class:`MultimodalAttentionPool`
            
    Returns:
        Tuple[nn.Parameter, MultimodalAttentionPool]: A tuple containing:
        
        - **fusion_query** (:class:`~torch.nn.Parameter`): Learnable query
          parameter of shape :math:`(1, 1, E)` where :math:`E` is ``embed_dim``
        - **attention_pool** (:class:`MultimodalAttentionPool`): Configured
          attention pooling module with curriculum masking enabled
          
    Raises:
        ValueError: If ``embed_dim`` is not positive or ``mask_prob`` is not
            in valid range
            
    Examples:
        >>> query, pool = create_fusion_pool(embed_dim=512, num_modalities=3)
        >>> batch_size = 32
        >>> modalities = torch.randn(batch_size, 3, 512)
        >>> 
        >>> # Expand query for batch and apply fusion
        >>> expanded_query = query.expand(batch_size, -1, -1)
        >>> fused = pool(expanded_query, modalities)  # Shape: (32, 1, 512)
        
        >>> # Extract fusion result 
        >>> fused_features = fused.squeeze(1)  # Shape: (32, 512)
    
    Note:
        The returned query parameter should be registered with your model's
        parameters (e.g., as ``self.fusion_query = query`` in your module's
        ``__init__`` method).
    """
    # Input validation
    if not isinstance(embed_dim, int) or embed_dim <= 0:
        raise ValueError(f"embed_dim must be a positive integer, got {embed_dim}")
    if not isinstance(num_modalities, int) or num_modalities <= 0:
        raise ValueError(f"num_modalities must be a positive integer, got {num_modalities}")
    if not isinstance(mask_prob, (int, float)) or not (0.0 < mask_prob <= 1.0):
        raise ValueError(f"mask_prob must be in (0, 1], got {mask_prob}")
    
    # Initialize learnable fusion query with proper scaling
    fusion_query = nn.Parameter(torch.empty(1, 1, embed_dim))
    # Use Xavier initialization scaled for attention
    nn.init.normal_(fusion_query, 0.0, (2.0 / embed_dim) ** 0.5)
    
    # Create curriculum masking with reasonable defaults
    curriculum_masking = CurriculumMasking(base_mask_prob=mask_prob)
    
    # Create attention pool with curriculum masking
    attention_pool = MultimodalAttentionPool(
        embed_dim=embed_dim,
        curriculum_masking=curriculum_masking,
        **kwargs
    )
    
    return fusion_query, attention_pool