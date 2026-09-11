# -*- coding: utf-8 -*-
"""
Evaluation Functions

This module contains functions for evaluating model performance,
calculating mAP scores, and handling missing modality scenarios.
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from .data_setup import simulate_missing_modalities_improved, simulate_missing_images, simulate_missing_text, device

def calculate_map_score(y_pred, y_true):
    """Calculate mAP score for multi-label classification."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Apply sigmoid to logits
    y_pred_prob = 1 / (1 + np.exp(-y_pred))
    
    try:
        # Only calculate for classes that appear in ground truth
        valid_classes = y_true.sum(axis=0) > 0
        if not valid_classes.any():
            return 0.0
        
        map_score = average_precision_score(
            y_true[:, valid_classes], 
            y_pred_prob[:, valid_classes], 
            average='macro'
        )
        return map_score
    except ValueError:
        return 0.0

def evaluate_model(model, val_loader, missing_ratio=0.0, missing_type='both'):
    """Evaluate model with mAP score.
    
    Args:
        missing_type: 'both', 'images', or 'text' - what to make missing
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Eval {missing_ratio*100:.0f}% {missing_type}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if missing_ratio > 0:
                # Apply different missing data patterns
                if missing_type == 'images':
                    batch = simulate_missing_images(batch, missing_ratio)
                elif missing_type == 'text':
                    batch = simulate_missing_text(batch, missing_ratio)
                else:  # missing_type == 'both'
                    batch = simulate_missing_modalities_improved(batch, missing_ratio)
            
            # Handle different model types
            if hasattr(model, 'fusion_type') and model.fusion_type == 'aecf':
                logits, _ = model(batch)
            elif hasattr(model, 'fusion_layer') and hasattr(model.fusion_layer, 'last_fusion_info'):
                logits, _ = model(batch)
            else:
                logits = model(batch)
            
            all_preds.append(logits.cpu())
            all_labels.append(batch['label'].cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return calculate_map_score(all_preds, all_labels)

def evaluate_robustness_comprehensive(model, val_loader, missing_ratios, model_name):
    """Evaluate robustness across missing modality ratios for different scenarios."""
    print(f"\nEvaluating {model_name} robustness...")
    results = {
        'both': {},
        'images': {},
        'text': {}
    }
    
    # Test complete data first
    map_score = evaluate_model(model, val_loader, 0.0, 'both')
    results['both'][0.0] = map_score
    results['images'][0.0] = map_score
    results['text'][0.0] = map_score
    print(f"  Complete data: mAP={map_score:.4f}")
    
    for ratio in missing_ratios:
        if ratio == 0.0:
            continue
            
        # Test missing images only
        map_score_img = evaluate_model(model, val_loader, ratio, 'images')
        results['images'][ratio] = map_score_img
        print(f"  {ratio*100:.0f}% images missing: mAP={map_score_img:.4f}")
        
        # Test missing text only
        map_score_txt = evaluate_model(model, val_loader, ratio, 'text')
        results['text'][ratio] = map_score_txt
        print(f"  {ratio*100:.0f}% text missing: mAP={map_score_txt:.4f}")
        
        # Test mixed missing (original behavior)
        map_score_both = evaluate_model(model, val_loader, ratio, 'both')
        results['both'][ratio] = map_score_both
        print(f"  {ratio*100:.0f}% both missing: mAP={map_score_both:.4f}")
    
    return results

def evaluate_robustness(model, val_loader, missing_ratios, model_name):
    """Legacy function - calls comprehensive evaluation but returns mixed results for compatibility."""
    comprehensive_results = evaluate_robustness_comprehensive(model, val_loader, missing_ratios, model_name)
    return comprehensive_results['both']  # Return mixed results for backward compatibility

def debug_predictions(model, val_loader):
    """Debug model predictions with detailed analysis."""
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Debug input features
        print(f"Input features - Image norm: {torch.norm(batch['image'], dim=1).mean():.4f}, Text norm: {torch.norm(batch['text'], dim=1).mean():.4f}")
        
        # Check if this is an AECF model
        if hasattr(model, 'fusion_type') and model.fusion_type == 'aecf':
            logits, info = model(batch)
            # Convert tensor values to scalars for printing
            entropy_val = info.get('entropy', 0.0)
            if torch.is_tensor(entropy_val):
                if entropy_val.numel() > 1:
                    entropy_val = entropy_val.mean().item()  # Take mean if it's a vector
                else:
                    entropy_val = entropy_val.item()
            
            masking_val = info.get('masking_rate', 0.0)
            if torch.is_tensor(masking_val):
                if masking_val.numel() > 1:
                    masking_val = masking_val.mean().item()  # Take mean if it's a vector
                else:
                    masking_val = masking_val.item()
            print(f"AECF - Entropy: {entropy_val:.4f}, Masking: {masking_val:.4f}")
            if 'attention_weights' in info:
                try:
                    att_weights = info['attention_weights']
                    print(f"Attention weights shape: {att_weights.shape}")
                    
                    # Handle different possible shapes
                    if att_weights.dim() >= 3:  # [batch, heads, seq_len] or similar 
                        att_weights = att_weights.mean(dim=1)  # Average over heads
                    if att_weights.dim() >= 2:  # [batch, seq_len]
                        att_weights = att_weights.mean(dim=0)  # Average over batch
                    
                    # Ensure we have at least 2 elements for image/text
                    if att_weights.numel() >= 2:
                        print(f"AECF - Attention weights: image={att_weights[0]:.3f}, text={att_weights[1]:.3f}")
                    else:
                        print(f"AECF - Attention weights: {att_weights}")
                except Exception as e:
                    print(f"AECF - Could not parse attention weights: {e}")
        else:
            logits = model(batch)
        
        probs = torch.sigmoid(logits)
        batch_map = calculate_map_score(logits, batch['label'])
        
        print(f"Logits: [{logits.min():.3f}, {logits.max():.3f}]")
        print(f"Probs: [{probs.min():.3f}, {probs.max():.3f}] (avg: {probs.mean():.3f})")
        print(f"GT avg: {batch['label'].mean():.3f}, Batch mAP: {batch_map:.4f}")