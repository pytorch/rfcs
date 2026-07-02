# -*- coding: utf-8 -*-
"""
Multi-Architecture Experiment Framework

This module contains the framework for testing AECF across multiple architectures
and fusion methods, demonstrating its effectiveness as a drop-in replacement.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from .architectures import (
    SimpleMLPArchitecture, DeepMLPArchitecture, CNNTextArchitecture,
    MultiScaleArchitecture, ResNetLikeArchitecture
)
from .fusion_layers import AECFFusion
from .evaluation import evaluate_model
from .data_setup import device

class MultiArchitectureExperiment:
    """Framework to test AECF across multiple architectures."""
    
    def __init__(self, image_dim: int, text_dim: int, num_classes: int):
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.num_classes = num_classes
        
        # Define architectures to test
        self.architectures = {
            'SimpleMLP': SimpleMLPArchitecture,
            'DeepMLP': DeepMLPArchitecture,
            'CNNText': CNNTextArchitecture,
            'MultiScale': MultiScaleArchitecture,
            'ResNetLike': ResNetLikeArchitecture,
        }
        
        # Define fusion methods to compare
        self.fusion_methods = ['concat', 'aecf', 'attention', 'transformer']
        
        self.results = defaultdict(dict)
    
    def create_model(self, arch_name: str, fusion_method: str):
        """Create a model with specified architecture and fusion method."""
        arch_class = self.architectures[arch_name]
        return arch_class(
            self.image_dim, 
            self.text_dim, 
            self.num_classes, 
            fusion_method
        )
    
    def train_and_evaluate(self, model, train_loader, val_loader, 
                          epochs: int = 8, model_name: str = "Model"):
        """Train and evaluate a single model."""
        model = model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()
        
        best_map = 0.0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_losses = []
            
            for batch in tqdm(train_loader, desc=f"  Epoch {epoch+1}", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                # Handle AECF models with entropy loss
                if isinstance(model.fusion_layer, AECFFusion):
                    logits, fusion_info = model(batch)
                    loss = criterion(logits, batch['label'])
                    
                    # Add entropy regularization for AECF
                    if 'entropy' in fusion_info:
                        entropy_loss = model.fusion_layer.curriculum_masking.entropy_loss(
                            fusion_info['entropy']
                        )
                        loss += 0.01 * entropy_loss
                else:
                    logits = model(batch)
                    loss = criterion(logits, batch['label'])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation
            val_map = evaluate_model(model, val_loader)
            if val_map > best_map:
                best_map = val_map
        
        return best_map
    
    def run_comprehensive_experiment(self, train_loader, val_loader, 
                                   epochs_per_model: int = 8):
        """Run experiment across all architectures and fusion methods."""
        print("🚀 Starting Multi-Architecture AECF Drop-in Experiment")
        print(f"Testing {len(self.architectures)} architectures × {len(self.fusion_methods)} fusion methods")
        print("="*80)
        
        for arch_name in self.architectures:
            print(f"\n🏗️  Testing Architecture: {arch_name}")
            print("-" * 50)
            
            for fusion_method in self.fusion_methods:
                print(f"  🔧 Fusion Method: {fusion_method}")
                
                try:
                    # Create model
                    model = self.create_model(arch_name, fusion_method)
                    model_name = f"{arch_name}_{fusion_method}"
                    
                    # Train and evaluate
                    map_score = self.train_and_evaluate(
                        model, train_loader, val_loader, 
                        epochs_per_model, model_name
                    )
                    
                    self.results[arch_name][fusion_method] = map_score
                    print(f"    ✅ Final mAP: {map_score:.4f}")
                    
                except Exception as e:
                    print(f"    ❌ Failed: {e}")
                    self.results[arch_name][fusion_method] = 0.0
        
        return self.results
    
    def analyze_results(self):
        """Analyze and display results."""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE RESULTS ANALYSIS")
        print("="*80)
        
        # Create results table
        print(f"\n{'Architecture':<15} {'Concat':<8} {'AECF':<8} {'Attention':<10} {'Bilinear':<9} {'Transformer':<12} {'AECF vs Concat':<15}")
        print("-" * 95)
        
        aecf_wins = 0
        total_comparisons = 0
        improvements = []
        
        for arch_name in self.architectures:
            results = self.results[arch_name]
            
            # Get scores
            concat_score = results.get('concat', 0.0)
            aecf_score = results.get('aecf', 0.0)
            attention_score = results.get('attention', 0.0)
            bilinear_score = results.get('bilinear', 0.0)
            transformer_score = results.get('transformer', 0.0)
            
            # Calculate improvement
            improvement = ((aecf_score - concat_score) / concat_score * 100) if concat_score > 0 else 0
            improvements.append(improvement)
            
            # Check if AECF wins
            if aecf_score > concat_score:
                aecf_wins += 1
            total_comparisons += 1
            
            print(f"{arch_name:<15} {concat_score:<8.4f} {aecf_score:<8.4f} {attention_score:<10.4f} "
                  f"{bilinear_score:<9.4f} {transformer_score:<12.4f} {improvement:>+10.1f}%")
        
        # Summary statistics
        avg_improvement = np.mean(improvements) if improvements else 0
        win_rate = (aecf_wins / total_comparisons * 100) if total_comparisons > 0 else 0
        
        print("\n" + "="*80)
        print("📈 SUMMARY STATISTICS")
        print("="*80)
        print(f"🎯 AECF Win Rate: {aecf_wins}/{total_comparisons} ({win_rate:.1f}%)")
        print(f"📊 Average Improvement: {avg_improvement:+.1f}%")
        print(f"🏆 Best Individual Improvement: {max(improvements):+.1f}%")
        print(f"📉 Worst Individual Result: {min(improvements):+.1f}%")
        
        return {
            'results_table': dict(self.results),
            'aecf_win_rate': win_rate,
            'average_improvement': avg_improvement,
            'improvements': improvements
        }

def test_robustness_on_top_architectures(experiment, train_loader, val_loader, missing_ratios):
    """Test missing modality robustness on top performing architectures."""
    print("\n🔍 Testing robustness on top AECF architectures...")
    
    # Find top 3 architectures by AECF performance
    aecf_scores = {arch: results.get('aecf', 0.0) 
                   for arch, results in experiment.results.items()}
    top_archs = sorted(aecf_scores.keys(), 
                      key=lambda x: aecf_scores[x], reverse=True)[:3]
    
    robustness_results = {}
    
    for arch_name in top_archs:
        print(f"\n🧪 Testing {arch_name} robustness...")
        
        # Test both baseline and AECF versions
        for fusion_method in ['concat', 'aecf']:
            print(f"  Training {fusion_method} fusion...")
            
            model = experiment.create_model(arch_name, fusion_method)
            
            # Quick training (fewer epochs for robustness testing)
            experiment.train_and_evaluate(
                model, train_loader, val_loader, 
                epochs=6, model_name=f"{arch_name}_{fusion_method}"
            )
            
            # Test robustness
            arch_results = {}
            for ratio in missing_ratios:
                map_score = evaluate_model(model, val_loader, ratio)
                arch_results[ratio] = map_score
                print(f"    {ratio*100:.0f}% missing: mAP={map_score:.4f}")
            
            robustness_results[f"{arch_name}_{fusion_method}"] = arch_results
    
    return robustness_results

def print_robustness_comparison(robustness_results, missing_ratios):
    """Print robustness comparison table."""
    print("\n" + "="*60)
    print("🛡️  ROBUSTNESS COMPARISON")
    print("="*60)
    
    # Group by architecture
    arch_groups = {}
    for model_name, results in robustness_results.items():
        arch = model_name.rsplit('_', 1)[0]
        fusion = model_name.rsplit('_', 1)[1]
        
        if arch not in arch_groups:
            arch_groups[arch] = {}
        arch_groups[arch][fusion] = results
    
    for arch_name, fusion_results in arch_groups.items():
        print(f"\n🏗️  {arch_name}")
        print(f"{'Missing %':<10} {'Baseline':<10} {'AECF':<10} {'Improvement':<12}")
        print("-" * 45)
        
        for ratio in missing_ratios:
            baseline = fusion_results.get('concat', {}).get(ratio, 0.0)
            aecf = fusion_results.get('aecf', {}).get(ratio, 0.0)
            improvement = ((aecf - baseline) / baseline * 100) if baseline > 0 else 0
            
            print(f"{ratio*100:>6.0f}%{'':4} {baseline:<10.4f} {aecf:<10.4f} {improvement:>+8.1f}%")