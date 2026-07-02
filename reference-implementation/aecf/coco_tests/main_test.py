# -*- coding: utf-8 -*-
"""
Main Test Runner

This is the main entry point for running the comprehensive AECF multi-architecture
benchmark. It demonstrates AECF's effectiveness as a drop-in fusion layer.
"""

import torch
from .data_setup import setup_data
from .legacy_models import MultimodalClassifier
from .experiments import MultiArchitectureExperiment, test_robustness_on_top_architectures, print_robustness_comparison
from .training_utils import train_model, plot_and_summarize_results, create_comprehensive_report, save_comprehensive_results, print_final_summary
from .evaluation import evaluate_robustness, debug_predictions

def main():
    """Run the comprehensive AECF multi-architecture benchmark."""
    print("🚀 Starting Comprehensive AECF Multi-Architecture Benchmark")
    print("This experiment demonstrates AECF's effectiveness as a drop-in fusion layer")
    print("="*80)
    
    # Setup data - will use normalized existing .pt files if available
    data_result = setup_data(batch_size=256)  # Smaller batch for multi-architecture testing
    
    # Handle different return formats
    if len(data_result) == 3:
        train_loader, val_loader, test_loader = data_result
        print("📊 Using train, validation, and test sets")
    else:
        train_loader, val_loader = data_result
        test_loader = None
        print("📊 Using train and validation sets")
    
    # Get dimensions
    sample_batch = next(iter(train_loader))
    img_dim = sample_batch['image'].size(-1)
    txt_dim = sample_batch['text'].size(-1)
    num_classes = sample_batch['label'].size(-1)
    
    print(f"Dimensions - Image: {img_dim}D, Text: {txt_dim}D, Classes: {num_classes}")
    
    # Debug feature statistics
    print(f"\n🔍 Feature Analysis:")
    img_batch = sample_batch['image']
    txt_batch = sample_batch['text']
    print(f"Image features - mean: {img_batch.mean():.4f}, std: {img_batch.std():.4f}, norm: {torch.norm(img_batch, dim=1).mean():.4f}")
    print(f"Text features - mean: {txt_batch.mean():.4f}, std: {txt_batch.std():.4f}, norm: {torch.norm(txt_batch, dim=1).mean():.4f}")
    print(f"Cross-modal similarity: {torch.nn.functional.cosine_similarity(img_batch[:100], txt_batch[:100]).mean():.4f}")
    
    # ========================================================================
    # Part 1: Original Single-Architecture Comparison
    # ========================================================================
    
    print("\n" + "="*80)
    print("📚 PART 1: ORIGINAL SINGLE-ARCHITECTURE COMPARISON")
    print("="*80)
    
    # Create original models for backward compatibility
    baseline_model = MultimodalClassifier(img_dim, txt_dim, num_classes, fusion_type='baseline')
    aecf_model = MultimodalClassifier(img_dim, txt_dim, num_classes, fusion_type='aecf')
    
    print("\n📚 Training Enhanced Baseline...")
    train_model(baseline_model, train_loader, val_loader, epochs=12, model_name="Enhanced Baseline")
    
    print("\n📚 Training Fixed AECF...")
    train_model(aecf_model, train_loader, val_loader, epochs=12, model_name="Fixed AECF")
    
    # Debug models
    print("\n🔍 Debugging original models...")
    print("Baseline:")
    debug_predictions(baseline_model, val_loader)
    print("\nAECF:")
    debug_predictions(aecf_model, val_loader)
    
    # Evaluate robustness on original models
    missing_ratios = [0.0, 0.2, 0.4, 0.6]
    
    baseline_results = evaluate_robustness(baseline_model, val_loader, missing_ratios, "Enhanced Baseline")
    aecf_results = evaluate_robustness(aecf_model, val_loader, missing_ratios, "Fixed AECF")
    
    # Show original results
    original_avg_improvement = plot_and_summarize_results(baseline_results, aecf_results, missing_ratios)
    
    # ========================================================================
    # Part 2: Multi-Architecture Drop-in Testing
    # ========================================================================
    
    print("\n" + "="*80)
    print("🏗️  PART 2: MULTI-ARCHITECTURE DROP-IN TESTING")
    print("="*80)
    
    # Create multi-architecture experiment
    experiment = MultiArchitectureExperiment(img_dim, txt_dim, num_classes)
    
    # Run comprehensive test across all architectures and fusion methods
    print("\nTesting AECF as drop-in replacement across multiple architectures...")
    multi_arch_results = experiment.run_comprehensive_experiment(
        train_loader, val_loader, epochs_per_model=8
    )
    
    # Analyze multi-architecture results
    multi_arch_analysis = experiment.analyze_results()
    
    # ========================================================================
    # Part 3: Robustness Testing on Top Architectures
    # ========================================================================
    
    print("\n" + "="*80)
    print("🛡️  PART 3: ROBUSTNESS TESTING ON TOP ARCHITECTURES")
    print("="*80)
    
    # Test robustness on top performing architectures
    robustness_results = test_robustness_on_top_architectures(
        experiment, train_loader, val_loader, missing_ratios
    )
    
    # Display robustness comparison
    print_robustness_comparison(robustness_results, missing_ratios)
    
    # ========================================================================
    # Part 4: Comprehensive Analysis and Reporting
    # ========================================================================
    
    print("\n" + "="*80)
    print("📊 PART 4: COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # Create comprehensive report
    create_comprehensive_report(original_avg_improvement, multi_arch_analysis, robustness_results)
    
    # Save all results
    comprehensive_results = save_comprehensive_results(
        original_avg_improvement, multi_arch_analysis, robustness_results, 
        experiment, missing_ratios
    )
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    
    print_final_summary(original_avg_improvement, multi_arch_analysis)
    
    # Optional: Test set evaluation if available
    if test_loader:
        print("\n🧪 Additional test set evaluation on original models")
        from .evaluation import evaluate_model
        test_baseline = evaluate_model(baseline_model, test_loader)
        test_aecf = evaluate_model(aecf_model, test_loader)
        print(f"Test set - Baseline mAP: {test_baseline:.4f}, AECF mAP: {test_aecf:.4f}")
        improvement = (test_aecf - test_baseline) / test_baseline * 100 if test_baseline > 0 else 0
        print(f"Test improvement: {improvement:+.1f}%")
    
    return comprehensive_results

if __name__ == "__main__":
    main()