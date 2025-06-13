# -*- coding: utf-8 -*-
"""
Data Setup and Utilities

This module handles COCO dataset setup, feature loading, normalization,
and missing modality simulation for AECF testing.
"""

import os
import subprocess
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# GPU Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def install_packages():
    """Install required packages."""
    packages = ["open-clip-torch", "pycocotools", "transformers", "scikit-learn"]
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install packages if needed
if not os.environ.get('DOCKER_CONTAINER'):
    install_packages()

# Import AECF Components - handle different import paths
try:
    from datasets import CocoDataset, ClipFeatureDataset, check_existing_features, make_clip_loaders, verify_clip_features, simulate_missing_modalities
    from AECFLayer import MultimodalAttentionPool, CurriculumMasking
except ImportError:
    try:
        from aecf.datasets import CocoDataset, ClipFeatureDataset, check_existing_features, make_clip_loaders, verify_clip_features, simulate_missing_modalities
        from aecf.AECFLayer import MultimodalAttentionPool, CurriculumMasking
    except ImportError:
        print("❌ Could not import AECF components. Ensure they are in the path.")
        sys.exit(1)

print("✅ AECF components imported successfully")

class NormalizedClipFeatureDataset(ClipFeatureDataset):
    """Dataset that normalizes CLIP features to unit norm."""
    
    def __init__(self, features_file, normalize_features=True):
        super().__init__(features_file)
        
        if normalize_features:
            print("🔧 Normalizing features to unit norm...")
            
            # Normalize image features
            img_norms = torch.norm(self.images, dim=1, keepdim=True)
            self.images = self.images / (img_norms + 1e-8)
            
            # Normalize text features  
            txt_norms = torch.norm(self.texts, dim=1, keepdim=True)
            self.texts = self.texts / (txt_norms + 1e-8)
            
            print(f"   Image features normalized: norm = {torch.norm(self.images, dim=1).mean():.3f}")
            print(f"   Text features normalized: norm = {torch.norm(self.texts, dim=1).mean():.3f}")

def make_normalized_clip_loaders(train_file, val_file, test_file=None, batch_size=512, num_workers=0):
    """Create loaders with normalized CLIP features."""
    
    train_dataset = NormalizedClipFeatureDataset(train_file, normalize_features=True)
    val_dataset = NormalizedClipFeatureDataset(val_file, normalize_features=True)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    
    if test_file:
        test_dataset = NormalizedClipFeatureDataset(test_file, normalize_features=True)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False
        )
        return train_loader, val_loader, test_loader
    
    return train_loader, val_loader

def setup_data(coco_root="./coco_tests/coco_features/", batch_size=512):
    """Setup COCO dataset using existing CLIP features or extract if needed."""
    print("Setting up COCO dataset...")

    # First check for existing pre-extracted features
    train_file, val_file, test_file = check_existing_features(coco_root)
    
    if train_file and val_file:
        print("🎯 Using existing CLIP features from current directory")
        
        try:
            # Verify the features look reasonable
            train_valid = verify_clip_features(train_file)
            val_valid = verify_clip_features(val_file)
            
            if not train_valid or not val_valid:
                print("❌ Feature validation failed, falling back to standard pipeline")
                train_file, val_file, test_file = None, None, None
            else:
                test_valid = False
                if test_file:
                    test_valid = verify_clip_features(test_file)
                    if not test_valid:
                        print("⚠️  Test file validation failed, proceeding without test set")
                        test_file = None
                
                # Create data loaders with normalization
                try:
                    if test_file:
                        train_loader, val_loader, test_loader = make_normalized_clip_loaders(
                            train_file=train_file,
                            val_file=val_file,
                            test_file=test_file,
                            batch_size=batch_size
                        )
                        print("✅ Normalized data loaders created successfully (with test set)")
                        return train_loader, val_loader, test_loader
                    else:
                        train_loader, val_loader = make_normalized_clip_loaders(
                            train_file=train_file,
                            val_file=val_file,
                            batch_size=batch_size
                        )
                        print("✅ Normalized data loaders created successfully")
                        return train_loader, val_loader
                        
                except Exception as e:
                    print(f"❌ Error creating data loaders: {e}")
                    print("   Falling back to standard pipeline")
                    train_file, val_file, test_file = None, None, None
                    
        except Exception as e:
            print(f"❌ Error with existing features: {e}")
            print("   Falling back to standard pipeline")
            train_file, val_file, test_file = None, None, None
    
    # Fallback: Use standard pipeline
    if not train_file or not val_file:
        print("⚠️  Using standard pipeline (download COCO + extract features)")
        
        try:
            from datasets import setup_aecf_data_pipeline
        except ImportError:
            try:
                from aecf.datasets import setup_aecf_data_pipeline
            except ImportError:
                print("❌ Could not import setup_aecf_data_pipeline")
                sys.exit(1)
        
        return setup_aecf_data_pipeline(coco_root, batch_size=batch_size)

def simulate_missing_modalities_improved(batch, missing_prob=0.3):
    """Improved missing modality simulation."""
    batch_size = batch['image'].size(0)
    
    # Create random masks
    img_missing = torch.rand(batch_size) < missing_prob
    txt_missing = torch.rand(batch_size) < missing_prob
    
    # Ensure at least one modality remains per sample
    both_missing = img_missing & txt_missing
    if both_missing.any():
        # Randomly keep one modality for samples with both missing
        keep_img = torch.rand(both_missing.sum()) > 0.5
        img_missing[both_missing] = ~keep_img
        txt_missing[both_missing] = keep_img
    
    # Apply masks by zeroing out features
    batch_copy = batch.copy()
    
    if img_missing.any():
        batch_copy['image'][img_missing] = 0.0
    if txt_missing.any():
        batch_copy['text'][txt_missing] = 0.0
    
    return batch_copy

def simulate_missing_images(batch, missing_prob=0.3):
    """Simulate missing images only."""
    batch_size = batch['image'].size(0)
    img_missing = torch.rand(batch_size) < missing_prob
    
    batch_copy = batch.copy()
    if img_missing.any():
        batch_copy['image'][img_missing] = 0.0
    
    return batch_copy

def simulate_missing_text(batch, missing_prob=0.3):
    """Simulate missing text only."""
    batch_size = batch['text'].size(0)
    txt_missing = torch.rand(batch_size) < missing_prob
    
    batch_copy = batch.copy()
    if txt_missing.any():
        batch_copy['text'][txt_missing] = 0.0
    
    return batch_copy