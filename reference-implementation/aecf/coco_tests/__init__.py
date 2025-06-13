"""
COCO Tests Package

Organized test suite for AECF (Adaptive Ensemble Curriculum Fusion) evaluation
Split from the original monolithic test_full.py for better organization.

Usage:
    from coco_tests.main_test import main
    main()  # Run the comprehensive benchmark

Or import specific components:
    from coco_tests.data_setup import setup_data
    from coco_tests.evaluation import evaluate_model
    from coco_tests.experiments import MultiArchitectureExperiment
    from coco_tests.legacy_models import MultimodalClassifier
"""

__version__ = "1.0.0"

# Import main components for easy access
from .main_test import main
from .data_setup import setup_data
from .evaluation import evaluate_model, calculate_map_score
from .experiments import MultiArchitectureExperiment
from .legacy_models import MultimodalClassifier
from .training_utils import train_model

__all__ = [
    'main',
    'setup_data', 
    'evaluate_model',
    'calculate_map_score',
    'MultiArchitectureExperiment',
    'MultimodalClassifier',
    'train_model'
]