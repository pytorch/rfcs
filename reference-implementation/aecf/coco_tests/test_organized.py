#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Organized Test Runner

This script runs the comprehensive AECF benchmark using the organized
modular structure from the coco_tests package.
"""

# Import the main test function from the organized package
from coco_tests import main

if __name__ == "__main__":
    print("=" * 80)
    print("🎯 RUNNING ORGANIZED AECF COMPREHENSIVE BENCHMARK")
    print("=" * 80)
    print("Using the new modular test structure from coco_tests/")
    print()
    
    # Run the comprehensive benchmark
    results = main()
    
    print("\n" + "=" * 80)
    print("✅ ORGANIZED TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("The test suite has been successfully split into organized modules:")
    print("  📁 ./")
    print("  ├── 📄 __init__.py           - Package initialization") 
    print("  ├── 📄 data_setup.py         - Data loading and preprocessing")
    print("  ├── 📄 evaluation.py         - Model evaluation and metrics")
    print("  ├── 📄 fusion_layers.py      - Different fusion implementations")
    print("  ├── 📄 architectures.py      - Network architectures")
    print("  ├── 📄 legacy_models.py      - Backward compatibility models")
    print("  ├── 📄 experiments.py        - Multi-architecture experiments")
    print("  ├── 📄 training_utils.py     - Training and analysis utilities")
    print("  └── 📄 main_test.py          - Main test runner")
    print()
    print("Original test_full.py functionality is now organized and maintainable!")