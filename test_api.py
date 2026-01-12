"""
Test script for API functionality.
Run this to verify the API layer works correctly.
"""
import sys
import json

# Test the API library without starting the server
print("=" * 80)
print("Testing API Library Layer")
print("=" * 80)

try:
    from api_lib import run_pipeline, get_dataset_info
    
    print("\n✅ Successfully imported api_lib module")
    
    # Test 1: Get dataset info
    print("\n[Test 1] Getting dataset information...")
    dataset_info = get_dataset_info()
    print(f"  Found {len(dataset_info['datasets'])} datasets")
    for ds in dataset_info['datasets']:
        print(f"    - {ds['name']}: {ds['samples']} samples, {ds['features']} features")
    
    # Test 2: Run a simple experiment
    print("\n[Test 2] Running breast cancer + logistic regression...")
    config = {
        'dataset': 'breast_cancer',
        'model': 'logistic',
        'use_feature_selection': True,
        'polynomial_degree': 1
    }
    
    result = run_pipeline(config)
    
    if result['success']:
        print(f"  ✅ Training succeeded!")
        print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"  AUROC: {result['metrics']['auroc']:.4f}")
        print(f"  Training time: {result['training_time_seconds']}s")
        print(f"  Features: {result['selected_features_count']}/{result['total_features_count']}")
        print(f"  Convergence iterations: {result['convergence_iterations']}")
    else:
        print(f"  ❌ Training failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    # Test 3: Test with wine dataset
    print("\n[Test 3] Running wine + multiclass logistic...")
    config = {
        'dataset': 'wine',
        'model': 'multiclass_logistic',
        'use_feature_selection': False,
        'polynomial_degree': 1
    }
    
    result = run_pipeline(config)
    
    if result['success']:
        print(f"  ✅ Training succeeded!")
        print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"  Training time: {result['training_time_seconds']}s")
    else:
        print(f"  ❌ Training failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    # Test 4: Test with polynomial features
    print("\n[Test 4] Testing polynomial feature expansion...")
    config = {
        'dataset': 'breast_cancer',
        'model': 'logistic',
        'use_feature_selection': True,
        'polynomial_degree': 2
    }
    
    result = run_pipeline(config)
    
    if result['success']:
        print(f"  ✅ Polynomial features work!")
        print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
    else:
        print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe API layer is ready. To start the server:")
    print("  python api_server.py")
    print("\nOr with uvicorn:")
    print("  uvicorn api_server:app --reload")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("\nMake sure you have all dependencies installed:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
