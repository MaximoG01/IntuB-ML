import sys
sys.path.insert(0, '.')

from app import create_app
from app.ml_nn import evaluate_nn, TORCH_AVAILABLE

app = create_app()

print(f"TORCH_AVAILABLE: {TORCH_AVAILABLE}")

with app.app_context():
    try:
        print("\nTesting evaluate_nn()...")
        if TORCH_AVAILABLE:
            result = evaluate_nn(min_samples=20)
            print(f"✓ NN evaluation successful!")
            print(f"  Accuracy: {result.get('accuracy_val', 'N/A')}")
            print(f"  AUC: {result.get('auc_val', 'N/A')}")
            print(f"  Device: {result.get('torch_device', 'N/A')}")
            print(f"  Has w1: {'w1' in result}")
            print(f"  Has b1: {'b1' in result}")
        else:
            print("PyTorch not available, skipping NN test")
    except Exception as e:
        import traceback
        print(f"✗ Error: {e}")
        traceback.print_exc()
