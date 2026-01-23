#!/usr/bin/env python3
"""
Train NN model locally and save weights as JSON for deployment.
Run this locally with PyTorch installed, then commit the JSON to git.
"""

import json
from app import create_app, db
from app.ml_nn import train_hybrid_distilled_nn

def main():
    app = create_app()
    with app.app_context():
        print("Training NN model with current database...")
        model, metrics = train_hybrid_distilled_nn(min_samples=50)
        
        # Prepare model data for serialization
        model_data = {
            "version": 1,
            "input_dim": metrics.input_dim,
            "hidden_dim": metrics.hidden_dim,
            "weights": {
                "fc1": {
                    "weight": metrics.w1,
                    "bias": metrics.b1,
                },
                "fc2": {
                    "weight": metrics.w2,
                    "bias": metrics.b2,
                }
            },
            "metrics": {
                "n_train": metrics.n_train,
                "n_val": metrics.n_val,
                "accuracy_val": metrics.accuracy_val,
                "auc_val": metrics.auc_val,
                "brier_val": metrics.brier_val,
                "epochs": metrics.epochs,
                "alpha_distill": metrics.alpha_distill,
            }
        }
        
        # Save to JSON
        model_path = "app/data/trained_model.json"
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"\n✓ Model saved to {model_path}")
        print(f"  Input dim: {metrics.input_dim}")
        print(f"  Hidden dim: {metrics.hidden_dim}")
        print(f"  Train samples: {metrics.n_train}")
        print(f"  Validation samples: {metrics.n_val}")
        print(f"  Accuracy: {metrics.accuracy_val:.4f}")
        print(f"  AUC: {metrics.auc_val:.4f}")
        print(f"  Brier: {metrics.brier_val:.4f}")
        print("\nNow commit this file and push to Heroku:")
        print("  git add app/data/trained_model.json")
        print("  git commit -m 'Add trained NN model for Heroku deployment'")
        print("  git push origin main")
        print("  git push heroku main")

if __name__ == "__main__":
    main()
