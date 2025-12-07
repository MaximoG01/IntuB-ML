# train_logistic_offline.py
from app import create_app
from app.ml import evaluate_logistic

def main():
    app = create_app()
    with app.app_context():
        metrics = evaluate_logistic(min_samples=50)
        print("Logistic regression metrics (offline):")
        for k, v in metrics.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
