# train_nn.py
from app import create_app
from app.ml_nn import train_nn, predict_single
from app.models import IntubationRecord
from app.ml import build_feature_vector

def main():
    app = create_app()
    with app.app_context():
        model, metrics = train_nn(min_samples=50, epochs=50)
        print("Validation metrics:", metrics)

        # optional: test on a random record from DB
        rec = IntubationRecord.query.filter(
            IntubationRecord.difficult_binary.isnot(None)
        ).first()
        if rec:
            fv = build_feature_vector(rec)
            proba = predict_single(model, fv)
            print(
                f"Example record #{rec.id}: predicted P(difficult)={proba:.3f}, "
                f"true difficult={rec.difficult_binary}"
            )

if __name__ == "__main__":
    main()
