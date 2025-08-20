import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .data import load_dataset
from .evaluate import evaluate
from . import utils

logger = utils.setup_logger("train")


def train_model() -> dict:
    logger.info("Loading dataset...")
    X_train, X_test, y_train, y_test = load_dataset()

    logger.info("Building pipeline...")
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=300, n_jobs=None)),
        ]
    )

    logger.info("Fitting...")
    pipe.fit(X_train, y_train)

    logger.info("Evaluating...")
    y_pred = pipe.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    logger.info(f"Metrics: {metrics}")

    logger.info(f"Saving model to {utils.MODEL_PATH}")
    utils.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, utils.MODEL_PATH)

    logger.info(f"Saving metrics to {utils.METRICS_PATH}")
    utils.save_json(utils.METRICS_PATH, metrics)

    return metrics


def main():
    metrics = train_model()
    logger.info("Training complete.")
    print(metrics)


if __name__ == "__main__":
    main()
