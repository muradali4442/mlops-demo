from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from src import utils
from src.train import train_model
from contextlib import asynccontextmanager

utils_logger = utils.setup_logger("api")


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


IRIS_TARGETS = ["setosa", "versicolor", "virginica"]
_model = None


def load_or_train_model():
    global _model
    if utils.MODEL_PATH.exists():
        utils_logger.info(f"Loading model from {utils.MODEL_PATH}")
        _model = joblib.load(utils.MODEL_PATH)
    else:
        utils_logger.warning("Model not found. Training a new model...")
        train_model()
        _model = joblib.load(utils.MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_or_train_model()
    yield  # (place shutdown/flush logic here if needed)


app = FastAPI(title="Iris Classifier API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_available": utils.MODEL_PATH.exists()}


@app.post("/predict")
def predict(feats: IrisFeatures):
    if _model is None:
        load_or_train_model()
    X = [[feats.sepal_length, feats.sepal_width, feats.petal_length, feats.petal_width]]
    pred = int(_model.predict(X)[0])
    return {"prediction": IRIS_TARGETS[pred], "class_index": pred}
