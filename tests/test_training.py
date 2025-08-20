from src.train import train_model


def test_training_creates_artifacts(tmp_path, monkeypatch):
    # redirect artifact paths to temp directory
    monkeypatch.setattr("src.utils.ARTIFACTS_DIR", tmp_path, raising=False)
    monkeypatch.setattr(
        "src.utils.MODEL_PATH", tmp_path / "model.joblib", raising=False
    )
    monkeypatch.setattr(
        "src.utils.METRICS_PATH", tmp_path / "metrics.json", raising=False
    )

    metrics = train_model()
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
    assert metrics["accuracy"] > 0.7
