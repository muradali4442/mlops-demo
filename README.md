# MLOps GitHub CI/CD Demo

A minimal, end-to-end example showing:
- Model training (scikit-learn) with a reproducible pipeline.
- FastAPI prediction service (Dockerized).
- GitHub Actions for CI (tests, lint) and CD (build/push Docker image to GHCR).
- Simple monitoring hooks and structured logs.

## Stack
- Python 3.11, scikit-learn, FastAPI, Uvicorn
- Docker, GitHub Actions (CI/CD)
- Optional: GHCR (GitHub Container Registry) for images

## Repo Layout
```
.
├── src/
│   ├── __init__.py
│   ├── train.py            # trains a model, writes to models/model.joblib
│   ├── data.py             # data loading utilities (Iris dataset)
│   ├── evaluate.py         # evaluation helpers
│   └── utils.py            # logging, paths
├── app/
│   └── main.py             # FastAPI app serving /predict
├── tests/
│   ├── test_api.py
│   └── test_training.py
├── models/                 # artifacts (created by training)
├── Dockerfile
├── Makefile
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── .github/workflows/
    ├── ci.yml
    └── cd.yml
```

## Quickstart (Local)
1) **Create & activate a venv (recommended):**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) **Install deps:**
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

3) **Train the model:**
```bash
python -m src.train
```

4) **Run tests:**
```bash
pytest -q
```

5) **Start the API:**
```bash
uvicorn app.main:app --reload --port 8000
```

6) **Call the API:**
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

## Docker (Local)
```bash
docker build -t mlops-github-demo:latest .
docker run -p 8000:8000 mlops-github-demo:latest
```

## GitHub Actions: CI & CD
- **CI** runs on PRs & pushes to `main`: installs, lints, tests; then builds & pushes an image to GHCR (`ghcr.io/<owner>/<repo>`).
- **CD** runs on release published or manual dispatch; it builds & pushes an image tagged with the release version and `latest`.

### Setup GHCR (no extra token needed)
- Ensure your repo is public or you have permission to push packages.
- Actions will use `GITHUB_TOKEN` automatically — we set `packages: write` permissions in the workflow.

### Deploying the Image
This demo doesn't bind to a specific platform. Deploy the GHCR image to any Docker-compatible target (Render, Railway, AWS ECS, Azure Web Apps for Containers, GCP Cloud Run, Kubernetes, etc.). For example, to run on a VM:
```bash
docker login ghcr.io -u <your-username> -p <a-personal-access-token-or-ghcr-token>
docker pull ghcr.io/<owner>/<repo>:latest
docker run -p 8000:8000 ghcr.io/<owner>/<repo>:latest
```

## Notes
- The app will auto-train **once** on startup if no model is found (useful for CI/tests). In real MLOps, you'd separate training from serving; here it's optional and for convenience.
- Replace Iris with your dataset by editing `src/data.py` and `src/train.py`.
