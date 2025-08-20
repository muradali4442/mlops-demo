.PHONY: install train serve test lint fmt docker-build docker-run

install:
	pip install -r requirements.txt -r requirements-dev.txt

train:
	python -m src.train

serve:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	ruff check .
	black --check .

fmt:
	black .

docker-build:
	docker build -t mlops-github-demo:latest .

docker-run:
	docker run -p 8000:8000 mlops-github-demo:latest
