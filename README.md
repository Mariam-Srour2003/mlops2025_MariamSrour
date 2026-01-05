# mlops2025_firstname1_firstname2

## Overview ✅
This repository implements an end-to-end ML project for the New York City Taxi Trip Duration dataset (Kaggle). The implementation follows the MLOps course requirements: clean src layout, packaging with `uv`, Docker + `docker-compose`, MLflow integration, and SageMaker training & batch inference pipelines.

---

## What is implemented 🔧
- Project structure using `src/` layout and package name `ml-project` ✅
- Scripts for each pipeline stage in `scripts/`: 
  - `preprocess.py` — data cleaning, missing values handling, datetime features, duplicates removal ✅
  - `feature_engineering.py` — distance & datetime features, categorical encoding, numeric scaling ✅
  - `train.py` — trains multiple models, selects best based on metric, optional MLflow logging & registry ✅
  - `batch_inference.py` — loads model, runs batch predictions, saves outputs to `outputs/` with date naming ✅
- Class-based pipeline in `src/ml_project/pipelines/pipeline.py` that orchestrates end-to-end flow (preprocess → features → train → inference) ✅
- SageMaker pipeline scripts: `run_training_pipeline.py` and `run_batch_inference_pipeline.py` ✅
- Docker + `docker-compose.yml` with an `app` service and an optional `mlflow` service ✅
- MLflow integration for experiment tracking and optional model registry ✅
- Tests using `pytest` covering preprocess, features, and class-based components ✅
- `uv.lock` present for deterministic dependency management ✅

---

## Project structure 📁
```
src/
  ml_project/
    data/
    preprocess/
    features/
    train/
    inference/
    pipelines/
tests/
scripts/
Dockerfile
docker-compose.yml
pyproject.toml
uv.lock
README.md
```

---

## How to run locally (recommended) ▶️
1. Create & activate a virtualenv and install dependencies with `uv` (see project `uv.lock`).

2. Run the full pipeline (train + inference):

```bash
uv run train
# or
python -m scripts.taxi_pipeline run --config config/config.yaml
```

3. Run inference only:

```bash
uv run inference
# or
python -m ml_project.cli inference --config config/config.yaml
```

Notes:
- `uv run train` calls the CLI entrypoint and runs the pipeline (preprocessing, feature engineering, training, saving artifacts, and running batch inference on test data).
- The trained model artifact will be saved to `artifacts/best_model.pkl` and predictions will be saved to `outputs/` with a timestamped filename.

---

## How to run with Docker 🐳
Build and start services:

```bash
docker-compose build --no-cache
docker-compose up -d
```

Run training in the `app` container:

```bash
docker-compose run --rm app train
```

Run inference in the `app` container:

```bash
docker-compose run --rm app inference
```

The `mlflow` service (optional) runs on port `5000` and stores experiments under `./mlruns`.

---

## SageMaker pipelines ☁️
- Training pipeline: `run_training_pipeline.py` — defines preprocessing, feature engineering and training steps using SageMaker Processing and Training steps.
- Batch inference pipeline: `run_batch_inference_pipeline.py` — takes raw CSV input, runs preprocessing + feature engineering and executes batch inference.

To deploy pipelines, configure your AWS credentials and run:

```bash
python run_training_pipeline.py
python run_batch_inference_pipeline.py
```

(Ensure an appropriate SageMaker execution role and S3 bucket are configured.)

---

## Models & Metrics 🎯
- Models trained: Linear, Ridge, Lasso, RandomForest, GradientBoosting (configurable) ✅
- Selected metric: **RMSE** (set in `config/config.yaml`) — RMSE is suitable for measuring average prediction error in units of the target (seconds) and penalizes larger errors.
- The best model is selected based on the configured metric and optionally registered with MLflow model registry.

---

## Tests ✅
Run tests with `pytest` or via `uv`:

```bash
uv run pytest
# or
pytest -q
```

Unit tests cover preprocess, feature engineering, the trainer, inference pipeline and CLI smoke tests.

---

## What to improve / future work 💡
- Add a minimal GitHub Actions CI workflow to run a fast test matrix & lint on PRs (optional).
- Add more robust integration tests for the SageMaker pipelines (requires AWS credentials).
- Add end-to-end checks for `uv run train` and `uv run inference` across Docker and local setups.

---

## Team contributions
- List contributions in the README (commits & PRs) or in the repository under `CONTRIBUTING.md` — update manually with names & roles.

---

If you want, I can open a PR that implements the CLI, updates the packaging entry-points (so `uv run train` / `uv run inference` work), and adds the README (done). I can also add a small CI job next.