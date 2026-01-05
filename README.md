# 🚖 mlops2025 

![Project Banner](docs/images/header.png)

---

## ✨About 
A complete end-to-end MLOps project for the New York City Taxi Trip Duration dataset. The repo includes preprocessing, feature engineering, model training and evaluation, batch inference, MLflow tracking, and optional SageMaker pipelines for training and batch inference.

---

## 📥Data 
- Dataset: NYC Taxi Trip Duration (Kaggle) — raw dataset is not committed to the repo.
- Expected locations:
  - Local runs: Put your CSVs in a `data/raw/` directory or configure `config/config.yaml` to point to your data sources.
  - SageMaker runs: Data is read from S3 (configure `s3_bucket` and `s3_prefix` in `config/config.yaml`).
- Preprocessed and intermediate artifacts are saved to `artifacts/` and predictions to `outputs/`.

**Note:** For privacy and CI speed, raw production datasets are not tracked in this repo — use small sample subsets for tests if needed.

---

## ✅Features 
- End-to-end pipeline: preprocess → feature engineering → train → inference
- Multiple models: Linear, Ridge, Lasso, RandomForest, GradientBoosting (configurable)
- MLflow experiment tracking (optional local `mlflow` service)
- Optional SageMaker pipelines for training and batch inference
- Dockerized environment for reproducible runs

---

## 📁Project structure 
```
├─ config/                  # yaml configs (config/config.yaml)
├─ scripts/                 # runnable scripts (train, preprocess, inference, pipeline wrappers)
├─ src/                     # package: ml_project (core logic, pipelines, models)
├─ tests/                   # unit & integration tests
├─ artifacts/               # saved models & intermediate artifacts
├─ outputs/                 # predictions
├─ mlruns/                  # MLflow experiment tracking (if used)
├─ run_training_pipeline.py
├─ run_batch_inference_pipeline.py
├─ Dockerfile
├─ docker-compose.yml
└─ README.md
```

---

## ⚙️Getting started 
**Prerequisites**
- Python 3.9+ (create a venv)
- `uv` for running tasks (project uses `uv.lock`)
- Docker (optional, for containerized runs)

**Install dependencies**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -U pip
pip install -r requirements.txt   # or use `uv` tooling if configured
```

**Configuration**
- Edit `config/config.yaml` to set paths, model hyperparameters, S3 bucket & role for SageMaker runs, and experiment names.

---

## ▶️Running locally 
**Full pipeline (train + inference)**
```bash
# recommended: use project's uv tasks
uv run train
# or run the pipeline script directly
python run.py --config config/config.yaml --run train
```

**Train only**
```bash
python -m scripts.train --config config/config.yaml
```

**Batch inference (local)**
```bash
uv run inference
# or
python -m scripts.batch_inference --config config/config.yaml --input data/raw/sample.csv
```

Output predictions are saved to `outputs/` with timestamped filenames.

---

## 🐳Docker 
Build and start services:
```bash
docker-compose build --no-cache
docker-compose up -d
```
Run training or inference inside the container:
```bash
docker-compose run --rm app train
# or
docker-compose run --rm app inference
```
The optional `mlflow` service runs on port `5000` and stores runs under `./mlruns`.

---

## ☁️SageMaker pipelines 
**Quick steps**
1. Configure AWS credentials (profile or env vars) with access to S3 and SageMaker.
2. Set `role_arn` in `config/config.yaml` or export `SAGEMAKER_ROLE`.
3. Run the training pipeline:
```bash
python run_training_pipeline.py --config config/config.yaml
```
4. Run batch inference (SageMaker Batch Transform):
```bash
python run_batch_inference_pipeline.py --config config/config.yaml --input s3://your-bucket/path/to/input.csv
```

**IAM Role note:** The project uses a `SageMakerExecutionRole` with permissions for SageMaker and S3. See the IAM Role section in this README for details and best practices (least privilege, scoped S3 access, logging/CloudWatch).

---

## 🔧What is implemented 
- Project structure using `src/` layout and package name `ml-project`
- Scripts for each pipeline stage in `scripts/`: 
  - `preprocess.py` — data cleaning, missing values handling, datetime features, duplicates removal
  - `feature_engineering.py` — distance & datetime features, categorical encoding, numeric scaling
  - `train.py` — trains multiple models, selects best based on metric, optional MLflow logging & registry
  - `batch_inference.py` — loads model, runs batch predictions, saves outputs to `outputs/` with date naming
- Class-based pipeline in `src/ml_project/pipelines/pipeline.py` that orchestrates end-to-end flow (preprocess → features → train → inference)
- SageMaker pipeline scripts: `run_training_pipeline.py` and `run_batch_inference_pipeline.py`
- Docker + `docker-compose.yml` with an `app` service and an optional `mlflow` service
- MLflow integration for experiment tracking and optional model registry
- Tests using `pytest` covering preprocess, features, and class-based components

---

## 📁Project structure 
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

## ▶️How to run locally (recommended) 
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

## 🐳How to run with Docker 
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

## ☁️SageMaker pipelines 
- Training pipeline: `run_training_pipeline.py` — defines preprocessing, feature engineering and training steps using SageMaker Processing and Training steps.
- Batch inference pipeline: `run_batch_inference_pipeline.py` — takes raw CSV input, runs preprocessing + feature engineering and executes batch inference.

To deploy pipelines, configure your AWS credentials and run:

```bash
python run_training_pipeline.py
python run_batch_inference_pipeline.py
```

(Ensure an appropriate SageMaker execution role and S3 bucket are configured.)

---

## 🔐IAM Role for SageMaker 
The IAM Role used in this project is **SageMakerExecutionRole**, which grants the necessary permissions to interact with various AWS services, specifically Amazon SageMaker and Amazon S3.

**Role Overview:**

- **Role Name:** `SageMakerExecutionRole`

- **Purpose:** This role allows the SageMaker service to execute machine learning workflows such as training, inference, and data processing.

**Permissions:**

- **Amazon SageMaker Full Access:** Grants SageMaker the ability to create, run, and manage ML models and pipelines.

- **Amazon S3 Full Access:** Provides full access to read and write data in Amazon S3 buckets, which is required for storing datasets, model artifacts, and inference results.

- **Additional Permissions:** The role may also include permissions for logging and monitoring, such as CloudWatch access, to track and troubleshoot pipeline executions.

**Role ARN:**

This role is associated with the ARN:

`arn:aws:iam::029937870282:role/SageMakerExecutionRole`

**Best Practices:**

- **Least Privilege Principle:** The permissions granted to this role should adhere to the least privilege principle, ensuring that only the necessary access is provided.

- **Secure Access:** The role should be restricted to only those who need it and regularly audited for security compliance.

- **Scoped Permissions:** In production environments, it is recommended to scope the permissions further if possible (e.g., limiting the access to specific S3 buckets).

**Usage:**

This IAM role is used by the SageMaker pipelines during training and inference. Specifically, it allows SageMaker to:

- Access training data from S3

- Save model artifacts and predictions to S3

- Run processing steps (e.g., preprocessing, feature engineering) on specified instances

By using this IAM role, we ensure that our SageMaker workflows have the required permissions to interact with AWS resources securely and efficiently.

---

## 🎯Models & Metrics 
- **Models trained:** Linear, Ridge, Lasso, RandomForest, GradientBoosting (configurable)
- **Selected metric:** **RMSE** (set in `config/config.yaml`) — RMSE measures average error in seconds and penalizes large errors.
- **Model selection:** The best model is selected based on the configured metric and can be optionally registered in MLflow model registry.

---

## ✅Tests 
Run tests with `pytest` or via `uv`:

```bash
uv run pytest
# or
pytest -q
```

**Run a single test file**
```bash
pytest tests/classes/test_preprocess_class.py -q
# or (pass args through uv):
uv run pytest -- tests/classes/test_preprocess_class.py -q
```

**Run a single test function (node id)**
```bash
pytest tests/classes/test_preprocess_class.py::test_handle_missing_values -q
```

**Test coverage:**
- Preprocessing & feature engineering
- Trainer & model selection
- Inference pipeline
- CLI smoke tests

**Tips:** Use markers for slow/integration tests and exclude them in CI if they require cloud access.

---

## 🔧Scripts & Files 
- `run_training_pipeline.py` — SageMaker training pipeline wrapper
- `run_batch_inference_pipeline.py` — SageMaker batch inference wrapper
- `run.py` — convenience entrypoint for local pipeline runs
- `scripts/preprocess.py`, `scripts/feature_engineering.py`, `scripts/train.py`, `scripts/batch_inference.py` — stage scripts
- `src/ml_project/` — core package (models, features, preprocessing, pipeline orchestration)
- `config/config.yaml` — central configuration

---

If you'd like, I can open a PR with this README + placeholder graphic, add a CI job that runs `pytest` on every PR, or generate a few example graphics/screenshots from the MLflow UI for the README. Which would you prefer next?