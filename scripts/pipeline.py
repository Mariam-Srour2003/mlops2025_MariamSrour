
import sys
from pathlib import Path

# Ensure project root is on sys.path so running the script directly (e.g. `uv run scripts/pipeline.py`)
# works even when the package is not installed into the environment.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from omegaconf import OmegaConf

from scripts.preprocess import preprocess
from scripts.feature_engineering import feature_engineering
from scripts.train import train_models, save_model, load_model
from scripts.batch_inference import batch_inference

from sklearn.model_selection import train_test_split
import os


def run_pipeline():

    cfg = OmegaConf.create({
        "train": {
            "metric": "mae",
            "test_size": 0.2,
            "seed": 42
        },
        "paths": {
            "train_csv": "src/ml_project/data/train.csv",
            "test_csv": "src/ml_project/data/test.csv",
            "artifact_dir": "artifacts/",
            "output_dir": "outputs/"
        }
    })

    print("\n=== Loading training data ===")
    df = pd.read_csv(cfg.paths.train_csv)

    train_df, valid_df = train_test_split(
        df,
        test_size=cfg.train.test_size,
        random_state=cfg.train.seed
    )

    # PREPROCESS
    print("\n=== Preprocessing ===")
    train_df = preprocess(train_df)
    valid_df = preprocess(valid_df, is_train=False)

    # FEATURE ENGINEERING
    print("\n=== Feature engineering ===")
    X_train, y_train, _ = feature_engineering(train_df, fit=True)
    X_valid, y_valid, _ = feature_engineering(valid_df, fit=False)

    # TRAIN MODELS
    print("\n=== Training models ===")
    model, best_model_name = train_models(
        X_train,
        y_train,
        X_valid,
        y_valid,
        metric=cfg.train.metric
    )

    # SAVE MODEL ARTIFACT
    print("\n=== Saving best model ===")
    os.makedirs(cfg.paths.artifact_dir, exist_ok=True)
    save_model(model, f"{cfg.paths.artifact_dir}best_model.pkl")

    # RUN BATCH INFERENCE
    print("\n=== Running batch inference ===")
    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    best_model = load_model(f"{cfg.paths.artifact_dir}best_model.pkl")
    test_df = pd.read_csv(cfg.paths.test_csv)
    test_df = preprocess(test_df, is_train=False)

    output_df = batch_inference(
        best_model,
        test_df,
        save_path=cfg.paths.output_dir
    )

    print("\n=== Pipeline Complete ===")
    print(f"Best model: {best_model_name}")
    print(f"Predictions saved to: {cfg.paths.output_dir}")


if __name__ == "__main__":
    run_pipeline()
