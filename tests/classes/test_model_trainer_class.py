import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from src.ml_project.train.trainer import ModelTrainer

from unittest.mock import patch

# Fixtures
@pytest.fixture
def small_regression_data():
    X, y = make_regression(n_samples=20, n_features=5, noise=0.1, random_state=42)
    # Split into train/valid
    X_train, X_valid = X[:15], X[15:]
    y_train, y_valid = y[:15], y[15:]
    return X_train, y_train, X_valid, y_valid


@pytest.fixture
def trainer():
    return ModelTrainer(metric="mae", model_config={
        "linear": {},
        "ridge": {"alpha": 1.0},
        "lasso": {"alpha": 0.1},
        "rf": {"n_estimators": 5, "max_depth": 3, "n_jobs": 1},
        "gb": {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1},
    })


# Tests
def test_build_models(trainer):
    models = trainer._build_models()
    assert isinstance(models, dict)
    assert all(hasattr(m, "fit") for m in models.values())
    assert set(models.keys()) == {"linear", "ridge", "lasso", "rf", "gb"}


def test_evaluate(small_regression_data, trainer):
    X_train, y_train, X_valid, y_valid = small_regression_data
    model = LinearRegression()
    model.fit(X_train, y_train)

    metrics = trainer.evaluate(model, X_valid, y_valid)
    assert set(metrics.keys()) == {"mae", "rmse", "r2"}
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert -1 <= metrics["r2"] <= 1


@patch("mlflow.start_run")  # mock MLflow so it doesn't actually log
@patch("mlflow.log_params")
@patch("mlflow.log_metric")
@patch("mlflow.set_experiment")
def test_train(mock_set_exp, mock_log_metric, mock_log_params, mock_start_run, small_regression_data, trainer):
    X_train, y_train, X_valid, y_valid = small_regression_data

    best_model, best_name = trainer.train(X_train, y_train, X_valid, y_valid)

    assert best_model is not None
    assert isinstance(best_name, str)
    # Best score should be numeric
    assert isinstance(trainer.best_score, float)


def test_maybe_update_best_logic(small_regression_data, trainer):
    X_train, y_train, X_valid, y_valid = small_regression_data
    model = LinearRegression()
    model.fit(X_train, y_train)

    metrics = {"mae": 0.5, "rmse": 1.0, "r2": 0.9}

    # Initial best
    trainer.best_score = float("inf")
    trainer.metric = "mae"

    trainer._maybe_update_best(model, "linear", metrics)
    assert trainer.best_model == model
    assert trainer.best_model_name == "linear"
    assert trainer.best_score == metrics["mae"]

    # Not better score
    metrics2 = {"mae": 0.6, "rmse": 1.1, "r2": 0.8}
    trainer._maybe_update_best(model, "ridge", metrics2)
    assert trainer.best_model_name == "linear"  # should remain the same


def test_save_and_load_model(tmp_path, small_regression_data):
    from sklearn.linear_model import LinearRegression
    X_train, y_train, _, _ = small_regression_data

    model = LinearRegression()
    model.fit(X_train, y_train)

    save_path = tmp_path / "test_model.pkl"
    saved_path = ModelTrainer.save_model(model, path=str(save_path))
    assert saved_path == str(save_path)

    loaded_model = ModelTrainer.load_model(path=str(save_path))
    assert hasattr(loaded_model, "predict")
    np.testing.assert_allclose(
        model.predict(X_train), loaded_model.predict(X_train)
    )
