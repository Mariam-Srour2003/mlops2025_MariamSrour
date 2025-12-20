import pandas as pd
import os
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from omegaconf import OmegaConf
from scripts.preprocess import preprocess
from scripts.feature_engineering import feature_engineering

# Evaluation
def evaluate(model, X, y):
    preds = model.predict(X)
    return {
        "mae": mean_absolute_error(y, preds),
        "rmse": mean_squared_error(y, preds, squared=False),
        "r2": r2_score(y, preds)
    }

# Train multiple models
def train_models(X_train, y_train, X_valid, y_valid, metric="mae", config=None):
    default_config = {
        "linear": {},
        "ridge": {"alpha": 1.0},
        "lasso": {"alpha": 0.1},
        "rf": {"n_estimators":10, "max_depth":12, "n_jobs": -1},
        "gb": {"n_estimators":10, "max_depth":3, "learning_rate":0.1}
    }
    cfg = config if config is not None else default_config

    models = {
        "linear": LinearRegression(**cfg.get("linear",{})),
        "ridge": Ridge(**cfg.get("ridge",{})),
        "lasso": Lasso(**cfg.get("lasso",{})),
        "rf": RandomForestRegressor(**cfg.get("rf",{})),
        "gb": GradientBoostingRegressor(**cfg.get("gb",{}))
    }

    best_model = None
    best_score = float('inf') if metric in ["mae","rmse"] else -float('inf')
    best_name = None

    mlflow.set_experiment("NYC-Taxi-Trip-ML")

    for name, model in models.items():
        print(f"Training model: {name}")
        with mlflow.start_run(run_name=name):
            mlflow.log_params(cfg.get(name, {}))
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_valid, y_valid)
            for k,v in metrics.items():
                mlflow.log_metric(k,v)
            current_score = metrics.get(metric)
            if (metric in ["mae","rmse"] and current_score < best_score) or \
               (metric=="r2" and current_score > best_score):
                best_model = model
                best_score = current_score
                best_name = name

    print(f"Best model = {best_name} | {metric.upper()} = {best_score:.4f}")
    return best_model, best_name

# Save / Load
def save_model(model, path="best_model.pkl"):
    joblib.dump(model, path)
    return path

def load_model(path="best_model.pkl"):
    return joblib.load(path)
