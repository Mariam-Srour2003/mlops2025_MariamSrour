import os
import pytest
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error

from scripts.train import (
    evaluate,
    train_models,
    save_model,
    load_model,
)

# Dummy dataset fixture
@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = pd.DataFrame({
        "f1": np.random.rand(20),
        "f2": np.random.rand(20),
        "f3": np.random.rand(20),
    })
    y = X["f1"]*10 + np.random.rand(20)   # predictable target
    X_train, X_valid = X.iloc[:15], X.iloc[15:]
    y_train, y_valid = y.iloc[:15], y.iloc[15:]
    return X_train, y_train, X_valid, y_valid


# Test evaluate()
def test_evaluate_function(sample_data):
    from sklearn.linear_model import LinearRegression
    X_train, y_train, X_valid, y_valid = sample_data

    model = LinearRegression()
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_valid, y_valid)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["mae"] >= 0


# Test model training + selection
def test_train_models(sample_data):
    X_train, y_train, X_valid, y_valid = sample_data

    best_model, best_name = train_models(
        X_train, y_train, X_valid, y_valid,
        metric="mae"
    )

    assert best_model is not None
    assert best_name in ["linear","ridge","lasso","rf","gb"]


# Test save + load model
def test_save_and_load_model(tmp_path, sample_data):
    from sklearn.linear_model import LinearRegression
    
    X_train, y_train, X_valid, y_valid = sample_data

    model = LinearRegression()
    model.fit(X_train, y_train)

    save_path = tmp_path / "test_model.pkl"
    save_model(model, path=str(save_path))

    assert os.path.exists(save_path)

    loaded_model = load_model(str(save_path))
    preds_loaded = loaded_model.predict(X_valid)
    preds_original = model.predict(X_valid)

    assert np.allclose(preds_loaded, preds_original)
