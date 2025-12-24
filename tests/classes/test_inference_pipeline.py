import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.ml_project.inference.inference import InferencePipeline

# Fixtures
@pytest.fixture
def sample_df():
    data = {
        "pickup_latitude": [40.7, 40.8],
        "pickup_longitude": [-73.9, -73.95],
        "dropoff_latitude": [40.75, 40.82],
        "dropoff_longitude": [-73.98, -73.96],
        "pickup_datetime": ["2025-12-20 12:00:00", "2025-12-20 13:00:00"],
        "passenger_count": [1, 2],
        "trip_duration": [1800, 2700],
        "vendor_id": ["V1", "V2"],
        "store_and_fwd_flag": ["N", "Y"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = np.array([2000, 2500])
    return model

@pytest.fixture
def pipeline(mock_model):
    return InferencePipeline(model=mock_model)

# Tests
def test_run_creates_predictions(sample_df, pipeline):
    # Mock transform to return numeric data directly
    X_mock = sample_df.drop(columns=["trip_duration"])
    y_mock = sample_df["trip_duration"]
    columns_mock = list(X_mock.columns)

    with patch.object(pipeline.feature_engineer, "transform", return_value=(X_mock, y_mock, columns_mock)):
        result = pipeline.run(sample_df, fit=False, is_train=True)

        # Check columns
        assert "prediction" in result.columns
        assert "timestamp" in result.columns

        # Check prediction values
        np.testing.assert_array_equal(result["prediction"].values, [2000, 2500])

def test_run_saves_to_file(tmp_path, sample_df, pipeline):
    save_file = tmp_path / "predictions.csv"

    X_mock = sample_df.drop(columns=["trip_duration"])
    y_mock = sample_df["trip_duration"]
    columns_mock = list(X_mock.columns)

    with patch.object(pipeline.feature_engineer, "transform", return_value=(X_mock, y_mock, columns_mock)):
        result = pipeline.run(sample_df, save_path=str(save_file), fit=False, is_train=True)

        # Check file exists
        assert save_file.exists()

        # Read back and check columns
        saved_df = pd.read_csv(save_file)
        assert "prediction" in saved_df.columns
        assert "timestamp" in saved_df.columns

def test_run_saves_to_directory(tmp_path, sample_df, pipeline):
    save_dir = tmp_path / "outputs"
    save_dir.mkdir()

    X_mock = sample_df.drop(columns=["trip_duration"])
    y_mock = sample_df["trip_duration"]
    columns_mock = list(X_mock.columns)

    with patch.object(pipeline.feature_engineer, "transform", return_value=(X_mock, y_mock, columns_mock)):
        result = pipeline.run(sample_df, save_path=str(save_dir), fit=False, is_train=True)

        # File should be created in directory with date prefix
        files = list(save_dir.glob("*_predictions.csv"))
        assert len(files) == 1
        saved_df = pd.read_csv(files[0])
        assert "prediction" in saved_df.columns
        assert "timestamp" in saved_df.columns
