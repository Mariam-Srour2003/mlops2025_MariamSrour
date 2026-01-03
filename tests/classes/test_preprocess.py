import pandas as pd
import pytest
from src.ml_project.preprocess.processor import Preprocessor


# Sample data
data = {
    "id": [1, 2, 3, 4, 4],
    "pickup_latitude": [40.7, 40.8, None, 91, 40.7],
    "pickup_longitude": [-73.9, -73.95, -73.85, -180.5, -73.9],
    "dropoff_latitude": [40.75, 40.82, 40.80, 40.7, 40.75],
    "dropoff_longitude": [-73.98, -73.96, -73.9, -73.85, -73.98],
    "pickup_datetime": [
        "2025-12-20 12:00:00",
        "2025-12-20 13:00:00",
        "invalid",
        "2025-12-20 14:00:00",
        "2025-12-20 12:00:00",
    ],
    "dropoff_datetime": [
        "2025-12-20 12:30:00",
        "2025-12-20 13:45:00",
        "2025-12-20 14:00:00",
        "2025-12-20 14:30:00",
        "2025-12-20 12:30:00",
    ],
    "passenger_count": [1, None, 2, 3, 1],
    "trip_duration": [1800, 2700, -500, 3600, 1800],
}

df = pd.DataFrame(data)


# Tests
def test_validate_required_columns():
    p = Preprocessor()  # Remove is_train=True from constructor
    p.validate_required_columns(df)  # This passes for train data
    
    # Test missing column raises error
    with pytest.raises(ValueError):
        p.validate_required_columns(df.drop(columns=["pickup_latitude"]))


def test_drop_missing_locations():
    p = Preprocessor()
    cleaned = p.drop_missing_locations(df)
    assert cleaned["pickup_latitude"].isnull().sum() == 0
    assert cleaned["pickup_longitude"].isnull().sum() == 0
    assert len(cleaned) == 4  # Should drop 1 row with None pickup_latitude


def test_fill_missing_passenger_count():
    p = Preprocessor()
    filled = p.fill_missing_passenger_count(df)
    assert filled["passenger_count"].isnull().sum() == 0
    assert filled.loc[1, "passenger_count"] == 1


def test_remove_invalid_coordinates():
    p = Preprocessor()
    cleaned = p.remove_invalid_coordinates(df)
    assert cleaned["pickup_latitude"].between(-90, 90).all()
    assert cleaned["dropoff_latitude"].between(-90, 90).all()
    assert cleaned["pickup_longitude"].between(-180, 180).all()
    assert cleaned["dropoff_longitude"].between(-180, 180).all()
    assert len(cleaned) == 3  # Drops invalid lat=91 and lon=-180.5


def test_remove_invalid_trip_durations():
    p = Preprocessor()
    cleaned = p.remove_invalid_trip_durations(df)
    assert (cleaned["trip_duration"] > 0).all()
    assert len(cleaned) == 4  # Drops row with -500 duration


def test_parse_datetime():
    p = Preprocessor()
    parsed = p.parse_datetime(df)
    assert parsed["pickup_datetime"].dtype == "datetime64[ns]"
    # Should drop the invalid datetime row
    assert parsed["pickup_datetime"].notnull().all()
    assert len(parsed) == 4  # Drops "invalid" datetime


def test_remove_duration_outliers():
    p = Preprocessor()
    # Create data with outliers
    outlier_df = pd.DataFrame({
        "trip_duration": [100, 200, 300, 10000, 15000]
    })
    cleaned = p.remove_duration_outliers(outlier_df)
    assert len(cleaned) == 3  # Removes 2 extreme outliers


def test_extract_time_features():
    p = Preprocessor()
    # Clean datetime first
    df_clean = p.parse_datetime(df)
    df_features = p.extract_time_features(df_clean)
    assert "pickup_hour" in df_features.columns
    assert "pickup_day" in df_features.columns
    assert "pickup_weekday" in df_features.columns
    assert "pickup_month" in df_features.columns
    # Fix: pandas returns int32 for small datasets, check integer type instead
    assert pd.api.types.is_integer_dtype(df_features["pickup_hour"])


def test_remove_duplicates():
    p = Preprocessor()
    cleaned = p.remove_duplicates(df)
    assert len(cleaned) == 4  # Removes duplicate id=4
    assert cleaned["id"].nunique() == 4


def test_full_preprocess_run():
    p = Preprocessor()  # Remove is_train=True from constructor
    processed = p.run(df, is_train=True)
    
    # Check all expected transformations
    assert "pickup_hour" in processed.columns
    assert processed.shape[0] > 0  # Should have some data left
    assert processed["trip_duration"].min() > 0  # No negative durations
    assert processed["passenger_count"].notnull().all()  # No missing passengers
    assert processed["pickup_latitude"].notnull().all()  # No missing locations
    assert processed["pickup_latitude"].between(-90, 90).all()  # Valid coordinates
    assert processed["id"].nunique() == len(processed)  # No duplicates
    assert pd.api.types.is_datetime64_any_dtype(processed["pickup_datetime"])


def test_full_preprocess_run_test_mode():
    """Test preprocess in test mode (no trip_duration cleaning)"""
    p = Preprocessor()
    processed = p.run(df, is_train=False)
    
    assert "pickup_hour" in processed.columns
    assert processed.shape[0] > 0
    # Should NOT filter trip_duration since is_train=False
    assert len(processed) == 3  # Only location/datetime cleaning
