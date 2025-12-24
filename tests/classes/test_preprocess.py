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
    p = Preprocessor(is_train=True)
    p.validate_required_columns(df)

    with pytest.raises(ValueError):
        p.validate_required_columns(df.drop(columns=["pickup_latitude"]))


def test_drop_missing_locations():
    p = Preprocessor()
    cleaned = p.drop_missing_locations(df)
    assert cleaned["pickup_latitude"].isnull().sum() == 0


def test_fill_missing_passenger_count():
    p = Preprocessor()
    filled = p.fill_missing_passenger_count(df)
    assert filled["passenger_count"].isnull().sum() == 0
    assert filled.loc[1, "passenger_count"] == 1


def test_remove_invalid_coordinates():
    p = Preprocessor()
    cleaned = p.remove_invalid_coordinates(df)

    assert cleaned["pickup_latitude"].between(-90, 90).all()
    assert cleaned["pickup_longitude"].between(-180, 180).all()


def test_remove_invalid_trip_durations():
    p = Preprocessor()
    cleaned = p.remove_invalid_trip_durations(df)
    assert (cleaned["trip_duration"] > 0).all()


def test_parse_datetime():
    p = Preprocessor()
    parsed = p.parse_datetime(df)

    assert parsed["pickup_datetime"].dtype == "datetime64[ns]"
    assert parsed["pickup_datetime"].notnull().all()


def test_full_preprocess_run():
    p = Preprocessor(is_train=True)
    processed = p.run(df)

    assert "pickup_hour" in processed.columns
    assert processed["trip_duration"].min() > 0
    assert processed["passenger_count"].notnull().all()
    assert processed["id"].nunique() == len(processed)
