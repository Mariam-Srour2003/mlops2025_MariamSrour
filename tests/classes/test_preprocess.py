import pandas as pd
import pytest

from scripts.preprocess import (
    preprocess,
    validate_required_columns,
    drop_missing_locations,
    fill_missing_passenger_count,
    remove_invalid_coordinates,
    remove_invalid_trip_durations,
    remove_duration_outliers,
    parse_datetime,
    extract_time_features,
    remove_duplicates,
)

# Sample data for testing
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
    validate_required_columns(df)  # should pass

    with pytest.raises(ValueError):
        validate_required_columns(df.drop(columns=["pickup_latitude"]))


def test_drop_missing_locations():
    cleaned = drop_missing_locations(df)
    assert cleaned["pickup_latitude"].isnull().sum() == 0
    assert cleaned["pickup_longitude"].isnull().sum() == 0


def test_fill_missing_passenger_count():
    filled = fill_missing_passenger_count(df)
    assert filled["passenger_count"].isnull().sum() == 0
    assert filled.loc[1, "passenger_count"] == 1


def test_remove_invalid_coordinates():
    cleaned = remove_invalid_coordinates(df)
    assert cleaned["pickup_latitude"].between(-90, 90).all()
    assert cleaned["pickup_longitude"].between(-180, 180).all()
    assert cleaned["dropoff_latitude"].between(-90, 90).all()
    assert cleaned["dropoff_longitude"].between(-180, 180).all()


def test_remove_invalid_trip_durations():
    cleaned = remove_invalid_trip_durations(df)
    assert (cleaned["trip_duration"] > 0).all()


def test_remove_duration_outliers():
    cleaned = remove_duration_outliers(df, lower=0.01, upper=0.99)
    q1 = df["trip_duration"].quantile(0.01)
    q99 = df["trip_duration"].quantile(0.99)
    assert cleaned["trip_duration"].between(q1, q99).all()


def test_parse_datetime():
    parsed = parse_datetime(df)
    assert parsed["pickup_datetime"].dtype == "datetime64[ns]"
    assert parsed["pickup_datetime"].notnull().all()


def test_extract_time_features():
    parsed = parse_datetime(df)
    features = extract_time_features(parsed)

    for col in ["pickup_hour", "pickup_day", "pickup_weekday", "pickup_month"]:
        assert col in features.columns


def test_remove_duplicates():
    deduped = remove_duplicates(df)
    assert deduped["id"].nunique() == len(deduped)


def test_full_preprocess():
    processed = preprocess(df)

    # feature checks
    assert "pickup_hour" in processed.columns
    assert "pickup_weekday" in processed.columns

    # data validity
    assert processed["trip_duration"].min() > 0
    assert processed["pickup_latitude"].between(-90, 90).all()
    assert processed["pickup_longitude"].between(-180, 180).all()
    assert processed["passenger_count"].notnull().all()

    # no duplicate ids
    assert processed["id"].nunique() == len(processed)
