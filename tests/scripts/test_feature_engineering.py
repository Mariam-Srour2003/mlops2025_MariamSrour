import pandas as pd
import numpy as np
import pytest

from scripts.feature_engineering import (
    build_distance_feature,
    build_datetime_features,
    encode_categorical,
    scale_numeric,
    feature_engineering,
)

# Dummy dataset for testing
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": [1,2,3],
        "pickup_latitude": [40.7, 40.8, 40.76],
        "pickup_longitude": [-73.9, -73.95, -73.98],
        "dropoff_latitude": [40.75, 40.82, 40.81],
        "dropoff_longitude": [-73.98, -73.96, -73.99],
        "pickup_datetime": [
            "2025-12-20 12:00:00",
            "2025-12-21 09:00:00",
            "2025-12-22 22:30:00",
        ],
        "dropoff_datetime":[
            "2025-12-20 12:10:00",
            "2025-12-21 09:15:00",
            "2025-12-22 22:45:00",
        ],
        "vendor_id":[1,2,2],
        "store_and_fwd_flag":[None, "Y","N"],
        "passenger_count":[1,1,2],
        "trip_duration":[1000, 1500, 1800]
    })


# Test distance feature
def test_distance_feature(sample_df):
    df = build_distance_feature(sample_df.copy())
    assert 'distance_km' in df.columns
    assert df['distance_km'].isnull().sum() == 0
    assert (df['distance_km'] > 0).all()


# Test datetime feature extraction
def test_datetime_features(sample_df):
    df = build_datetime_features(sample_df.copy())
    for col in ['pickup_hour','pickup_day','pickup_weekday','pickup_month']:
        assert col in df.columns
        assert df[col].notnull().all()


# Test categorical encoding
def test_encode_categorical_fit(sample_df):
    df = encode_categorical(sample_df.copy(), fit=True)
    assert "vendor_id" not in df.columns
    assert "store_and_fwd_flag" not in df.columns
    assert len([c for c in df.columns if "vendor_id_" in c]) > 0
    assert len([c for c in df.columns if "store_and_fwd_flag_" in c]) > 0


def test_encode_categorical_transform(sample_df):
    df1 = encode_categorical(sample_df.copy(), fit=True)
    df2 = encode_categorical(sample_df.copy(), fit=False)
    assert list(df1.columns) == list(df2.columns)


# Test scaling
def test_scaling_fit(sample_df):
    df = build_distance_feature(sample_df.copy())
    df = build_datetime_features(df)
    df = encode_categorical(df, fit=True)
    df = scale_numeric(df, fit=True)
    assert abs(df['distance_km'].mean()) < 1e-6


def test_scaling_transform(sample_df):
    df1 = build_distance_feature(sample_df.copy())
    df1 = build_datetime_features(df1)
    df1 = encode_categorical(df1, fit=True)
    df1 = scale_numeric(df1, fit=True)

    df2 = build_distance_feature(sample_df.copy())
    df2 = build_datetime_features(df2)
    df2 = encode_categorical(df2, fit=False)
    df2 = scale_numeric(df2, fit=False)

    assert list(df1.columns) == list(df2.columns)


# Test full feature engineering pipeline
def test_feature_engineering_pipeline(sample_df):
    X, y, cols = feature_engineering(sample_df.copy(), fit=True, is_train=True)
    assert y is not None
    assert len(X) == len(sample_df)
    assert len(cols) == X.shape[1]
    assert 'distance_km' in X.columns
    assert X.isnull().sum().sum() == 0
