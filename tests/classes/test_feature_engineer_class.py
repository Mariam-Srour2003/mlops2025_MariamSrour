import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.ml_project.features.engineer import FeatureEngineer

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": ["id1", "id2"],
        "vendor_id": [1, 2],
        "pickup_datetime": [
            "2016-01-01 10:30:00",
            "2016-01-02 22:45:00"
        ],
        "dropoff_datetime": [
            "2016-01-01 10:50:00",
            "2016-01-02 23:10:00"
        ],
        "pickup_latitude": [40.7128, 40.7580],
        "pickup_longitude": [-74.0060, -73.9855],
        "dropoff_latitude": [40.7306, 40.7612],
        "dropoff_longitude": [-73.9352, -73.9776],
        "passenger_count": [1, 2],
        "store_and_fwd_flag": ["N", "Y"],
        "trip_duration": [1200, 1500],
    })


@pytest.fixture
def feature_engineer():
    return FeatureEngineer(chunk_size=1)


# Distance feature
def test_build_distance_feature(sample_df, feature_engineer):
    df = feature_engineer.build_distance_feature(sample_df)

    assert "distance_km" in df.columns
    assert df["distance_km"].notna().all()
    assert (df["distance_km"] > 0).all()


# Datetime features
def test_build_datetime_features(sample_df, feature_engineer):
    df = feature_engineer.build_datetime_features(sample_df)

    for col in ["pickup_hour", "pickup_day", "pickup_weekday", "pickup_month"]:
        assert col in df.columns
        assert df[col].notna().all()


# Categorical encoding
def test_encode_categorical_fit(sample_df, feature_engineer):
    df = feature_engineer.encode_categorical(sample_df, fit=True)

    assert feature_engineer.ohe is not None
    assert isinstance(feature_engineer.ohe, OneHotEncoder)

    # original categorical columns removed
    for col in feature_engineer.categorical_cols:
        assert col not in df.columns


def test_encode_categorical_transform(sample_df, feature_engineer):
    feature_engineer.encode_categorical(sample_df, fit=True)
    df = feature_engineer.encode_categorical(sample_df, fit=False)

    assert df.shape[0] == sample_df.shape[0]


def test_encode_categorical_without_fit_raises(sample_df, feature_engineer):
    with pytest.raises(RuntimeError):
        feature_engineer.encode_categorical(sample_df, fit=False)


# Numeric scaling
def test_scale_numeric_fit(sample_df, feature_engineer):
    df = feature_engineer.build_distance_feature(sample_df)
    df = feature_engineer.build_datetime_features(df)

    df_scaled = feature_engineer.scale_numeric(df, fit=True)

    assert feature_engineer.scaler is not None
    assert isinstance(feature_engineer.scaler, StandardScaler)

    # mean approximately zero
    np.testing.assert_allclose(
        df_scaled[feature_engineer.numeric_cols].mean().values,
        np.zeros(len(feature_engineer.numeric_cols)),
        atol=1e-6,
    )


def test_scale_numeric_without_fit_raises(sample_df, feature_engineer):
    with pytest.raises(RuntimeError):
        feature_engineer.scale_numeric(sample_df, fit=False)


# Full pipeline
def test_transform_train_mode(sample_df, feature_engineer):
    X, y, feature_names = feature_engineer.transform(
        sample_df,
        fit=True,
        is_train=True,
        save=False,
    )

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(feature_names) == X.shape[1]
    assert "trip_duration" not in X.columns


def test_transform_inference_mode(sample_df, feature_engineer):
    # fit first
    feature_engineer.transform(sample_df, fit=True, is_train=True)

    X, y, feature_names = feature_engineer.transform(
        sample_df,
        fit=False,
        is_train=False,
    )

    assert y is None
    assert isinstance(X, pd.DataFrame)
    assert len(feature_names) == X.shape[1]
