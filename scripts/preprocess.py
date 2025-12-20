import pandas as pd
import numpy as np

# Validation helpers
def validate_required_columns(df, is_train=True):
    required = [
        'pickup_latitude','pickup_longitude', 'dropoff_latitude','dropoff_longitude', 
        'pickup_datetime'
    ]

    if is_train:
        required.append("trip_duration")

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df

# Missing values
def drop_missing_locations(df):
    return df.dropna(subset=[
        'pickup_latitude','pickup_longitude',
        'dropoff_latitude','dropoff_longitude'
    ])

def fill_missing_passenger_count(df):
    if 'passenger_count' in df.columns:
        df['passenger_count'] = df['passenger_count'].fillna(1)
    return df

# Location validity
def remove_invalid_coordinates(df):
    return df[
        (df['pickup_longitude'].between(-180,180)) &
        (df['dropoff_longitude'].between(-180,180)) &
        (df['pickup_latitude'].between(-90,90)) &
        (df['dropoff_latitude'].between(-90,90))
    ]

# Trip duration cleaning
def remove_invalid_trip_durations(df):
    return df[df['trip_duration'] > 0]

def remove_duration_outliers(df, lower=0.01, upper=0.99):
    q1 = df['trip_duration'].quantile(lower)
    q99 = df['trip_duration'].quantile(upper)
    return df[(df['trip_duration'] >= q1) & (df['trip_duration'] <= q99)]

# Time feature extraction
def parse_datetime(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    if 'dropoff_datetime' in df.columns:
        df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], errors='coerce')
    df = df.dropna(subset=['pickup_datetime'])
    return df

def extract_time_features(df):
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_day'] = df['pickup_datetime'].dt.day
    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
    df['pickup_month'] = df['pickup_datetime'].dt.month
    return df

# Duplicates
def remove_duplicates(df):
    if 'id' in df.columns:
        return df.drop_duplicates(subset=['id'])
    return df.drop_duplicates()

# Final Full pipeline
def preprocess(df, is_train=True):
    df = validate_required_columns(df, is_train=is_train)
    df = drop_missing_locations(df)
    df = fill_missing_passenger_count(df)
    df = remove_invalid_coordinates(df)

    if is_train:
        df = remove_invalid_trip_durations(df)
        df = remove_duration_outliers(df)

    df = remove_duplicates(df)
    df = parse_datetime(df)
    df = extract_time_features(df)
    df = df.reset_index(drop=True)
    return df
