import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ohe = None
scaler = None

# Distance features
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def build_distance_feature(df, chunk_size=100000):
    distances = np.zeros(len(df))
    for start in range(0, len(df), chunk_size):
        end = start + chunk_size
        distances[start:end] = haversine_vectorized(
            df['pickup_latitude'].iloc[start:end].values,
            df['pickup_longitude'].iloc[start:end].values,
            df['dropoff_latitude'].iloc[start:end].values,
            df['dropoff_longitude'].iloc[start:end].values
        )
    df['distance_km'] = distances
    df['distance_km'] = df['distance_km'].fillna(df['distance_km'].median())
    return df

# Datetime features
def build_datetime_features(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_day'] = df['pickup_datetime'].dt.day
    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
    df['pickup_month'] = df['pickup_datetime'].dt.month
    datetime_cols = ['pickup_hour','pickup_day','pickup_weekday','pickup_month']
    for col in datetime_cols:
        df[col] = df[col].fillna(df[col].median())
    return df

# Encoding
def encode_categorical(df, fit=False):
    global ohe
    cat_cols = ['vendor_id', 'store_and_fwd_flag']
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    if fit:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = ohe.fit_transform(df[cat_cols])
    else:
        encoded = ohe.transform(df[cat_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(cat_cols),
        index=df.index
    )
    df = df.drop(columns=cat_cols)
    df = pd.concat([df, encoded_df], axis=1)
    return df

# Scaling
def scale_numeric(df, fit=False):
    global scaler
    num_cols = [
        'distance_km', 'pickup_hour','pickup_day',
        'pickup_weekday','pickup_month','passenger_count'
    ]
    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    return df

# Feature engineering pipeline
def feature_engineering(df, fit=False, save=False, is_train=True):
    df = build_distance_feature(df)
    df = build_datetime_features(df)
    df = encode_categorical(df, fit=fit)
    df = scale_numeric(df, fit=fit)

    X = df.drop(columns=['trip_duration','id','pickup_datetime','dropoff_datetime'], errors='ignore')
    y = df['trip_duration'] if is_train and 'trip_duration' in df.columns else None

    if save:
        df.to_csv("processed_features.csv", index=False)

    return X, y, list(X.columns)
