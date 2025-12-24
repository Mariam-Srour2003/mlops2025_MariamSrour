import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureEngineer:
    """
    Feature engineering for NYC Taxi Trip Duration dataset.
    Handles:
    - distance features
    - datetime features
    - categorical encoding
    - numeric scaling
    """

    def __init__(
        self,
        categorical_cols=None,
        numeric_cols=None,
        chunk_size: int = 100_000
    ):
        self.categorical_cols = categorical_cols or [
            "vendor_id",
            "store_and_fwd_flag"
        ]

        self.numeric_cols = numeric_cols or [
            "distance_km",
            "pickup_hour",
            "pickup_day",
            "pickup_weekday",
            "pickup_month",
            "passenger_count"
        ]

        self.chunk_size = chunk_size

        # learned components
        self.ohe = None
        self.scaler = None

    # Distance features
    @staticmethod
    def haversine_vectorized(lat1, lon1, lat2, lon2):
        R = 6371.0088
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def build_distance_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        distances = np.zeros(len(df))

        for start in range(0, len(df), self.chunk_size):
            end = start + self.chunk_size
            distances[start:end] = self.haversine_vectorized(
                df["pickup_latitude"].iloc[start:end].values,
                df["pickup_longitude"].iloc[start:end].values,
                df["dropoff_latitude"].iloc[start:end].values,
                df["dropoff_longitude"].iloc[start:end].values,
            )

        df = df.copy()
        df["distance_km"] = distances
        df["distance_km"] = df["distance_km"].fillna(df["distance_km"].median())

        return df

    # Datetime features
    def build_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["pickup_datetime"] = pd.to_datetime(
            df["pickup_datetime"], errors="coerce"
        )

        df["pickup_hour"] = df["pickup_datetime"].dt.hour
        df["pickup_day"] = df["pickup_datetime"].dt.day
        df["pickup_weekday"] = df["pickup_datetime"].dt.weekday
        df["pickup_month"] = df["pickup_datetime"].dt.month

        for col in ["pickup_hour", "pickup_day", "pickup_weekday", "pickup_month"]:
            df[col] = df[col].fillna(df[col].median())

        return df

    # Categorical encoding
    def encode_categorical(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()
        df[self.categorical_cols] = df[self.categorical_cols].fillna("Unknown")

        if fit:
            self.ohe = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
            encoded = self.ohe.fit_transform(df[self.categorical_cols])
        else:
            if self.ohe is None:
                raise RuntimeError("OneHotEncoder not fitted.")
            encoded = self.ohe.transform(df[self.categorical_cols])

        encoded_df = pd.DataFrame(
            encoded,
            columns=self.ohe.get_feature_names_out(self.categorical_cols),
            index=df.index
        )

        df = df.drop(columns=self.categorical_cols)
        df = pd.concat([df, encoded_df], axis=1)

        return df

    # Numeric scaling
    def scale_numeric(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()

        if fit:
            self.scaler = StandardScaler()
            df[self.numeric_cols] = self.scaler.fit_transform(df[self.numeric_cols])
        else:
            if self.scaler is None:
                raise RuntimeError("Scaler not fitted.")
            df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])

        return df

    # Full pipeline
    def transform(
        self,
        df: pd.DataFrame,
        fit: bool = False,
        is_train: bool = True,
        save: bool = False,
        save_path: str = "processed_features.csv",
    ):
        df = self.build_distance_feature(df)
        df = self.build_datetime_features(df)
        df = self.encode_categorical(df, fit=fit)
        df = self.scale_numeric(df, fit=fit)

        X = df.drop(
            columns=[
                "trip_duration",
                "id",
                "pickup_datetime",
                "dropoff_datetime",
            ],
            errors="ignore",
        )

        y = df["trip_duration"] if is_train and "trip_duration" in df.columns else None

        if save:
            df.to_csv(save_path, index=False)

        return X, y, list(X.columns)
