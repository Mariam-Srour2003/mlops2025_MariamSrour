import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Optional
from ml_project.features.engineer import FeatureEngineer
from ml_project.train.trainer import ModelTrainer  # <- class-based

class InferencePipeline:
    """
    Handles batch inference for a trained model:
    - Applies feature engineering via FeatureEngineer class
    - Generates predictions
    - Optionally saves predictions to CSV
    """

    def __init__(self, model=None, model_name: str = None, stage: str = "None"):
        """
        Args:
            model: trained ML model (sklearn-like API) OR
            model_name: MLflow registered model name to load (e.g., "NYC_Taxi_gb")
            stage: MLflow stage ("None", "Production", "Staging")
        """
        self.feature_engineer = FeatureEngineer()
        
        if model_name:
            # Load from MLflow Model Registry by name
            import mlflow.sklearn
            self.model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
            print(f"✅ Loaded MLflow model: models:/{model_name}/{stage}")
        elif model is not None:
            self.model = model
            print("✅ Using provided model")
        else:
            raise ValueError("Must provide either 'model' or 'model_name'")



    def run(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None,
        is_train: bool = False,
        fit: bool = False,
    ) -> pd.DataFrame:
        """
        Run batch inference on a dataframe

        Args:
            df (pd.DataFrame): input raw data
            save_path (str, optional): directory or filename to save predictions
            is_train (bool): whether the df contains 'trip_duration'
            fit (bool): whether to fit the encoders/scalers (only True for training)

        Returns:
            pd.DataFrame: dataframe with predictions and timestamp
        """
        # Feature engineering
        X, _, _ = self.feature_engineer.transform(df, fit=fit, is_train=is_train)

        # Predictions
        preds = self.model.predict(X)
        df = df.copy()
        df['prediction'] = preds
        df['timestamp'] = datetime.now()

        # Save if path provided
        if save_path:
            self._save(df, save_path)

        return df

    def _save(self, df: pd.DataFrame, save_path: str):
        """
        Save predictions dataframe to CSV
        """
        if os.path.isdir(save_path):
            date_str = datetime.now().strftime("%Y%m%d")
            save_file = os.path.join(save_path, f"{date_str}_predictions.csv")
        else:
            save_file = save_path

        df.to_csv(save_file, index=False)
        print(f"Predictions saved to {save_file}")
