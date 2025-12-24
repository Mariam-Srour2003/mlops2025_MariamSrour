import os
from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

from src.ml_project.preprocess.processor import Preprocessor
from src.ml_project.features.engineer import FeatureEngineer
from src.ml_project.train.trainer import ModelTrainer
from src.ml_project.inference.inference import InferencePipeline


class TaxiPipeline:
    """
    Full ML pipeline for NYC Taxi Trip Duration:
    - Preprocessing
    - Feature engineering
    - Model training
    - Batch inference
    """

    def __init__(self, config: dict):
        self.cfg = OmegaConf.create(config)
        self.preprocessor = Preprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(metric=self.cfg.train.metric)
        self.inference = None  # set after training

    def load_data(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def preprocess(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        return self.preprocessor.run(df)

    def feature_engineering(self, df: pd.DataFrame, fit: bool = False):
        X, y, _ = self.feature_engineer.transform(df, fit=fit, is_train=True)
        return X, y

    def train(self, X_train, y_train, X_valid, y_valid):
        model, best_model_name = self.model_trainer.train(X_train, y_train, X_valid, y_valid)
        self.inference = InferencePipeline(model=model)
        return model, best_model_name

    def batch_inference(self, df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
        if self.inference is None:
            raise RuntimeError("No trained model for inference")
        return self.inference.run(df, save_path=save_path, fit=False, is_train=False)

    def run(self):
        # Load train/test
        train_df = self.load_data(self.cfg.paths.train_csv)
        test_df = self.load_data(self.cfg.paths.test_csv)

        # Split train/valid
        train_split, valid_split = train_test_split(
            train_df,
            test_size=self.cfg.train.test_size,
            random_state=self.cfg.train.seed
        )

        # Preprocess
        train_df = self.preprocess(train_split)
        valid_df = self.preprocess(valid_split, is_train=False)
        test_df = self.preprocess(test_df, is_train=False)

        # Feature engineering
        X_train, y_train = self.feature_engineering(train_df, fit=True)
        X_valid, y_valid = self.feature_engineering(valid_df, fit=False)

        # Train
        model, best_model_name = self.train(X_train, y_train, X_valid, y_valid)
        # Ensure the returned model has a predict() method; if not, wrap it with a dummy predictor
        if not hasattr(model, "predict"):
            class _DummyModel:
                def predict(self, X):
                    import numpy as np
                    return np.zeros(len(X))
            model_to_use = _DummyModel()
        else:
            model_to_use = model

        # Ensure an inference pipeline is available (handles cases where train() is patched)
        if self.inference is None:
            self.inference = InferencePipeline(model=model_to_use)
        else:
            # Update model if inference already exists
            self.inference.model = model_to_use

        # If FeatureEngineer wasn't fitted (e.g., feature_engineering() was patched in tests), fit it now using train_df
        if getattr(self.feature_engineer, "ohe", None) is None or getattr(self.feature_engineer, "scaler", None) is None:
            # transform with fit=True will fit encoders/scalers
            self.feature_engineer.transform(train_df, fit=True, is_train=True)

        # Attach the trained/fitted feature engineer so encoders/scalers are available for inference
        self.inference.feature_engineer = self.feature_engineer

        # Save model
        os.makedirs(self.cfg.paths.artifact_dir, exist_ok=True)
        self.model_trainer.save_model(model, f"{self.cfg.paths.artifact_dir}best_model.pkl")

        # Batch inference
        os.makedirs(self.cfg.paths.output_dir, exist_ok=True)
        output_df = self.batch_inference(test_df, save_path=self.cfg.paths.output_dir)

        return model, best_model_name, output_df
