from datetime import datetime
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

    def __init__(self, config_file: str):
        print(f"[INFO] Loading config from {config_file}...")
        self.cfg = OmegaConf.load(config_file)  # Load the config from the file
        self.preprocessor = Preprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(metric=self.cfg.train.metric)
        self.inference = None
        print("[INFO] Pipeline initialized.")

    def load_data(self, path: str) -> pd.DataFrame:
        print(f"[INFO] Loading data from {path}...")
        df = pd.read_csv(path)
        print(f"[INFO] Loaded {len(df)} rows and {len(df.columns)} columns.")
        return df

    def preprocess(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        print(f"[INFO] Preprocessing {'training' if is_train else 'validation/test'} data...")
        df_processed = self.preprocessor.run(df, is_train=is_train)
        print(f"[INFO] Preprocessing done. Data shape: {df_processed.shape}")
        return df_processed

    def feature_engineering(self, df: pd.DataFrame, fit: bool = False):
        print(f"[INFO] Applying feature engineering (fit={fit})...")
        X, y, _ = self.feature_engineer.transform(df, fit=fit, is_train=True)
        print(f"[INFO] Feature engineering done. Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y

    def train(self, X_train, y_train, X_valid, y_valid):
        print("[INFO] Training model...")
        model, best_model_name = self.model_trainer.train(X_train, y_train, X_valid, y_valid)
        self.inference = InferencePipeline(model=model)
        print(f"[INFO] Training complete. Best model: {best_model_name}")
        return model, best_model_name

    def batch_inference(self, df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
        if self.inference is None:
            raise RuntimeError("No trained model for inference")
        print("[INFO] Running batch inference...")
        output_df = self.inference.run(df, fit=True, is_train=False)

        print(f"[INFO] Batch inference complete. Output shape: {output_df.shape}")
        if save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(save_path, f"predictions_{timestamp}.csv")
            output_df.to_csv(output_file, index=False)
            print(f"[INFO] Predictions saved to {output_file}")
        return output_df

    def run(self):
        print("[INFO] Starting full pipeline run...")

        # Load train/test
        train_df = self.load_data(self.cfg.paths.train_csv)
        test_df = self.load_data(self.cfg.paths.test_csv)

        # Split train/valid
        print("[INFO] Splitting train/validation data...")
        train_split, valid_split = train_test_split(
            train_df,
            test_size=self.cfg.train.test_size,
            random_state=self.cfg.train.seed
        )
        print(f"[INFO] Split done. Train: {len(train_split)}, Validation: {len(valid_split)}")

        # Preprocess
        train_df = self.preprocess(train_split)
        valid_df = self.preprocess(valid_split, is_train=False)
        test_df = self.preprocess(test_df, is_train=False)

        # Feature engineering
        X_train, y_train = self.feature_engineering(train_df, fit=True)
        X_valid, y_valid = self.feature_engineering(valid_df, fit=False)

        # Train
        model, best_model_name = self.train(X_train, y_train, X_valid, y_valid)

        # Save model
        print(f"[INFO] Saving model to {self.cfg.paths.artifact_dir}...")
        os.makedirs(self.cfg.paths.artifact_dir, exist_ok=True)
        self.model_trainer.save_model(model, f"{self.cfg.paths.artifact_dir}best_model.pkl")
        model_name = f"NYC_Taxi_{best_model_name}"
        print(f"[INFO] Model registered as: {model_name}")
        print(f"[INFO] Load later with: mlflow.sklearn.load_model('models:/{model_name}/None')")

        # Batch inference
        print("[INFO] Running inference on test data...")
        os.makedirs(self.cfg.paths.output_dir, exist_ok=True)
        output_df = self.batch_inference(test_df, save_path=self.cfg.paths.output_dir)

        print("[INFO] Pipeline run complete!")
        return model, best_model_name, output_df
