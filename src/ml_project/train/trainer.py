import mlflow
import joblib
import time
from typing import Dict, Tuple
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelTrainer:
    """
    Handles:
    - model training
    - evaluation
    - model selection
    - MLflow logging
    - model persistence
    """

    def __init__(
        self,
        metric: str = "mae",
        experiment_name: str = "NYC-Taxi-Trip-ML",
        model_config: Dict = None,
    ):
        self.metric = metric
        self.experiment_name = experiment_name
        self.model_config = model_config or self._default_model_config()

        self.best_model = None
        self.best_model_name = None
        self.best_score = (
            float("inf") if metric in ["mae", "rmse"] else -float("inf")
        )
        # Track the run id corresponding to the current best model (for registry)
        self.best_run_id = None
        # Track whether MLflow is reachable/enabled
        self._mlflow_available = True

    # Default model configuration
    @staticmethod
    def _default_model_config() -> Dict:
        return {
            "linear": {},
            "ridge": {"alpha": 1.0},
            "lasso": {"alpha": 0.1},
            "rf": {"n_estimators": 200, "max_depth": 12, "n_jobs": -1},
            "gb": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1},
        }

    # Model registry
    def _build_models(self) -> Dict:
        return {
            "linear": LinearRegression(**self.model_config.get("linear", {})),
            "ridge": Ridge(**self.model_config.get("ridge", {})),
            "lasso": Lasso(**self.model_config.get("lasso", {})),
            "rf": RandomForestRegressor(**self.model_config.get("rf", {})),
            "gb": GradientBoostingRegressor(**self.model_config.get("gb", {})),
        }

    # Evaluation
    @staticmethod
    def evaluate(model, X, y) -> Dict:
        preds = model.predict(X)
        return {
            "mae": mean_absolute_error(y, preds),
            "rmse": mean_squared_error(y, preds) ** 0.5,
            "r2": r2_score(y, preds),
        }

    # Training loop
    def train(self, X_train, y_train, X_valid, y_valid) -> Tuple[object, str]:
        # Try to set MLflow experiment with retries; if unavailable, proceed without MLflow
        retries = 3
        delay = 5
        for attempt in range(1, retries + 1):
            try:
                mlflow.set_experiment(self.experiment_name)
                break
            except Exception as e:
                print(f"⚠️ Unable to contact MLflow (attempt {attempt}/{retries}): {e}")
                if attempt == retries:
                    print("⚠️ MLflow unavailable; proceeding without MLflow logging.")
                    self._mlflow_available = False
                else:
                    time.sleep(delay)

        # Infer signature for model logging (best-effort)
        signature = None
        try:
            signature = infer_signature(X_train, self._build_models()["linear"].fit(X_train, y_train).predict(X_valid))
        except Exception as e:
            print(f"⚠️ Failed to infer signature: {e}")

        models = self._build_models()

        for name, model in models.items():
            print(f"Training model: {name}")

            if self._mlflow_available:
                try:
                    with mlflow.start_run(run_name=name):
                        mlflow.log_params(self.model_config.get(name, {}))

                        model.fit(X_train, y_train)

                        # Log model to MLflow with signature (guarded)
                        try:
                            mlflow.sklearn.log_model(
                                sk_model=model,
                                artifact_path="model",
                                signature=signature,
                                input_example=(X_valid.iloc[:5] if hasattr(X_valid, "iloc") else X_valid[:5])
                            )
                        except Exception as e:
                            print(f"⚠️ MLflow logging failed for model {name}: {e}")

                        metrics = self.evaluate(model, X_valid, y_valid)
                        for k, v in metrics.items():
                            try:
                                mlflow.log_metric(k, v)
                            except Exception as e:
                                print(f"⚠️ Failed to log metric {k} to MLflow: {e}")

                        # Update best model and record run id if it is the best
                        prev_best = self.best_model_name
                        self._maybe_update_best(model, name, metrics)
                        if self.best_model_name == name:
                            try:
                                self.best_run_id = mlflow.active_run().info.run_id
                            except Exception as e:
                                print(f"⚠️ Could not get active MLflow run id: {e}")
                except Exception as e:
                    print(f"⚠️ Unexpected MLflow error during training {name}: {e}")
                    print("Proceeding with training without MLflow for this model.")
                    # fallback to local training
                    model.fit(X_train, y_train)
                    metrics = self.evaluate(model, X_valid, y_valid)
                    self._maybe_update_best(model, name, metrics)
            else:
                # Train without MLflow
                model.fit(X_train, y_train)
                metrics = self.evaluate(model, X_valid, y_valid)
                self._maybe_update_best(model, name, metrics)

        # Register best model to Model Registry if MLflow is available and we have a run id
        if self.best_model_name and self.best_run_id and self._mlflow_available:
            model_uri = f"runs:/{self.best_run_id}/model"
            registered_name = f"NYC_Taxi_{self.best_model_name}"
            try:
                mlflow.register_model(model_uri, registered_name)
                print(f"Registered best model '{registered_name}' from run {self.best_run_id}")
            except Exception as e:
                print(f"⚠️ Failed to register best model '{registered_name}': {e}")
        elif self.best_model_name and not self._mlflow_available:
            print("⚠️ Best model found but MLflow unavailable; skipping registry registration.")
        elif self.best_model_name:
            print("⚠️ Found a best model but couldn't determine its run id; skipping registry registration.")

        print(f"Best model = {self.best_model_name} | {self.metric.upper()} = {self.best_score:.4f}")
        return self.best_model, self.best_model_name

    def load_model_by_name(self, model_name: str, stage: str = "None") -> object:
        """Load model from MLflow Model Registry by name"""
        model_uri = f"models:/{model_name}/{stage}"
        return mlflow.sklearn.load_model(model_uri)

    # Best model selection logic
    def _maybe_update_best(self, model, name: str, metrics: Dict):
        current_score = metrics[self.metric]

        is_better = (
            current_score < self.best_score
            if self.metric in ["mae", "rmse"]
            else current_score > self.best_score
        )

        if is_better:
            self.best_model = model
            self.best_model_name = name
            self.best_score = current_score

    # Persistence
    @staticmethod
    def save_model(model, path: str = "best_model.pkl") -> str:
        joblib.dump(model, path)
        return path

    @staticmethod
    def load_model(path: str = "best_model.pkl"):
        return joblib.load(path)
