from pathlib import Path
import argparse
import sys
import os

# Ensure project root on path when run as module
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml_project.pipelines.pipeline import TaxiPipeline
from ml_project.train.trainer import ModelTrainer
from ml_project.inference.inference import InferencePipeline
from omegaconf import OmegaConf


def train(config_path: str = "config/config.yaml") -> None:
    """Run the full training pipeline (preprocess, features, train, inference).

    This command is exposed via `uv run train`.
    """
    pipeline = TaxiPipeline(config_path)
    pipeline.run()


def inference(config_path: str = "config/config.yaml") -> None:
    """Run inference only using saved model artifacts and the test CSV.

    This command is exposed via `uv run inference`.
    """
    cfg = OmegaConf.load(config_path)
    pipeline = TaxiPipeline(config_path)

    # Load test data
    test_df = pipeline.load_data(cfg.paths.test_csv)

    # Load model artifact
    model_path = Path(cfg.paths.artifact_dir) / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Run `uv run train` first.`")

    model = ModelTrainer.load_model(str(model_path))

    # Run batch inference and save to outputs
    inf = InferencePipeline(model=model)
    os_output_dir = cfg.paths.output_dir
    inf.run(test_df, save_path=os_output_dir, is_train=False, fit=False)


def main():
    parser = argparse.ArgumentParser(description="CLI for ml-project: train or inference")
    parser.add_argument("command", choices=["train", "inference"], help="Command to run")
    parser.add_argument("--config", "-c", default="config/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    if args.command == "train":
        train(args.config)
    elif args.command == "inference":
        inference(args.config)


if __name__ == "__main__":
    main()
