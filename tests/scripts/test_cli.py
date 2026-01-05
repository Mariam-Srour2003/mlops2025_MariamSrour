import pytest
import os
from pathlib import Path

import ml_project.cli as cli


def test_cli_functions_exist():
    assert hasattr(cli, "train") and callable(cli.train)
    assert hasattr(cli, "inference") and callable(cli.inference)


def test_inference_missing_model_raises(tmp_path, monkeypatch):
    # Create a temporary config that points to non-existing artifact dir
    cfg = {
        "paths": {
            "train_csv": str(tmp_path / "train.csv"),
            "test_csv": str(tmp_path / "test.csv"),
            "artifact_dir": str(tmp_path / "artifacts"),
            "output_dir": str(tmp_path / "outputs")
        },
        "train": {"test_size": 0.2, "seed": 42, "metric": "rmse"}
    }

    cfg_path = tmp_path / "config.yaml"
    import yaml
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    # Ensure artifact dir does not exist
    if os.path.exists(cfg["paths"]["artifact_dir"]):
        import shutil
        shutil.rmtree(cfg["paths"]["artifact_dir"])

    with pytest.raises(FileNotFoundError):
        cli.inference(str(cfg_path))
