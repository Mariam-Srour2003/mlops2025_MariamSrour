import pytest
import pandas as pd
from unittest.mock import MagicMock, patch, Mock
from src.ml_project.pipelines.pipeline import TaxiPipeline
from omegaconf import OmegaConf


@pytest.fixture
def config(tmp_path):
    config_dict = {
        "train": {"metric": "mae", "test_size": 0.2, "seed": 42},
        "paths": {
            "train_csv": str(tmp_path / "train.csv"),
            "test_csv": str(tmp_path / "test.csv"),
            "artifact_dir": str(tmp_path / "artifacts"),
            "output_dir": str(tmp_path / "outputs")
        }
    }
    # Convert to OmegaConf DictConfig to match what production code expects
    return OmegaConf.create(config_dict)


@pytest.fixture
def sample_csvs(config):
    # Create dummy train/test CSVs
    train_data = pd.DataFrame({
        "id": [1, 2],
        "pickup_latitude": [40.7, 40.8],
        "pickup_longitude": [-73.9, -73.95],
        "dropoff_latitude": [40.75, 40.82],
        "dropoff_longitude": [-73.98, -73.96],
        "pickup_datetime": ["2025-12-20 12:00:00", "2025-12-20 13:00:00"],
        "passenger_count": [1, 2],
        "trip_duration": [1800, 2700],
        "vendor_id": ["V1","V2"],
        "store_and_fwd_flag": ["N","Y"]
    })
    test_data = train_data.copy()
    train_data.to_csv(config["paths"]["train_csv"], index=False)
    test_data.to_csv(config["paths"]["test_csv"], index=False)
    return config


def test_pipeline_run(sample_csvs):
    cfg = sample_csvs
    
    # Patch OmegaConf.load to return our OmegaConf config object
    with patch('omegaconf.OmegaConf.load', return_value=cfg):
        # Create temp config file path for the constructor
        with patch('src.ml_project.pipelines.pipeline.os') as mock_os:
            mock_os.path.join.return_value = "fake_config.yaml"
            
            pipeline = TaxiPipeline("fake_config.yaml")
        
        # Patch heavy components
        with patch.object(pipeline, "load_data") as mock_load_data, \
             patch.object(pipeline, "preprocess", return_value=pd.DataFrame()), \
             patch.object(pipeline, "feature_engineering", 
                         side_effect=lambda df, fit=False: (
                             pd.DataFrame({"feature": [1,2]}), 
                             pd.Series([1800, 2700])
                         )), \
             patch.object(pipeline, "train", 
                         return_value=(MagicMock(), "mock_model")), \
             patch.object(pipeline.model_trainer, "save_model"), \
             patch.object(pipeline, "batch_inference", 
                         return_value=pd.DataFrame({
                             "id": [1, 2],
                             "prediction": [1750, 2650],
                             "timestamp": ["2025-01-03 15:00:00", "2025-01-03 15:01:00"]
                         })):
            
            # Mock load_data to return dataframes with expected columns
            mock_load_data.side_effect = [
                pd.DataFrame({"id": [1,2], "trip_duration": [1800, 2700]}),  # train
                pd.DataFrame({"id": [1,2]})  # test
            ]
            
            model, best_model_name, output_df = pipeline.run()
            
            # Assertions
            assert best_model_name == "mock_model"
            assert not output_df.empty
            assert "prediction" in output_df.columns
            assert "timestamp" in output_df.columns
            assert len(output_df) == 2
            mock_load_data.assert_any_call(cfg.paths.train_csv)
            mock_load_data.assert_any_call(cfg.paths.test_csv)
