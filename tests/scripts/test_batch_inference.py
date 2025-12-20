import os
import pandas as pd
import numpy as np
from datetime import datetime

from scripts.batch_inference import batch_inference
from scripts.feature_engineering import feature_engineering


class DummyModel:
    def predict(self, X):
        return np.ones(len(X)) * 5.0


def build_df():
    return pd.DataFrame({
        "id":[1,2,3],
        "pickup_latitude":[40.7,40.8,40.76],
        "pickup_longitude":[-73.9,-73.95,-73.97],
        "dropoff_latitude":[40.75,40.82,40.81],
        "dropoff_longitude":[-73.98,-73.96,-73.99],
        "pickup_datetime":[
            "2025-12-20 10:00:00",
            "2025-12-21 12:00:00",
            "2025-12-22 08:00:00",
        ],
        "passenger_count":[1,2,1],
        "vendor_id":[1,2,2],
        "store_and_fwd_flag":["N","Y",None],
    })


# fixture to fit encoders before inference
def fit_feature_engineering():
    df = build_df()
    feature_engineering(df.copy(), fit=True, save=False, is_train=False)



def test_batch_inference_output():
    fit_feature_engineering()

    df = build_df()
    model = DummyModel()

    result = batch_inference(model, df.copy())

    assert "prediction" in result.columns
    assert "timestamp" in result.columns
    assert result["prediction"].eq(5.0).all()



def test_batch_inference_save_dir(tmp_path):
    fit_feature_engineering()

    df = build_df()
    model = DummyModel()

    save_dir = tmp_path / "preds"
    save_dir.mkdir()

    result = batch_inference(model, df.copy(), save_path=str(save_dir))

    saved_files = list(save_dir.iterdir())
    assert len(saved_files) == 1
    assert saved_files[0].suffix == ".csv"



def test_batch_inference_save_file(tmp_path):
    fit_feature_engineering()

    df = build_df()
    model = DummyModel()

    save_file = tmp_path / "preds.csv"

    batch_inference(model, df.copy(), save_path=str(save_file))

    assert os.path.exists(save_file)

    loaded = pd.read_csv(save_file)
    assert "prediction" in loaded.columns
