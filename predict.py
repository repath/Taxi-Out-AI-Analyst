"""
predict.py
Production Inference Script for Taxi-Out Time Prediction.

Allows predicting taxi-out duration for new flights using the trained Random Forest, XGBoost, or baseline Linear Regression model
and saved preprocessors/encoders.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

CONTINUOUS_FEATURES = [
    'DEP_DELAY',
    'DISTANCE',
    'CONGESTION_1H',
    'CONGESTION_30M',
    'SIN_HOUR',
    'COS_HOUR',
    'ORIGIN_TAXI_MEAN',
    'CARRIER_TAXI_MEAN'
]

CATEGORICAL_FEATURES = [
    'OP_UNIQUE_CARRIER',
    'ORIGIN',
    'DAY_OF_WEEK'
]

def load_deployment_artifacts(model_type="rf"):
    """Loads trained pipeline and target encoder maps."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    
    if model_type == "lr":
        model_file = os.path.join(models_dir, "linear_regression_baseline.joblib")
    elif model_type == "xgb":
        model_file = os.path.join(models_dir, "xgboost_regressor.joblib")
    else:
        model_file = os.path.join(models_dir, "random_forest_regressor.joblib")
        
    encoder_file = os.path.join(models_dir, "target_encoders.json")
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found at: {model_file}. Please run 'python src/train_models.py' first.")
    if not os.path.exists(encoder_file):
        raise FileNotFoundError(f"Encoder file not found at: {encoder_file}. Please run 'python src/feature_engineering.py' first.")
        
    pipeline = joblib.load(model_file)
    # Avoid the sklearn/joblib parallel configuration warning during dashboard inference.
    # The saved model remains unchanged; only runtime worker count is set to one.
    if hasattr(pipeline, "named_steps") and "regressor" in pipeline.named_steps:
        regressor = pipeline.named_steps["regressor"]
        if hasattr(regressor, "n_jobs"):
            regressor.set_params(n_jobs=1)
    with open(encoder_file, "r") as f:
        target_encoders = json.load(f)
        
    return pipeline, target_encoders

def prepare_flight_features(flight_dict, target_encoders):
    """
    Transforms raw flight input dictionary into the required feature schema.
    flight_dict requires:
      - FL_DATE: 'YYYY-MM-DD'
      - OP_UNIQUE_CARRIER: e.g. 'AA'
      - ORIGIN: e.g. 'DFW'
      - DEST: e.g. 'ORD'
      - CRS_DEP_TIME: e.g. 830 (HHMM)
      - DEP_TIME: e.g. 845 (HHMM)
      - DEP_DELAY: e.g. 15.0 (minutes)
      - DISTANCE: e.g. 802.0 (miles)
      - CONGESTION_1H: e.g. 12 (number of departures in past hour)
      - CONGESTION_30M: (optional, defaults to CONGESTION_1H // 2)
    """
    df = pd.DataFrame([flight_dict])
    
    # 1. Temporal features
    hour = int(df['CRS_DEP_TIME'].iloc[0] // 100)
    hour = max(0, min(23, hour))
    df['HOUR'] = hour
    
    fl_date = pd.to_datetime(df['FL_DATE'].iloc[0])
    df['DAY_OF_WEEK'] = fl_date.dayofweek
    df['MONTH'] = fl_date.month
    
    df['SIN_HOUR'] = np.sin(2 * np.pi * hour / 24.0)
    df['COS_HOUR'] = np.cos(2 * np.pi * hour / 24.0)
    
    # 2. Congestion defaults if not provided
    if 'CONGESTION_1H' not in df.columns:
        df['CONGESTION_1H'] = 8.0 # default hub congestion
    if 'CONGESTION_30M' not in df.columns:
        df['CONGESTION_30M'] = df['CONGESTION_1H'] / 2.0
        
    # 3. Target Encodings
    global_mean = target_encoders.get('global_mean', 20.0)
    orig_enc = target_encoders.get('origin_encoding', {})
    carrier_enc = target_encoders.get('carrier_encoding', {})
    
    df['ORIGIN_TAXI_MEAN'] = orig_enc.get(df['ORIGIN'].iloc[0], global_mean)
    df['CARRIER_TAXI_MEAN'] = carrier_enc.get(df['OP_UNIQUE_CARRIER'].iloc[0], global_mean)
    
    feature_cols = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
    return df[feature_cols]

def predict_single_flight(flight_info, model_type="rf"):
    """
    Runs end-to-end inference on a single flight specification.
    """
    pipeline, encoders = load_deployment_artifacts(model_type=model_type)
    X = prepare_flight_features(flight_info, encoders)
    pred_taxi_out = float(pipeline.predict(X)[0])
    
    # Estimate wheels-off time
    dep_time = flight_info.get('DEP_TIME', flight_info.get('CRS_DEP_TIME'))
    h = int(dep_time) // 100
    m = int(dep_time) % 100
    tot_m = (h * 60 + m + int(round(pred_taxi_out))) % 1440
    est_wheels_off = f"{tot_m // 60:02d}:{tot_m % 60:02d}"
    
    return {
        "flight": f"{flight_info.get('OP_UNIQUE_CARRIER', 'FL')} {flight_info.get('ORIGIN', 'ORG')}->{flight_info.get('DEST', 'DST')}",
        "predicted_taxi_out_minutes": round(pred_taxi_out, 1),
        "scheduled_dep": f"{flight_info.get('CRS_DEP_TIME') // 100:02d}:{flight_info.get('CRS_DEP_TIME') % 100:02d}",
        "actual_dep_time": f"{int(dep_time) // 100:02d}:{int(dep_time) % 100:02d}",
        "estimated_wheels_off": est_wheels_off,
        "model_used": {
            "lr": "Linear Regression Baseline",
            "rf": "Random Forest Regressor",
            "xgb": "XGBoost Regressor"
        }.get(model_type, "Random Forest Regressor")
    }

if __name__ == "__main__":
    # Example operational test cases
    test_flights = [
        {
            "name": "DFW Morning Peak Push (AA to ORD)",
            "data": {
                "FL_DATE": "2025-11-15",
                "OP_UNIQUE_CARRIER": "AA",
                "ORIGIN": "DFW",
                "DEST": "ORD",
                "CRS_DEP_TIME": 830,
                "DEP_TIME": 835,
                "DEP_DELAY": 5.0,
                "DISTANCE": 802.0,
                "CONGESTION_1H": 18,
                "CONGESTION_30M": 10
            }
        },
        {
            "name": "DFW Midday Off-Peak (AA to MIA)",
            "data": {
                "FL_DATE": "2025-11-15",
                "OP_UNIQUE_CARRIER": "AA",
                "ORIGIN": "DFW",
                "DEST": "MIA",
                "CRS_DEP_TIME": 1400,
                "DEP_TIME": 1400,
                "DEP_DELAY": 0.0,
                "DISTANCE": 1121.0,
                "CONGESTION_1H": 5,
                "CONGESTION_30M": 2
            }
        },
        {
            "name": "JFK Evening Peak (B6 to LAX)",
            "data": {
                "FL_DATE": "2025-11-20",
                "OP_UNIQUE_CARRIER": "B6",
                "ORIGIN": "JFK",
                "DEST": "LAX",
                "CRS_DEP_TIME": 1845,
                "DEP_TIME": 1910,
                "DEP_DELAY": 25.0,
                "DISTANCE": 2475.0,
                "CONGESTION_1H": 22,
                "CONGESTION_30M": 12
            }
        }
    ]
    
    print("\n" + "=" * 65)
    print("TAXI-OUT PREDICTION DEMONSTRATION")
    print("=" * 65)
    for sample in test_flights:
        print(f"\nScenario: {sample['name']}")
        res = predict_single_flight(sample['data'], model_type="rf")
        for k, v in res.items():
            print(f"  {k:28s}: {v}")
