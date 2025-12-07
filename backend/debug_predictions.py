"""
Debug script - test on vehicles with actual failures
"""

import pandas as pd
import numpy as np
import pickle
import os

models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

print("=" * 70)
print("🔍 DIAGNOSTIC: Testing on FAILURE Vehicles")
print("=" * 70)

# Load models
with open(os.path.join(models_dir, 'xgboost_failure_model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(models_dir, 'feature_scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(models_dir, 'feature_columns.pkl'), 'rb') as f:
    feature_columns = pickle.load(f)

# Load data
data_path = r"C:\Users\ACER\Desktop\LampPost_Automotive\data\telematics_data.csv"
df = pd.read_csv(data_path)

print(f"✓ Data loaded: {len(df)} rows, {df['vehicle_id'].nunique()} vehicles")
print(f"✓ Failure cases: {df['failure_label'].sum()}")

# Find vehicles with failures
failure_vehicles = df[df['failure_label'] == 1]['vehicle_id'].unique()
print(f"✓ Vehicles with failures: {len(failure_vehicles)}")
print(f"  Examples: {failure_vehicles[:5]}")

# Feature engineering
def compute_features(readings_df):
    if len(readings_df) < 4:
        return None
    f = readings_df.copy()
    f['bearing_temp_delta'] = f['bearing_temp'].diff()
    f['engine_temp_delta'] = f['engine_temp'].diff()
    f['brake_pressure_delta'] = f['brake_pressure'].diff()
    w = 3
    f['bearing_temp_rolling_mean'] = f['bearing_temp'].rolling(w, min_periods=1).mean()
    f['bearing_temp_rolling_std'] = f['bearing_temp'].rolling(w, min_periods=1).std().fillna(0)
    f['bearing_temp_rolling_max'] = f['bearing_temp'].rolling(w, min_periods=1).max()
    f['engine_temp_rolling_mean'] = f['engine_temp'].rolling(w, min_periods=1).mean()
    f['engine_temp_rolling_std'] = f['engine_temp'].rolling(w, min_periods=1).std().fillna(0)
    f['brake_pressure_rolling_mean'] = f['brake_pressure'].rolling(w, min_periods=1).mean()
    f['brake_pressure_rolling_std'] = f['brake_pressure'].rolling(w, min_periods=1).std().fillna(0)
    f['bearing_temp_cumsum_delta'] = f['bearing_temp_delta'].cumsum()
    f['bearing_temp_anomaly_score'] = (
        (f['bearing_temp'] - f['bearing_temp_rolling_mean']) / 
        (f['bearing_temp_rolling_std'] + 1)
    )
    return f

# Test on failure vehicles
print("\n" + "=" * 70)
print("🧪 Testing Predictions on Failure Vehicles")
print("=" * 70)

predictions_normal = {}
predictions_failure = {}

for vehicle_id in df['vehicle_id'].unique():
    try:
        vehicle_data = df[df['vehicle_id'] == vehicle_id]
        
        # Get last 10 readings (when failure is most likely to be detected)
        recent_data = vehicle_data.tail(10).copy()
        
        if len(recent_data) < 4:
            continue
        
        engineered = compute_features(recent_data)
        if engineered is None:
            continue
        
        latest = engineered.iloc[-1]
        
        # Extract features
        feature_vals = [latest[col] if col in latest.index else 0 for col in feature_columns]
        X = np.array([feature_vals])
        X_scaled = scaler.transform(X)
        
        risk_score = float(model.predict_proba(X_scaled)[0, 1])
        
        # Check if this vehicle had failures
        has_failure = vehicle_data['failure_label'].max() == 1
        
        if has_failure:
            predictions_failure[vehicle_id] = risk_score
            print(f"🔴 {vehicle_id} (HAS FAILURE): Risk {risk_score:.6f}")
        else:
            predictions_normal[vehicle_id] = risk_score
            
    except Exception as e:
        pass

# Statistics
print("\n" + "=" * 70)
print("📊 Analysis Results")
print("=" * 70)

print(f"\nNormal vehicles (no failures):")
print(f"  Count: {len(predictions_normal)}")
if predictions_normal:
    normal_risks = list(predictions_normal.values())
    print(f"  Mean risk: {np.mean(normal_risks):.6f}")
    print(f"  Max risk: {np.max(normal_risks):.6f}")

print(f"\nFailure vehicles (with failures):")
print(f"  Count: {len(predictions_failure)}")
if predictions_failure:
    failure_risks = list(predictions_failure.values())
    print(f"  Mean risk: {np.mean(failure_risks):.6f}")
    print(f"  Max risk: {np.max(failure_risks):.6f}")
    print(f"  Min risk: {np.min(failure_risks):.6f}")
    
    # Check separation
    avg_normal = np.mean(list(predictions_normal.values())) if predictions_normal else 0
    avg_failure = np.mean(failure_risks)
    
    print(f"\n🎯 Model Discrimination:")
    print(f"  Average normal risk: {avg_normal:.6f}")
    print(f"  Average failure risk: {avg_failure:.6f}")
    print(f"  Separation: {avg_failure - avg_normal:.6f}")
    
    if avg_failure > avg_normal:
        print(f"  ✅ Model correctly separates failure vs normal!")
    else:
        print(f"  ⚠️  Model doesn't separate well - all risks too low")

print("\n" + "=" * 70)
