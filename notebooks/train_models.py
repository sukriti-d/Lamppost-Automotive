"""
Train XGBoost failure prediction model using telematics data.
This model learns to predict mechanical failures 2-3 days in advance.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import pickle
import os

print("=" * 60)
print("🤖 TRAINING FAILURE PREDICTION MODEL")
print("=" * 60)

# Load data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'telematics_data.csv')
df = pd.read_csv(data_path)
print(f"\n📊 Loaded {len(df)} records from telematics data")

# ============================================
# FEATURE ENGINEERING
# ============================================
print("\n🔧 Performing feature engineering...")

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sort by vehicle and timestamp
df = df.sort_values(['vehicle_id', 'timestamp']).reset_index(drop=True)

# For each vehicle, compute rolling statistics (trends over time)
def engineer_features(vehicle_df):
    """
    Create advanced features from raw telematics.
    Rolling stats capture sensor trends (e.g., temperature rising).
    """
    features = vehicle_df.copy()
    
    # 1. RATE OF CHANGE (how fast sensor values changing?)
    features['bearing_temp_delta'] = features['bearing_temp'].diff()
    features['engine_temp_delta'] = features['engine_temp'].diff()
    features['brake_pressure_delta'] = features['brake_pressure'].diff()
    
    # 2. ROLLING STATISTICS (3-hour window = last 3 readings)
    window = 3
    features['bearing_temp_rolling_mean'] = features['bearing_temp'].rolling(window, min_periods=1).mean()
    features['bearing_temp_rolling_std'] = features['bearing_temp'].rolling(window, min_periods=1).std().fillna(0)
    features['bearing_temp_rolling_max'] = features['bearing_temp'].rolling(window, min_periods=1).max()
    
    features['engine_temp_rolling_mean'] = features['engine_temp'].rolling(window, min_periods=1).mean()
    features['engine_temp_rolling_std'] = features['engine_temp'].rolling(window, min_periods=1).std().fillna(0)
    
    features['brake_pressure_rolling_mean'] = features['brake_pressure'].rolling(window, min_periods=1).mean()
    features['brake_pressure_rolling_std'] = features['brake_pressure'].rolling(window, min_periods=1).std().fillna(0)
    
    # 3. CUMULATIVE STATISTICS (overall trend)
    features['bearing_temp_cumsum_delta'] = features['bearing_temp_delta'].cumsum()
    
    # 4. ANOMALY INDICATOR (deviation from rolling mean)
    features['bearing_temp_anomaly_score'] = (
        (features['bearing_temp'] - features['bearing_temp_rolling_mean']) / 
        (features['bearing_temp_rolling_std'] + 1)
    )
    
    return features

# Apply feature engineering to each vehicle
engineered_data = []
for vehicle_id in df['vehicle_id'].unique():
    vehicle_data = df[df['vehicle_id'] == vehicle_id].copy()
    engineered_vehicle = engineer_features(vehicle_data)
    engineered_data.append(engineered_vehicle)

df_engineered = pd.concat(engineered_data, ignore_index=True)

# Drop rows with NaN (first few readings of each vehicle have NaN from rolling stats)
df_engineered = df_engineered.dropna()

print(f"✓ Engineered features: {df_engineered.shape[1]} total columns")
print(f"✓ Clean records for training: {len(df_engineered)}")

# ============================================
# PREPARE TRAINING DATA
# ============================================
print("\n🎯 Preparing training data...")

# Select features for model
feature_columns = [
    'engine_temp', 'rpm', 'brake_pressure', 'bearing_temp', 'fuel_level', 'odometer',
    'bearing_temp_delta', 'engine_temp_delta', 'brake_pressure_delta',
    'bearing_temp_rolling_mean', 'bearing_temp_rolling_std', 'bearing_temp_rolling_max',
    'engine_temp_rolling_mean', 'engine_temp_rolling_std',
    'brake_pressure_rolling_mean', 'brake_pressure_rolling_std',
    'bearing_temp_cumsum_delta', 'bearing_temp_anomaly_score'
]

X = df_engineered[feature_columns]
y = df_engineered['failure_label']

print(f"✓ Features: {X.shape[1]} columns")
print(f"✓ Samples: {len(X)} rows")
print(f"✓ Failure rate: {y.mean()*100:.2f}% (imbalanced - normal, which is realistic)")

# Normalize features (scale to 0-1 range)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Train samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")

# ============================================
# TRAIN XGBOOST MODEL
# ============================================
print("\n🚀 Training XGBoost model (this may take 1-2 minutes)...")

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    verbosity=0
)

# Train without early stopping (more compatible)
model.fit(X_train, y_train, verbose=False)

print("✓ Model training complete!")

# ============================================
# EVALUATE MODEL
# ============================================
print("\n📊 Model Performance:")
print("-" * 60)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Metrics
print(f"Accuracy: {model.score(X_test, y_test):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Failure']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
print("\n🎯 Top 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:<35} {row['importance']:.4f}")

# ============================================
# SAVE MODEL & SCALER
# ============================================
print("\n💾 Saving models...")

models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(models_dir, exist_ok=True)

# Save XGBoost model
model_path = os.path.join(models_dir, 'xgboost_failure_model.pkl')
pickle.dump(model, open(model_path, 'wb'))
print(f"✓ XGBoost model saved: {model_path}")

# Save scaler
scaler_path = os.path.join(models_dir, 'feature_scaler.pkl')
pickle.dump(scaler, open(scaler_path, 'wb'))
print(f"✓ Feature scaler saved: {scaler_path}")

# Save feature columns
features_path = os.path.join(models_dir, 'feature_columns.pkl')
pickle.dump(feature_columns, open(features_path, 'wb'))
print(f"✓ Feature columns saved: {features_path}")

print("\n" + "=" * 60)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 60)
