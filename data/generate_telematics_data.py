"""
Generate synthetic telematics data for model training & demo.
This simulates 100 vehicles over 30 days with realistic sensor readings.
Some vehicles trend toward failures (bearing, brake, engine issues).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_normal_vehicle_data(vehicle_id, days=30, anomaly=None):
    """
    Generate normal telematics readings for a vehicle.
    
    Args:
        vehicle_id: Unique vehicle identifier (e.g., "MH-02-AB-5847")
        days: Number of days to simulate
        anomaly: Type of anomaly ('bearing_failure', 'brake_failure', 'engine_failure', None)
    
    Returns:
        DataFrame with telematics readings
    """
    
    data = []
    start_time = datetime(2025, 11, 1, 8, 0)
    
    # Each vehicle generates ~24 readings/day (hourly intervals during driving hours)
    for day in range(days):
        # 8 AM to 8 PM driving (assuming ~1 trip per hour = 12 hours)
        for hour in range(0, 12):  # 0-11 = 8 AM to 7 PM
            timestamp = start_time + timedelta(days=day, hours=hour)
            
            # Normal baseline values
            engine_temp = np.random.normal(90, 5)  # Mean 90°C, std 5°C
            rpm = np.random.normal(2500, 400)
            brake_pressure = np.random.normal(45, 2)
            bearing_temp = np.random.normal(38, 3)
            fuel_level = max(10, np.random.normal(60, 10))
            odometer = 40000 + (day * 50) + np.random.normal(0, 10)
            
            # Inject anomalies based on failure type
            if anomaly == 'bearing_failure' and day >= 20:
                # Bearing temperature trends upward as day approaches day 25
                bearing_temp += (day - 20) * 2 + np.random.normal(0, 1)
                bearing_temp = min(65, bearing_temp)  # Cap at 65°C
                
            elif anomaly == 'brake_failure' and day >= 20:
                # Brake pressure fluctuates and drops
                brake_pressure -= (day - 20) * 0.5 + np.random.normal(0, 0.5)
                brake_pressure = max(30, brake_pressure)
                
            elif anomaly == 'engine_failure' and day >= 20:
                # Engine temperature spikes and RPM becomes erratic
                engine_temp += (day - 20) * 1.5 + np.random.normal(0, 2)
                rpm += np.random.normal(0, 200)
            
            # Determine if failure occurred (only after day 25 with anomalies)
            failure_label = 1 if (anomaly and day >= 25) else 0
            
            data.append({
                'timestamp': timestamp,
                'vehicle_id': vehicle_id,
                'engine_temp': round(engine_temp, 2),
                'rpm': round(rpm, 0),
                'brake_pressure': round(brake_pressure, 2),
                'bearing_temp': round(bearing_temp, 2),
                'fuel_level': round(fuel_level, 1),
                'odometer': round(odometer, 0),
                'failure_label': failure_label
            })
    
    return pd.DataFrame(data)

# Generate data for 100 vehicles
print("🚗 Generating telematics data for 100 vehicles...")

all_data = []

# Generate 85 normal vehicles (no failures)
for i in range(85):
    vehicle_id = f"VH-{i:03d}"
    df = generate_normal_vehicle_data(vehicle_id, days=30, anomaly=None)
    all_data.append(df)
    print(f"✓ Generated {vehicle_id} (Normal)")

# Generate 5 bearing failure vehicles
for i in range(85, 90):
    vehicle_id = f"VH-{i:03d}"
    df = generate_normal_vehicle_data(vehicle_id, days=30, anomaly='bearing_failure')
    all_data.append(df)
    print(f"✓ Generated {vehicle_id} (Bearing Failure)")

# Generate 5 brake failure vehicles
for i in range(90, 95):
    vehicle_id = f"VH-{i:03d}"
    df = generate_normal_vehicle_data(vehicle_id, days=30, anomaly='brake_failure')
    all_data.append(df)
    print(f"✓ Generated {vehicle_id} (Brake Failure)")

# Generate 5 engine failure vehicles
for i in range(95, 100):
    vehicle_id = f"VH-{i:03d}"
    df = generate_normal_vehicle_data(vehicle_id, days=30, anomaly='engine_failure')
    all_data.append(df)
    print(f"✓ Generated {vehicle_id} (Engine Failure)")

# Combine all data
final_df = pd.concat(all_data, ignore_index=True)

# Save to CSV
output_path = os.path.join(os.path.dirname(__file__), 'telematics_data.csv')
final_df.to_csv(output_path, index=False)

print(f"\n✅ Data generation complete!")
print(f"📊 Total records: {len(final_df)}")
print(f"🚗 Unique vehicles: {final_df['vehicle_id'].nunique()}")
print(f"⚠️ Failure cases: {final_df['failure_label'].sum()}")
print(f"💾 Saved to: {output_path}")

# Display sample
print("\n📋 Sample data:")
print(final_df.head(10))
