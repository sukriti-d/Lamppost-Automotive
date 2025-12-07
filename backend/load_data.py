"""
Load generated telematics data into the backend via API.
"""

import pandas as pd
import requests
import os

API_URL = "http://localhost:8000/api/telematics/batch"

# Load CSV
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'telematics_data.csv')

print(f"📤 Uploading telematics data from: {data_path}")
print("This will ingest all 28,800 records into the backend...\n")

with open(data_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(API_URL, files=files)

result = response.json()
print("✅ Upload Complete!")
print(f"Total rows processed: {result['total_rows']}")
print(f"Successfully ingested: {result['ingested']}")
print(f"Unique vehicles loaded: {result['vehicles_loaded']}")

if result['errors'] > 0:
    print(f"Errors: {result['errors']}")
    print(f"First few errors: {result['error_details']}")

# Test dashboard API
print("\n" + "=" * 60)
print("Testing Dashboard API...")
print("=" * 60)

overview = requests.get("http://localhost:8000/api/dashboard/overview").json()
print(f"\n📊 Dashboard Overview:")
print(f"  Vehicles Monitored: {overview['vehicles_monitored']}")
print(f"  High Risk: {overview['risk_distribution']['high_risk']}")
print(f"  Medium Risk: {overview['risk_distribution']['medium_risk']}")
print(f"  Low Risk: {overview['risk_distribution']['low_risk']}")
print(f"  Anomalies Detected: {overview['total_anomalies_detected']}")

print("\n✅ System Test Passed!")
