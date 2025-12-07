"""
Fast data ingestion script - loads telematics CSV into running backend.
"""

import pandas as pd
import requests
import time

API_URL = "http://localhost:8000/api/telematics/batch"

print("=" * 70)
print("📤 UPLOADING TELEMATICS DATA TO BACKEND")
print("=" * 70)

# Load CSV
data_path = r"C:\Users\ACER\Desktop\LampPost_Automotive\data\telematics_data.csv"

print(f"\n📁 Loading CSV from: {data_path}")

try:
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} records")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    exit(1)

print("\n🚀 Uploading to backend API...")
print(f"   API: {API_URL}")

try:
    # Upload via API
    with open(data_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(API_URL, files=files, timeout=60)
    
    result = response.json()
    
    print("\n" + "=" * 70)
    print("✅ UPLOAD COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Results:")
    print(f"   Total rows processed: {result['total_rows']}")
    print(f"   Successfully ingested: {result['ingested']}")
    print(f"   Unique vehicles loaded: {result['vehicles_loaded']}")
    print(f"   Errors: {result['errors']}")
    
    if result['errors'] > 0:
        print(f"\n   First few errors:")
        for err in result['error_details'][:3]:
            print(f"   - {err}")
    
    # Test dashboard API
    print("\n" + "=" * 70)
    print("🔍 TESTING DASHBOARD API")
    print("=" * 70)
    
    time.sleep(2)  # Give backend time to process
    
    overview = requests.get("http://localhost:8000/api/dashboard/overview").json()
    queue = requests.get("http://localhost:8000/api/dashboard/queue?limit=10").json()
    
    print(f"\n📊 Dashboard Overview:")
    print(f"   Vehicles Monitored: {overview['vehicles_monitored']}")
    print(f"   High Risk: {overview['risk_distribution']['high_risk']} 🔴")
    print(f"   Medium Risk: {overview['risk_distribution']['medium_risk']} 🟡")
    print(f"   Low Risk: {overview['risk_distribution']['low_risk']} 🟢")
    print(f"   Anomalies Detected: {overview['total_anomalies_detected']}")
    print(f"   Breakdowns Prevented: {overview['breakdowns_prevented_estimate']}")
    
    if queue['queue']:
        print(f"\n⚠️ High-Risk Queue ({len(queue['queue'])} vehicles):")
        for i, vehicle in enumerate(queue['queue'][:5], 1):
            print(f"   {i}. {vehicle['vehicle_id']} - Risk: {vehicle['risk_score']:.2f} ({vehicle['severity']}) | TTF: {vehicle['ttf_days']} days")
    
    print("\n" + "=" * 70)
    print("✅ DATA LOADING SUCCESSFUL!")
    print("🌐 Refresh dashboard at http://localhost:3000")
    print("=" * 70 + "\n")
    
except Exception as e:
    print(f"\n❌ Error uploading data: {e}")
    print(f"\nMake sure:")
    print(f"  1. Backend is running at http://localhost:8000")
    print(f"  2. CSV file exists at: {data_path}")
    print(f"  3. Internet connection is stable")
    exit(1)
