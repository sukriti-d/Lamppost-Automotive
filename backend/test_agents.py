"""
Test agentic orchestration workflow.
"""

from agents_orchestration import MasterAgent
import json

print("\n" + "=" * 70)
print("🧪 TESTING AGENTIC ORCHESTRATION")
print("=" * 70)

# Create master agent
master = MasterAgent()

# Test data: Vehicle with bearing failure trend
vehicle_data = {
    'vehicle_id': 'VH-085',
    'model': 'Maruti Swift',
    'owner_name': 'Rajesh Sharma',
    'owner_phone': '+91-98765-43210',
    'service_center': 'XYZ Motors, Kharadi'
}

telematics_readings = [
    {'bearing_temp': 38, 'engine_temp': 92, 'rpm': 2500, 'brake_pressure': 45},
    {'bearing_temp': 42, 'engine_temp': 93, 'rpm': 2400, 'brake_pressure': 45},
    {'bearing_temp': 48, 'engine_temp': 94, 'rpm': 2600, 'brake_pressure': 44},
    {'bearing_temp': 55, 'engine_temp': 95, 'rpm': 2700, 'brake_pressure': 43},
    {'bearing_temp': 62, 'engine_temp': 96, 'rpm': 2800, 'brake_pressure': 42},
]

ml_prediction = {
    'risk_score': 0.95,
    'predicted_failure_type': 'bearing',
    'time_to_failure_days': 2
}

# Run workflow
result = master.orchestrate_workflow(vehicle_data, telematics_readings, ml_prediction)

# Print result
print("\n" + "=" * 70)
print("✅ WORKFLOW COMPLETE")
print("=" * 70)

print("\nWorkflow Decisions:")
for agent, decision in result['agent_decisions'].items():
    print(f"\n{agent.upper()}:")
    print(f"  {json.dumps(decision, indent=4, default=str)[:200]}...")

print("\nSecurity Audit Trail:")
for entry in result.get('audit_trail', []):
    print(f"  {entry['timestamp']}: {entry['agent_name']} - {entry['action']} - {entry['decision']}")

print("\n" + "=" * 70)
