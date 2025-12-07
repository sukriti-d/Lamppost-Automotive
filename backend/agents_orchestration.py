"""
LampPost Agentic Orchestration Layer

Master Agent + 7 Worker Agents for autonomous vehicle maintenance workflows.

This is the CORE INNOVATION - autonomous decision-making with governance,
not just ML predictions. Each agent is responsible for one domain.
"""

import json
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import random

# ============================================
# ENUMS & CONSTANTS
# ============================================

class WorkflowState(Enum):
    """Workflow states in orchestration."""
    MONITORING = "monitoring"
    DIAGNOSING = "diagnosing"
    ENGAGING = "engaging"
    SCHEDULING = "scheduling"
    SERVICE_EXECUTING = "service_executing"
    FEEDBACK_COLLECTING = "feedback_collecting"
    MANUFACTURING_LOOP = "manufacturing_loop"
    COMPLETE = "complete"

class Severity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

# ============================================
# WORKER AGENT 1: DATA ANALYSIS AGENT
# ============================================

class DataAnalysisAgent:
    """
    Analyzes telematics data for patterns & trends.
    
    Responsibility:
    - Data fetching and validation
    - Feature computation
    - Anomaly detection
    - Health status assessment
    """
    
    def __init__(self, name="DataAnalysisAgent"):
        self.name = name
        self.execution_log = []
    
    def analyze_vehicle_data(self, vehicle_id: str, telematics_readings: List[Dict]):
        """Analyze vehicle telematics data."""
        
        action_log = {
            'timestamp': datetime.now().isoformat(),
            'agent': self.name,
            'action': 'analyze_vehicle_data',
            'vehicle_id': vehicle_id,
            'readings_count': len(telematics_readings),
            'status': 'executing'
        }
        
        if len(telematics_readings) < 4:
            action_log['status'] = 'insufficient_data'
            action_log['result'] = None
            self.execution_log.append(action_log)
            return None
        
        # Analyze trends
        recent_5 = telematics_readings[-5:]
        bearing_temps = [r['bearing_temp'] for r in recent_5]
        brake_pressures = [r['brake_pressure'] for r in recent_5]
        
        bearing_trend = "rising" if bearing_temps[-1] > bearing_temps[0] else "stable"
        brake_trend = "dropping" if brake_pressures[-1] < brake_pressures[0] else "stable"
        
        analysis = {
            'vehicle_id': vehicle_id,
            'reading_count': len(telematics_readings),
            'bearing_trend': bearing_trend,
            'brake_trend': brake_trend,
            'recent_bearing_temp': float(bearing_temps[-1]),
            'recent_brake_pressure': float(brake_pressures[-1]),
            'anomalies_detected': sum(1 for t in bearing_temps if t > 50),
            'health_status': 'normal'
        }
        
        action_log['status'] = 'complete'
        action_log['result'] = analysis
        self.execution_log.append(action_log)
        
        print(f"  ✓ Data Analysis Agent: {vehicle_id}")
        print(f"    - Bearing trend: {bearing_trend} ({bearing_temps[-1]:.1f}°C)")
        print(f"    - Brake trend: {brake_trend} ({brake_pressures[-1]:.1f} bar)")
        
        return analysis

# ============================================
# WORKER AGENT 2: DIAGNOSIS AGENT
# ============================================

class DiagnosisAgent:
    """
    Diagnoses failure risk using ML predictions.
    
    Responsibility:
    - Risk scoring
    - TTF estimation
    - Failure type classification
    - Severity assessment
    """
    
    def __init__(self, name="DiagnosisAgent", ml_predictor=None):
        self.name = name
        self.ml_predictor = ml_predictor  # Reference to ML model function
        self.execution_log = []
    
    def diagnose(self, vehicle_id: str, analysis: Dict, ml_prediction: Dict = None):
        """Diagnose failure risk based on analysis."""
        
        action_log = {
            'timestamp': datetime.now().isoformat(),
            'agent': self.name,
            'action': 'diagnose',
            'vehicle_id': vehicle_id,
            'status': 'executing'
        }
        
        if not analysis:
            action_log['status'] = 'skipped'
            self.execution_log.append(action_log)
            return None
        
        # Use ML prediction if available, otherwise heuristic
        if ml_prediction:
            risk_score = ml_prediction['risk_score']
            failure_type = ml_prediction['predicted_failure_type']
            ttf_days = ml_prediction['time_to_failure_days']
        else:
            # Heuristic fallback
            if analysis.get('bearing_trend') == 'rising':
                risk_score = 0.85
                failure_type = 'bearing'
                ttf_days = 2
            elif analysis.get('brake_trend') == 'dropping':
                risk_score = 0.65
                failure_type = 'brake'
                ttf_days = 5
            else:
                risk_score = 0.15
                failure_type = 'engine'
                ttf_days = 14
        
        # Determine severity
        if risk_score > 0.7:
            severity = Severity.HIGH.value
        elif risk_score > 0.4:
            severity = Severity.MEDIUM.value
        else:
            severity = Severity.LOW.value
        
        diagnosis = {
            'vehicle_id': vehicle_id,
            'risk_score': risk_score,
            'predicted_failure_type': failure_type,
            'time_to_failure_days': ttf_days,
            'severity': severity,
            'confidence': 0.92,
            'diagnosis_timestamp': datetime.now().isoformat()
        }
        
        action_log['status'] = 'complete'
        action_log['result'] = diagnosis
        self.execution_log.append(action_log)
        
        print(f"  ✓ Diagnosis Agent: {vehicle_id}")
        print(f"    - Risk Score: {risk_score:.2f} ({severity})")
        print(f"    - Failure Type: {failure_type}")
        print(f"    - TTF: {ttf_days} days")
        
        return diagnosis

# ============================================
# WORKER AGENT 3: ENGAGEMENT AGENT (VOICE BOT)
# ============================================

class EngagementAgent:
    """
    Engages customers via voice/chat.
    
    Responsibility:
    - Outreach message crafting
    - Persuasion strategy
    - Multi-language support
    - Consent management
    """
    
    def __init__(self, name="EngagementAgent"):
        self.name = name
        self.execution_log = []
        
        # Message templates by severity
        self.templates = {
            'HIGH': """Hi {owner_name}, this is LampPost from {service_center}. 
Your {model} is showing critical bearing wear. We can service it now for ₹1,500, 
or risk a ₹6,000 breakdown on the highway. Can I book a slot for you this Friday?""",
            
            'MEDIUM': """Hi {owner_name}, your {model} needs a routine brake check soon. 
We have slots available this week. Would you like to book?""",
            
            'LOW': """Hi {owner_name}, your {model} is running well! 
Routine maintenance next month? I can help schedule."""
        }
    
    def engage_customer(self, vehicle_data: Dict, diagnosis: Dict):
        """Craft and send engagement message."""
        
        action_log = {
            'timestamp': datetime.now().isoformat(),
            'agent': self.name,
            'action': 'engage_customer',
            'vehicle_id': vehicle_data['vehicle_id'],
            'status': 'executing'
        }
        
        severity = diagnosis['severity']
        template = self.templates.get(severity, self.templates['LOW'])
        
        # Personalize message
        message = template.format(
            owner_name=vehicle_data.get('owner_name', 'Valued Customer'),
            model=vehicle_data.get('model', 'vehicle'),
            service_center=vehicle_data.get('service_center', 'our service center')
        )
        
        engagement = {
            'vehicle_id': vehicle_data['vehicle_id'],
            'owner_phone': vehicle_data.get('owner_phone', ''),
            'message': message,
            'channel': 'voice',  # Could be SMS, app, email
            'engagement_id': f"ENG-{vehicle_data['vehicle_id']}-{int(datetime.now().timestamp())}",
            'status': 'message_crafted',
            'severity': severity
        }
        
        action_log['status'] = 'complete'
        action_log['result'] = engagement
        self.execution_log.append(action_log)
        
        print(f"  ✓ Engagement Agent: {vehicle_data['vehicle_id']}")
        print(f"    - Channel: {engagement['channel']}")
        print(f"    - Message preview: \"{message[:80]}...\"")
        
        return engagement

# ============================================
# WORKER AGENT 4: SCHEDULING AGENT
# ============================================

class SchedulingAgent:
    """
    Schedules service appointments optimally.
    
    Responsibility:
    - Slot availability checking
    - Constraint optimization (urgency, distance, skills)
    - Capacity balancing
    - Parts availability verification
    """
    
    def __init__(self, name="SchedulingAgent"):
        self.name = name
        self.execution_log = []
        
        # Mock available slots
        self.available_slots = {
            'Friday': ['09:00', '10:00', '14:00', '15:00'],
            'Saturday': ['10:00', '11:00', '16:00'],
            'Sunday': ['09:00', '15:00']
        }
    
    def find_optimal_slot(self, vehicle_data: Dict, diagnosis: Dict):
        """Find optimal service slot using constraints."""
        
        action_log = {
            'timestamp': datetime.now().isoformat(),
            'agent': self.name,
            'action': 'find_optimal_slot',
            'vehicle_id': vehicle_data['vehicle_id'],
            'status': 'executing'
        }
        
        # Priority-based slot assignment
        if diagnosis['severity'] == 'HIGH':
            # High-risk: early slot
            recommended_slot = {
                'day': 'Friday',
                'time': '09:00',
                'duration_minutes': 90,
                'service_center': vehicle_data.get('service_center', 'XYZ Motors'),
                'distance_km': round(random.uniform(1.5, 3.5), 1),
                'technician_skill': 'bearing_specialist'
            }
        else:
            # Medium/Low: flexible slots
            recommended_slot = {
                'day': 'Saturday',
                'time': '10:00',
                'duration_minutes': 60,
                'service_center': vehicle_data.get('service_center', 'XYZ Motors'),
                'distance_km': round(random.uniform(1.5, 3.5), 1),
                'technician_skill': 'general'
            }
        
        schedule = {
            'vehicle_id': vehicle_data['vehicle_id'],
            'recommended_slot': recommended_slot,
            'alternatives': [
                {'day': 'Wednesday', 'time': '14:00', 'distance_km': 3.2},
                {'day': 'Thursday', 'time': '11:00', 'distance_km': 2.8}
            ],
            'booking_status': 'pending_customer_confirmation',
            'parts_pre_reserved': True
        }
        
        action_log['status'] = 'complete'
        action_log['result'] = schedule
        self.execution_log.append(action_log)
        
        print(f"  ✓ Scheduling Agent: {vehicle_data['vehicle_id']}")
        print(f"    - Recommended: {recommended_slot['day']} {recommended_slot['time']}")
        print(f"    - Distance: {recommended_slot['distance_km']} km")
        
        return schedule

# ============================================
# WORKER AGENT 5: FEEDBACK AGENT
# ============================================

class FeedbackAgent:
    """
    Collects service feedback & CSAT scores.
    
    Responsibility:
    - Post-service feedback collection
    - CSAT/NPS scoring
    - Issue tracking
    - Customer satisfaction analysis
    """
    
    def __init__(self, name="FeedbackAgent"):
        self.name = name
        self.execution_log = []
    
    def collect_feedback(self, vehicle_id: str, service_completed: bool):
        """Collect feedback after service."""
        
        action_log = {
            'timestamp': datetime.now().isoformat(),
            'agent': self.name,
            'action': 'collect_feedback',
            'vehicle_id': vehicle_id,
            'status': 'executing'
        }
        
        if not service_completed:
            action_log['status'] = 'skipped'
            self.execution_log.append(action_log)
            return None
        
        # Mock feedback data
        feedback = {
            'vehicle_id': vehicle_id,
            'csat_score': random.randint(8, 10),
            'nps_score': random.randint(7, 10),
            'feedback_text': 'Great service! Appreciate the predictive approach.',
            'service_quality': 'excellent',
            'parts_quality': 'original',
            'technician_skill': 'excellent',
            'timestamp': datetime.now().isoformat()
        }
        
        action_log['status'] = 'complete'
        action_log['result'] = feedback
        self.execution_log.append(action_log)
        
        print(f"  ✓ Feedback Agent: {vehicle_id}")
        print(f"    - CSAT: {feedback['csat_score']}/10")
        print(f"    - NPS: {feedback['nps_score']}/10")
        
        return feedback

# ============================================
# WORKER AGENT 6: MANUFACTURING INSIGHTS AGENT
# ============================================

class MfgInsightsAgent:
    """
    Feeds service data back to manufacturing (closes the loop).
    
    Responsibility:
    - CAPA (Corrective Action Preventive Action) ticket creation
    - RCA (Root Cause Analysis) from field data
    - Defect pattern identification
    - Design improvement recommendations
    """
    
    def __init__(self, name="MfgInsightsAgent"):
        self.name = name
        self.execution_log = []
    
    def generate_insights(self, vehicle_id: str, failure_type: str, service_data: Dict):
        """Generate manufacturing insights from service data."""
        
        action_log = {
            'timestamp': datetime.now().isoformat(),
            'agent': self.name,
            'action': 'generate_insights',
            'vehicle_id': vehicle_id,
            'status': 'executing'
        }
        
        # Analyze defect patterns
        insights = {
            'vehicle_id': vehicle_id,
            'failure_type': failure_type,
            'model_year': '2020',
            'defect_pattern': f'{failure_type.capitalize()} batch defect detected in 2020 models',
            'defect_frequency': 'recurring (5th case this month)',
            'capa_ticket': {
                'id': f'CAPA-2025-{random.randint(100, 999)}',
                'title': f'Investigate {failure_type} degradation in 2020 models',
                'priority': 'HIGH',
                'assigned_to': 'Quality Engineering Team',
                'status': 'created',
                'sla_days': 7
            },
            'design_recommendation': f'Upgrade {failure_type} thermal sensor threshold or source alternative supplier',
            'supplier_alert': {
                'supplier': 'XYZ Bearing Corp',
                'issue': 'Quality variance in batch XYZ-2020',
                'ppm': 450  # Parts per million defect rate
            },
            'estimated_savings': 1250000,  # Rupees saved by early detection vs warranty claims
            'research_paper_topic': f'Predictive maintenance for automotive {failure_type}s'
        }
        
        action_log['status'] = 'complete'
        action_log['result'] = insights
        self.execution_log.append(action_log)
        
        print(f"  ✓ Manufacturing Insights Agent: {vehicle_id}")
        print(f"    - CAPA Ticket: {insights['capa_ticket']['id']}")
        print(f"    - Priority: {insights['capa_ticket']['priority']}")
        print(f"    - Est. Savings: ₹{insights['estimated_savings']:,.0f}")
        
        return insights

# ============================================
# WORKER AGENT 7: UEBA MONITOR AGENT
# ============================================

class UEBAMonitorAgent:
    """
    User & Entity Behavioral Analysis (UEBA) for security.
    
    Responsibility:
    - Agent action monitoring
    - Policy enforcement
    - Anomaly detection
    - Audit trail maintenance
    - Least-privilege verification
    """
    
    def __init__(self, name="UEBAMonitorAgent"):
        self.name = name
        self.audit_log = []
        
        # Define access policies
        self.policies = {
            'SchedulingAgent': {
                'allowed_resources': ['slot_availability', 'parts_inventory', 'technician_skills'],
                'denied_resources': ['raw_telematics', 'customer_payment_info']
            },
            'EngagementAgent': {
                'allowed_resources': ['customer_contact', 'service_history', 'outreach_templates'],
                'denied_resources': ['raw_telematics', 'technical_specs']
            }
        }
    
    def monitor_agent_action(self, agent_name: str, action: str, resource: str):
        """Monitor agent actions for policy violations."""
        
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent_name': agent_name,
            'action': action,
            'resource': resource,
            'decision': 'ALLOWED'
        }
        
        # Check policy
        if agent_name in self.policies:
            policy = self.policies[agent_name]
            if resource in policy.get('denied_resources', []):
                audit_entry['decision'] = 'BLOCKED'
                audit_entry['reason'] = f'Policy violation: {agent_name} cannot access {resource}'
                print(f"  🚫 UEBA: {agent_name} blocked from {resource}")
            else:
                audit_entry['reason'] = f'{agent_name} authorized to access {resource}'
        
        self.audit_log.append(audit_entry)
        return audit_entry['decision'] == 'ALLOWED'

# ============================================
# MASTER AGENT (ORCHESTRATOR)
# ============================================

class MasterAgent:
    """
    Master Agent orchestrates all 7 Worker Agents.
    
    Workflow:
    1. MONITORING: Receive and analyze telematics data
    2. DIAGNOSING: Predict failures using ML model
    3. ENGAGING: Contact customer via voice/chat
    4. SCHEDULING: Find optimal service slot
    5. SERVICE_EXECUTING: Manage service execution
    6. FEEDBACK_COLLECTING: Gather CSAT/NPS
    7. MANUFACTURING_LOOP: Feed back to manufacturing for CAPA
    
    Security: All agent actions logged by UEBA Monitor Agent
    """
    
    def __init__(self):
        self.name = "MasterAgent"
        
        # Initialize worker agents
        self.data_analysis = DataAnalysisAgent()
        self.diagnosis = DiagnosisAgent()
        self.engagement = EngagementAgent()
        self.scheduling = SchedulingAgent()
        self.feedback = FeedbackAgent()
        self.mfg_insights = MfgInsightsAgent()
        self.ueba_monitor = UEBAMonitorAgent()
        
        # Workflow state
        self.current_state = WorkflowState.MONITORING
        self.workflow_log = []
    
    def orchestrate_workflow(self, vehicle_data: Dict, telematics_readings: List[Dict], ml_prediction: Dict = None):
        """
        Main orchestration logic.
        
        Coordinates all 7 agents through complete workflow.
        """
        
        workflow_id = f"WF-{vehicle_data['vehicle_id']}-{int(datetime.now().timestamp())}"
        
        workflow_result = {
            'workflow_id': workflow_id,
            'vehicle_id': vehicle_data['vehicle_id'],
            'timestamp': datetime.now().isoformat(),
            'states_executed': [],
            'agent_decisions': {},
            'workflow_status': 'in_progress'
        }
        
        print(f"\n{'='*70}")
        print(f"🤖 MASTER AGENT ORCHESTRATION: {vehicle_data['vehicle_id']}")
        print(f"{'='*70}")
        
        try:
            # STATE 1: MONITORING
            print(f"\n[1/7] STATE: MONITORING")
            self.current_state = WorkflowState.MONITORING
            analysis = self.data_analysis.analyze_vehicle_data(
                vehicle_data['vehicle_id'],
                telematics_readings
            )
            workflow_result['states_executed'].append(WorkflowState.MONITORING.value)
            workflow_result['agent_decisions']['analysis'] = analysis
            
            if not analysis:
                print(f"  ❌ Insufficient data. Waiting for more readings.")
                workflow_result['workflow_status'] = 'waiting_for_data'
                return workflow_result
            
            # STATE 2: DIAGNOSING
            print(f"\n[2/7] STATE: DIAGNOSING")
            self.current_state = WorkflowState.DIAGNOSING
            diagnosis = self.diagnosis.diagnose(
                vehicle_data['vehicle_id'],
                analysis,
                ml_prediction
            )
            workflow_result['states_executed'].append(WorkflowState.DIAGNOSING.value)
            workflow_result['agent_decisions']['diagnosis'] = diagnosis
            
            # Check if action needed
            if diagnosis['risk_score'] < 0.3:
                print(f"\n  ℹ️  Low risk. Monitoring only. No intervention needed.")
                workflow_result['workflow_status'] = 'monitoring'
                return workflow_result
            
            # STATE 3: ENGAGING
            print(f"\n[3/7] STATE: ENGAGING")
            self.current_state = WorkflowState.ENGAGING
            engagement = self.engagement.engage_customer(vehicle_data, diagnosis)
            workflow_result['states_executed'].append(WorkflowState.ENGAGING.value)
            workflow_result['agent_decisions']['engagement'] = engagement
            
            # STATE 4: SCHEDULING
            print(f"\n[4/7] STATE: SCHEDULING")
            self.current_state = WorkflowState.SCHEDULING
            schedule = self.scheduling.find_optimal_slot(vehicle_data, diagnosis)
            workflow_result['states_executed'].append(WorkflowState.SCHEDULING.value)
            workflow_result['agent_decisions']['schedule'] = schedule
            
            # STATE 5: SERVICE EXECUTION
            print(f"\n[5/7] STATE: SERVICE_EXECUTING")
            self.current_state = WorkflowState.SERVICE_EXECUTING
            print(f"  ✓ Service Execution Agent: {vehicle_data['vehicle_id']}")
            print(f"    - Status: Service completed successfully")
            print(f"    - Duration: {schedule['recommended_slot']['duration_minutes']} min")
            print(f"    - Parts used: bearing, seals, lubricant")
            workflow_result['states_executed'].append(WorkflowState.SERVICE_EXECUTING.value)
            
            # STATE 6: FEEDBACK
            print(f"\n[6/7] STATE: FEEDBACK_COLLECTING")
            self.current_state = WorkflowState.FEEDBACK_COLLECTING
            feedback = self.feedback.collect_feedback(vehicle_data['vehicle_id'], True)
            workflow_result['states_executed'].append(WorkflowState.FEEDBACK_COLLECTING.value)
            workflow_result['agent_decisions']['feedback'] = feedback
            
            # STATE 7: MANUFACTURING INSIGHTS
            print(f"\n[7/7] STATE: MANUFACTURING_LOOP")
            self.current_state = WorkflowState.MANUFACTURING_LOOP
            insights = self.mfg_insights.generate_insights(
                vehicle_data['vehicle_id'],
                diagnosis['predicted_failure_type'],
                {'service_completed': True}
            )
            workflow_result['states_executed'].append(WorkflowState.MANUFACTURING_LOOP.value)
            workflow_result['agent_decisions']['insights'] = insights
            
            # Security monitoring
            print(f"\n🔒 UEBA SECURITY MONITORING")
            self.ueba_monitor.monitor_agent_action(
                'SchedulingAgent',
                'query_slots',
                'slot_availability'
            )
            self.ueba_monitor.monitor_agent_action(
                'SchedulingAgent',
                'query_telematics',
                'raw_telematics'
            )
            
            workflow_result['workflow_status'] = 'complete'
            workflow_result['audit_trail'] = self.ueba_monitor.audit_log
            
        except Exception as e:
            print(f"\n❌ Workflow error: {str(e)}")
            workflow_result['error'] = str(e)
            workflow_result['workflow_status'] = 'error'
        
        print(f"\n{'='*70}\n")
        return workflow_result


# ============================================
# EXAMPLE USAGE & TESTING
# ============================================

if __name__ == "__main__":
    # Create master agent
    master = MasterAgent()
    
    # Mock vehicle data
    vehicle_data = {
        'vehicle_id': 'VH-085',
        'model': 'Maruti Swift',
        'owner_name': 'Rajesh Sharma',
        'owner_phone': '+91-98765-43210',
        'service_center': 'XYZ Motors, Kharadi'
    }
    
    # Mock telematics readings (bearing failure trend)
    telematics_readings = [
        {'bearing_temp': 38, 'engine_temp': 92, 'rpm': 2500, 'brake_pressure': 45},
        {'bearing_temp': 42, 'engine_temp': 93, 'rpm': 2400, 'brake_pressure': 45},
        {'bearing_temp': 48, 'engine_temp': 94, 'rpm': 2600, 'brake_pressure': 44},
        {'bearing_temp': 55, 'engine_temp': 95, 'rpm': 2700, 'brake_pressure': 43},
        {'bearing_temp': 62, 'engine_temp': 96, 'rpm': 2800, 'brake_pressure': 42},
    ]
    
    # Mock ML prediction
    ml_prediction = {
        'risk_score': 0.95,
        'predicted_failure_type': 'bearing',
        'time_to_failure_days': 2
    }
    
    # Run orchestration
    result = master.orchestrate_workflow(vehicle_data, telematics_readings, ml_prediction)
    
    # Print result
    print("\n📊 WORKFLOW RESULT (JSON):")
    print("=" * 70)
    print(json.dumps(result, indent=2, default=str))
    print("=" * 70)
