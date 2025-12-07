"""
LampPost Agentic Orchestration Layer

Master Agent + 7 Worker Agents coordinating vehicle maintenance workflows.
This is the CORE INNOVATION of LampPost - autonomous decision-making.

Workers:
1. Data Analysis Agent - Fetch & analyze telematics
2. Diagnosis Agent - Score failure risk
3. Engagement Agent - Craft outreach messages
4. Scheduling Agent - Find optimal service slots
5. Feedback Agent - Collect & track CSAT
6. Mfg Insights Agent - Manufacturing loop closure
7. UEBA Monitor Agent - Security & compliance
"""

from enum import Enum
from typing import Optional, Dict, List
from datetime import datetime
import json

# ============================================
# ENUMS & TYPES
# ============================================

class WorkflowState(Enum):
    """States in the orchestration workflow."""
    MONITORING = "monitoring"
    DIAGNOSING = "diagnosing"
    ENGAGING = "engaging"
    SCHEDULING = "scheduling"
    SERVICE_EXECUTING = "service_executing"
    FEEDBACK_COLLECTING = "feedback_collecting"
    COMPLETE = "complete"

class Severity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

# ============================================
# WORKER AGENTS
# ============================================

class DataAnalysisAgent:
    """
    Analyzes telematics data for patterns & trends.
    Responsibility: Data fetching, preprocessing, feature computation.
    """
    
    def __init__(self, name="DataAnalysisAgent"):
        self.name = name
        self.execution_log = []
    
    def analyze_vehicle_data(self, vehicle_id: str, telematics_readings: List[Dict]):
        """
        Analyze vehicle telematics data.
        
        Returns:
            - anomalies detected
            - trends identified
            - health status
        """
        action = {
            'timestamp': datetime.now().isoformat(),
            'agent': self.name,
            'action': 'analyze_vehicle_data',
            'vehicle_id': vehicle_id,
            'readings_count': len(telematics_readings),
            'status': 'executing'
        }
        
        if len(telematics_readings) < 4:
            action['status'] = 'insufficient_data'
            self.execution_log.append(action)
            return None
        
        # Analyze trends
        last_5_bearing_temps = [r['bearing_temp'] for r in telematics_readings[-5:]]
        bearing_trend = "rising" if last_5_bearing_temps[-1] > last_5_bearing_temps[0] else "stable"
        
        analysis = {
            'vehicle_id': vehicle_id,
            'reading_count': len(telematics_readings),
            'bearing_trend': bearing_trend,
            'recent_anomalies': sum(1 for r in telematics_readings[-5:] if r['bearing_temp'] > 50),
            'health_status': 'normal'
        }
        
        action['status'] = 'complete'
        action['result'] = analysis
        self.execution_log.append(action)
        
        return analysis


class DiagnosisAgent:
    """
    Diagnoses failure risk using ML predictions.
    Responsibility: Risk scoring, TTF estimation, severity assessment.
    """
    
    def __init__(self, name="DiagnosisAgent", predictor=None):
        self.name = name
        self.predictor = predictor  # ML model wrapper
        self.execution_log = []
    
    def diagnose(self, vehicle_id: str, analysis: Dict):
        """
        Diagnose failure risk based on analysis.
        
        Returns:
            - risk_score (0-1)
            - failure_type (bearing, brake, engine)
            - ttf_days (time to failure)
            - severity (HIGH, MEDIUM, LOW)
        """
        action = {
            'timestamp': datetime.now().isoformat(),
            'agent': self.name,
            'action': 'diagnose',
            'vehicle_id': vehicle_id,
            'status': 'executing'
        }
        
        if not analysis:
            action['status'] = 'skipped'
            self.execution_log.append(action)
            return None
        
        # Simulate ML prediction (in real system, calls XGBoost model)
        if analysis.get('bearing_trend') == 'rising':
            risk_score = 0.75  # High risk
            failure_type = 'bearing'
            ttf_days = 2
            severity = Severity.HIGH
        else:
            risk_score = 0.25  # Low risk
            failure_type = 'engine'
            ttf_days = 14
            severity = Severity.LOW
        
        diagnosis = {
            'vehicle_id': vehicle_id,
            'risk_score': risk_score,
            'predicted_failure_type': failure_type,
            'time_to_failure_days': ttf_days,
            'severity': severity.value,
            'confidence': 0.92
        }
        
        action['status'] = 'complete'
        action['result'] = diagnosis
        self.execution_log.append(action)
        
        return diagnosis


class EngagementAgent:
    """
    Engages customers via voice/chat.
    Responsibility: Outreach messaging, persuasion, consent management.
    """
    
    def __init__(self, name="EngagementAgent"):
        self.name = name
        self.execution_log = []
        self.engagement_templates = {
            'HIGH': "Hi {{owner_name}}, your {{model}} is showing critical bearing wear. We can service it now for ₹1,500 or wait—but a roadside breakdown costs ₹6,000+. Can I book an appointment?",
            'MEDIUM': "Hi {{owner_name}}, your {{model}} needs a routine brake check soon. Any preference this week?",
            'LOW': "Hi {{owner_name}}, your {{model}} is running great! Routine maintenance next month?"
        }
    
    def engage_customer(self, vehicle_data: Dict, diagnosis: Dict):
        """
        Craft & send engagement message to customer.
        
        Returns:
            - message
            - channel (voice, sms, app)
            - engagement_id
        """
        action = {
            'timestamp': datetime.now().isoformat(),
            'agent': self.name,
            'action': 'engage_customer',
            'vehicle_id': vehicle_data['vehicle_id'],
            'status': 'executing'
        }
        
        severity = diagnosis['severity']
        template = self.engagement_templates.get(severity, self.engagement_templates['LOW'])
        
        message = template.replace('{{owner_name}}', vehicle_data.get('owner_name', 'Valued Customer'))
        message = message.replace('{{model}}', vehicle_data.get('model', 'vehicle'))
        
        engagement = {
            'vehicle_id': vehicle_data['vehicle_id'],
            'owner_phone': vehicle_data.get('owner_phone'),
            'message': message,
            'channel': 'voice',  # Could be SMS, app, email
            'engagement_id': f"ENG-{vehicle_data['vehicle_id']}-{datetime.now().timestamp()}",
            'status': 'message_crafted'
        }
        
        action['status'] = 'complete'
        action['result'] = engagement
        self.execution_log.append(action)
        
        return engagement


class SchedulingAgent:
    """
    Schedules service appointments.
    Responsibility: Slot optimization, capacity management, confirmation.
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
        """
        Find optimal service slot.
        
        Returns:
            - recommended_slot
            - alternatives
            - estimated_duration
        """
        action = {
            'timestamp': datetime.now().isoformat(),
            'agent': self.name,
            'action': 'find_optimal_slot',
            'vehicle_id': vehicle_data['vehicle_id'],
            'status': 'executing'
        }
        
        # Urgent cases get priority (Friday morning preferred)
        if diagnosis['severity'] == 'HIGH':
            recommended_slot = {
                'day': 'Friday',
                'time': '09:00',
                'duration_minutes': 90,
                'service_center': 'XYZ Motors, Kharadi',
                'distance_km': 2.1
            }
        else:
            recommended_slot = {
                'day': 'Saturday',
                'time': '10:00',
                'duration_minutes': 60,
                'service_center': 'XYZ Motors, Kharadi',
                'distance_km': 2.1
            }
        
        schedule = {
            'vehicle_id': vehicle_data['vehicle_id'],
            'recommended_slot': recommended_slot,
            'alternatives': [
                {'day': 'Wednesday', 'time': '14:00', 'distance_km': 3.2},
                {'day': 'Thursday', 'time': '11:00', 'distance_km': 2.8}
            ],
            'booking_status': 'pending_customer_confirmation'
        }
        
        action['status'] = 'complete'
        action['result'] = schedule
        self.execution_log.append(action)
        
        return schedule


class FeedbackAgent:
    """
    Collects service feedback & CSAT.
    Responsibility: Post-service feedback, rating collection, issue tracking.
    """
    
    def __init__(self, name="FeedbackAgent"):
        self.name = name
        self.execution_log = []
    
    def collect_feedback(self, vehicle_id: str, service_completed: bool):
        """
        Collect feedback after service.
        
        Returns:
            - csat_score
            - nps_score
            - feedback_text
        """
        action = {
            'timestamp': datetime.now().isoformat(),
            'agent': self.name,
            'action': 'collect_feedback',
            'vehicle_id': vehicle_id,
            'status': 'executing'
        }
        
        if not service_completed:
            action['status'] = 'skipped'
            self.execution_log.append(action)
            return None
        
        feedback = {
            'vehicle_id': vehicle_id,
            'csat_score': 9,  # 1-10 scale
            'nps_score': 8,   # Net Promoter Score
            'feedback_text': 'Great service, very professional. Appreciate the predictive approach.',
            'timestamp': datetime.now().isoformat()
        }
        
        action['status'] = 'complete'
        action['result'] = feedback
        self.execution_log.append(action)
        
        return feedback


class MfgInsightsAgent:
    """
    Feeds service data back to manufacturing (close the loop).
    Responsibility: CAPA ticket creation, defect analysis, design recommendations.
    """
    
    def __init__(self, name="MfgInsightsAgent"):
        self.name = name
        self.execution_log = []
    
    def generate_insights(self, vehicle_id: str, failure_type: str, service_data: Dict):
        """
        Generate manufacturing insights from service data.
        
        Returns:
            - defect_pattern
            - capa_ticket
            - design_recommendation
        """
        action = {
            'timestamp': datetime.now().isoformat(),
            'agent': self.name,
            'action': 'generate_insights',
            'vehicle_id': vehicle_id,
            'status': 'executing'
        }
        
        # Analyze defect pattern
        insights = {
            'vehicle_id': vehicle_id,
            'failure_type': failure_type,
            'defect_pattern': 'Bearing batch XYZ-2020 models trending failures',
            'capa_ticket': {
                'id': 'CAPA-2025-001',
                'title': 'Investigate bearing degradation in 2020 models',
                'priority': 'HIGH',
                'assigned_to': 'Quality Engineering'
            },
            'design_recommendation': 'Upgrade bearing thermal sensor threshold or source alternate bearing supplier',
            'estimated_savings': 1250000  # Rupees saved by early detection
        }
        
        action['status'] = 'complete'
        action['result'] = insights
        self.execution_log.append(action)
        
        return insights


class UEBAMonitorAgent:
    """
    User & Entity Behavioral Analysis (UEBA) for security.
    Responsibility: Anomaly detection, policy enforcement, audit logging.
    """
    
    def __init__(self, name="UEBAMonitorAgent"):
        self.name = name
        self.execution_log = []
        self.audit_log = []
    
    def monitor_agent_actions(self, agent_name: str, action: str, resource: str, status: str):
        """
        Monitor other agents for policy violations.
        
        Returns:
            - status (allowed / blocked)
            - reason
        """
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent_name': agent_name,
            'action': action,
            'resource': resource,
            'status': status
        }
        
        # Example policy: Scheduling Agent should NOT access raw telematics
        if agent_name == 'SchedulingAgent' and resource == 'raw_telematics':
            audit_entry['decision'] = 'BLOCKED'
            audit_entry['reason'] = 'Scheduling Agent policy: Cannot access raw telematics (least privilege)'
        else:
            audit_entry['decision'] = 'ALLOWED'
            audit_entry['reason'] = None
        
        self.audit_log.append(audit_entry)
        
        return audit_entry['decision'] == 'ALLOWED', audit_entry


# ============================================
# MASTER AGENT (ORCHESTRATOR)
# ============================================

class MasterAgent:
    """
    Master Agent coordinates all Worker Agents.
    
    Workflow:
    1. MONITORING: Receive telematics data
    2. DIAGNOSING: Run diagnostics & predict failures
    3. ENGAGING: Contact customer via voice
    4. SCHEDULING: Find & book service slot
    5. SERVICE_EXECUTING: Manage service execution
    6. FEEDBACK_COLLECTING: Gather CSAT
    7. COMPLETE: Close the loop with manufacturing
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
        self.decision_log = []
    
    def orchestrate_workflow(self, vehicle_data: Dict, telematics_readings: List[Dict]):
        """
        Main orchestration logic - coordinates all agents in workflow.
        
        Returns:
            - workflow_result with all decisions
            - next_actions
        """
        workflow_id = f"WF-{vehicle_data['vehicle_id']}-{datetime.now().timestamp()}"
        workflow_result = {
            'workflow_id': workflow_id,
            'vehicle_id': vehicle_data['vehicle_id'],
            'timestamp': datetime.now().isoformat(),
            'states_executed': [],
            'decisions': {}
        }
        
        print(f"\n{'='*60}")
        print(f"🤖 MASTER AGENT ORCHESTRATION: {vehicle_data['vehicle_id']}")
        print(f"{'='*60}")
        
        try:
            # STATE 1: MONITORING
            print(f"\n[1/7] STATE: MONITORING")
            self.current_state = WorkflowState.MONITORING
            print(f"  → Analyzing telematics data...")
            analysis = self.data_analysis.analyze_vehicle_data(
                vehicle_data['vehicle_id'],
                telematics_readings
            )
            workflow_result['states_executed'].append(WorkflowState.MONITORING.value)
            workflow_result['decisions']['analysis'] = analysis
            
            if not analysis:
                print(f"  ❌ Insufficient data. Waiting for more readings.")
                return workflow_result
            
            print(f"  ✅ Analysis complete. Trend: {analysis['bearing_trend']}")
            
            # STATE 2: DIAGNOSING
            print(f"\n[2/7] STATE: DIAGNOSING")
            self.current_state = WorkflowState.DIAGNOSING
            print(f"  → Running failure prediction model...")
            diagnosis = self.diagnosis.diagnose(vehicle_data['vehicle_id'], analysis)
            workflow_result['states_executed'].append(WorkflowState.DIAGNOSING.value)
            workflow_result['decisions']['diagnosis'] = diagnosis
            
            print(f"  ✅ Diagnosis: Risk {diagnosis['risk_score']:.2f} ({diagnosis['severity']}) | TTF: {diagnosis['time_to_failure_days']} days")
            
            # Check if action needed (only if HIGH or MEDIUM risk)
            if diagnosis['risk_score'] < 0.3:
                print(f"\n  ℹ️ Low risk. No action needed. Continuing monitoring...")
                workflow_result['action_required'] = False
                return workflow_result
            
            # STATE 3: ENGAGING
            print(f"\n[3/7] STATE: ENGAGING")
            self.current_state = WorkflowState.ENGAGING
            print(f"  → Crafting customer engagement message...")
            engagement = self.engagement.engage_customer(vehicle_data, diagnosis)
            workflow_result['states_executed'].append(WorkflowState.ENGAGING.value)
            workflow_result['decisions']['engagement'] = engagement
            
            print(f"  ✅ Message crafted:")
            print(f"     \"{engagement['message'][:100]}...\"")
            print(f"  → Sending via {engagement['channel']}...")
            print(f"  ✅ Message sent to {engagement['owner_phone']}")
            
            # STATE 4: SCHEDULING
            print(f"\n[4/7] STATE: SCHEDULING")
            self.current_state = WorkflowState.SCHEDULING
            print(f"  → Finding optimal service slot...")
            schedule = self.scheduling.find_optimal_slot(vehicle_data, diagnosis)
            workflow_result['states_executed'].append(WorkflowState.SCHEDULING.value)
            workflow_result['decisions']['schedule'] = schedule
            
            slot = schedule['recommended_slot']
            print(f"  ✅ Optimal slot found:")
            print(f"     📅 {slot['day']} at {slot['time']}")
            print(f"     📍 {slot['service_center']} ({slot['distance_km']} km away)")
            print(f"     ⏱️  Estimated duration: {slot['duration_minutes']} minutes")
            
            # STATE 5: SERVICE EXECUTION (Mock)
            print(f"\n[5/7] STATE: SERVICE_EXECUTING")
            self.current_state = WorkflowState.SERVICE_EXECUTING
            print(f"  → Customer confirmed appointment")
            print(f"  → Service execution in progress...")
            print(f"  ✅ Service completed successfully")
            
            # STATE 6: FEEDBACK
            print(f"\n[6/7] STATE: FEEDBACK_COLLECTING")
            self.current_state = WorkflowState.FEEDBACK_COLLECTING
            print(f"  → Collecting post-service feedback...")
            feedback = self.feedback.collect_feedback(vehicle_data['vehicle_id'], True)
            workflow_result['states_executed'].append(WorkflowState.FEEDBACK_COLLECTING.value)
            workflow_result['decisions']['feedback'] = feedback
            
            if feedback:
                print(f"  ✅ Feedback received:")
                print(f"     CSAT: {feedback['csat_score']}/10")
                print(f"     NPS: {feedback['nps_score']}/10")
            
            # STATE 7: MANUFACTURING INSIGHTS
            print(f"\n[7/7] STATE: COMPLETE (Manufacturing Insights)")
            self.current_state = WorkflowState.COMPLETE
            print(f"  → Analyzing for manufacturing insights...")
            insights = self.mfg_insights.generate_insights(
                vehicle_data['vehicle_id'],
                diagnosis['predicted_failure_type'],
                {'service_data': 'mock'}
            )
            workflow_result['states_executed'].append(WorkflowState.COMPLETE.value)
            workflow_result['decisions']['insights'] = insights
            
            if insights['capa_ticket']:
                print(f"  ✅ CAPA ticket created:")
                print(f"     ID: {insights['capa_ticket']['id']}")
                print(f"     Title: {insights['capa_ticket']['title']}")
                print(f"     Priority: {insights['capa_ticket']['priority']}")
            
            print(f"\n  💰 Estimated savings from early detection: ₹{insights['estimated_savings']:,.0f}")
            
            workflow_result['action_required'] = True
            workflow_result['workflow_status'] = 'complete'
            
        except Exception as e:
            print(f"\n❌ Workflow error: {str(e)}")
            workflow_result['error'] = str(e)
            workflow_result['workflow_status'] = 'error'
        
        print(f"\n{'='*60}\n")
        return workflow_result


# ============================================
# EXAMPLE USAGE
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
    
    # Mock telematics readings (simulate bearing failure trend)
    telematics_readings = [
        {'bearing_temp': 38, 'engine_temp': 92, 'rpm': 2500, 'brake_pressure': 45},
        {'bearing_temp': 42, 'engine_temp': 93, 'rpm': 2400, 'brake_pressure': 45},
        {'bearing_temp': 48, 'engine_temp': 94, 'rpm': 2600, 'brake_pressure': 44},
        {'bearing_temp': 55, 'engine_temp': 95, 'rpm': 2700, 'brake_pressure': 43},
        {'bearing_temp': 62, 'engine_temp': 96, 'rpm': 2800, 'brake_pressure': 42},
    ]
    
    # Run orchestration
    result = master.orchestrate_workflow(vehicle_data, telematics_readings)
    
    # Print result
    print("\n📊 WORKFLOW RESULT:")
    print(json.dumps(result, indent=2, default=str))
