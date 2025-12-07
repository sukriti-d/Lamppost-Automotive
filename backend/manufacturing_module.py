"""
Manufacturing Insights & CAPA Loop

Connects service data to manufacturing quality improvements.
"""

from datetime import datetime
from typing import Dict, List
from enum import Enum

# ============================================
# ENUMS
# ============================================

class CapaPriority(Enum):
    """CAPA ticket priority."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class DefectCategory(Enum):
    """Defect categories."""
    DESIGN = "design_defect"
    MANUFACTURING = "manufacturing_defect"
    ASSEMBLY = "assembly_defect"
    MATERIAL = "material_defect"
    SERVICE = "service_related"

# ============================================
# MANUFACTURING MODULE
# ============================================

class ManufacturingInsightsModule:
    """
    Analyzes field data and creates manufacturing improvement tickets.
    
    Closes the feedback loop: Service → QA → Design
    """
    
    def __init__(self):
        self.name = "ManufacturingInsightsModule"
        self.capa_tickets = []
        self.defect_patterns = {}
        self.supplier_alerts = []
        self.design_recommendations = []
    
    def analyze_defect_patterns(self, vehicle_failures: List[Dict]) -> Dict:
        """
        Analyze patterns across multiple vehicle failures.
        
        Detect:
        - Batch defects (same model_year, same component)
        - Recurring defects (same issue repeating)
        - Root causes (design vs manufacturing vs usage)
        """
        
        pattern_analysis = {
            'total_failures': len(vehicle_failures),
            'defect_patterns': {},
            'root_causes': {},
            'affected_batch': None,
            'ppm': 0  # Parts per million
        }
        
        # Group by failure type & model year
        failure_groups = {}
        for failure in vehicle_failures:
            key = f"{failure['predicted_failure_type']}-{failure.get('model_year', '2020')}"
            if key not in failure_groups:
                failure_groups[key] = []
            failure_groups[key].append(failure)
        
        # Analyze patterns
        for pattern_key, failures in failure_groups.items():
            if len(failures) >= 3:  # Pattern threshold
                pattern_analysis['defect_patterns'][pattern_key] = {
                    'count': len(failures),
                    'percentage': (len(failures) / len(vehicle_failures)) * 100,
                    'severity': 'HIGH' if len(failures) > 10 else 'MEDIUM'
                }
        
        # Calculate PPM
        pattern_analysis['ppm'] = int((len(vehicle_failures) / 36000) * 1000000)
        
        print(f"\n📊 DEFECT PATTERN ANALYSIS")
        print(f"   Total failures: {pattern_analysis['total_failures']}")
        print(f"   PPM: {pattern_analysis['ppm']}")
        print(f"   Patterns detected: {len(pattern_analysis['defect_patterns'])}")
        
        return pattern_analysis
    
    def create_capa_ticket(self, failure_data: Dict, pattern_analysis: Dict) -> Dict:
        """
        Create CAPA (Corrective Action Preventive Action) ticket.
        
        Sent to Quality Engineering & Design teams.
        """
        
        ticket_id = f"CAPA-2025-{len(self.capa_tickets) + 1:04d}"
        
        # Determine priority based on pattern
        if pattern_analysis['ppm'] > 1000:
            priority = CapaPriority.CRITICAL.value
        elif pattern_analysis['ppm'] > 500:
            priority = CapaPriority.HIGH.value
        elif pattern_analysis['ppm'] > 100:
            priority = CapaPriority.MEDIUM.value
        else:
            priority = CapaPriority.LOW.value
        
        capa_ticket = {
            'ticket_id': ticket_id,
            'priority': priority,
            'title': f"Investigate {failure_data['predicted_failure_type']} defects in {failure_data.get('model_year', '2020')} models",
            'description': f"""
            Recurring {failure_data['predicted_failure_type']} failures detected in field.
            
            Affected: {pattern_analysis['total_failures']} vehicles
            PPM (Parts Per Million): {pattern_analysis['ppm']}
            Pattern: {failure_data['predicted_failure_type']} degradation
            
            First 5 failures:
            - VH-085: Risk 0.95, TTF 2 days
            - VH-086: Risk 0.98, TTF 2 days
            - VH-087: Risk 0.99, TTF 2 days
            - VH-088: Risk 0.98, TTF 2 days
            - VH-089: Risk 0.99, TTF 2 days
            """,
            'assigned_to': 'Quality Engineering Team',
            'created_at': datetime.now().isoformat(),
            'due_date': self._calculate_due_date(priority),
            'status': 'created',
            'root_cause_analysis': None,
            'corrective_actions': [],
            'preventive_actions': []
        }
        
        self.capa_tickets.append(capa_ticket)
        
        print(f"\n🎫 CAPA TICKET CREATED")
        print(f"   ID: {ticket_id}")
        print(f"   Priority: {priority}")
        print(f"   Assigned: {capa_ticket['assigned_to']}")
        
        return capa_ticket
    
    def generate_rca(self, failure_data: Dict) -> Dict:
        """
        Generate Root Cause Analysis (RCA).
        
        Determines: Design issue? Manufacturing? Material? Assembly? Service?
        """
        
        failure_type = failure_data['predicted_failure_type']
        
        rca = {
            'failure_type': failure_type,
            'root_cause_category': None,
            'root_cause_description': '',
            'evidence': [],
            'hypothesis': None,
            'probability': 0.0
        }
        
        # RCA logic
        if failure_type == 'bearing':
            rca['root_cause_category'] = DefectCategory.MATERIAL.value
            rca['hypothesis'] = 'Bearing material fatigue due to thermal cycling'
            rca['evidence'] = [
                'Bearing temperature rising consistently',
                'Temperature exceeds OEM spec by 15-20°C',
                'Occurs in batch from specific supplier (XYZ Bearing Corp)',
                'PPM in 2020 models: 450 (vs normal <50)'
            ]
            rca['probability'] = 0.92
        
        elif failure_type == 'brake':
            rca['root_cause_category'] = DefectCategory.ASSEMBLY.value
            rca['hypothesis'] = 'Brake pressure calibration during assembly'
            rca['evidence'] = [
                'Brake pressure dropping linearly',
                'Occurs across multiple units from same production week',
                'Pressure sensor readings +0.5 bar offset'
            ]
            rca['probability'] = 0.85
        
        elif failure_type == 'engine':
            rca['root_cause_category'] = DefectCategory.DESIGN.value
            rca['hypothesis'] = 'Thermostat threshold too low for hot climates'
            rca['evidence'] = [
                'Engine temperature increasing during city driving',
                'Occurs primarily in high-temperature regions (Rajasthan, Karnataka)',
                'Design spec: 95°C, Field observed: 98-100°C'
            ]
            rca['probability'] = 0.78
        
        print(f"\n🔍 ROOT CAUSE ANALYSIS")
        print(f"   Component: {failure_type}")
        print(f"   Category: {rca['root_cause_category']}")
        print(f"   Hypothesis: {rca['hypothesis']}")
        print(f"   Confidence: {rca['probability']:.0%}")
        
        return rca
    
    def create_supplier_alert(self, failure_data: Dict, rca: Dict) -> Dict:
        """Alert supplier if defect is supplier-related."""
        
        if rca['root_cause_category'] != DefectCategory.MATERIAL.value:
            return None
        
        alert = {
            'alert_id': f"SUPP-{len(self.supplier_alerts) + 1:04d}",
            'supplier_name': 'XYZ Bearing Corp',
            'supplier_contact': 'quality@xyzbearing.com',
            'component': failure_data['predicted_failure_type'],
            'batch_id': '2020-BATCH-001',
            'issue': f"Quality variance in {failure_data['predicted_failure_type']} batch",
            'ppm': 450,
            'acceptable_ppm': 50,
            'action_required': 'Immediate batch review & quality audit',
            'deadline': self._calculate_due_date('CRITICAL'),
            'severity': 'HIGH'
        }
        
        self.supplier_alerts.append(alert)
        
        print(f"\n⚠️  SUPPLIER ALERT CREATED")
        print(f"   Supplier: {alert['supplier_name']}")
        print(f"   PPM: {alert['ppm']} (Target: {alert['acceptable_ppm']})")
        
        return alert
    
    def generate_design_recommendation(self, failure_data: Dict, rca: Dict) -> Dict:
        """Generate design improvement recommendation."""
        
        failure_type = failure_data['predicted_failure_type']
        
        recommendation = {
            'recommendation_id': f"DESIGN-{len(self.design_recommendations) + 1:04d}",
            'component': failure_type,
            'current_spec': None,
            'recommended_spec': None,
            'justification': '',
            'estimated_cost_impact': 0,
            'estimated_warranty_savings': 0,
            'implementation_effort': 'MEDIUM',
            'priority': 'HIGH'
        }
        
        if failure_type == 'bearing':
            recommendation['current_spec'] = 'Bearing material: SKF 6205'
            recommendation['recommended_spec'] = 'Bearing material: SKF 6205 HC (High-temp variant)'
            recommendation['justification'] = 'Upgrade thermal capability for hot climate operation'
            recommendation['estimated_cost_impact'] = 250  # Rupees per unit
            recommendation['estimated_warranty_savings'] = 2000
            recommendation['implementation_effort'] = 'LOW'
        
        elif failure_type == 'engine':
            recommendation['current_spec'] = 'Thermostat threshold: 95°C'
            recommendation['recommended_spec'] = 'Thermostat threshold: 92°C (climate-aware calibration)'
            recommendation['justification'] = 'Reduce engine temp variance in high-temp regions'
            recommendation['estimated_cost_impact'] = 50
            recommendation['estimated_warranty_savings'] = 1500
            recommendation['implementation_effort'] = 'MEDIUM'
        
        self.design_recommendations.append(recommendation)
        
        print(f"\n💡 DESIGN RECOMMENDATION")
        print(f"   Component: {failure_type}")
        print(f"   Recommended: {recommendation['recommended_spec']}")
        print(f"   Est. Warranty Savings: ₹{recommendation['estimated_warranty_savings']:,.0f}")
        
        return recommendation
    
    def _calculate_due_date(self, priority: str) -> str:
        """Calculate due date based on priority."""
        days = {
            'CRITICAL': 3,
            'HIGH': 7,
            'MEDIUM': 14,
            'LOW': 30
        }
        from datetime import timedelta
        due = datetime.now() + timedelta(days=days.get(priority, 14))
        return due.isoformat()
