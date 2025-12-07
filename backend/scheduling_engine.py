from datetime import datetime, timedelta
from typing import Dict, List
import random

class SchedulingEngine:
    """Intelligent service scheduling based on risk predictions."""
    
    def __init__(self):
        self.service_slots = {
            'Monday': ['09:00', '10:30', '14:00', '15:30'],
            'Tuesday': ['09:00', '10:30', '14:00', '15:30'],
            'Wednesday': ['09:00', '10:30', '14:00', '15:30'],
            'Thursday': ['09:00', '10:30', '14:00', '15:30'],
            'Friday': ['09:00', '10:30', '14:00', '15:30'],
            'Saturday': ['10:00', '11:30'],
        }
        self.bookings = {}
    
    def get_optimal_slot(self, vehicle_id: str, risk_score: float) -> Dict:
        """Get optimal service slot based on risk urgency."""
        
        # Urgency determines days ahead
        if risk_score > 0.8:
            days_ahead = 1  # Tomorrow
            priority = 'CRITICAL'
        elif risk_score > 0.6:
            days_ahead = 2
            priority = 'HIGH'
        elif risk_score > 0.4:
            days_ahead = 3
            priority = 'MEDIUM'
        else:
            days_ahead = 7
            priority = 'LOW'
        
        # Find first available slot
        for i in range(days_ahead, days_ahead + 7):
            target_date = datetime.now() + timedelta(days=i)
            day_name = target_date.strftime('%A')
            
            if day_name == 'Sunday':
                continue
            
            slots = self.service_slots.get(day_name, [])
            
            for slot_time in slots:
                slot_key = f"{target_date.date()}_{slot_time}"
                
                if slot_key not in self.bookings:
                    self.bookings[slot_key] = []
                
                if len(self.bookings[slot_key]) < 3:  # Max 3 per slot
                    self.bookings[slot_key].append(vehicle_id)
                    
                    return {
                        'vehicle_id': vehicle_id,
                        'recommended_date': target_date.date().isoformat(),
                        'recommended_time': slot_time,
                        'day': day_name,
                        'priority': priority,
                        'urgency_days': days_ahead,
                        'estimated_duration_minutes': 60 if risk_score > 0.7 else 45,
                        'service_type': self._get_service_type(risk_score),
                        'confirmation_code': f"SVC-{vehicle_id}-{slot_key}",
                        'parts_reserved': True
                    }
        
        return {'error': 'No slots available'}
    
    def _get_service_type(self, risk_score: float) -> str:
        """Determine service type based on risk."""
        if risk_score > 0.8:
            return 'EMERGENCY_INSPECTION'
        elif risk_score > 0.6:
            return 'PREVENTIVE_MAINTENANCE'
        else:
            return 'ROUTINE_CHECKUP'
    
    def get_occupancy(self) -> Dict:
        """Get service center occupancy stats."""
        total_slots = sum(len(slots) for slots in self.service_slots.values()) * 7
        booked = sum(len(vehicles) for vehicles in self.bookings.values())
        
        return {
            'total_slots': total_slots,
            'booked_slots': booked,
            'occupancy_percent': round((booked / total_slots * 100), 2),
            'next_available': min(
                (k for k in self.bookings.keys() if len(self.bookings[k]) < 3),
                default='No slots'
            )
        }

scheduler = SchedulingEngine()
