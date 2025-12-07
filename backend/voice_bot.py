"""
Voice Bot Integration Module

Handles outbound calls, IVR interactions, transcription, and NLU.
Currently mocked (no Twilio keys needed) for prototype.
"""

from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional
import json

# ============================================
# ENUMS
# ============================================

class CallStatus(Enum):
    """States of voice call."""
    INITIATED = "initiated"
    RINGING = "ringing"
    CONNECTED = "connected"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NO_ANSWER = "no_answer"

class CallOutcome(Enum):
    """Outcomes of voice interaction."""
    BOOKING_CONFIRMED = "booking_confirmed"
    BOOKING_RESCHEDULED = "booking_rescheduled"
    CALLBACK_REQUESTED = "callback_requested"
    CUSTOMER_DECLINED = "customer_declined"
    TRANSFERRED_TO_AGENT = "transferred_to_agent"

class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    HINDI = "hi"
    MARATHI = "mr"
    GUJARATI = "gu"
    KANNADA = "kn"
    TAMIL = "ta"
    TELUGU = "te"

# ============================================
# VOICE BOT ENGINE
# ============================================

class VoiceBotEngine:
    """
    Multi-lingual voice bot for customer engagement.
    
    Features:
    - TTS (Text-to-Speech) with regional language support
    - ASR (Automatic Speech Recognition)
    - IVR (Interactive Voice Response)
    - Call recording & transcription
    - Natural Language Understanding (NLU)
    """
    
    def __init__(self):
        self.name = "VoiceBotEngine"
        self.call_logs = []
        
        # Message templates in multiple languages
        self.messages = {
            Language.ENGLISH.value: {
                'greeting': 'Hi {name}, this is LampPost from {service_center}.',
                'issue_intro': 'We detected your {model} might need {failure_type} service soon.',
                'urgency_high': 'Early action now costs ₹{early_cost}. Risk of breakdown costs ₹{late_cost} plus your safety.',
                'urgency_medium': 'Routine maintenance recommended. Would you like to book a slot?',
                'booking_offer': 'We have slots available {day} at {time}. Can I confirm?',
                'confirmation': 'Great! Your appointment is confirmed for {day} at {time} at {center}.',
                'goodbye': 'Thank you for choosing LampPost. Your safety is our priority.'
            },
            Language.HINDI.value: {
                'greeting': 'नमस्ते {name}, यह लैम्पपोस्ट {service_center} से बोल रहा हूँ।',
                'issue_intro': 'हमने आपकी {model} को {failure_type} सर्विस की जरूरत देखी है।',
                'urgency_high': 'अभी सर्विस करवाएं तो ₹{early_cost}, नहीं तो ₹{late_cost} + खतरा।',
                'urgency_medium': 'नियमित सर्विस की सिफारिश है। क्या बुकिंग करेंगे?',
                'booking_offer': '{day} को {time} पर स्लॉट उपलब्ध है। कन्फर्म करूँ?',
                'confirmation': 'शानदार! आपकी बुकिंग {day} {time} पर {center} में कन्फर्म है।',
                'goodbye': 'धन्यवाद। आपकी सुरक्षा हमारी प्राथमिकता है।'
            },
            Language.MARATHI.value: {
                'greeting': 'नमस्कार {name}, मी लॅम्पपोस्ट {service_center} कडून बोलत आहे।',
                'issue_intro': 'आपल्या {model} ला {failure_type} सेवेची आवश्यकता आहे.',
                'urgency_high': 'आता सेवा ₹{early_cost}, अन्यथा ₹{late_cost} + धोका।',
                'urgency_medium': 'नियमित सेवा शिफारस आहे। स्लॉट बुक करू?',
                'booking_offer': '{day} रोजी {time} ला स्लॉट आहे। कन्फर्म करू?',
                'confirmation': 'बरोबर! आपली बुकिंग {day} {time} ला {center} येथे कन्फर्म आहे।',
                'goodbye': 'धन्यवाद। आपली सुरक्षा आमचे प्राथमिकता आहे।'
            }
        }
    
    def initiate_call(self, vehicle_data: Dict, diagnosis: Dict, language: str = 'en'):
        """Initiate outbound call to customer."""
        
        call_id = f"CALL-{vehicle_data['vehicle_id']}-{int(datetime.now().timestamp())}"
        
        call_log = {
            'call_id': call_id,
            'vehicle_id': vehicle_data['vehicle_id'],
            'phone_number': vehicle_data.get('owner_phone', ''),
            'initiated_at': datetime.now().isoformat(),
            'status': CallStatus.INITIATED.value,
            'language': language,
            'duration_seconds': 0,
            'transcript': '',
            'outcome': None,
            'booking_confirmed': False
        }
        
        print(f"\n📞 VOICE BOT: Initiating call to {vehicle_data['vehicle_id']}")
        print(f"   Phone: {call_log['phone_number']}")
        print(f"   Language: {language}")
        
        return call_log
    
    def generate_voice_script(self, vehicle_data: Dict, diagnosis: Dict, language: str = 'en') -> str:
        """Generate personalized voice script."""
        
        msgs = self.messages.get(language, self.messages['en'])
        
        # Build script
        script = ""
        script += msgs['greeting'].format(
            name=vehicle_data.get('owner_name', 'Customer'),
            service_center=vehicle_data.get('service_center', 'our service center')
        ) + "\n"
        
        script += msgs['issue_intro'].format(
            model=vehicle_data.get('model', 'vehicle'),
            failure_type=diagnosis['predicted_failure_type']
        ) + "\n"
        
        # Add urgency-based message
        if diagnosis['severity'] == 'HIGH':
            script += msgs['urgency_high'].format(
                early_cost=1500,
                late_cost=6000
            ) + "\n"
        else:
            script += msgs['urgency_medium'] + "\n"
        
        script += msgs['booking_offer'].format(
            day='Friday',
            time='9:00 AM',
        ) + "\n"
        
        return script
    
    def process_asr_transcript(self, transcript: str) -> Dict:
        """Process speech-to-text output using NLU."""
        
        # Simple NLU - can be upgraded to BERT/GPT
        nlu_result = {
            'raw_transcript': transcript,
            'intent': 'unknown',
            'entities': {},
            'confidence': 0.0,
            'action': None
        }
        
        transcript_lower = transcript.lower()
        
        # Intent classification
        if any(word in transcript_lower for word in ['yes', 'yeah', 'okay', 'ok', 'book', 'confirm', 'haan', 'bilkul']):
            nlu_result['intent'] = 'booking_confirmation'
            nlu_result['action'] = 'confirm_booking'
            nlu_result['confidence'] = 0.95
        elif any(word in transcript_lower for word in ['no', 'nope', 'not', 'nahi', 'later']):
            nlu_result['intent'] = 'booking_decline'
            nlu_result['action'] = 'reschedule_option'
            nlu_result['confidence'] = 0.92
        elif any(word in transcript_lower for word in ['when', 'time', 'slot', 'available', 'kab']):
            nlu_result['intent'] = 'slot_inquiry'
            nlu_result['action'] = 'list_alternatives'
            nlu_result['confidence'] = 0.88
        elif any(word in transcript_lower for word in ['talk', 'agent', 'person', 'human', 'kisi se baat']):
            nlu_result['intent'] = 'escalation_request'
            nlu_result['action'] = 'transfer_to_agent'
            nlu_result['confidence'] = 0.90
        
        return nlu_result
    
    def generate_response(self, nlu_result: Dict, language: str = 'en') -> str:
        """Generate voice bot response based on NLU."""
        
        msgs = self.messages.get(language, self.messages['en'])
        
        action = nlu_result.get('action')
        
        if action == 'confirm_booking':
            return msgs['confirmation'].format(
                day='Friday',
                time='9:00 AM',
                center='XYZ Motors, Kharadi'
            )
        elif action == 'reschedule_option':
            return 'No problem! I can call you back next week. What time works best?'
        elif action == 'list_alternatives':
            return 'We have slots: Friday 9 AM, Saturday 10 AM, or next Wednesday 2 PM. Which works?'
        elif action == 'transfer_to_agent':
            return 'Of course! Connecting you to our service advisor. Please hold...'
        else:
            return 'I didn\'t quite catch that. Can you repeat?'
    
    def end_call(self, call_log: Dict, outcome: str, duration_seconds: int = 0) -> Dict:
        """End call and log results."""
        
        call_log['status'] = CallStatus.COMPLETED.value
        call_log['outcome'] = outcome
        call_log['duration_seconds'] = duration_seconds
        call_log['ended_at'] = datetime.now().isoformat()
        
        print(f"   Status: {CallStatus.COMPLETED.value}")
        print(f"   Outcome: {outcome}")
        print(f"   Duration: {duration_seconds}s")
        
        self.call_logs.append(call_log)
        return call_log

# ============================================
# VOICE BOT SESSION MANAGER
# ============================================

class VoiceBotSessionManager:
    """Manages multi-turn voice bot conversations."""
    
    def __init__(self):
        self.sessions = {}  # session_id -> conversation state
    
    def create_session(self, call_id: str, vehicle_id: str) -> str:
        """Create new session."""
        session_id = f"SESSION-{call_id}"
        self.sessions[session_id] = {
            'call_id': call_id,
            'vehicle_id': vehicle_id,
            'turns': [],
            'booking_confirmed': False,
            'created_at': datetime.now().isoformat()
        }
        return session_id
    
    def add_turn(self, session_id: str, bot_message: str, customer_transcript: str, nlu_result: Dict):
        """Log conversation turn."""
        self.sessions[session_id]['turns'].append({
            'bot_message': bot_message,
            'customer_input': customer_transcript,
            'nlu_intent': nlu_result.get('intent'),
            'nlu_confidence': nlu_result.get('confidence'),
            'timestamp': datetime.now().isoformat()
        })
    
    def end_session(self, session_id: str, booking_confirmed: bool = False) -> Dict:
        """End session."""
        session = self.sessions[session_id]
        session['booking_confirmed'] = booking_confirmed
        session['ended_at'] = datetime.now().isoformat()
        return session
