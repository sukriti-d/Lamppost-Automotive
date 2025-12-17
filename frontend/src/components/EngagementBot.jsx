import React, { useState, useEffect } from 'react';
import './EngagementBot.css';

const API_BASE = 'http://localhost:8000';

export default function EngagementBot() {
  const [vehicles, setVehicles] = useState([]);
  const [selectedVehicle, setSelectedVehicle] = useState(null);
  const [conversation, setConversation] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [scheduledSlot, setScheduledSlot] = useState(null);
  const [engagementStep, setEngagementStep] = useState('vehicle_selection');

  useEffect(() => {
    fetchHighRiskVehicles();
  }, []);

  const fetchHighRiskVehicles = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/dashboard/queue?limit=10`);
      const data = await res.json();
      setVehicles(data.queue.filter(v => v.severity === 'HIGH'));
    } catch (err) {
      console.error('Error fetching vehicles:', err);
    }
  };

  const handleSelectVehicle = async (vehicle) => {
    setSelectedVehicle(vehicle);
    setLoading(true);
    setConversation([]);

    try {
      const res = await fetch(`${API_BASE}/api/engagement/initiate-outreach?vehicle_id=${vehicle.vehicle_id}`, {
        method: 'POST'
      });
      const outreach = await res.json();

      setSessionId(outreach.session_id);
      setScheduledSlot(outreach.recommended_slot);
      
      setConversation([
        {
          role: 'bot',
          message: outreach.initial_message,
          timestamp: new Date().toLocaleTimeString()
        }
      ]);

      setEngagementStep('waiting_response');
    } catch (err) {
      console.error('Error initiating outreach:', err);
      alert('Error initiating outreach');
    } finally {
      setLoading(false);
    }
  };

  const handleSendResponse = async () => {
    if (!userInput.trim() || !sessionId) return;

    const newConversation = [
      ...conversation,
      {
        role: 'user',
        message: userInput,
        timestamp: new Date().toLocaleTimeString()
      }
    ];
    setConversation(newConversation);
    setUserInput('');

    try {
      const res = await fetch(
        `${API_BASE}/api/engagement/submit-response?session_id=${sessionId}&customer_response=${encodeURIComponent(userInput)}`,
        { method: 'POST' }
      );
      const result = await res.json();

      let botResponse = '';
      
      if (result.processing_result.intent === 'accept') {
        botResponse = `Great! ✅ We have you down for ${scheduledSlot.recommended_date} at ${scheduledSlot.recommended_time}. Confirmation code: ${scheduledSlot.confirmation_code}`;
        setEngagementStep('booking_confirmed');
      } else if (result.processing_result.intent === 'decline') {
        botResponse = "No problem! We'll reach out again in a few days. Safe travels!";
        setEngagementStep('declined');
      } else if (result.processing_result.intent === 'inquiry') {
        botResponse = `Available slots: ${scheduledSlot.recommended_date} at ${scheduledSlot.recommended_time} (${scheduledSlot.priority} priority). Would this work for you?`;
      } else {
        botResponse = result.follow_up_message;
      }

      setConversation([
        ...newConversation,
        {
          role: 'bot',
          message: botResponse,
          timestamp: new Date().toLocaleTimeString()
        }
      ]);
    } catch (err) {
      console.error('Error processing response:', err);
    }
  };

  const handleConfirmBooking = async () => {
    if (!selectedVehicle || !scheduledSlot) return;

    try {
      const res = await fetch(
        `${API_BASE}/api/scheduling/book-appointment?vehicle_id=${selectedVehicle.vehicle_id}&date=${scheduledSlot.recommended_date}&time=${scheduledSlot.recommended_time}`,
        { method: 'POST' }
      );
      const booking = await res.json();

      setConversation([
        ...conversation,
        {
          role: 'bot',
          message: `✅ Booking confirmed!\n📍 Location: ${booking.service_center}\n👨‍🔧 Technician: ${booking.assigned_technician}\n⏱️ Duration: ${booking.estimated_duration} minutes\n📦 Parts reserved: ${booking.parts_list.join(', ')}`,
          timestamp: new Date().toLocaleTimeString()
        }
      ]);

      setEngagementStep('booking_complete');
    } catch (err) {
      console.error('Error confirming booking:', err);
      alert('Error confirming booking');
    }
  };

  const handleReset = () => {
    setSelectedVehicle(null);
    setConversation([]);
    setUserInput('');
    setSessionId(null);
    setScheduledSlot(null);
    setEngagementStep('vehicle_selection');
  };

  return (
    <div className="engagement-bot">
      <h1>💬 Customer Engagement Bot</h1>

      <div className="engagement-container">
        {engagementStep === 'vehicle_selection' ? (
          <div className="vehicle-selection">
            <h2>Select High-Risk Vehicle to Engage</h2>
            {vehicles.length === 0 ? (
              <p style={{ color: '#9ca3af', textAlign: 'center' }}>No high-risk vehicles available</p>
            ) : (
              <div className="vehicle-list">
                {vehicles.map((vehicle) => (
                  <div
                    key={vehicle.vehicle_id}
                    className="vehicle-card"
                    onClick={() => handleSelectVehicle(vehicle)}
                  >
                    <div className="vehicle-info">
                      <p className="vehicle-id">🚗 {vehicle.vehicle_id}</p>
                      <p className="owner-name">{vehicle.owner_name}</p>
                      <p className="model">{vehicle.model}</p>
                    </div>
                    <div className="vehicle-metrics">
                      <span className="risk-badge high">⚠️ {(vehicle.risk_score * 100).toFixed(0)}% Risk</span>
                      <span className="ttf">🕐 {vehicle.ttf_days} days</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="chat-interface">
            <div className="chat-header">
              <div className="vehicle-info-header">
                <h2>💬 {selectedVehicle?.owner_name}</h2>
                <p>{selectedVehicle?.vehicle_id} • {selectedVehicle?.model}</p>
              </div>
              <button className="btn-reset" onClick={handleReset}>
                ← Back
              </button>
            </div>

            <div className="chat-messages">
              {conversation.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role}`}>
                  <div className="message-content">
                    <p>{msg.message}</p>
                    <span className="timestamp">{msg.timestamp}</span>
                  </div>
                </div>
              ))}

              {engagementStep === 'booking_confirmed' && scheduledSlot && (
                <div className="slot-preview">
                  <h3>📅 Recommended Slot</h3>
                  <div className="slot-details">
                    <p><strong>Date:</strong> {scheduledSlot.recommended_date}</p>
                    <p><strong>Time:</strong> {scheduledSlot.recommended_time}</p>
                    <p><strong>Priority:</strong> {scheduledSlot.priority}</p>
                    <p><strong>Technician:</strong> {scheduledSlot.assigned_technician}</p>
                    <p><strong>Duration:</strong> {scheduledSlot.estimated_duration_minutes} min</p>
                    <p><strong>Parts:</strong> {scheduledSlot.parts_list?.join(', ') || 'N/A'}</p>
                  </div>
                  <button className="btn-confirm" onClick={handleConfirmBooking}>
                    ✅ Confirm Appointment
                  </button>
                </div>
              )}
            </div>

            {engagementStep === 'waiting_response' && (
              <div className="chat-input">
                <input
                  type="text"
                  value={userInput}
                  onChange={(e) => setUserInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSendResponse()}
                  placeholder="Type response (Yes, No, When?, etc.)"
                />
                <button onClick={handleSendResponse} disabled={!userInput.trim()}>
                  Send
                </button>
              </div>
            )}

            {engagementStep === 'booking_complete' && (
              <div className="booking-success">
                <h3>✅ Booking Complete!</h3>
                <p>Customer will receive SMS reminder 24 hours before appointment.</p>
                <button className="btn-new-engagement" onClick={handleReset}>
                  → Next Vehicle
                </button>
              </div>
            )}

            {engagementStep === 'declined' && (
              <div className="engagement-declined">
                <p>Customer declined engagement. Follow-up scheduled for next week.</p>
                <button className="btn-new-engagement" onClick={handleReset}>
                  → Next Vehicle
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
