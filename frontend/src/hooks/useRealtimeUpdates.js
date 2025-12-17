import { useEffect, useCallback } from 'react';

const WS_URL = 'ws://localhost:8000/ws/dashboard';

export function useRealtimeUpdates(onUpdate) {
  useEffect(() => {
    let ws = null;
    let reconnectAttempts = 0;
    const maxReconnect = 5;

    const connect = () => {
      try {
        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
          console.log('✓ WebSocket connected');
          reconnectAttempts = 0;
          // Send initial subscription
          ws.send(JSON.stringify({ action: 'ping' }));
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log('📡 Real-time update:', data);

            // Trigger callback on any update
            if (data.type === 'alert' || data.type === 'analytics_update') {
              onUpdate(data);
            }
          } catch (err) {
            console.error('Message parse error:', err);
          }
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
        };

        ws.onclose = () => {
          console.log('⚠️ WebSocket disconnected');
          
          // Auto-reconnect with exponential backoff
          if (reconnectAttempts < maxReconnect) {
            reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
            console.log(`🔄 Reconnecting in ${delay}ms...`);
            setTimeout(connect, delay);
          }
        };
      } catch (err) {
        console.error('Connection error:', err);
      }
    };

    connect();

    return () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [onUpdate]);
}
