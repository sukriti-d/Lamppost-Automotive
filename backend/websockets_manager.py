"""
WebSocket manager for real-time dashboard updates.
"""

from fastapi import WebSocket
from typing import Set, Dict, List
import json
import asyncio
from datetime import datetime

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscription_map: Dict[WebSocket, List[str]] = {}  # ws -> [vehicle_ids]
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.subscription_map[websocket] = []
        print(f"✓ WebSocket connected. Total: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket):
        """Close WebSocket connection."""
        self.active_connections.discard(websocket)
        del self.subscription_map[websocket]
        print(f"✓ WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast_vehicle_update(self, vehicle_id: str, data: Dict):
        """Broadcast vehicle update to all subscribed clients."""
        message = {
            'type': 'vehicle_update',
            'vehicle_id': vehicle_id,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        for connection in self.active_connections:
            # Only send if client subscribed to this vehicle
            if vehicle_id in self.subscription_map.get(connection, []):
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print(f"Error sending message: {e}")
    
    async def broadcast_dashboard_metric(self, metric: str, value: any):
        """Broadcast dashboard metric update."""
        message = {
            'type': 'metric_update',
            'metric': metric,
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting metric: {e}")
    
    async def broadcast_alert(self, alert_type: str, vehicle_id: str, message: str):
        """Broadcast alert to all clients."""
        alert = {
            'type': 'alert',
            'alert_type': alert_type,  # 'high_risk', 'anomaly', 'breakdown'
            'vehicle_id': vehicle_id,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        for connection in self.active_connections:
            try:
                await connection.send_json(alert)
            except Exception as e:
                print(f"Error sending alert: {e}")

# Global connection manager
manager = ConnectionManager()
