import React, { useState, useEffect } from 'react';
import { useRealtimeUpdates } from '../hooks/useRealtimeUpdates';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import './AnalyticsDashboard.css';

const API_BASE = 'http://localhost:8000';

export default function AnalyticsDashboard() {
  const [riskTrend, setRiskTrend] = useState([]);
  const [componentFailures, setComponentFailures] = useState([]);
  const [sensorAnomalies, setSensorAnomalies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [alerts, setAlerts] = useState([]);

  // Real-time updates via WebSocket
  const handleRealtimeUpdate = (data) => {
    console.log('Real-time event:', data);
    
    // Add alert to toast
    if (data.type === 'alert') {
      const newAlert = {
        id: Date.now(),
        message: data.message,
        type: data.alert_type,
        timestamp: new Date()
      };
      setAlerts(prev => [newAlert, ...prev].slice(0, 5));
      
      // Auto-dismiss after 5s
      setTimeout(() => {
        setAlerts(prev => prev.filter(a => a.id !== newAlert.id));
      }, 5000);
    }
    
    // Refresh analytics on update
    if (data.type === 'analytics_update') {
      fetchAnalytics();
    }
  };

  useRealtimeUpdates(handleRealtimeUpdate);

  useEffect(() => {
    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchAnalytics = async () => {
    try {
      setLoading(true);
      const [trendRes, componentsRes, sensorsRes] = await Promise.all([
        fetch(`${API_BASE}/api/analytics/risk-trend?days=7`),
        fetch(`${API_BASE}/api/analytics/component-failures`),
        fetch(`${API_BASE}/api/analytics/sensor-anomalies`),
      ]);

      if (!trendRes.ok || !componentsRes.ok || !sensorsRes.ok) {
        throw new Error('Failed to fetch analytics');
      }

      const trendData = await trendRes.json();
      const componentsData = await componentsRes.json();
      const sensorsData = await sensorsRes.json();

      setRiskTrend(trendData.trend || []);
      setComponentFailures(componentsData.components || []);
      setSensorAnomalies(sensorsData.sensors || []);
      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('Analytics fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const COLORS = ['#208096', '#E67E22', '#E74C3C', '#9B59B6', '#1ABC9C'];

  if (loading && riskTrend.length === 0) {
    return <div className="analytics-loading">Loading analytics...</div>;
  }

  return (
    <div className="analytics-dashboard">
      <div className="analytics-header">
        <h1>🔍 Advanced Analytics Dashboard</h1>
        <div className="header-info">
          <span className="ws-status">● Live Updates Connected</span>
          {lastUpdate && (
            <span className="last-update">
              Last updated: {lastUpdate.toLocaleTimeString()}
            </span>
          )}
        </div>
      </div>

      {/* Toast Alerts */}
      {alerts.length > 0 && (
        <div className="alerts-container">
          {alerts.map(alert => (
            <div key={alert.id} className={`alert alert-${alert.type}`}>
              {alert.message}
            </div>
          ))}
        </div>
      )}
      
      {error && <div className="analytics-error">Error: {error}</div>}

      <div className="analytics-grid">
        {/* Risk Trend Chart */}
        <div className="analytics-card">
          <h2>Risk Score Trend (7 Days)</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={riskTrend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="avg_risk_score"
                stroke="#208096"
                name="Avg Risk"
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="max_risk_score"
                stroke="#E74C3C"
                name="Max Risk"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Component Failures Distribution */}
        <div className="analytics-card">
          <h2>Component Failure Distribution</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={componentFailures}
                dataKey="failure_count"
                nameKey="component"
                cx="50%"
                cy="50%"
                outerRadius={100}
                label
              >
                {componentFailures.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Component Failure Count Bar Chart */}
        <div className="analytics-card">
          <h2>Failure Count by Component</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={componentFailures}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="component" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="failure_count" fill="#208096" name="Failures" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Sensor Anomalies Table */}
        <div className="analytics-card analytics-table">
          <h2>Anomalies by Sensor</h2>
          <table>
            <thead>
              <tr>
                <th>Sensor</th>
                <th>HIGH</th>
                <th>MEDIUM</th>
                <th>LOW</th>
                <th>Total</th>
              </tr>
            </thead>
            <tbody>
              {sensorAnomalies.map((sensor, idx) => (
                <tr key={idx}>
                  <td className="sensor-name">{sensor.sensor}</td>
                  <td className="high">{sensor.HIGH || 0}</td>
                  <td className="medium">{sensor.MEDIUM || 0}</td>
                  <td className="low">{sensor.LOW || 0}</td>
                  <td className="total">{sensor.total}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Statistics Cards */}
        <div className="analytics-stats">
          <StatCard
            title="Total Risk Trend Readings"
            value={riskTrend.length}
            icon="📈"
          />
          <StatCard
            title="Total Failures Tracked"
            value={componentFailures.reduce((sum, c) => sum + c.failure_count, 0)}
            icon="🔧"
          />
          <StatCard
            title="Anomalies Detected"
            value={sensorAnomalies.reduce((sum, s) => sum + s.total, 0)}
            icon="⚠️"
          />
          <StatCard
            title="Avg Risk Score"
            value={
              riskTrend.length > 0
                ? (
                    riskTrend.reduce((sum, t) => sum + t.avg_risk_score, 0) /
                    riskTrend.length
                  ).toFixed(3)
                : '0.000'
            }
            icon="📊"
          />
        </div>
      </div>

      <button className="refresh-btn" onClick={fetchAnalytics} disabled={loading}>
        {loading ? '⟳ Refreshing...' : '⟳ Manual Refresh'}
      </button>
    </div>
  );
}

function StatCard({ title, value, icon }) {
  return (
    <div className="stat-card">
      <div className="stat-icon">{icon}</div>
      <div className="stat-content">
        <p className="stat-title">{title}</p>
        <p className="stat-value">{value}</p>
      </div>
    </div>
  );
}
