import React, { useState, useEffect } from 'react';

const API_BASE = 'http://localhost:8000';

export default function ReportsTab() {
  const [reports, setReports] = useState({
    executive: null,
    technical: null
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchReports = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [execRes, techRes] = await Promise.all([
        fetch(`${API_BASE}/api/reports/executive-summary`),
        fetch(`${API_BASE}/api/reports/technical-analysis`)
      ]);

      if (!execRes.ok || !techRes.ok) {
        throw new Error('Failed to fetch reports');
      }

      const exec = await execRes.json();
      const tech = await techRes.json();

      setReports({ executive: exec, technical: tech });
    } catch (err) {
      console.error('Report fetch error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReports();
  }, []);

  const handleExportJSON = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/reports/export-json`);
      const data = await res.json();
      
      const element = document.createElement('a');
      element.href = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(data, null, 2));
      element.download = `LampPost_Report_${new Date().toISOString().split('T')[0]}.json`;
      element.click();
    } catch (err) {
      console.error('Export error:', err);
      alert('Failed to export report');
    }
  };

  if (loading) {
    return <div className="reports-tab loading">Loading reports...</div>;
  }

  if (error) {
    return (
      <div className="reports-tab error">
        <p>Error loading reports: {error}</p>
        <button onClick={fetchReports}>Retry</button>
      </div>
    );
  }

  return (
    <div className="reports-tab">
      <h1>📋 Reports & Analytics</h1>

      <div className="reports-grid">
        {/* Executive Summary */}
        {reports.executive && (
          <div className="report-card">
            <h2>Executive Summary</h2>
            <div className="metrics-grid">
              <MetricBox 
                label="Vehicles Monitored" 
                value={reports.executive?.key_metrics?.total_vehicles_monitored || 0}
                icon="🚗"
              />
              <MetricBox 
                label="High Risk" 
                value={reports.executive?.key_metrics?.high_risk_vehicles || 0}
                icon="⚠️"
              />
              <MetricBox 
                label="Anomalies" 
                value={reports.executive?.key_metrics?.total_anomalies_detected || 0}
                icon="📊"
              />
              <MetricBox 
                label="System Uptime" 
                value={reports.executive?.key_metrics?.system_reliability || 'N/A'}
                icon="✅"
              />
            </div>

            {reports.executive?.recommendations && (
              <div className="recommendations">
                <h3>Key Recommendations</h3>
                <ul>
                  {reports.executive.recommendations.map((rec, idx) => (
                    <li key={idx}>{rec}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Technical Report */}
        {reports.technical && (
          <div className="report-card">
            <h2>Technical Analysis</h2>
            <div className="technical-info">
              {reports.technical?.failure_analysis && (
                <>
                  <p>
                    <strong>Most Common Failure:</strong>{' '}
                    {reports.technical.failure_analysis.most_common_failure || 'N/A'}
                  </p>
                  <p>
                    <strong>Total Failure Types:</strong>{' '}
                    {Object.keys(reports.technical.failure_analysis.failure_type_distribution || {}).length}
                  </p>
                </>
              )}
              
              {reports.technical?.model_insights && (
                <>
                  <p>
                    <strong>Features Tracked:</strong>{' '}
                    {reports.technical.model_insights.features_tracked || 'N/A'}
                  </p>
                  <p>
                    <strong>Prediction Horizon:</strong>{' '}
                    {reports.technical.model_insights.prediction_horizon || 'N/A'}
                  </p>
                  <p>
                    <strong>Model Confidence:</strong>{' '}
                    {(reports.technical.model_insights.model_confidence_avg * 100).toFixed(1)}%
                  </p>
                </>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="report-actions">
        <button onClick={fetchReports} disabled={loading} className="btn-primary">
          {loading ? '⟳ Loading...' : '🔄 Refresh Reports'}
        </button>
        <button onClick={handleExportJSON} className="btn-secondary">
          💾 Export as JSON
        </button>
      </div>
    </div>
  );
}

function MetricBox({ label, value, icon }) {
  return (
    <div className="metric-box">
      <div className="metric-icon">{icon}</div>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
    </div>
  );
}
