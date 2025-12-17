import React, { useState, useEffect } from 'react';
import axios from 'axios';

import { AlertCircle, TrendingDown, Users, AlertTriangle, BarChartBig } from 'lucide-react';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import ReportsTab from './components/ReportsTab';
import './components/ReportsTab.css';
import EngagementBot from './components/EngagementBot';
import './components/EngagementBot.css';
import './App.css';

const API_BASE_URL = 'http://localhost:8000/api';

function App() {
  // Tab state
  const [activeTab, setActiveTab] = useState('overview');

  // Existing state
  const [overview, setOverview] = useState(null);
  const [riskQueue, setRiskQueue] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshCount, setRefreshCount] = useState(0);

  // Fetch Dashboard Data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        const [overviewRes, queueRes, metricsRes] = await Promise.all([
          axios.get(`${API_BASE_URL}/dashboard/overview`),
          axios.get(`${API_BASE_URL}/dashboard/queue?limit=20`),
          axios.get(`${API_BASE_URL}/dashboard/metrics`)
        ]);

        setOverview(overviewRes.data);
        setRiskQueue(queueRes.data.queue);
        setMetrics(metricsRes.data);
      } catch (err) {
        setError(err.message || 'Failed to fetch data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [refreshCount]);

  if (loading && !overview) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
        Loading LampPost Dashboard...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
        Connection Error: {error}
      </div>
    );
  }

  const riskDistributionData = [
    { name: 'High Risk', value: overview?.risk_distribution?.high_risk || 0, color: '#ef4444' },
    { name: 'Medium Risk', value: overview?.risk_distribution?.medium_risk || 0, color: '#f59e0b' },
    { name: 'Low Risk', value: overview?.risk_distribution?.low_risk || 0, color: '#10b981' }
  ];

  const breakdownTrendData = [
    { day: "Mon", prevented: 12, avoided: 8 },
    { day: "Tue", prevented: 15, avoided: 10 },
    { day: "Wed", prevented: 18, avoided: 12 },
    { day: "Thu", prevented: 14, avoided: 9 },
    { day: "Fri", prevented: 20, avoided: 15 },
    { day: "Sat", prevented: 11, avoided: 7 },
    { day: "Sun", prevented: 9, avoided: 5 }
  ];

  return (
    <div className="bg-gray-900 text-white min-h-screen">

      {/* Header with Tab Menu */}
      <header className="bg-gray-800 border-b border-gray-700 shadow-lg">
        <div className="max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
          <h1 className="text-3xl font-bold text-cyan-400 flex items-center gap-2">
            <AlertTriangle className="h-8 w-8" />
            LampPost Automotive
          </h1>

          <div className="flex gap-4">
            <button
              className={`px-4 py-2 rounded-lg font-semibold transition border ${
                activeTab === 'overview'
                  ? 'bg-cyan-600 border-cyan-500'
                  : 'border-gray-600 hover:border-cyan-400'
              }`}
              onClick={() => setActiveTab('overview')}
            >📊 Overview</button>

            <button
              className={`px-4 py-2 rounded-lg font-semibold transition border ${
                activeTab === 'analytics'
                  ? 'bg-cyan-600 border-cyan-500'
                  : 'border-gray-600 hover:border-cyan-400'
              }`}
              onClick={() => setActiveTab('analytics')}
            >🔍 Analytics</button>

            
            <button 
              className={`px-4 py-2 rounded-lg font-semibold transition border ${
                activeTab === 'engagement'
                  ? 'bg-cyan-600 border-cyan-500'
                  : 'border-gray-600 hover:border-cyan-400'
              }`}
              onClick={() => setActiveTab('engagement')}
            >
              💬 Engagement Bot
            </button>

            {/* 📋 Reports Tab → Added */}
            <button
              className={`px-4 py-2 rounded-lg font-semibold transition border ${
                activeTab === 'reports'
                  ? 'bg-cyan-600 border-cyan-500'
                  : 'border-gray-600 hover:border-cyan-400'
              }`}
              onClick={() => setActiveTab('reports')}
            >
              📋 Reports
            </button>

            <button
              onClick={() => setRefreshCount(r => r + 1)}
              className="bg-cyan-600 hover:bg-cyan-700 px-4 py-2 rounded-lg font-semibold transition"
            >
              🔄 Refresh
            </button>
          </div>
        </div>
      </header>

      {/* 🔀 Tab Switching */}
      {activeTab === 'analytics' ? (
        <AnalyticsDashboard />
      ) : activeTab === 'reports' ? (
        <ReportsTab />
      ) : activeTab === 'engagement' ? (
        <EngagementBot />
      ) : null}

       : (
        <main className="max-w-7xl mx-auto px-6 py-8">

          {/* Dashboard KPI Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            {/* Vehicles Monitored */}
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 hover:border-cyan-500 transition">
              <p className="text-gray-400 text-sm font-semibold">Vehicles Monitored</p>
              <h3 className="text-3xl font-bold">{overview?.vehicles_monitored}</h3>
              <Users className="h-10 w-10 text-cyan-500 opacity-50 mt-2" />
            </div>

            {/* High Risk */}
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 hover:border-red-500 transition">
              <p className="text-gray-400 text-sm font-semibold">High Risk Vehicles</p>
              <h3 className="text-3xl font-bold text-red-400">
                {overview?.risk_distribution?.high_risk}
              </h3>
              <AlertCircle className="h-10 w-10 text-red-500 opacity-50 mt-2" />
            </div>

            {/* Anomalies */}
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 hover:border-yellow-500 transition">
              <p className="text-gray-400 text-sm font-semibold">Anomalies This Week</p>
              <h3 className="text-3xl font-bold text-yellow-400">
                {overview?.total_anomalies_detected}
              </h3>
              <AlertTriangle className="h-10 w-10 text-yellow-500 opacity-50 mt-2" />
            </div>

            {/* Prevented */}
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 hover:border-green-500 transition">
              <p className="text-gray-400 text-sm font-semibold">Breakdowns Prevented</p>
              <h3 className="text-3xl font-bold text-green-400">
                {overview?.breakdowns_prevented_estimate}
              </h3>
              <TrendingDown className="h-10 w-10 text-green-500 opacity-50 mt-2" />
            </div>
          </div>

          {/* Risk Pie + Breakdown Trend */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
              <h2 className="text-xl font-bold text-cyan-400 mb-4">Risk Distribution</h2>
              <ResponsiveContainer height={250}>
                <PieChart>
                  <Pie data={riskDistributionData} dataKey="value" outerRadius={100}>
                    {riskDistributionData.map((e, i) => (
                      <Cell key={i} fill={e.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
              <h2 className="text-xl font-bold text-cyan-400 mb-4">Breakdown Prevention Trend</h2>
              <ResponsiveContainer height={250}>
                <BarChart data={breakdownTrendData}>
                  <XAxis dataKey="day" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="prevented" fill="#10b981" />
                  <Bar dataKey="avoided" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* High Risk Vehicles Table */}
          <div className="mt-8 bg-gray-800 border border-gray-700 rounded-lg p-6">
            <h2 className="text-xl font-bold mb-6 text-cyan-400">🔴 High-Risk Vehicles Queue</h2>

            {riskQueue.length === 0 ? (
              <div className="text-center py-12 text-gray-400">
                <p>No vehicles with predictions yet. Upload telematics data to start analysis.</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="border-b border-gray-700">
                    <tr>
                      <th className="text-left py-3 px-4 font-semibold text-gray-300">Vehicle ID</th>
                      <th className="text-left py-3 px-4 font-semibold text-gray-300">Model</th>
                      <th className="text-left py-3 px-4 font-semibold text-gray-300">Owner</th>
                      <th className="text-right py-3 px-4 font-semibold text-gray-300">Risk Score</th>
                      <th className="text-left py-3 px-4 font-semibold text-gray-300">Failure Type</th>
                      <th className="text-center py-3 px-4 font-semibold text-gray-300">TTF (Days)</th>
                      <th className="text-center py-3 px-4 font-semibold text-gray-300">Action</th>
                    </tr>
                  </thead>

                  <tbody>
                    {riskQueue.map((vehicle, index) => (
                      <tr key={index} className="border-b border-gray-700 hover:bg-gray-700 transition">
                        <td className="py-3 px-4 font-mono text-cyan-400">{vehicle.vehicle_id}</td>
                        <td className="py-3 px-4 text-gray-300">{vehicle.model}</td>
                        <td className="py-3 px-4 text-gray-300">{vehicle.owner_name}</td>
                        <td className="py-3 px-4 text-right">
                          <div className="flex items-center justify-end gap-2">
                            <div
                              className="w-2 h-2 rounded-full"
                              style={{
                                backgroundColor:
                                  vehicle.risk_score > 0.7
                                    ? '#ef4444'
                                    : vehicle.risk_score > 0.4
                                    ? '#f59e0b'
                                    : '#10b981'
                              }}
                            />
                            <span className="font-bold">
                              {(vehicle.risk_score * 100).toFixed(0)}%
                            </span>
                          </div>
                        </td>
                        <td className="py-3 px-4 capitalize text-gray-300">{vehicle.failure_type}</td>
                        <td className="py-3 px-4 text-center font-semibold">
                          <span className="bg-red-900 text-red-200 px-2 py-1 rounded">
                            {vehicle.ttf_days} days
                          </span>
                        </td>
                        <td className="py-3 px-4 text-center">
                          <button className="bg-cyan-600 hover:bg-cyan-700 text-white px-3 py-1 rounded text-xs font-semibold transition">
                            Engage
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>


          {/* Footer */}
          <div className="mt-8 text-center text-gray-400 text-sm border-t border-gray-700 pt-6">
            <p>Last Updated: {new Date(overview?.timestamp).toLocaleString()}</p>
            <p className="mt-2">
              Model Accuracy: {(metrics?.model_statistics?.model_accuracy * 100).toFixed(2)}%
              &nbsp;| ROC-AUC: {metrics?.model_statistics?.roc_auc}
            </p>
          </div>

        </main>
      )}
    </div>
  );
}

export default App;
