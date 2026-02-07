import { useEffect, useState } from 'react'
import './App.css'

interface Metrics {
  p50_latency_ms: number;
  p99_latency_ms: number;
  action_distribution: Record<string, number>;
  policy_version: string;
}

function App() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [lastUpdate, setLastUpdate] = useState<string>("");

  const fetchMetrics = async () => {
    try {
      const response = await fetch('http://localhost:8000/metrics');
      const data = await response.json();
      setMetrics(data);
      setLastUpdate(new Date().toLocaleTimeString());
    } catch (error) {
      console.error("Error fetching metrics:", error);
    }
  };

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 2000);
    return () => clearInterval(interval);
  }, []);

  if (!metrics) return <div className="loading">Connecting to StreamSafe Core...</div>;

  return (
    <div className="dashboard">
      <header>
        <h1>StreamSafe-RL Observer</h1>
        <span className="status">● Live</span>
      </header>
      
      <div className="grid">
        <div className="card latency">
          <h2>System Latency</h2>
          <div className="metric-row">
            <div className="metric">
              <span className="label">P50 (Median)</span>
              <span className="value">{metrics.p50_latency_ms}ms</span>
            </div>
            <div className="metric">
              <span className="label">P99 (Slowest 1%)</span>
              <span className="value danger">{metrics.p99_latency_ms}ms</span>
            </div>
          </div>
        </div>

        <div className="card actions">
          <h2>Action Distribution</h2>
          <ul>
            {Object.entries(metrics.action_distribution).map(([action, count]) => (
              <li key={action} className={count > 0 ? "active" : ""}>
                <span className="action-name">{action}</span>
                <span className="action-count">{count}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="card info">
          <h2>System Status</h2>
          <p><strong>Policy:</strong> {metrics.policy_version}</p>
          <p><strong>Last Sync:</strong> {lastUpdate}</p>
        </div>
      </div>
    </div>
  )
}

export default App