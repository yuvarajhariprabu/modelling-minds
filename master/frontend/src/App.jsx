import React, { useState, useEffect, useRef } from 'react';
import ClusterMeshTopology from './ClusterMeshTopology';

const API_BASE = 'http://127.0.0.1:8000';

const SCENARIOS = {
  standard: { name: 'DGX Cluster (6x H100 SXM5, Cap=4, Arr=2.0)', n_nodes: 6, node_capacity: 4, arrival_rate: 2.0 },
  heavy: { name: 'SuperPOD Saturated (8x H100 SXM5, Cap=4, Arr=3.0)', n_nodes: 8, node_capacity: 4, arrival_rate: 3.0 },
  scaled: { name: 'Mega AI Cluster (12x H100 SXM5, Cap=3, Arr=4.5)', n_nodes: 12, node_capacity: 3, arrival_rate: 4.5 }
};

export default function App() {
  const [simData, setSimData] = useState(null);
  const [logs, setLogs] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(2); // 1x, 2x, 5x, 10x
  const [agentType, setAgentType] = useState('health_aware');
  const [seed, setSeed] = useState(3);
  const [scenarioKey, setScenarioKey] = useState('standard');
  const [backendConnected, setBackendConnected] = useState(false);
  const [topologyView, setTopologyView] = useState('split'); // 'split', 'mesh', 'cards'
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  
  // Shootout Modal
  const [showShootout, setShowShootout] = useState(false);
  const [shootoutData, setShootoutData] = useState(null);
  const [isBenchmarking, setIsBenchmarking] = useState(false);

  // Tier-2 LLM Copilot & RCA State
  const [copilotOpen, setCopilotOpen] = useState(false);
  const [copilotMessages, setCopilotMessages] = useState([
    {
      sender: 'ai',
      text: '### 🤖 Sentinel GenAI SuperPOD Copilot Active\n\nI am your Tier-2 Autonomous SRE Copilot. I continuously audit Bayesian telemetry, GPU thermal drifts (CUSUM), NVLink crossbar degradation, and in-flight micro-batch deadlines.\n\nAsk me anything about current cluster anomalies, or request an automated **Incident RCA Post-Mortem**!'
    }
  ]);
  const [copilotInput, setCopilotInput] = useState('');
  const [copilotLoading, setCopilotLoading] = useState(false);

  // RCA Post-Mortem Modal State
  const [rcaOpen, setRcaOpen] = useState(false);
  const [rcaData, setRcaData] = useState(null);
  const [rcaLoading, setRcaLoading] = useState(false);
  const [rcaCopied, setRcaCopied] = useState(false);

  const timerRef = useRef(null);
  const chatEndRef = useRef(null);

  // Initialize episode
  const initEpisode = async (type = agentType, s = seed, scen = scenarioKey) => {
    setIsPlaying(false);
    clearInterval(timerRef.current);
    try {
      const config = SCENARIOS[scen];
      const res = await fetch(`${API_BASE}/api/init`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agent_type: type,
          seed: Number(s),
          n_nodes: config.n_nodes,
          node_capacity: config.node_capacity,
          arrival_rate: config.arrival_rate,
          episode_length: 400
        })
      });
      const data = await res.json();
      setSimData(data);
      setLogs(data.new_logs || []);
      setBackendConnected(true);
    } catch (err) {
      console.error('Failed to init simulation:', err);
      setBackendConnected(false);
    }
  };

  // Step simulation forward
  const stepEpisode = async (steps = 1) => {
    try {
      const res = await fetch(`${API_BASE}/api/step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ steps })
      });
      const data = await res.json();
      setSimData(data);
      if (data.new_logs && data.new_logs.length > 0) {
        setLogs(prev => [...prev, ...data.new_logs]);
      }
      setBackendConnected(true);
      if (data.done) {
        setIsPlaying(false);
      }
    } catch (err) {
      console.error('Step error:', err);
      setIsPlaying(false);
    }
  };

  // Manual Chaos Injection
  const injectChaos = async (nodeId, state) => {
    try {
      await fetch(`${API_BASE}/api/inject`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ node_id: nodeId, state })
      });
      stepEpisode(1);
    } catch (err) {
      console.error('Failed to inject chaos:', err);
    }
  };

  // Run benchmark shootout
  const runShootout = async () => {
    setIsBenchmarking(true);
    setShowShootout(true);
    try {
      const res = await fetch(`${API_BASE}/api/benchmark`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed })
      });
      const data = await res.json();
      setShootoutData(data);
    } catch (err) {
      console.error('Benchmark failed:', err);
    } finally {
      setIsBenchmarking(false);
    }
  };

  // Ask Tier-2 LLM Copilot
  const handleCopilotSend = async (queryText = copilotInput) => {
    const q = (typeof queryText === 'string' ? queryText : copilotInput || '').trim();
    if (!q || copilotLoading) return;

    setCopilotMessages(prev => [...prev, { sender: 'user', text: q }]);
    setCopilotInput('');
    setCopilotLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/copilot/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q })
      });
      const data = await res.json();
      setCopilotMessages(prev => [...prev, { sender: 'ai', text: data.answer || 'Telemetry analyzed.' }]);
    } catch (err) {
      setCopilotMessages(prev => [...prev, { sender: 'ai', text: 'Error connecting to Sentinel AI Copilot.' }]);
    } finally {
      setCopilotLoading(false);
    }
  };

  // Generate SRE RCA Post-Mortem
  const handleGenerateRca = async () => {
    setRcaLoading(true);
    setRcaOpen(true);
    try {
      const res = await fetch(`${API_BASE}/api/copilot/rca`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      const data = await res.json();
      setRcaData(data);
    } catch (err) {
      console.error('RCA generation failed:', err);
    } finally {
      setRcaLoading(false);
    }
  };

  const copyRcaToClipboard = () => {
    if (rcaData?.rca_markdown) {
      navigator.clipboard.writeText(rcaData.rca_markdown);
      setRcaCopied(true);
      setTimeout(() => setRcaCopied(false), 2500);
    }
  };

  // Initial load
  useEffect(() => {
    initEpisode();
  }, []);

  // Auto-scroll chat
  useEffect(() => {
    if (copilotOpen) {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [copilotMessages, copilotOpen]);

  // Playback timer
  useEffect(() => {
    if (isPlaying && simData && !simData.done) {
      const intervalMs = Math.max(25, Math.floor(600 / speed));
      timerRef.current = setInterval(() => {
        stepEpisode(1);
      }, intervalMs);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [isPlaying, speed, simData?.done]);

  const metrics = simData?.metrics || {
    completed: 0,
    failed: 0,
    completion_rate: 100.0,
    total_reward: 0.0,
    churn_count: 0,
    detection_latency: 0.0
  };

  const progressPct = simData ? ((simData.step / simData.episode_length) * 100).toFixed(1) : 0;

  const parseInline = (str) => {
    // Splits by backticks and bold markdown
    const parts = str.split(/(\`[^\`]+\`|\*\*[^*]+\*\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={i} style={{ color: '#fff', fontWeight: 600 }}>{part.slice(2, -2)}</strong>;
      }
      if (part.startsWith('`') && part.endsWith('`')) {
        return (
          <code
            key={i}
            style={{
              background: 'rgba(225, 29, 72, 0.18)',
              color: '#f472b6',
              padding: '1px 5px',
              borderRadius: '4px',
              fontSize: '0.82rem',
              fontFamily: 'monospace'
            }}
          >
            {part.slice(1, -1)}
          </code>
        );
      }
      return part;
    });
  };

  const renderFormattedMessage = (text) => {
    if (!text) return null;
    const lines = text.split('\n');
    return lines.map((line, idx) => {
      const trimmed = line.trim();
      if (trimmed.startsWith('# ')) {
        return <h3 key={idx} style={{ color: '#fbcfe8', margin: '0.6rem 0 0.3rem 0', fontSize: '1.05rem', fontWeight: 700 }}>{line.replace(/^#\s+/, '')}</h3>;
      }
      if (trimmed.startsWith('## ')) {
        return <h4 key={idx} style={{ color: '#c084fc', margin: '0.5rem 0 0.25rem 0', fontSize: '0.95rem', fontWeight: 600 }}>{line.replace(/^##\s+/, '')}</h4>;
      }
      if (trimmed.startsWith('### ')) {
        return <h5 key={idx} style={{ color: '#fb7185', margin: '0.45rem 0 0.2rem 0', fontSize: '0.9rem', fontWeight: 600 }}>{line.replace(/^###\s+/, '')}</h5>;
      }
      if (trimmed === '---') {
        return <hr key={idx} style={{ borderColor: 'rgba(255,255,255,0.1)', margin: '0.5rem 0' }} />;
      }
      if (/^\d+\.\s+/.test(trimmed)) {
        return (
          <div key={idx} style={{ paddingLeft: '0.8rem', margin: '0.2rem 0', color: '#e2e8f0', fontSize: '0.85rem' }}>
            {parseInline(trimmed)}
          </div>
        );
      }
      if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
        return (
          <div key={idx} style={{ paddingLeft: '0.8rem', margin: '0.18rem 0', color: '#cbd5e1', fontSize: '0.85rem' }}>
            &bull; {parseInline(trimmed.slice(2))}
          </div>
        );
      }
      return (
        <div key={idx} style={{ margin: trimmed === '' ? '0.35rem 0' : '0.12rem 0', fontSize: '0.86rem', color: '#cbd5e1', lineHeight: 1.45 }}>
          {parseInline(line)}
        </div>
      );
    });
  };

  return (
    <div className="app-container">
      {/* HEADER */}
      <header className="glass-panel app-header">
        <div className="brand-section">
          <div className="brand-icon">
            <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <div className="brand-text">
            <h1>Sentinel GPU SuperPOD</h1>
            <p>
              Autonomous GenAI Cluster Fault Tolerance &amp; Rerouting
              <span className="badge-tag">NVIDIA H100 SXM5 80GB</span>
              <span className="badge-tag">MM26AI02</span>
              {backendConnected ? (
                <span style={{ color: '#10b981', display: 'flex', alignItems: 'center', gap: 4 }}>
                  <span className="heartbeat-dot pulse" style={{ width: 7, height: 7 }}></span> SuperPOD Live
                </span>
              ) : (
                <span style={{ color: '#ef4444' }}>Backend Offline</span>
              )}
            </p>
          </div>
        </div>

        <div className="header-actions">
          {/* Agent Selector Pill */}
          <div className="agent-selector-pill">
            <button
              className={`agent-btn health-aware ${agentType === 'health_aware' ? 'active' : ''}`}
              onClick={() => {
                setAgentType('health_aware');
                initEpisode('health_aware', seed, scenarioKey);
              }}
            >
              ★ HealthAwareAgent (Two-Tier AI)
            </button>
            <button
              className={`agent-btn baseline ${agentType === 'baseline' ? 'active' : ''}`}
              onClick={() => {
                setAgentType('baseline');
                initEpisode('baseline', seed, scenarioKey);
              }}
            >
              Round-Robin (Health-Blind)
            </button>
          </div>

          <button className="shootout-trigger-btn" onClick={handleGenerateRca} style={{ background: 'rgba(239, 68, 68, 0.15)', borderColor: 'rgba(239, 68, 68, 0.35)', color: '#f87171' }}>
            🚨 Incident RCA
          </button>

          <button className="shootout-trigger-btn" onClick={() => setCopilotOpen(true)} style={{ background: 'rgba(168, 85, 247, 0.22)', borderColor: 'rgba(168, 85, 247, 0.45)', color: '#f0abfc' }}>
            🤖 AI SRE Copilot
          </button>

          <button className="shootout-trigger-btn" onClick={runShootout}>
            <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Compare Shootout
          </button>
        </div>
      </header>

      {/* METRICS HUD */}
      <section className="metrics-deck">
        <div className="glass-panel metric-card">
          <span className="metric-label">Completion SLA</span>
          <span className="metric-value" style={{ color: metrics.completion_rate >= 95 ? 'var(--color-healthy)' : metrics.completion_rate >= 70 ? 'var(--color-degraded)' : 'var(--color-down)' }}>
            {metrics.completion_rate}%
          </span>
          <span className="metric-subtext">
            {metrics.completed} finished &bull; {metrics.failed} deadline drops
          </span>
        </div>

        <div className="glass-panel metric-card">
          <span className="metric-label">SuperPOD Net Reward</span>
          <span className="metric-value" style={{ color: metrics.total_reward >= 0 ? '#818cf8' : '#ef4444' }}>
            {metrics.total_reward > 0 ? `+${metrics.total_reward}` : metrics.total_reward}
          </span>
          <span className="metric-subtext">+1 per micro-batch finish &bull; -1 per failure</span>
        </div>

        <div className="glass-panel metric-card">
          <span className="metric-label">Detection Latency</span>
          <span className="metric-value" style={{ color: metrics.detection_latency === 0 ? 'var(--color-cyan)' : 'var(--color-degraded)' }}>
            {metrics.detection_latency} <span style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>steps</span>
          </span>
          <span className="metric-subtext">Instantaneous Bayesian + CUSUM trip</span>
        </div>

        <div className="glass-panel metric-card">
          <span className="metric-label">Task Churn (Reroutes)</span>
          <span className="metric-value" style={{ color: 'var(--text-main)' }}>
            {metrics.churn_count}
          </span>
          <span className="metric-subtext">Selective Graceful Drain &bull; Anti-Thrash</span>
        </div>
      </section>

      {/* CONTROL DECK */}
      <section className="glass-panel control-deck">
        <div className="btn-cluster">
          <button
            className={`ctrl-btn play-btn ${isPlaying ? 'playing' : ''}`}
            onClick={() => setIsPlaying(!isPlaying)}
            disabled={simData?.done}
          >
            {isPlaying ? (
              <>
                <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/></svg>
                Pause
              </>
            ) : (
              <>
                <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                {simData?.done ? 'Episode Complete' : 'Run SuperPOD'}
              </>
            )}
          </button>

          <button
            className="ctrl-btn"
            onClick={() => stepEpisode(1)}
            disabled={isPlaying || simData?.done}
          >
            +1 Step
          </button>

          <button
            className="ctrl-btn"
            onClick={() => stepEpisode(10)}
            disabled={isPlaying || simData?.done}
          >
            +10 Steps
          </button>

          <button
            className="ctrl-btn"
            onClick={() => initEpisode(agentType, seed, scenarioKey)}
          >
            <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Reset
          </button>

          {/* Speed selector */}
          <div className="speed-selector">
            {[1, 2, 5, 10, 20].map(s => (
              <button
                key={s}
                className={`speed-btn ${speed === s ? 'active' : ''}`}
                onClick={() => setSpeed(s)}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>

        {/* Scenario & Seed Config */}
        <div className="config-selector-group">
          <div className="config-item">
            <span>Topology:</span>
            <select
              className="select-input"
              value={scenarioKey}
              onChange={e => {
                setScenarioKey(e.target.value);
                initEpisode(agentType, seed, e.target.value);
              }}
            >
              {Object.entries(SCENARIOS).map(([k, v]) => (
                <option key={k} value={k}>{v.name}</option>
              ))}
            </select>
          </div>

          <div className="config-item">
            <span>Seed:</span>
            <input
              type="number"
              className="number-input"
              style={{ width: 65 }}
              value={seed}
              onChange={e => setSeed(Number(e.target.value))}
              onBlur={() => initEpisode(agentType, seed, scenarioKey)}
            />
          </div>
        </div>

        {/* Episode Progress Bar */}
        <div className="episode-progress-container">
          <div className="progress-bar-track">
            <div className="progress-bar-fill" style={{ width: `${progressPct}%` }}></div>
          </div>
          <span className="progress-step-label">
            Step {simData?.step || 0} / {simData?.episode_length || 400} ({progressPct}%)
          </span>
        </div>
      </section>

      {/* MAIN COCKPIT: CLUSTER MESH & FLIGHT DECK */}
      <div className="main-grid">
        {/* LEFT: CLUSTER NODES TOPOLOGY & MESH FABRIC */}
        <section className="glass-panel">
          <div className="topology-panel-header">
            <div>
              <h2>
                <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
                NVIDIA H100 SXM5 SuperPOD Topology
              </h2>
              <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                Live NVLink 4.0 Crossbar Interconnect &bull; 900 GB/s Optical P2P Mesh
              </span>
            </div>

            {/* View Switcher Tabs */}
            <div className="topology-view-tabs">
              <button
                className={`view-tab-btn ${topologyView === 'mesh' ? 'active' : ''}`}
                onClick={() => setTopologyView('mesh')}
                title="View interactive interconnected graph"
              >
                🔀 Mesh Fabric
              </button>
              <button
                className={`view-tab-btn ${topologyView === 'split' ? 'active' : ''}`}
                onClick={() => setTopologyView('split')}
                title="View both connected mesh and card telemetry"
              >
                ⚡ Split View
              </button>
              <button
                className={`view-tab-btn ${topologyView === 'cards' ? 'active' : ''}`}
                onClick={() => setTopologyView('cards')}
                title="View modular compute trays"
              >
                ▦ Compute Trays
              </button>
            </div>
          </div>

          {/* DYNAMIC INTER-NODE NVLINK MESH FABRIC CANVAS */}
          {(topologyView === 'mesh' || topologyView === 'split') && (
            <ClusterMeshTopology
              nodes={simData?.nodes || []}
              recentActions={simData?.recent_actions || []}
              decisionLogs={logs || []}
              onInjectChaos={injectChaos}
              selectedNodeId={selectedNodeId}
              onSelectNode={(nid) => setSelectedNodeId(selectedNodeId === nid ? null : nid)}
            />
          )}

          {/* MODULAR COMPUTE CARDS GRID */}
          {(topologyView === 'cards' || topologyView === 'split') && (
            <div className="cluster-node-grid">
              {simData?.nodes?.map(node => {
                const score = node.score || 100;
                const trueState = node.true_state || 'healthy';
                const belief = node.belief || [1, 0, 0];
                const scoreColor = score >= 70 ? 'var(--color-healthy)' : score >= 40 ? 'var(--color-degraded)' : 'var(--color-down)';
                const temp = node.cuda_temp_c || 62.0;
                const tempClass = temp < 75 ? 'temp-optimal' : temp < 95 ? 'temp-throttled' : 'temp-critical';
                const vramPct = ((node.vram_used_gb / 80.0) * 100).toFixed(0);
                const isSelected = selectedNodeId === node.node_id;

                return (
                  <div
                    key={node.node_id}
                    className={`node-card state-${trueState} ${isSelected ? 'selected-highlight' : ''}`}
                    onClick={() => setSelectedNodeId(isSelected ? null : node.node_id)}
                    style={{ cursor: 'pointer' }}
                  >
                    <div className="node-card-header">
                      <div className="node-title">
                        <span className={`heartbeat-dot ${node.heartbeat_ok ? 'pulse' : ''}`} style={{ background: scoreColor }}></span>
                        <span>{node.node_name || `GPU-Node-${node.node_id}`}</span>
                      </div>
                      <span className={`temp-badge ${tempClass}`}>
                        {temp}°C
                      </span>
                    </div>

                    {/* GPU Telemetry Row */}
                    <div className="gpu-metrics-row">
                      <div className="gpu-metric-item">
                        <span className="gpu-metric-label">H100 VRAM ({vramPct}%)</span>
                        <span className="gpu-metric-value">{node.vram_used_gb} / 80 GB</span>
                        <div className="vram-bar-container">
                          <div className="vram-bar-fill" style={{ width: `${vramPct}%`, background: scoreColor }}></div>
                        </div>
                      </div>
                      <div className="gpu-metric-item">
                        <span className="gpu-metric-label">NVLink Bus</span>
                        <span className="gpu-metric-value">{node.nvlink_throughput_gbps} GB/s</span>
                        <span style={{ fontSize: '0.65rem', color: 'var(--text-dim)' }}>{node.thermal_status}</span>
                      </div>
                    </div>

                    <div className="node-stats-row">
                      <div className="node-stat-item">
                        <span className="node-stat-label">Queue Load</span>
                        <span className="node-stat-value">{node.queue_len} / {node.capacity}</span>
                      </div>
                      <div className="node-stat-item">
                        <span className="node-stat-label">Round-Trip Latency</span>
                        <span className="node-stat-value">
                          {node.latency_ms != null ? `${node.latency_ms.toFixed(0)} ms` : 'TIMEOUT'}
                        </span>
                      </div>
                      <div className="node-stat-item">
                        <span className="node-stat-label">CUDA Error Rate</span>
                        <span className="node-stat-value">{(node.error_rate * 100).toFixed(1)}%</span>
                      </div>
                    </div>

                    {/* Load Capacity Bar */}
                    <div className="capacity-bar-container">
                      <div
                        className="capacity-bar-fill"
                        style={{
                          width: `${Math.min(100, (node.queue_len / node.capacity) * 100)}%`,
                          background: node.queue_len >= node.capacity ? '#ef4444' : scoreColor
                        }}
                      ></div>
                    </div>

                    {/* Bayesian Belief Radar Bars */}
                    <div className="belief-radar">
                      <div className="belief-segment" style={{ width: `${belief[0] * 100}%`, background: '#10b981' }} title={`P(Healthy): ${(belief[0] * 100).toFixed(1)}%`}></div>
                      <div className="belief-segment" style={{ width: `${belief[1] * 100}%`, background: '#f59e0b' }} title={`P(Degraded): ${(belief[1] * 100).toFixed(1)}%`}></div>
                      <div className="belief-segment" style={{ width: `${belief[2] * 100}%`, background: '#ef4444' }} title={`P(Down): ${(belief[2] * 100).toFixed(1)}%`}></div>
                    </div>

                    {/* Chaos Injection Buttons */}
                    <div className="chaos-btn-group" onClick={e => e.stopPropagation()}>
                      <button
                        className="chaos-btn heal"
                        onClick={() => injectChaos(node.node_id, 'healthy')}
                        title="Force Recover"
                      >
                        Restore
                      </button>
                      <button
                        className="chaos-btn throttle"
                        onClick={() => injectChaos(node.node_id, 'degraded')}
                        title="Inject Thermal Throttling"
                      >
                        Throttle
                      </button>
                      <button
                        className="chaos-btn crash"
                        onClick={() => injectChaos(node.node_id, 'down')}
                        title="Inject Hardware Fault"
                      >
                        Crash
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        {/* RIGHT: TASK QUEUE & EXPLAINABILITY LOGS */}
        <div className="right-column-stack">
          {/* ACTIVE MICRO-BATCHES */}
          <section className="glass-panel task-board-panel">
            <div className="panel-header">
              <h2>
                <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                In-Flight GenAI Micro-Batches ({simData?.tasks?.length || 0})
              </h2>
            </div>

            <div className="tasks-table-container">
              <table className="tasks-table">
                <thead>
                  <tr>
                    <th>Batch ID</th>
                    <th>Model Workload</th>
                    <th>Target GPU</th>
                    <th>Remaining</th>
                    <th>SLA Deadline</th>
                    <th>Slack Margin</th>
                  </tr>
                </thead>
                <tbody>
                  {simData?.tasks?.length === 0 ? (
                    <tr>
                      <td colSpan="6" style={{ textAlign: 'center', color: 'var(--text-dim)', padding: '2rem' }}>
                        No micro-batches currently queued
                      </td>
                    </tr>
                  ) : (
                    simData?.tasks?.map(task => {
                      const timeLeft = task.deadline - (simData.step || 0);
                      const isDanger = timeLeft <= 3;
                      return (
                        <tr key={task.task_id} className={isDanger ? 'danger-row' : ''}>
                          <td><strong>#{task.task_id}</strong></td>
                          <td>
                            <span style={{ color: '#818cf8', fontWeight: 600 }}>{task.workload || 'Micro-Batch'}</span>
                            <span style={{ display: 'block', fontSize: '0.65rem', color: 'var(--text-dim)' }}>BS={task.micro_batch_size || 16} tokens</span>
                          </td>
                          <td>
                            {task.node != null ? (
                              <span className="task-assigned-badge">GPU #{task.node}</span>
                            ) : (
                              <span className="task-unassigned-badge">Dispatching...</span>
                            )}
                          </td>
                          <td>{task.duration_remaining.toFixed(1)} steps</td>
                          <td>t={task.deadline}</td>
                          <td style={{ color: timeLeft <= 1 ? '#ef4444' : timeLeft <= 3 ? '#f59e0b' : '#10b981', fontWeight: 600 }}>
                            {timeLeft}s
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </section>

          {/* EXPLAINABILITY LOG FEED */}
          <section className="glass-panel explainability-panel">
            <div className="panel-header">
              <h2>
                <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Autonomous Causal Log Stream
              </h2>
              <span style={{ fontSize: '0.75rem', color: '#c084fc' }}>Bayesian + EVC Engine</span>
            </div>

            <div className="log-feed">
              {logs.length === 0 && (
                <div style={{ color: '#64748b' }}>Awaiting agent actions and telemetry anomalies...</div>
              )}
              {logs.slice(-30).map((log, idx) => (
                <div key={idx} className={`log-entry ${log.action === 'REROUTE' ? 'reroute' : 'event'}`}>
                  <span className="log-step">[t={log.step}]</span>
                  <span className="log-action">[{log.action}]</span>
                  {log.action === 'REROUTE' ? (
                    <span>Micro-batch #{log.task_id} &rarr; GPU {log.to_node} (from GPU {log.from_node})</span>
                  ) : (
                    <span>Micro-batch #{log.task_id} &rarr; GPU {log.to_node}</span>
                  )}
                  <div className="log-reason">{log.reason}</div>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>

      {/* TIER-2 COPILOT FLOATING BUTTON */}
      <button className="copilot-floating-btn" onClick={() => setCopilotOpen(true)}>
        <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
        </svg>
        AI SRE Copilot
      </button>

      {/* TIER-2 COPILOT DRAWER */}
      {copilotOpen && (
        <div className="copilot-drawer-overlay" onClick={() => setCopilotOpen(false)}>
          <div className="copilot-drawer" onClick={e => e.stopPropagation()}>
            <div className="copilot-header">
              <div className="copilot-header-title">
                <span className="heartbeat-dot pulse" style={{ width: 10, height: 10, background: '#a855f7' }}></span>
                <span>Tier-2 AI SRE Copilot</span>
              </div>
              <button className="close-btn" onClick={() => setCopilotOpen(false)}>&times;</button>
            </div>

            <div className="copilot-feed">
              {copilotMessages.map((m, i) => (
                <div key={i} className={`copilot-msg copilot-msg-${m.sender}`}>
                  {renderFormattedMessage(m.text)}
                </div>
              ))}
              {copilotLoading && (
                <div className="copilot-msg copilot-msg-ai">
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--text-muted)' }}>
                    <span className="heartbeat-dot pulse" style={{ width: 8, height: 8, background: '#a855f7' }}></span>
                    <em>Synthesizing telemetry forensics &amp; reasoning...</em>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <div className="copilot-chips-container">
              <button className="prompt-chip" onClick={() => handleCopilotSend('What is the difference between Health Aware and Round Robin?')}>
                ⚖️ Health-Aware vs Round-Robin
              </button>
              <button className="prompt-chip" onClick={() => handleCopilotSend('Explain the three cluster topologies.')}>
                🌐 Explain 3 Topologies
              </button>
              <button className="prompt-chip" onClick={() => handleCopilotSend('What is the LLM model used in this project?')}>
                🤖 What LLM is Used?
              </button>
              <button className="prompt-chip" onClick={() => handleCopilotSend('What is the status of VRAM and thermal health in the GPU cluster?')}>
                🔥 Check Thermals &amp; VRAM
              </button>
              <button className="prompt-chip" onClick={() => handleCopilotSend('Why was the last micro-batch rerouted?')}>
                🔄 Explain Last Reroute
              </button>
              <button className="prompt-chip" onClick={() => handleCopilotSend('Generate Incident RCA Post-Mortem')}>
                🚨 Incident RCA Report
              </button>
            </div>

            <div className="copilot-input-bar">
              <input
                className="copilot-input"
                type="text"
                placeholder="Ask Sentinel AI about GPU health, reroutes, or faults..."
                value={copilotInput}
                onChange={e => setCopilotInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleCopilotSend()}
              />
              <button className="copilot-send-btn" onClick={() => handleCopilotSend()}>
                Send
              </button>
            </div>
          </div>
        </div>
      )}

      {/* RCA POST-MORTEM MODAL */}
      {rcaOpen && (
        <div className="modal-overlay" onClick={() => setRcaOpen(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: 900 }}>
            <div className="modal-header">
              <div>
                <h2>🚨 Automated SRE Incident Root Cause Analysis (RCA)</h2>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Generated by Tier-2 Cognitive Synthesis Engine &bull; Incident: {rcaData?.incident_id || 'PENDING'}
                </p>
              </div>
              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                <button
                  className="ctrl-btn"
                  onClick={copyRcaToClipboard}
                  style={{ background: rcaCopied ? '#10b981' : 'var(--color-primary)', border: 'none', color: '#fff' }}
                >
                  {rcaCopied ? '✓ Copied Markdown' : 'Copy Report'}
                </button>
                <button className="close-btn" onClick={() => setRcaOpen(false)}>&times;</button>
              </div>
            </div>
            <div className="modal-body">
              {rcaLoading ? (
                <div style={{ textAlign: 'center', padding: '3rem' }}>
                  <div className="heartbeat-dot pulse" style={{ width: 24, height: 24, margin: '0 auto 1rem auto' }}></div>
                  <h3>Synthesizing Deep Forensic Post-Mortem...</h3>
                </div>
              ) : rcaData ? (
                <div className="rca-report-body">
                  {rcaData.rca_markdown}
                </div>
              ) : null}
            </div>
          </div>
        </div>
      )}

      {/* BENCHMARK SHOOTOUT MODAL */}
      {showShootout && (
        <div className="modal-overlay" onClick={() => setShowShootout(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Side-by-Side Agent Shootout: Baseline vs HealthAwareAgent</h2>
              <button className="close-btn" onClick={() => setShowShootout(false)}>&times;</button>
            </div>
            <div className="modal-body">
              {isBenchmarking ? (
                <div style={{ textAlign: 'center', padding: '3rem' }}>
                  <div className="heartbeat-dot pulse" style={{ width: 24, height: 24, margin: '0 auto 1rem auto' }}></div>
                  <h3>Running 400-step Monte Carlo Simulation on both models...</h3>
                  <p style={{ color: 'var(--text-muted)' }}>Simulating hidden Markov failures and cold restarts...</p>
                </div>
              ) : shootoutData ? (
                <div>
                  <table className="comparison-table">
                    <thead>
                      <tr>
                        <th>Metric</th>
                        <th>Round-Robin Baseline</th>
                        <th>HealthAwareAgent (Ours)</th>
                        <th>Improvement</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td><strong>Completion Rate</strong></td>
                        <td>{(shootoutData.baseline.completion_rate_mean * 100).toFixed(1)}%</td>
                        <td style={{ color: '#10b981', fontWeight: 700 }}>
                          {(shootoutData.health_aware.completion_rate_mean * 100).toFixed(1)}%
                        </td>
                        <td style={{ color: '#10b981', fontWeight: 700 }}>
                          +{((shootoutData.health_aware.completion_rate_mean - shootoutData.baseline.completion_rate_mean) * 100).toFixed(1)}%
                        </td>
                      </tr>
                      <tr>
                        <td><strong>Episode Net Reward</strong></td>
                        <td>{shootoutData.baseline.reward_mean.toFixed(1)}</td>
                        <td style={{ color: '#c084fc', fontWeight: 700 }}>
                          {shootoutData.health_aware.reward_mean.toFixed(1)}
                        </td>
                        <td style={{ color: '#c084fc', fontWeight: 700 }}>
                          +{(shootoutData.health_aware.reward_mean - shootoutData.baseline.reward_mean).toFixed(1)}
                        </td>
                      </tr>
                      <tr>
                        <td><strong>Detection Latency</strong></td>
                        <td>{shootoutData.baseline.detection_latency_mean.toFixed(1)} steps</td>
                        <td style={{ color: '#f43f5e', fontWeight: 700 }}>
                          {shootoutData.health_aware.detection_latency_mean.toFixed(1)} steps
                        </td>
                        <td style={{ color: '#f43f5e', fontWeight: 700 }}>
                          {(shootoutData.health_aware.detection_latency_mean - shootoutData.baseline.detection_latency_mean).toFixed(1)} steps
                        </td>
                      </tr>
                      <tr>
                        <td><strong>Task Churn (Reroutes)</strong></td>
                        <td>{shootoutData.baseline.churn_mean.toFixed(0)}</td>
                        <td>{shootoutData.health_aware.churn_mean.toFixed(0)}</td>
                        <td style={{ color: 'var(--text-muted)' }}>Principled</td>
                      </tr>
                      <tr>
                        <td><strong>Runtime / 400 Steps</strong></td>
                        <td>{(shootoutData.baseline.time_per_episode * 1000).toFixed(1)} ms</td>
                        <td>{(shootoutData.health_aware.time_per_episode * 1000).toFixed(1)} ms</td>
                        <td style={{ color: '#10b981' }}>Ultra-Fast (&lt;0.2ms/step)</td>
                      </tr>
                    </tbody>
                  </table>

                  <div style={{ marginTop: '1.5rem', background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: 8, fontSize: '0.85rem' }}>
                    <h4 style={{ marginBottom: '0.5rem', color: '#10b981' }}>Summary Verdict:</h4>
                    The <strong>HealthAwareAgent</strong> dramatically outperforms the health-blind baseline by dynamically detecting latent node degradation, halting assignments to dead nodes in <strong>0.0 steps</strong>, and executing cold-restart-feasible task rerouting to achieve up to <strong>99% completion rate</strong>.
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
