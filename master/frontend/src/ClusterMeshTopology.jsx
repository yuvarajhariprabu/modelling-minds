import React, { useState, useMemo } from 'react';

/**
 * Dynamic NVLink 4.0 Crossbar Switch Fabric & Inter-Node Topology Canvas
 * Renders physical glowing interconnects, peer-to-peer data flows,
 * live task reroute migration beams, and interactive GPU node controls.
 */
export default function ClusterMeshTopology({
  nodes = [],
  recentActions = [],
  decisionLogs = [],
  onInjectChaos,
  selectedNodeId,
  onSelectNode
}) {
  const [hoveredNodeId, setHoveredNodeId] = useState(null);

  // Canvas geometry
  const width = 880;
  const height = 440;
  const cx = width / 2;
  const cy = height / 2;
  const rx = 340;
  const ry = 155;

  const n = nodes.length;

  // Calculate coordinates for each node in an elliptical SuperPOD ring
  const nodePositions = useMemo(() => {
    const posMap = {};
    nodes.forEach((node, i) => {
      const angle = (2 * Math.PI * i) / n - Math.PI / 2;
      const x = cx + rx * Math.cos(angle);
      const y = cy + ry * Math.sin(angle);
      posMap[node.node_id] = { x, y, angle, node };
    });
    return posMap;
  }, [nodes, n, cx, cy, rx, ry]);

  // Aggregate crossbar throughput calculation
  const totalThroughput = useMemo(() => {
    return nodes.reduce((sum, node) => sum + (node.nvlink_throughput_gbps || 0), 0);
  }, [nodes]);

  // Active or recent reroutes
  const activeReroutes = useMemo(() => {
    const reroutes = [];
    if (recentActions && recentActions.length > 0) {
      recentActions.forEach(a => {
        if (a.action === 'REROUTE' && a.from_node !== undefined && a.to_node !== undefined) {
          reroutes.push(a);
        }
      });
    }
    // Also look in decisionLogs for recent reroutes if none in actions
    if (reroutes.length === 0 && decisionLogs && decisionLogs.length > 0) {
      const logReroutes = decisionLogs.filter(l => l.action === 'REROUTE').slice(-2);
      logReroutes.forEach(l => reroutes.push(l));
    }
    return reroutes.slice(-3);
  }, [recentActions, decisionLogs]);

  // Generate inter-node peer links
  // 1. Ring links: (i -> i+1)
  // 2. Cross-diagonal links: (i -> (i + n/2) % n)
  // 3. Radial links: (i -> Central NVSwitch Hub)
  const links = useMemo(() => {
    const linkList = [];
    if (n < 2) return linkList;

    // Ring links
    for (let i = 0; i < n; i++) {
      const nextIdx = (i + 1) % n;
      const idA = nodes[i]?.node_id;
      const idB = nodes[nextIdx]?.node_id;
      if (idA !== undefined && idB !== undefined && nodePositions[idA] && nodePositions[idB]) {
        linkList.push({
          id: `ring-${idA}-${idB}`,
          type: 'ring',
          source: idA,
          target: idB,
          nodeA: nodes[i],
          nodeB: nodes[nextIdx],
          posA: nodePositions[idA],
          posB: nodePositions[idB],
        });
      }
    }

    // Secondary Crossbar mesh links
    for (let i = 0; i < n; i++) {
      const crossIdx = (i + Math.floor(n / 2)) % n;
      if (crossIdx > i) {
        const idA = nodes[i]?.node_id;
        const idB = nodes[crossIdx]?.node_id;
        if (idA !== undefined && idB !== undefined && nodePositions[idA] && nodePositions[idB]) {
          linkList.push({
            id: `cross-${idA}-${idB}`,
            type: 'cross',
            source: idA,
            target: idB,
            nodeA: nodes[i],
            nodeB: nodes[crossIdx],
            posA: nodePositions[idA],
            posB: nodePositions[idB],
          });
        }
      }
    }

    return linkList;
  }, [nodes, n, nodePositions]);

  // Helper for node state colors
  const getNodeColor = (node) => {
    const state = node?.true_state || 'healthy';
    if (state === 'down') return '#ef4444';
    if (state === 'degraded') return '#f59e0b';
    return '#10b981';
  };

  const activeNode = hoveredNodeId !== null
    ? nodes.find(n => n.node_id === hoveredNodeId)
    : selectedNodeId !== null
    ? nodes.find(n => n.node_id === selectedNodeId)
    : null;

  return (
    <div className="mesh-fabric-container">
      {/* FABRIC TELEMETRY HUD BAR */}
      <div className="fabric-hud-bar">
        <div className="fabric-hud-item">
          <span className="fabric-hud-dot live-pulse"></span>
          <span className="fabric-hud-label">INTERCONNECT FABRIC:</span>
          <span className="fabric-hud-val">NVIDIA NVSwitch 3.2 Tbps Crossbar</span>
        </div>
        <div className="fabric-hud-item">
          <span className="fabric-hud-label">AGGREGATE NVLINK SPEED:</span>
          <span className="fabric-hud-val highlight-cyan">{totalThroughput.toLocaleString()} GB/s</span>
        </div>
        <div className="fabric-hud-item">
          <span className="fabric-hud-label">ACTIVE INTERCONNECTS:</span>
          <span className="fabric-hud-val">{links.length * 2} P2P Optical Lanes</span>
        </div>
        <div className="fabric-hud-item">
          <span className="fabric-hud-label">FABRIC STATUS:</span>
          <span className="fabric-hud-val" style={{ color: nodes.some(n => n.true_state === 'down') ? '#ef4444' : nodes.some(n => n.true_state === 'degraded') ? '#f59e0b' : '#10b981' }}>
            {nodes.some(n => n.true_state === 'down') ? 'PARTIAL FABRIC DISRUPTION' : nodes.some(n => n.true_state === 'degraded') ? 'THERMAL THROTTLING ACTIVE' : 'NOMINAL ULTRA-LOW LATENCY (1.2µs)'}
          </span>
        </div>
      </div>

      {/* SVG TOPOLOGY CANVAS */}
      <div className="svg-canvas-wrapper">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="mesh-topology-svg"
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            {/* Radial glow filter */}
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>

            {/* Intense laser reroute glow */}
            <filter id="laserGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="6" result="blur1" />
              <feGaussianBlur stdDeviation="12" result="blur2" />
              <feMerge>
                <feMergeNode in="blur2" />
                <feMergeNode in="blur1" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>

            {/* Linear Gradients for Reroute Beams */}
            <linearGradient id="rerouteGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#f43f5e" stopOpacity="0.95" />
              <stop offset="50%" stopColor="#d946ef" stopOpacity="0.95" />
              <stop offset="100%" stopColor="#a855f7" stopOpacity="0.95" />
            </linearGradient>

            {/* NVSwitch Central Core Gradient */}
            <radialGradient id="switchCoreGrad" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#e11d48" stopOpacity="0.85" />
              <stop offset="55%" stopColor="#7e22ce" stopOpacity="0.6" />
              <stop offset="100%" stopColor="#140624" stopOpacity="0.95" />
            </radialGradient>
          </defs>

          {/* 1. BACKGROUND GRID MATRIX */}
          <g className="mesh-background-decorations" opacity="0.35">
            <ellipse cx={cx} cy={cy} rx={rx} ry={ry} fill="none" stroke="rgba(168, 85, 247, 0.15)" strokeWidth="1.5" strokeDasharray="4, 8" />
            <ellipse cx={cx} cy={cy} rx={rx * 0.55} ry={ry * 0.55} fill="none" stroke="rgba(225, 29, 72, 0.12)" strokeWidth="1" />
            <circle cx={cx} cy={cy} r="18" fill="none" stroke="rgba(192, 132, 252, 0.25)" strokeWidth="1" />
          </g>

          {/* 2. RADIAL LINKS (Every GPU to Central NVSwitch Fabric) */}
          <g className="radial-nvswitch-links">
            {nodes.map(node => {
              const pos = nodePositions[node.node_id];
              if (!pos) return null;
              const isDegraded = node.true_state === 'degraded';
              const isDown = node.true_state === 'down';
              const isHovered = hoveredNodeId === node.node_id;

              const strokeColor = isDown
                ? 'rgba(239, 68, 68, 0.3)'
                : isDegraded
                ? 'rgba(245, 158, 11, 0.55)'
                : isHovered
                ? '#f43f5e'
                : 'rgba(168, 85, 247, 0.35)';

              const strokeWidth = isHovered ? 3 : isDown ? 1.2 : 2;

              return (
                <g key={`radial-${node.node_id}`}>
                  {/* Base Wire */}
                  <line
                    x1={cx}
                    y1={cy}
                    x2={pos.x}
                    y2={pos.y}
                    stroke={strokeColor}
                    strokeWidth={strokeWidth}
                    strokeDasharray={isDown ? '4, 6' : undefined}
                    className={!isDown ? 'nvlink-wire' : 'nvlink-wire-dead'}
                  />

                  {/* Flowing Data Particle on Radial Wire */}
                  {!isDown && (
                    <circle r={isHovered ? 4 : 2.5} fill={isDegraded ? '#f59e0b' : '#f43f5e'} filter="url(#glow)">
                      <animateMotion
                        path={`M ${cx} ${cy} L ${pos.x} ${pos.y}`}
                        dur={isDegraded ? '3.5s' : isHovered ? '1.2s' : '2.0s'}
                        repeatCount="indefinite"
                      />
                    </circle>
                  )}
                </g>
              );
            })}
          </g>

          {/* 3. PEER-TO-PEER INTERCONNECT CABLES (Ring & Cross-Mesh) */}
          <g className="peer-interconnect-links">
            {links.map(link => {
              const { id, posA, posB, nodeA, nodeB, type } = link;
              const hasDown = nodeA.true_state === 'down' || nodeB.true_state === 'down';
              const hasDegraded = nodeA.true_state === 'degraded' || nodeB.true_state === 'degraded';
              const isConnectedToHovered = hoveredNodeId === link.source || hoveredNodeId === link.target;

              let stroke = hasDown
                ? 'rgba(239, 68, 68, 0.25)'
                : hasDegraded
                ? 'rgba(245, 158, 11, 0.45)'
                : isConnectedToHovered
                ? '#f43f5e'
                : type === 'ring'
                ? 'rgba(168, 85, 247, 0.4)'
                : 'rgba(225, 29, 72, 0.25)';

              let widthVal = isConnectedToHovered ? 3 : type === 'ring' ? 2 : 1.2;

              // Curved control point for subtle curvature
              const mx = (posA.x + posB.x) / 2;
              const my = (posA.y + posB.y) / 2;
              const dx = posB.x - posA.x;
              const dy = posB.y - posA.y;
              const dist = Math.sqrt(dx * dx + dy * dy);
              const curveFactor = type === 'cross' ? 0.08 : 0.04;
              const qx = mx - dy * curveFactor;
              const qy = my + dx * curveFactor;
              const pathD = `M ${posA.x} ${posA.y} Q ${qx} ${qy} ${posB.x} ${posB.y}`;

              return (
                <g key={id}>
                  <path
                    d={pathD}
                    fill="none"
                    stroke={stroke}
                    strokeWidth={widthVal}
                    strokeDasharray={hasDown ? '4, 8' : isConnectedToHovered ? '6, 4' : undefined}
                    className={!hasDown ? 'nvlink-peer-path' : ''}
                  />

                  {/* Active Data Pulse across peer wires */}
                  {!hasDown && (type === 'ring' || isConnectedToHovered) && (
                    <circle r={isConnectedToHovered ? 3.5 : 2} fill={hasDegraded ? '#f59e0b' : '#c084fc'}>
                      <animateMotion
                        path={pathD}
                        dur={hasDegraded ? '4.0s' : isConnectedToHovered ? '1.4s' : '2.8s'}
                        repeatCount="indefinite"
                      />
                    </circle>
                  )}
                </g>
              );
            })}
          </g>

          {/* 4. ACTIVE REROUTE LASER BEAMS */}
          <g className="reroute-laser-beams">
            {activeReroutes.map((reroute, idx) => {
              const fromPos = nodePositions[reroute.from_node];
              const toPos = nodePositions[reroute.to_node];
              if (!fromPos || !toPos) return null;

              // Arch high over the center
              const mx = (fromPos.x + toPos.x) / 2;
              const my = (fromPos.y + toPos.y) / 2;
              const qx = mx + (cx - mx) * 0.6;
              const qy = my + (cy - my) * 0.6 - 40;
              const laserPath = `M ${fromPos.x} ${fromPos.y} Q ${qx} ${qy} ${toPos.x} ${toPos.y}`;

              return (
                <g key={`reroute-${idx}-${reroute.task_id}`}>
                  {/* Glowing wide halo */}
                  <path
                    d={laserPath}
                    fill="none"
                    stroke="url(#rerouteGrad)"
                    strokeWidth="7"
                    strokeLinecap="round"
                    opacity="0.5"
                    filter="url(#laserGlow)"
                  />
                  {/* Core laser beam */}
                  <path
                    d={laserPath}
                    fill="none"
                    stroke="#ffffff"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeDasharray="8, 6"
                    className="laser-dash-flow"
                  />
                  {/* Animated traveling energy ball */}
                  <circle r="6" fill="#f43f5e" filter="url(#laserGlow)">
                    <animateMotion path={laserPath} dur="1.2s" repeatCount="indefinite" />
                  </circle>

                  {/* Floating Reroute Label Badge */}
                  <g transform={`translate(${qx}, ${qy - 14})`}>
                    <rect
                      x="-70"
                      y="-12"
                      width="140"
                      height="24"
                      rx="6"
                      fill="rgba(20, 7, 32, 0.92)"
                      stroke="#f43f5e"
                      strokeWidth="1.5"
                      filter="url(#glow)"
                    />
                    <text
                      x="0"
                      y="4"
                      textAnchor="middle"
                      fill="#fff"
                      fontSize="10"
                      fontWeight="bold"
                      fontFamily="var(--font-mono)"
                    >
                      ⚡ REROUTE #{reroute.task_id} ➔ GPU {reroute.to_node}
                    </text>
                  </g>
                </g>
              );
            })}
          </g>

          {/* 5. CENTRAL NVSWITCH FABRIC CORE */}
          <g className="nvswitch-central-hub" transform={`translate(${cx}, ${cy})`}>
            {/* Outer pulsating ring */}
            <circle r="52" fill="none" stroke="rgba(168, 85, 247, 0.35)" strokeWidth="1.5" strokeDasharray="6, 6" className="spin-slow" />
            <circle r="44" fill="none" stroke="rgba(225, 29, 72, 0.45)" strokeWidth="1" />
            {/* Core Hexagonal/Circle Switch */}
            <circle r="36" fill="url(#switchCoreGrad)" stroke="#e11d48" strokeWidth="2" filter="url(#glow)" />
            {/* Switch Icon */}
            <path
              d="M -12 -6 L -4 -6 L 0 -10 L 4 -6 L 12 -6 M -12 6 L -4 6 L 0 10 L 4 6 L 12 6 M -8 -12 L -8 12 M 8 -12 L 8 12"
              stroke="#fbcfe8"
              strokeWidth="1.8"
              strokeLinecap="round"
              fill="none"
            />
            <text y="18" textAnchor="middle" fill="#fbcfe8" fontSize="9" fontWeight="700" fontFamily="var(--font-mono)">
              NVSWITCH
            </text>
            <text y="28" textAnchor="middle" fill="#f472b6" fontSize="7.5" fontFamily="var(--font-mono)">
              3.2 Tbps Crossbar
            </text>
          </g>

          {/* 6. GPU NODES (Interactive Chips on the Ring) */}
          <g className="gpu-nodes-group">
            {nodes.map(node => {
              const pos = nodePositions[node.node_id];
              if (!pos) return null;
              const nid = node.node_id;
              const trueState = node.true_state || 'healthy';
              const stateColor = getNodeColor(node);
              const isHovered = hoveredNodeId === nid;
              const isSelected = selectedNodeId === nid;
              const temp = node.cuda_temp_c || 62.0;
              const vramUsed = node.vram_used_gb || 20.0;
              const vramPct = Math.min(100, Math.round((vramUsed / 80.0) * 100));
              const qLen = node.queue_len || 0;
              const capacity = node.capacity || 4;

              // Node radius
              const r = 34;

              // Circumference for VRAM ring gauge
              const circ = 2 * Math.PI * (r + 4);
              const vramOffset = circ - (vramPct / 100) * circ;

              return (
                <g
                  key={`node-chip-${nid}`}
                  transform={`translate(${pos.x}, ${pos.y})`}
                  className="interactive-node-g"
                  onMouseEnter={() => setHoveredNodeId(nid)}
                  onMouseLeave={() => setHoveredNodeId(null)}
                  onClick={() => onSelectNode && onSelectNode(nid)}
                  style={{ cursor: 'pointer' }}
                >
                  {/* Outer Glowing Health Halo */}
                  <circle
                    r={r + 9}
                    fill="none"
                    stroke={stateColor}
                    strokeWidth={isHovered || isSelected ? 3 : 1.5}
                    opacity={trueState === 'down' ? 0.3 : 0.65}
                    filter="url(#glow)"
                    className={trueState !== 'down' ? 'node-halo-pulse' : ''}
                  />

                  {/* VRAM Utilization Ring Meter */}
                  <circle
                    r={r + 4}
                    fill="none"
                    stroke="rgba(255, 255, 255, 0.1)"
                    strokeWidth="3"
                  />
                  <circle
                    r={r + 4}
                    fill="none"
                    stroke={vramPct > 85 ? '#ef4444' : '#a855f7'}
                    strokeWidth="3"
                    strokeDasharray={circ}
                    strokeDashoffset={vramOffset}
                    strokeLinecap="round"
                    transform="rotate(-90)"
                  />

                  {/* Node Core Body */}
                  <circle
                    r={r}
                    fill="#140624"
                    stroke={isHovered ? '#fff' : stateColor}
                    strokeWidth={isHovered ? 2.5 : 1.8}
                  />

                  {/* Micro GPU Die in center */}
                  <rect
                    x="-14"
                    y="-14"
                    width="28"
                    height="28"
                    rx="4"
                    fill={trueState === 'down' ? '#3f1515' : '#260f38'}
                    stroke={stateColor}
                    strokeWidth="1"
                  />

                  {/* GPU Label */}
                  <text
                    y="-2"
                    textAnchor="middle"
                    fill="#ffffff"
                    fontSize="11"
                    fontWeight="800"
                    fontFamily="var(--font-display)"
                  >
                    GPU {nid}
                  </text>

                  {/* Temperature & Load Pills */}
                  <text
                    y="9"
                    textAnchor="middle"
                    fill={temp > 85 ? '#ef4444' : temp > 75 ? '#f59e0b' : '#f472b6'}
                    fontSize="9"
                    fontWeight="700"
                    fontFamily="var(--font-mono)"
                  >
                    {temp}°C
                  </text>

                  {/* Queue Load Badge */}
                  <g transform={`translate(${r - 8}, ${-r + 4})`}>
                    <circle
                      r="9"
                      fill={qLen >= capacity ? '#ef4444' : '#2e0f42'}
                      stroke={qLen >= capacity ? '#fca5a5' : '#c084fc'}
                      strokeWidth="1"
                    />
                    <text
                      y="3"
                      textAnchor="middle"
                      fill="#fff"
                      fontSize="8"
                      fontWeight="bold"
                      fontFamily="var(--font-mono)"
                    >
                      {qLen}
                    </text>
                  </g>

                  {/* NVLink Bandwidth Badge Below Node */}
                  <g transform="translate(0, 48)">
                    <rect
                      x="-42"
                      y="-9"
                      width="84"
                      height="18"
                      rx="4"
                      fill="rgba(20, 7, 32, 0.9)"
                      stroke={stateColor}
                      strokeWidth="1"
                    />
                    <text
                      y="3.5"
                      textAnchor="middle"
                      fill={stateColor}
                      fontSize="8.5"
                      fontWeight="bold"
                      fontFamily="var(--font-mono)"
                    >
                      {trueState === 'down' ? 'OFFLINE' : `${node.nvlink_throughput_gbps} GB/s`}
                    </text>
                  </g>
                </g>
              );
            })}
          </g>
        </svg>
      </div>

      {/* QUICK NODE INSPECTION / CHAOS INJECTION POPUP */}
      {activeNode && (
        <div className="active-node-inspector-hud glass-panel">
          <div className="inspector-hud-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <span
                className="heartbeat-dot pulse"
                style={{ background: getNodeColor(activeNode) }}
              ></span>
              <h4>GPU #{activeNode.node_id} — {activeNode.node_name || 'NVIDIA H100 SXM5'}</h4>
            </div>
            <span
              className="badge-tag"
              style={{
                borderColor: getNodeColor(activeNode),
                color: getNodeColor(activeNode),
                textTransform: 'uppercase'
              }}
            >
              {activeNode.true_state || 'HEALTHY'} ({activeNode.thermal_status || 'OPTIMAL'})
            </span>
          </div>

          <div className="inspector-hud-body">
            <div className="inspector-stat">
              <span>H100 Core Temp:</span>
              <strong style={{ color: activeNode.cuda_temp_c > 85 ? '#ef4444' : '#f472b6' }}>
                {activeNode.cuda_temp_c || 65}°C
              </strong>
            </div>
            <div className="inspector-stat">
              <span>NVLink Fabric:</span>
              <strong>{activeNode.nvlink_throughput_gbps || 900} GB/s</strong>
            </div>
            <div className="inspector-stat">
              <span>VRAM Allocation:</span>
              <strong>{activeNode.vram_used_gb || 20} / 80 GB ({Math.round(((activeNode.vram_used_gb || 20) / 80) * 100)}%)</strong>
            </div>
            <div className="inspector-stat">
              <span>Bayesian Belief:</span>
              <strong style={{ color: '#10b981' }}>
                P(OK): {((activeNode.belief?.[0] || 1) * 100).toFixed(1)}%
              </strong>
            </div>
            <div className="inspector-stat">
              <span>Queue Load:</span>
              <strong>{activeNode.queue_len || 0} / {activeNode.capacity || 4} Tasks</strong>
            </div>
          </div>

          {/* Quick Chaos Buttons directly in HUD */}
          <div className="inspector-chaos-actions">
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Inject Fault:</span>
            <button
              className="chaos-btn heal"
              onClick={() => onInjectChaos(activeNode.node_id, 'healthy')}
            >
              ✓ Heal
            </button>
            <button
              className="chaos-btn throttle"
              onClick={() => onInjectChaos(activeNode.node_id, 'degraded')}
            >
              ⚠️ Throttle (88°C)
            </button>
            <button
              className="chaos-btn crash"
              onClick={() => onInjectChaos(activeNode.node_id, 'down')}
            >
              🚨 Crash (Bus Drop)
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
