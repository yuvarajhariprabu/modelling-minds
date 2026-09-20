# Cluster Sentinel AI — Mission Control Dashboard
### Frontend Web Application for Autonomous Agentic AI Cluster Management

An interactive, dark-mode real-time operations dashboard built with **React**, **Vite**, and **Vanilla CSS**. It visualizes cluster telemetry, Bayesian belief state distributions, autonomous rerouting decisions, and live explainability logs in real time.

---

## Key Features

1. **Live 3D-Look Node Mesh**:
   - Visualizes node health states: **Healthy** (Emerald), **Degraded** (Amber), and **Down** (Rose/Crimson).
   - Real-time capacity utilization gauges and progress bars.
   - Pulsing radar rings on nodes undergoing anomalies or failovers.

2. **Autonomous Bayesian Belief Radar**:
   - Displays continuous posterior probabilities: $P(\text{Healthy})$, $P(\text{Degraded})$, and $P(\text{Down})$.
   - Instant visual feedback on noisy telemetry filtering (heartbeats, latency spikes, error rates).

3. **Chaos Engineering Injection Matrix**:
   - Manually inject **Degraded** latency/packet drops or total node **Crashes** with 1 click.
   - Test how the autonomous AI agent reacts and recovers under live operational stress.

4. **Transparent Explainability Log Feed**:
   - Live stream of the agent's causal reasoning for every action.
   - Exact mathematical justification for every task reroute and cold restart decision.

5. **Head-to-Head Shootout Modal**:
   - Benchmarks the **HealthAwareAgent** against the health-blind **Round-Robin Baseline** across a 400-step episode.
   - Displays completion rates, detection latency, churn counts, and net reward comparisons.

---

## Getting Started

### 1. Prerequisites
- **Node.js**: v18 or later
- **Python**: 3.8+ (for the backend simulation server)

### 2. Start the Backend Simulation Server
From the project root:
```powershell
cd "..\..\"
python server.py
```
*(Runs on `http://127.0.0.1:8000`)*

### 3. Start the Frontend Dev Server
From this directory (`master/frontend`):
```powershell
npm run dev
```
*(Opens at `http://localhost:5173/`)*

---

## Tech Stack & Architecture

- **Framework**: React 18 + Vite (fast HMR, lightweight bundle)
- **Styling**: Pure CSS Design System (`src/index.css`) with curated HSL color tokens, dark glassmorphism, and responsive CSS grid.
- **Typography**: `Outfit` (Headings) and `JetBrains Mono` (Telemetry & Code)
- **API**: Native Fetch connecting to Python REST API (`/api/init`, `/api/step`, `/api/inject`, `/api/benchmark`).
