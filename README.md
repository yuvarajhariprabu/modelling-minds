# Keep the Cluster Alive: Detect, Reroute, Recover

### Track: Agentic AI | Problem Statement ID: MM26AI02

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![React 18](https://img.shields.io/badge/React-18-cyan.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-Bundler-purple.svg)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Executive Summary

In modern distributed cloud clusters, compute nodes fail silently, unpredictably, and independently: they drop heartbeats, suffer latency tail spikes, experience packet drops, or freeze completely without explicit failure notifications. Meanwhile, user workloads arrive continuously and are bounded by hard, unforgiving deadlines.

The **HealthAwareAgent** is an autonomous, fault-tolerant **Agentic AI controller** engineered to solve this **Partially Observable Markov Decision Process (POMDP)**. Operating strictly on noisy, real-time sensor streams (heartbeat status, round-trip latency, error rates, queue loads), it:

- Infers hidden node health states using **Bayesian Sequential Likelihood Filtering**.
- Dual-verifies silent failures by monitoring **empirical task progress stalls**.
- Performs cold-restart-optimal task rerouting backed by **mathematical deadline feasibility guards**.
- Dynamically balances cluster load with **Expected Value of Completion (EVC)** and **Shortest Remaining Processing Time (SRPT)** scheduling.

---

## 2. System Architecture

The agent executes a four-stage autonomous decision loop on every simulation step ($<0.2\text{ ms}$ wall-clock time):

```
                   +----------------------------------------------+
                   |           Per-Step Telemetry Obs             |
                   |  heartbeat_ok, latency_ms, error_rate, queue |
                   +----------------------------------------------+
                                          |
                                          v
    +----------------------------------------------------------------------------+
    | 1. MULTI-SIGNAL BAYESIAN & STATISTICAL BELIEF TRACKER                      |
    |    - Exponentially Weighted Moving Average (EWMA; α=0.4)                   |
    |    - Gaussian & Bernoulli Likelihoods: P(hb|S), P(lat|S), P(err|S)         |
    |    - Hidden Markov Transition Prior (Healthy <-> Degraded <-> Down)        |
    |    - Composite Health Score: H_i in [0, 100]                               |
    +----------------------------------------------------------------------------+
                                          |
                                          v
    +----------------------------------------------------------------------------+
    | 2. EMPIRICAL PROGRESS & DUAL VERIFICATION ENGINE                           |
    |    - Tracks task duration decrement across consecutive steps: ΔD           |
    |    - Detects zero-progress stalls (empirical proof of degraded/down state) |
    |    - Filters "zombie" tasks that have already missed deadline feasibility   |
    +----------------------------------------------------------------------------+
                                          |
                                          v
    +----------------------------------------------------------------------------+
    | 3. DEADLINE SLACK & COLD-RESTART OPTIMIZER                                 |
    |    - Feasibility Guard: Slack = Deadline - (Current_Step + Duration_Total) |
    |    - Rejects doomed reroutes to protect cluster capacity                   |
    |    - Anti-thrashing cooldowns (cooldown = 4 steps per task)                |
    +----------------------------------------------------------------------------+
                                          |
                                          v
    +----------------------------------------------------------------------------+
    | 4. CONGESTION-AWARE LOAD BALANCER & SCHEDULER                              |
    |    - Progress velocity: V_i = P(H)*1.0 + P(Deg)*0.25 + P(Down)*0.0         |
    |    - Capacity-ratio utility: Utility_i = V_i - 0.15 * (Load_i / Cap_i)     |
    |    - Shortest Remaining Processing Time (SRPT) dispatch                    |
    +----------------------------------------------------------------------------+
                                          |
                                          v
                   +----------------------------------------------+
                   |         Action Set: {task_id: node_id}       |
                   |    Transparent Explainability Log Export     |
                   +----------------------------------------------+
```

---

## 3. Mathematical Formulations

### 3.1 Bayesian Belief State Update

For each node $i$, the latent health state $S_i \in \{\text{Healthy}, \text{Degraded}, \text{Down}\}$ is updated using Bayes' theorem:

$$P(S_i = s \mid O_t) \propto P(O_t \mid S_i = s) \cdot \sum_{s' \in \mathcal{S}} P(S_i = s \mid S_{i, t-1} = s') P(S_{i, t-1} = s')$$

The emission likelihood $P(O_t \mid S)$ factors into independent sensor streams:

$$P(O_t \mid S) = P(\text{hb} \mid S) \cdot P(\text{latency} \mid S) \cdot P(\text{error\_rate} \mid S)$$

- **Heartbeat Likelihood**: Bernoulli trials with $p_H = 0.99$, $p_{Deg} = 0.85$, $p_{Down} = 0.05$.
- **Latency Likelihood**: Gaussian likelihood with missing-value penalty:
  $$P(\text{latency} \mid \text{Down}) \approx 0.95 \quad \text{if latency is None (timeout)}$$
- **Error Rate Likelihood**: Scaled normal distributions modeling typical healthy ($2\%$), degraded ($25\%$), and down ($95\%$) operating regions.

### 3.2 Dual Verification (Empirical Stalls)

Noisy telemetry alone can generate false alarms. The agent cross-examines telemetry against physical task progress:
$$\Delta D_j = D_{j, t-1} - D_{j, t}$$
If an in-flight task on node $i$ registers $\Delta D = 0$ for $\ge 2$ consecutive steps while the node claims to be responsive, the node is diagnosed as stalled, forcing an immediate belief penalty.

### 3.3 Cold Restart Decision Boundary

Rerouting a task resets its remaining duration to $D_{\text{total}}$ (simulating the cost of cold-starting a stateful workload on a new machine). Rerouting is approved **if and only if all three conditions are satisfied**:

1. **Unhealthy Node**: $H_{\text{current}} < 38.0$ OR $P(\text{Down}) > 0.50$.
2. **Cold-Restart Feasibility**:
   $$\text{Deadline} - (t + D_{\text{total}}) \ge 0$$
3. **Anti-Thrashing Cooldown**:
   $$t - t_{\text{last\_reroute}} \ge 4$$

---

## 4. Benchmark & Performance Results

### Official 5-Regime Multi-Environment Benchmark (`master/testing/`)

Evaluated across all 5 official competition environments across random seeds `[0, 1, 2]` (400 steps per episode):

| Environment | Nodes | Capacity | Arrival Rate | Deadline Slack | Round-Robin Baseline | HealthAwareAgent (Ours) | Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LIGHT** | 4 | 4 | 1.0 | 6–15 steps | 97.7% (+373.6 rew) | **100.0%** (+391.7 rew) | **+2.3% (Zero Failures)** |
| **SANDBOX** | 6 | 4 | 2.0 | 4–10 steps | 88.1% (+602.6 rew) | **99.0%** (+775.9 rew) | **+10.9% (+173.3 Reward)** |
| **MEDIUM** | 8 | 4 | 3.0 | 3–8 steps | 86.7% (+870.3 rew) | **99.2%** (+1171.2 rew) | **+12.5% (+300.9 Reward)** |
| **HARD** | 12 | 3 | 4.5 | 2–5 steps | 58.2% (+272.6 rew) | **76.0%** (+889.5 rew) | **+17.8% (+616.9 Reward)** |
| **EXTREME** | 16 | 2 | 6.0 | 1–3 steps | 15.3% (-1737.0 rew) | **32.6%** (-943.6 rew) | **+17.3% (2.1× Completed)** |
| **OVERALL** | — | — | — | — | 69.2% | **81.3%** | **+380.5 Mean Reward** |

- **Detection Latency**: Reduced from **11–19 steps** (baseline) down to **0.0 steps**.
- **Execution Speed**: $< 0.17\text{ ms}$ per decision step (~170ms for an entire 400-step episode).

---

## 5. Why No LLM is Used in the Real-Time Control Loop

A common question from judges: *"Did you use an LLM for routing tasks?"*

| Dimension | LLM (e.g., GPT-4 / Gemini) | Our Bayesian Agentic AI |
| :--- | :--- | :--- |
| **Inference Latency** | 500 ms – 3,000 ms per step (10,000× too slow for clusters) | **< 0.2 milliseconds** per step |
| **Determinism** | Stochastically hallucinates deadline math | **Exact mathematical proof**: $t + D_{\text{total}} \le \text{Deadline}$ |
| **Operational Cost** | Rate limits, API downtime, expensive per-step calls | **Zero dependencies, 100% offline & local** |
| **Explainability** | Black-box generated text | **Causal, probability-grounded log stream** |

*Note: In production, an asynchronous LLM copilot can sit on Tier 2 (the management plane) to read the agent's explainability logs and generate automated Root Cause Analysis (RCA) incident post-mortems for human SRE teams.*

---

## 6. Interactive Web Mission Control Dashboard

A full-featured dark-mode operations dashboard is included in `master/frontend/`:

- **Real-Time Node Mesh**: Live node cards with capacity gauges, pulsing heartbeat animations, and health indicators.
- **Chaos Engineering Matrix**: Inject latency degradations or total crashes on individual nodes with one click.
- **Autonomous Explainability Feed**: Live terminal streaming Bayesian posterior shifts and reroute justifications.
- **Compare Shootout Modal**: Real-time head-to-head evaluation against the baseline over 400 steps.

---

## 7. How to Run

### Option 1: Web Mission Control (Interactive Dashboard)

1. **Start Backend Simulation Server**:

   ```powershell
   python server.py
   ```

   *(Listens on `http://127.0.0.1:8000`)*

2. **Start Frontend Web App**:

   ```powershell
   cd master/frontend
   npm run dev
   ```

   *(Opens on `http://localhost:5173/`)*

### Option 2: Run Official Multi-Environment Test Suite

```powershell
python master/testing/test_all_environments.py
```

### Option 3: Run Standalone CLI Demos & Benchmarks

```powershell
# Head-to-Head Matchup
python run_example.py

# Live Diagnostic & Explainability Stream
python run_demo.py

# Statistical Benchmark Suite
python run_benchmark.py
```

---

## 8. Repository Structure

```
modelling mindes/
├── README.md                      <-- Project master documentation (this file)
├── JUDGES_PRESENTATION_GUIDE.md   <-- Pitch guide, slide structure, Q&A defense
├── DESIGN_CHEATSHEET.md           <-- Quick 30-second architectural summary
├── health_aware_agent.py          <-- SOTA Bayesian Agentic AI model
├── baseline_agent.py              <-- Reference health-blind benchmark agent
├── cluster_env.py                 <-- Hidden Markov cluster simulation environment
├── sandbox_env.py                 <-- Environment initialization wrapper
├── eval_harness.py                <-- Automated evaluation and scoring harness
├── server.py                      <-- Zero-dependency Python REST API server
├── run_example.py                 <-- CLI single-episode comparison
├── run_demo.py                    <-- CLI live telemetry and explainability demo
├── run_benchmark.py               <-- CLI multi-seed statistical evaluation
└── master/
    ├── frontend/                  <-- React + Vite Mission Control Dashboard
    │   ├── src/App.jsx            <-- Dashboard UI & chaos engineering controls
    │   ├── src/index.css          <-- Custom dark glassmorphism design system
    │   └── README.md              <-- Dedicated frontend documentation
    ├── testing/                   <-- Official 5-regime test environments
    │   ├── test_all_environments.py
    │   ├── test_env_light.py
    │   ├── test_env_sandbox.py
    │   ├── test_env_medium.py
    │   ├── test_env_hard.py
    │   └── test_env_extreme.py
    └── .vscode/                   <-- Editor configuration
```
