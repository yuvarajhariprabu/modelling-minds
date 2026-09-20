# Cluster Sentinel AI: Autonomous Health & Reroute Agent
> **Problem Statement ID:** MM26AI02 — *Keep the Cluster Alive: Detect, Reroute, Recover*  
> **Track:** Agentic AI | **Target System:** Distributed GPU SuperPOD Clusters

---

## 1. What Our Agentic AI Does
In high-throughput GPU clusters, nodes fail silently—experiencing thermal throttling, interconnect stalls, or complete unannounced crashes while workloads face hard SLA deadlines. 

The **`HealthAwareAgent`** is an autonomous control-plane agent that monitors noisy telemetry, infers hidden node degradation, and dynamically schedules and reroutes micro-batches to maximize completed tasks and minimize SLA violations.

---

## 2. Detection Approach (How It Sees Silent Failures)
Because the true node state is never explicitly revealed, the agent uses a multi-signal statistical detection pipeline:

1. **Bayesian Belief Inference**: Continuously updates health probabilities $P(\text{Healthy} \mid \text{Telemetry})$ using noisy observations (heartbeat consistency, exponential moving average latency, and error rates).
2. **CUSUM Change-Point Detection**: Uses cumulative sum statistical drift detection to identify subtle thermal throttling and NVLink packet drops $2\text{--}3$ steps before hard node failure.
3. **Task Stall Monitoring**: Tracks individual task remaining durations ($D_{\text{rem}}$). If an in-flight task makes zero progress across consecutive steps, the hosting node is immediately penalized as latent-degraded.

---

## 3. Adaptation Approach (How It Recovers & Schedules)
Rerouting a task incurs a **cold-restart penalty** (resets remaining work). The agent uses an **Expected Value of Completion (EVC)** policy:

1. **Selective Graceful Drain**: If an in-flight task on a degraded node only has $\le 1$ step remaining and sufficient deadline slack, the agent lets it finish rather than paying a cold restart penalty on a standby GPU.
2. **Proactive Cold Restart**: If the node is dead or the task will miss its deadline, the agent immediately reroutes it to the healthiest standby GPU.
3. **Congestion-Aware Routing**: Distributes new tasks to nodes with the highest health scores and lowest queue congestion, preventing node overload.

---

## 4. Evaluation Performance
Benchmarked across all 5 official competition difficulty environments:

| Environment | GPU Nodes | Capacity | Arrival Rate | Baseline Comp % | Our Agent Comp % | Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LIGHT** | 4 | 4 | 1.0 | 97.7% | **99.9%** | **+2.3% (Zero Drops)** |
| **SANDBOX** | 6 | 4 | 2.0 | 88.1% | **98.9%** | **+10.8%** |
| **MEDIUM** | 8 | 4 | 3.0 | 86.7% | **98.7%** | **+12.1%** |
| **HARD** | 12 | 3 | 4.5 | 58.2% | **76.0%** | **+17.8%** |
| **EXTREME** | 16 | 2 | 6.0 | 15.3% | **32.6%** | **2.1× Throughput** |

- **Decision Speed**: $<0.17\text{ ms}$ per step (100% compliant with competition wall-clock budget).
- **Dependencies**: Standard Python only (`math`, `collections`, `typing`). Zero external packages required.

---

## 5. Quick Start for Committee Evaluation

### Direct Agent Import (No UI / No Testing Dependencies Required):
```python
from health_aware_agent import HealthAwareAgent

# Evaluated directly by committee test harness:
agent = HealthAwareAgent(n_nodes=env.n_nodes, node_capacity=env.node_capacity)
obs = env.reset()
agent.reset()

for t in range(episode_length):
    actions = agent.act(obs)          # Returns {task_id: target_node_id}
    obs, reward, done, info = env.step(actions)
    agent.update(obs, reward, done, info)
```

### Run Benchmark Test Suite:
```bash
python master/testing/test_all_environments.py
```

### Optional: Launch Mission Control UI & AI Copilot:
```bash
python server.py
cd master/frontend && npm run dev
```
*(Live interface opens at `http://localhost:5173`)*
