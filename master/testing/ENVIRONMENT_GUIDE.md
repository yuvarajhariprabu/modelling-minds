# Testing Environment Guide

## Quick Reference: All Environments

| Environment | Nodes | Capacity | Arrival Rate | Deadline Slack | Failures | Use Case |
|-------------|-------|----------|--------------|----------------|----------|----------|
| **LIGHT** | 4 | 4 | 1.0 | 6–15 steps | Very stable | Debug & learn |
| **SANDBOX** | 6 | 4 | 2.0 | 4–10 steps | Balanced | Baseline testing |
| **MEDIUM** | 8 | 4 | 3.0 | 3–8 steps | More frequent | Real-world sim |
| **HARD** | 12 | 3 | 4.5 | 2–5 steps | Aggressive | Stress test |
| **EXTREME** | 16 | 2 | 6.0 | 1–3 steps | Very aggressive | Limit testing |

---

## Environment Details

### 🟢 LIGHT (4 nodes)
**When to use:** Developing and debugging your agent

**Configuration:**
- 4 nodes (easier to trace)
- Arrival rate: 1.0 (~1 task per step)
- Generous deadline slack: 6–15 steps
- Very stable failures (nodes stay healthy longer)
- Conservative Markov transitions

**Benefits:**
- Can see what happens when each node fails (few nodes)
- Lots of time to complete tasks (slack 6–15)
- Few simultaneous failures (good for learning)
- Runs fast (few nodes)

**Challenges:**
- Won't prepare you for high-load scenarios
- Failure patterns unrealistic (too stable)

**Expected Agent Performance:**
- Completion rate: 85–95%
- Easy to detect all failures
- Can reroute leisurely (lots of deadline margin)

**Recommended:** Use with `debug=True` to watch ground-truth node states!

---

### 🟡 SANDBOX (6 nodes)
**When to use:** Baseline testing (same as provided environment)

**Configuration:**
- 6 nodes
- Arrival rate: 2.0 (2 tasks per step)
- Moderate deadline slack: 4–10 steps
- Balanced failure dynamics (Markov chain as designed)

**Benefits:**
- Official reference environment
- Balanced difficulty (not too easy, not too hard)
- Good for tuning thresholds
- Provided as baseline

**Challenges:**
- Still easier than some real configurations
- May not reveal generalization issues

**Expected Agent Performance:**
- Completion rate: 75–85%
- Can detect most failures
- Needs some rerouting strategy

**Recommended:** Your primary testing ground. Multiple runs (5–10 seeds).

---

### 🟠 MEDIUM (8 nodes)
**When to use:** Testing scalability and higher load

**Configuration:**
- 8 nodes (2 more than sandbox)
- Arrival rate: 3.0 (50% higher than sandbox)
- Tight deadline slack: 3–8 steps
- Same failure dynamics, but affects more nodes

**Benefits:**
- Tests multi-node monitoring
- Higher load reveals rerouting pressure
- More realistic than sandbox
- Still manageable (8 nodes is reasonable)

**Challenges:**
- Harder to debug (more nodes to trace)
- Tighter deadlines (less margin for error)

**Expected Agent Performance:**
- Completion rate: 70–80%
- Needs faster failure detection
- Must reroute carefully (high load)

**Recommended:** After sandbox works well, test here.

---

### 🔴 HARD (12 nodes)
**When to use:** Stress testing and finding breaking points

**Configuration:**
- 12 nodes (double the sandbox)
- Arrival rate: 4.5 (2.25× sandbox)
- Very tight deadline slack: 2–5 steps
- **Aggressive failure dynamics:**
  - Nodes degrade more easily (0.97 vs 0.985 healthy retention)
  - Degraded nodes fail more often (0.10 vs 0.05 down probability)
  - Down nodes recover less often (0.01 vs 0.02)
- Reduced node capacity: 3 (vs 4 in sandbox)

**Benefits:**
- Tests detection speed (aggressive failures)
- Tests routing under extreme load
- Tests scalability (12 nodes)
- Reveals poor threshold choices

**Challenges:**
- Very difficult to get high completion rate
- Requires precise tuning
- Multiple simultaneous failures

**Expected Agent Performance:**
- Completion rate: 65–75% (good), 55–65% (acceptable)
- Must detect failures within 2–3 steps
- Heavy rerouting required

**Recommended:** Use this to verify your thresholds are tuned correctly.

---

### 🟣 EXTREME (16 nodes)
**When to use:** Absolute limit testing (find your agent's ceiling)

**Configuration:**
- 16 nodes (2.67× sandbox)
- Arrival rate: 6.0 (3× sandbox)
- Extremely tight deadline slack: 1–3 steps (vs 4–10)
- **Very aggressive failure dynamics:**
  - Nodes stay healthy only 95% (vs 98.5%)
  - Degraded nodes fail 20% of the time (vs 5%)
  - Down nodes rarely recover (only 0.5%)
- Very low node capacity: 2

**Benefits:**
- Absolute stress test
- Reveals if your approach scales
- Tests decision speed (milliseconds matter)
- Finds architectural issues

**Challenges:**
- Nearly impossible to achieve high completion rates
- May be beyond reasonable expectations
- Can reveal if your decision logic is O(n²) or worse

**Expected Agent Performance:**
- Completion rate: 50–70% (acceptable under extreme load)
- Multiple simultaneous failures always happening
- Rerouting becomes less effective (too many failures)
- May hit time budget limits (if decision logic is slow)

**Recommended:** Use only after other environments work well. This is a "stress ceiling" test.

---

## Which Environment Should I Use When?

### Phase 1: Development (Weeks 1–2)
```
Use LIGHT with debug=True
├─ Understand failure modes
├─ Verify scoring logic
├─ Test rerouting decisions
└─ Watch ground-truth vs predictions
```

### Phase 2: Baseline Tuning (Week 2)
```
Use SANDBOX (main testing)
├─ Tune health thresholds
├─ Optimize rerouting logic
├─ Run 10 seeds, measure completion rate
└─ Get to 75%+ completion
```

### Phase 3: Validation (Week 3)
```
Run on LIGHT, SANDBOX, MEDIUM (in sequence)
├─ Verify generalization
├─ Check if thresholds work across configs
└─ If scores drop >10%, revisit tuning
```

### Phase 4: Stress Testing (Week 3, Final)
```
Run on HARD and EXTREME
├─ Verify agent doesn't crash
├─ Check if decision logic is fast enough
├─ Measure scaling behavior
└─ Document performance ceiling
```

---

## How to Run Tests

### Single Environment
```python
from test_env_medium import make_medium_env
from health_aware_agent import HealthAwareAgent

env = make_medium_env(seed=42, debug=False)
agent = HealthAwareAgent(n_nodes=env.n_nodes, node_capacity=env.node_capacity)

obs = env.reset()
agent.reset()

for step in range(400):
    actions = agent.act(obs)
    obs, reward, done, info = env.step(actions)
    agent.update(obs, reward, done, info)
    if done:
        break

log = env.get_episode_log()
print(f"Completed: {log['completed_count']}, Failed: {log['failed_count']}")
```

### All Environments (Automated)
```bash
python test_all_environments.py
```
(After you uncomment the test code)

### Quick Comparison (5 seeds each)
```python
from test_all_environments import run_agent_on_env, print_results_table
from test_env_light import make_light_env
from test_env_sandbox import make_sandbox_env  # actually from sandbox_env
from test_env_medium import make_medium_env
from test_env_hard import make_hard_env
from test_env_extreme import make_extreme_env
from health_aware_agent import HealthAwareAgent

results = []
for env_func, name in [
    (make_light_env, "LIGHT"),
    (make_sandbox_env, "SANDBOX"),
    (make_medium_env, "MEDIUM"),
    (make_hard_env, "HARD"),
    (make_extreme_env, "EXTREME"),
]:
    result = run_agent_on_env(env_func, HealthAwareAgent, name, seeds=[0,1,2,3,4])
    results.append(result)
    print(f"✓ {name}: {result['avg_completion']:.1%} completion")

print_results_table(results)
```

---

## Interpreting Results

### Good Performance Pattern
```
LIGHT:     90% (easy, as expected)
SANDBOX:   78% (solid baseline)
MEDIUM:    75% (only -3% drop, good generalization)
HARD:      72% (only -6% drop, very good)
EXTREME:   68% (only -10% drop, excellent)
```
**Interpretation:** Your thresholds generalize well across configs!

### Poor Performance Pattern
```
LIGHT:     92% (good)
SANDBOX:   80% (okay)
MEDIUM:    65% (big drop -15%)
HARD:      45% (worse)
EXTREME:   20% (collapse)
```
**Interpretation:** Your agent has hardcoded assumptions. Likely:
- Health thresholds tuned to sandbox numbers
- Latency penalties assume specific node count
- Rerouting logic depends on specific capacity
- Decision logic may be too slow for many nodes

**Fix:** Review thresholds and move from "absolute numbers" to "relative signals"

---

## Threshold Tuning Tips

### If Too Many Tasks Complete Late (Low Completion Rate)
**Problem:** Not rerouting enough, or detecting too slowly

**Try:**
- Lower `REROUTE_HEALTH_THRESHOLD` (currently 40 → try 35)
- Lower `REROUTE_DEADLINE_URGENCY` (currently 10 → try 8)
- Increase latency penalty (currently -35 → try -45)

### If Too Much Thrashing (High Churn, Unnecessary Reroutes)
**Problem:** Rerouting too aggressively

**Try:**
- Raise `REROUTE_HEALTH_THRESHOLD` (currently 40 → try 45)
- Raise `REROUTE_DEADLINE_URGENCY` (currently 10 → try 12)
- Increase reroute cooldown (currently 5 → try 7)

### If Performance Collapses on Harder Environments
**Problem:** Thresholds are absolute (tuned to sandbox numbers)

**Try:**
- Use relative signals instead:
  - `if latency > baseline_latency * 5` instead of `> 250ms`
  - `if error_rate_trend_up_by_20%` instead of fixed thresholds
  - Score nodes relative to median, not absolute scale

### If Detection Latency is Too High
**Problem:** Takes >5 steps to detect failures

**Try:**
- Reduce window size (currently 5 → try 3)
- Increase trend penalty (currently -10 → try -20)
- Lower error rate thresholds (currently 0.5 → try 0.4)

---

## Environment File Structure

All test environments follow the same interface as `sandbox_env.py`:

```python
def make_XXX_env(seed=None, debug=False) -> ClusterEnv:
    """Returns a ClusterEnv with specific parameters."""
    return ClusterEnv(
        n_nodes=N_NODES,           # Varies by environment
        node_capacity=NODE_CAPACITY, # Varies by environment
        arrival_rate=ARRIVAL_RATE,   # Varies by environment
        duration_range=(3, 8),       # Same for all
        slack_range=(X, Y),          # Varies by environment
        health_transition=...,        # Optional, varies by environment
        episode_length=400,          # Same for all
        seed=seed,
        expose_health=debug,
    )
```

**Drop-in replacement:** You can use any environment without changing agent code!

---

## FAQ

**Q: Which environment should I submit against?**
A: Your agent will be evaluated on a *secret* environment with unknown parameters. Design to generalize, not to sandbox. Test on all 5 environments to verify.

**Q: What if my agent times out on EXTREME?**
A: Your decision logic might be O(n²) or worse. Optimize:
- Reduce history window size (from 5 to 3)
- Precompute node scores once per step (not per task)
- Use early-exit conditions in decision loops

**Q: Can I modify these environments?**
A: Yes! Create your own:
```python
def make_custom_env(seed=None, debug=False) -> ClusterEnv:
    return ClusterEnv(
        n_nodes=7,
        node_capacity=5,
        arrival_rate=2.5,
        slack_range=(3, 9),
        ...
    )
```

**Q: Should I test with debug=True or debug=False?**
A: Both!
- Development: `debug=True` to watch ground truth
- Testing: `debug=False` to simulate real evaluation (no ground truth)

**Q: How many seeds should I test?**
A: Minimum 5, ideally 10+. Stochastic environments need many runs.

---

## Performance Benchmarks

### Baseline (Round-Robin, No Health Awareness)
| Environment | Completion Rate |
|-------------|-----------------|
| LIGHT | 90% |
| SANDBOX | 70% |
| MEDIUM | 60% |
| HARD | 45% |
| EXTREME | 25% |

### Expected (Your Agent)
| Environment | Completion Rate | Target |
|-------------|-----------------|--------|
| LIGHT | 85–95% | >85% |
| SANDBOX | 75–85% | >75% |
| MEDIUM | 72–82% | >70% |
| HARD | 65–75% | >60% |
| EXTREME | 55–70% | >50% |

If your agent beats these targets, you're in great shape!

---

Good luck with testing! 🚀
