# Health-Aware Agent: Design Cheat Sheet

## The Core Problem (In 30 seconds)

```
Cluster nodes fail silently.
↓
You see only noisy telemetry (heartbeat, latency, error rate).
↓
You must detect failures + reroute tasks before deadline.
↓
Rerouting has a cost (cold restart).
↓
Be confident before rerouting.
```

---

## The Three Scoring Metrics

```
1️⃣  COMPLETION RATE (Most Important)
    Tasks that finish on time = +1 reward each
    Goal: Maximize completed_count / (completed_count + failed_count)

2️⃣  DETECTION LATENCY (Medium)
    Steps until you stop routing to a failed node
    Goal: Detect within 3–5 steps of failure

3️⃣  CHURN RATE (Least Important)
    Unnecessary reroutes = wasted cold restarts
    Goal: Only reroute when evidence + urgency justify it
```

---

## Signal-to-Action Mapping

```
┌─────────────────┬──────────┬──────────┬──────────┬─────────────────┐
│ Signal          │ HEALTHY  │ DEGRADED │ DOWN     │ Our Penalty     │
├─────────────────┼──────────┼──────────┼──────────┼─────────────────┤
│ Heartbeat OK    │ >90%     │ ~85%     │ ~5%      │ -50 if false    │
│ Latency         │ 50ms     │ 300ms    │ NULL     │ -35 if >250ms   │
│ Error Rate      │ 2%       │ 25%      │ 95%      │ -60 if >0.7     │
│ Queue Length    │ Normal   │ Backing  │ Stuck    │ -5 if full      │
│ Trend           │ Stable   │ Worsening│ Crashed  │ -10 if slope↑   │
└─────────────────┴──────────┴──────────┴──────────┴─────────────────┘

Score = 100 - Σ(penalties)

HEALTHY (70–100):    Route here!
DEGRADED (30–70):    Monitor, maybe assign.
FAILED (0–30):       Stop routing. Reroute if urgent.
```

---

## Decision Tree: Should I Reroute This Task?

```
Task on Node X, Deadline D, Current Step T

Q1: Is node X healthy? (score >= 40)
    YES → Don't reroute (leave it running)
    NO ↓

Q2: Is deadline urgent? (D - T < 10 steps)
    NO → Don't reroute (task might finish anyway)
    YES ↓

Q3: Haven't rerouted recently? (T - last_reroute_time > 5)
    NO → Don't reroute yet (cooldown active, avoid thrashing)
    YES ↓

✓ REROUTE → Move to healthiest available node with capacity
```
