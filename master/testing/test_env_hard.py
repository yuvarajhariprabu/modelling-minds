"""
HARD TEST ENVIRONMENT.
Significantly more challenging than sandbox:
- Many nodes (12 vs 6) → more to monitor, detect failures faster
- Very high arrival rate (4.5 vs 2.0) → constant deadline pressure
- Very tight deadlines (slack 2-5 vs 4-10) → no margin for error
- Aggressive failure rates (transition matrix modified)
  - Nodes degrade faster
  - Stay degraded longer
  - Fail more often

Usage:
    from test_env_hard import make_hard_env
    env = make_hard_env(seed=0, debug=False)
    obs = env.reset()
    ...

This environment tests:
- Can your agent scale to many nodes?
- Does it detect failures under extreme load?
- Can it make routing decisions fast enough?
- Does rerouting strategy hold up with very tight deadlines?
"""

import numpy as np
from cluster_env import ClusterEnv

N_NODES = 12
NODE_CAPACITY = 3  # Reduced capacity = tighter packing
EPISODE_LENGTH = 400


def make_hard_env(seed: int = None, debug: bool = False) -> ClusterEnv:
    """
    Hard difficulty: 12 nodes, 4.5 arrival rate, tight deadlines, aggressive failures.
    """
    
    # Modified failure dynamics: nodes fail/degrade more often
    # Original: [0.985, 0.012, 0.003] (healthy stay healthy)
    # Hard:     [0.97,  0.02,  0.01]  (more transitions)
    aggressive_transitions = np.array([
        [0.97,  0.02,  0.01 ],   # healthy   → healthy/degraded/down (easier to degrade)
        [0.08,  0.82,  0.10 ],   # degraded  → harder to recover, easier to fail
        [0.01,  0.04,  0.95 ],   # down      → harder to recover
    ])
    
    return ClusterEnv(
        n_nodes=N_NODES,
        node_capacity=NODE_CAPACITY,  # Reduced capacity
        arrival_rate=4.5,  # Much higher arrival rate
        duration_range=(3, 8),
        slack_range=(2, 5),  # Very tight deadlines (was 4-10)
        health_transition=aggressive_transitions,
        episode_length=EPISODE_LENGTH,
        seed=seed,
        expose_health=debug,
    )


def make_test_env(difficulty: str = "hard", seed: int = None, debug: bool = False):
    """Unified test environment maker."""
    if difficulty == "hard":
        return make_hard_env(seed=seed, debug=debug)
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")
