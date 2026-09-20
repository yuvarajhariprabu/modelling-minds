"""
EXTREME TEST ENVIRONMENT.
Maximum difficulty stress test:
- Many nodes (16 vs 6) → massive monitoring challenge
- Extreme arrival rate (6.0 vs 2.0) → constant deadline crisis
- Extremely tight deadlines (slack 1-3 vs 4-10) → almost no time margin
- Very aggressive failure rate (nodes fail very frequently)
  - High probability of transitioning to degraded/down
  - Hard to recover
  - Multiple simultaneous failures

Usage:
    from test_env_extreme import make_extreme_env
    env = make_extreme_env(seed=0, debug=False)
    obs = env.reset()
    ...

This environment tests absolute limits:
- Does your agent handle 16 nodes?
- Can it keep up with 6 tasks/step arriving?
- Does it make good decisions under extreme deadline pressure?
- Does it avoid panic rerouting when everything is on fire?
"""

import numpy as np
from cluster_env import ClusterEnv

N_NODES = 16
NODE_CAPACITY = 2  # Very low capacity = every node packed
EPISODE_LENGTH = 400


def make_extreme_env(seed: int = None, debug: bool = False) -> ClusterEnv:
    """
    Extreme difficulty: 16 nodes, 6.0 arrival rate, minimal slack, very frequent failures.
    """
    
    # Very aggressive transitions: nodes fail often
    # This is a chaotic environment where multiple nodes are down at any time
    extreme_transitions = np.array([
        [0.95,  0.04,  0.01 ],   # healthy quickly degrade
        [0.05,  0.75,  0.20 ],   # degraded → hard to stay/recover, easy to fail
        [0.005, 0.045, 0.95 ],   # down → usually stays down
    ])
    
    return ClusterEnv(
        n_nodes=N_NODES,
        node_capacity=NODE_CAPACITY,  # Very tight packing
        arrival_rate=6.0,  # Extreme arrival rate
        duration_range=(3, 8),
        slack_range=(1, 3),  # Minimal slack (was 4-10)
        health_transition=extreme_transitions,
        episode_length=EPISODE_LENGTH,
        seed=seed,
        expose_health=debug,
    )


def make_test_env(difficulty: str = "extreme", seed: int = None, debug: bool = False):
    """Unified test environment maker."""
    if difficulty == "extreme":
        return make_extreme_env(seed=seed, debug=debug)
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")
