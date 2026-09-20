"""
LIGHT TEST ENVIRONMENT.
Easier than sandbox for quick validation:
- Fewer nodes (4 vs 6) → simpler to debug
- Lower arrival rate (1.0 vs 2.0) → less pressure
- Generous deadlines (slack 6-15 vs 4-10) → more time margin
- Stable failures (nodes fail less frequently)

Usage:
    from test_env_light import make_light_env
    env = make_light_env(seed=0, debug=True)  # debug=True helpful here
    obs = env.reset()
    ...

This environment is great for:
- Understanding failure detection (fewer nodes = easier to trace)
- Testing rerouting logic in isolation (no time pressure)
- Debugging edge cases (generous deadlines)
- Quick iteration (few nodes = fast simulation)

Recommendation: Start here with debug=True to watch ground truth.
"""

import numpy as np
from cluster_env import ClusterEnv

N_NODES = 4
NODE_CAPACITY = 4
EPISODE_LENGTH = 400


def make_light_env(seed: int = None, debug: bool = False) -> ClusterEnv:
    """
    Light difficulty: 4 nodes, low arrival rate, generous deadlines, stable failures.
    """
    
    # Conservative failure dynamics: nodes stay healthy longer
    stable_transitions = np.array([
        [0.99,  0.008, 0.002],   # healthy → usually stay healthy
        [0.15,  0.80,  0.05 ],   # degraded → good chance to recover
        [0.05,  0.05,  0.90 ],   # down → might recover
    ])
    
    return ClusterEnv(
        n_nodes=N_NODES,
        node_capacity=NODE_CAPACITY,
        arrival_rate=1.0,  # Low arrival rate
        duration_range=(3, 8),
        slack_range=(6, 15),  # Generous slack (was 4-10)
        health_transition=stable_transitions,
        episode_length=EPISODE_LENGTH,
        seed=seed,
        expose_health=debug,
    )


def make_test_env(difficulty: str = "light", seed: int = None, debug: bool = False):
    """Unified test environment maker."""
    if difficulty == "light":
        return make_light_env(seed=seed, debug=debug)
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")
