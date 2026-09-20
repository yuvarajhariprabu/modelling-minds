"""
MEDIUM TEST ENVIRONMENT. 
More challenging than sandbox:
- More nodes (8 vs 6)
- Higher arrival rate (3.0 vs 2.0)
- Tighter deadlines (less slack)
- Same failure dynamics (but more nodes = more potential failures)

Usage:
    from test_env_medium import make_medium_env
    env = make_medium_env(seed=0, debug=False)
    obs = env.reset()
    ...
"""

from cluster_env import ClusterEnv

N_NODES = 8
NODE_CAPACITY = 4
EPISODE_LENGTH = 400


def make_medium_env(seed: int = None, debug: bool = False) -> ClusterEnv:
    """
    Medium difficulty: more nodes, higher arrival rate, tighter deadlines.
    
    Challenge:
    - 8 nodes instead of 6 (more to monitor)
    - 3.0 arrival rate instead of 2.0 (more tasks = more deadline pressure)
    - Slack reduced to 3-8 (was 4-10) (tighter deadlines)
    - Otherwise same failure dynamics
    
    This tests:
    - Can your agent handle more nodes?
    - Can it detect failures faster with higher load?
    - Does it reroute efficiently under pressure?
    """
    return ClusterEnv(
        n_nodes=N_NODES,
        node_capacity=NODE_CAPACITY,
        arrival_rate=3.0,  # More tasks arriving
        duration_range=(3, 8),
        slack_range=(3, 8),  # Tighter deadlines (was 4-10)
        episode_length=EPISODE_LENGTH,
        seed=seed,
        expose_health=debug,
    )


# Convenience wrapper for consistency
def make_test_env(difficulty: str = "medium", seed: int = None, debug: bool = False):
    """
    Unified test environment maker.
    
    difficulty: "medium" (this file)
    """
    if difficulty == "medium":
        return make_medium_env(seed=seed, debug=debug)
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")
