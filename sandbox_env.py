"""
SANDBOX environment. This is yours to build and test against as much as you
want -- unlimited episodes, no scoring, and an optional debug mode that
reveals ground-truth node health so you can sanity-check your own detection
logic.

Usage:
    from sandbox_env import make_sandbox_env
    env = make_sandbox_env(seed=0, debug=True)
    obs = env.reset()
    obs, reward, done, info = env.step({101: 2, 104: 0})
"""

try:
    from env.cluster_env import ClusterEnv
except ImportError:
    from cluster_env import ClusterEnv

N_NODES = 6
NODE_CAPACITY = 4
EPISODE_LENGTH = 400


def make_sandbox_env(seed: int = None, debug: bool = False) -> ClusterEnv:
    return ClusterEnv(
        n_nodes=N_NODES,
        node_capacity=NODE_CAPACITY,
        arrival_rate=2.0,
        duration_range=(3, 8),
        slack_range=(4, 10),
        episode_length=EPISODE_LENGTH,
        seed=seed,
        expose_health=debug,
    )
