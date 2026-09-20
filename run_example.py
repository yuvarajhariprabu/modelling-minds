"""
Demonstration comparison: Run both the Health-Blind Baseline and the HealthAwareAgent
against sandbox_env.py to directly demonstrate the fault-tolerance improvement.
Run: python run_example.py
"""

from sandbox_env import make_sandbox_env
from baseline_agent import RoundRobinNoHealthCheck
from health_aware_agent import HealthAwareAgent


def run_single_episode(agent_class, name: str, seed: int = 3):
    print(f"\n" + "=" * 70)
    print(f" Running: {name} (Seed={seed})")
    print("=" * 70)

    env = make_sandbox_env(seed=seed, debug=True)
    agent = agent_class(n_nodes=env.n_nodes, node_capacity=env.node_capacity)

    obs = env.reset()
    agent.reset()

    completed_before = 0
    failed_before = 0

    for t in range(env.episode_length):
        actions = agent.act(obs)
        obs, reward, done, info = env.step(actions)
        agent.update(obs, reward, done, info)

        if info.get("failed_this_step"):
            for node_id in info["failed_this_step"]:
                new_state = info["node_true_states"][node_id]
                if new_state != "healthy":
                    print(f"  [t={t:3d}] Node {node_id} transitioned -> {new_state.upper()}")

        if (t + 1) % 100 == 0:
            log = env.get_episode_log()
            d_completed = log["completed_count"] - completed_before
            d_failed = log["failed_count"] - failed_before
            completed_before, failed_before = log["completed_count"], log["failed_count"]
            print(f"    t={t+1:3d} | Completed(last 100)={d_completed:3d} | Failed(last 100)={d_failed:3d}")

        if done:
            break

    log = env.get_episode_log()
    tot_c = log["completed_count"]
    tot_f = log["failed_count"]
    rate = tot_c / (tot_c + tot_f) * 100.0
    return tot_c, tot_f, rate


if __name__ == "__main__":
    b_c, b_f, b_rate = run_single_episode(RoundRobinNoHealthCheck, "Baseline (Round-Robin Health-Blind)")
    h_c, h_f, h_rate = run_single_episode(HealthAwareAgent, "HealthAwareAgent (Autonomous Fault-Tolerant)")

    print("\n" + "=" * 70)
    print(" HEAD-TO-HEAD COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Baseline':<18} | {'HealthAwareAgent':<18}")
    print("-" * 70)
    print(f"{'Tasks Completed':<25} | {b_c:18d} | {h_c:18d}")
    print(f"{'Tasks Failed':<25} | {b_f:18d} | {h_f:18d}")
    print(f"{'Completion Rate':<25} | {b_rate:17.1f}% | {h_rate:17.1f}%")
    print(f"{'Failure Reduction':<25} | {'--':<18} | {((b_f - h_f)/b_f)*100:17.1f}%")
    print("=" * 70)
