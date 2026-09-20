"""
Interactive Demonstration & Visual Dashboard for HealthAwareAgent vs Baseline.
Runs a live episode with transparent explainability logs, showing:
  - When nodes fail (ground truth)
  - How the agent's multi-signal belief reacts
  - Exactly why tasks were rerouted (interpretability bonus)
  - Completion vs Failure timeline
"""

from sandbox_env import make_sandbox_env
from baseline_agent import RoundRobinNoHealthCheck
from health_aware_agent import HealthAwareAgent


def run_demo(seed: int = 3):
    print("=" * 85)
    print(f" LIVE CLUSTER DEMO: HealthAwareAgent (Seed={seed})")
    print(" Problem Statement MM26AI02 - Agentic Fault Tolerance & Recovery")
    print("=" * 85)

    env = make_sandbox_env(seed=seed, debug=True)
    agent = HealthAwareAgent(n_nodes=env.n_nodes, node_capacity=env.node_capacity, log_decisions=True)

    obs = env.reset()
    agent.reset()

    reroutes_logged = 0
    completed_prev, failed_prev = 0, 0

    for t in range(env.episode_length):
        if env._failure_events and env._failure_events[-1][0] == t:
            step, nid, new_state = env._failure_events[-1]
            print(f"\n[EVENT t={t:3d}] Node {nid} transitioned to '{new_state.upper()}'")

        actions = agent.act(obs)
        obs, reward, done, info = env.step(actions)
        agent.update(obs, reward, done, info)

        reroute_logs = [l for l in agent.decision_logs if l["action"] == "REROUTE"]
        while reroutes_logged < len(reroute_logs) and reroutes_logged < 8:
            log = reroute_logs[reroutes_logged]
            print(f"   --> [EXPLAINABILITY] Rerouted Task #{log['task_id']} from Node {log['from_node']} -> Node {log['to_node']}")
            print(f"       Reason: {log['reason']}")
            reroutes_logged += 1

        if (t + 1) % 100 == 0:
            log = env.get_episode_log()
            d_comp = log["completed_count"] - completed_prev
            d_fail = log["failed_count"] - failed_prev
            completed_prev, failed_prev = log["completed_count"], log["failed_count"]

            print(f"\n[CHECKPOINT t={t+1:3d}] Health Scores: " +
                  " ".join([f"N{i}:{agent.node_scores[i]:4.0f}" for i in range(env.n_nodes)]) +
                  f" | Last 100 Steps -> Done: {d_comp:3d}, Missed: {d_fail:2d}")

        if done:
            break

    log = env.get_episode_log()
    total_c = log["completed_count"]
    total_f = log["failed_count"]
    comp_rate = total_c / (total_c + total_f) * 100.0

    print("\n" + "=" * 85)
    print(" EPISODE SUMMARY - HEALTH AWARE AGENT")
    print("=" * 85)
    print(f" Total Tasks Completed : {total_c:4d}")
    print(f" Total Tasks Failed    : {total_f:4d}")
    print(f" Task Completion Rate  : {comp_rate:5.1f}%")
    print(f" Total Reroute Actions : {len([l for l in agent.decision_logs if l['action'] == 'REROUTE']):4d}")
    print("=" * 85)


if __name__ == "__main__":
    run_demo(seed=3)
