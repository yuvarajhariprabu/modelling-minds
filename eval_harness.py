"""
Evaluation Harness for Problem Statement MM26AI02:
"Keep the Cluster Alive: Detect, Reroute, Recover"

Measures:
  1. Completion Rate: completed / (completed + failed)
  2. Detection Latency: Average steps between node transitioning to DOWN
     and the agent stopping new assignments to that node.
  3. Churn Rate: Number of reroutes per task.
  4. Total Reward: Net sum of completion rewards (+1.0), failure penalties (-1.0),
     and unassigned holding costs (-0.01).
  5. Computational Efficiency: Mean wall-clock time per episode.
"""

import time
from typing import Dict, List, Type
import numpy as np

from agent_interface import BaseAgent
from cluster_env import ClusterEnv, DOWN, HEALTHY, DEGRADED
from sandbox_env import make_sandbox_env


def evaluate_agent(
    agent_class: Type[BaseAgent],
    agent_kwargs: dict = None,
    seeds: List[int] = None,
    env_config: dict = None,
    verbose: bool = False,
) -> dict:
    if agent_kwargs is None:
        agent_kwargs = {}
    if seeds is None:
        seeds = list(range(10))

    completion_rates = []
    total_rewards = []
    detection_latencies = []
    churn_counts = []
    run_times = []

    for seed in seeds:
        t_start = time.perf_counter()
        
        if env_config:
            env = ClusterEnv(seed=seed, expose_health=True, **env_config)
        else:
            env = make_sandbox_env(seed=seed, debug=True)

        agent = agent_class(n_nodes=env.n_nodes, node_capacity=env.node_capacity, **agent_kwargs)
        obs = env.reset()
        agent.reset()

        total_reward = 0.0
        total_reroutes = 0
        node_down_since: Dict[int, int] = {}
        down_events_detected_steps: List[int] = []

        for t in range(env.episode_length):
            for node_id, state in enumerate(env._node_states):
                if state == DOWN:
                    if node_id not in node_down_since:
                        node_down_since[node_id] = t
                else:
                    if node_id in node_down_since:
                        del node_down_since[node_id]

            task_lookup = {task["task_id"]: task for task in obs["tasks"]}
            actions = agent.act(obs)

            for tid, target_node in actions.items():
                if tid not in task_lookup:
                    continue
                current_task_node = task_lookup[tid]["node"]

                if current_task_node is not None and current_task_node != target_node:
                    total_reroutes += 1
                elif current_task_node is None:
                    if target_node in node_down_since:
                        steps_in_down = t - node_down_since[target_node] + 1
                        down_events_detected_steps.append(steps_in_down)

            obs, reward, done, info = env.step(actions)
            agent.update(obs, reward, done, info)
            total_reward += reward

            if done:
                break

        t_elapsed = time.perf_counter() - t_start
        log = env.get_episode_log()
        n_completed = log["completed_count"]
        n_failed = log["failed_count"]
        comp_rate = n_completed / max(1, n_completed + n_failed)

        mean_det_lat = float(np.mean(down_events_detected_steps)) if down_events_detected_steps else 0.0

        completion_rates.append(comp_rate)
        total_rewards.append(total_reward)
        detection_latencies.append(mean_det_lat)
        churn_counts.append(total_reroutes)
        run_times.append(t_elapsed)

        if verbose:
            print(
                f"Seed {seed:2d} | Completed: {n_completed:3d} | Failed: {n_failed:3d} | "
                f"CompRate: {comp_rate*100:5.1f}% | Reward: {total_reward:6.1f} | "
                f"Reroutes: {total_reroutes:3d} | DetLat: {mean_det_lat:4.1f} steps | "
                f"Time: {t_elapsed:.2f}s"
            )

    return {
        "completion_rate_mean": float(np.mean(completion_rates)),
        "completion_rate_std": float(np.std(completion_rates)),
        "reward_mean": float(np.mean(total_rewards)),
        "reward_std": float(np.std(total_rewards)),
        "detection_latency_mean": float(np.mean(detection_latencies)),
        "churn_mean": float(np.mean(churn_counts)),
        "time_per_episode": float(np.mean(run_times)),
    }
