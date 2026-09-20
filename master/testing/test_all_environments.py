"""
UNIFIED TEST RUNNER
Run HealthAwareAgent (and optional baseline comparison) across ALL official difficulty levels and seeds:
  1. LIGHT (4 nodes, low arrival rate, easy)
  2. SANDBOX (6 nodes, standard balanced)
  3. MEDIUM (8 nodes, higher load, tight deadlines)
  4. HARD (12 nodes, aggressive failure dynamics, stress test)
  5. EXTREME (16 nodes, 6.0 arrival rate, minimal slack, limit test)

Usage:
    python testing/test_all_environments.py
"""

import os
import sys

# Ensure root, master, and testing directories are in python path
TESTING_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_DIR = os.path.dirname(TESTING_DIR)
REPO_ROOT = os.path.dirname(MASTER_DIR)

for p in [REPO_ROOT, MASTER_DIR, TESTING_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sandbox_env import make_sandbox_env
from test_env_light import make_light_env
from test_env_medium import make_medium_env
from test_env_hard import make_hard_env
from test_env_extreme import make_extreme_env

from health_aware_agent import HealthAwareAgent
from baseline_agent import RoundRobinNoHealthCheck


def run_agent_on_env(env_factory, agent_class, env_name: str, seeds: list = None, debug: bool = False):
    """
    Run agent on given environment for multiple seeds.
    """
    if seeds is None:
        seeds = [0, 1, 2]

    results = {
        "env_name": env_name,
        "episodes": [],
        "avg_completion": 0.0,
        "avg_failed": 0.0,
        "avg_reward": 0.0,
    }

    for seed in seeds:
        environment = env_factory(seed=seed, debug=debug)
        ag = agent_class(n_nodes=environment.n_nodes, node_capacity=environment.node_capacity)

        obs = environment.reset()
        ag.reset()

        total_reward = 0.0
        for step in range(environment.episode_length):
            actions = ag.act(obs)
            obs, reward, done, info = environment.step(actions)
            ag.update(obs, reward, done, info)
            total_reward += reward

            if done:
                break

        log = environment.get_episode_log()
        completed = log["completed_count"]
        failed = log["failed_count"]
        total = completed + failed
        completion_rate = completed / total if total > 0 else 0.0

        results["episodes"].append({
            "seed": seed,
            "completed": completed,
            "failed": failed,
            "completion_rate": completion_rate,
            "total_reward": total_reward,
        })

    if results["episodes"]:
        avg_completion = sum(e["completion_rate"] for e in results["episodes"]) / len(results["episodes"])
        avg_failed = sum(e["failed"] for e in results["episodes"]) / len(results["episodes"])
        avg_reward = sum(e["total_reward"] for e in results["episodes"]) / len(results["episodes"])

        results["avg_completion"] = avg_completion
        results["avg_failed"] = avg_failed
        results["avg_reward"] = avg_reward

    return results


def print_results_table(agent_results: list, baseline_results: list = None):
    """Print results in a clean comparison table."""
    print("\n" + "=" * 95)
    print("TEST RESULTS: HealthAwareAgent Performance Across Difficulty Regimes")
    print("=" * 95)
    print(f"{'Environment':<18} {'Nodes':<7} {'Ours Comp %':<14} {'Ours Reward':<14} {'Base Comp %':<14} {'Advantage':<12}")
    print("-" * 95)

    node_map = {
        "LIGHT": 4,
        "SANDBOX": 6,
        "MEDIUM": 8,
        "HARD": 12,
        "EXTREME": 16,
    }

    for i, res in enumerate(agent_results):
        env_name = res["env_name"]
        key = env_name.split()[0]
        nodes = node_map.get(key, "?")

        ours_comp = f"{res['avg_completion']:.1%}"
        ours_rew = f"{res['avg_reward']:+.1f}"

        if baseline_results and i < len(baseline_results):
            base_res = baseline_results[i]
            base_comp = f"{base_res['avg_completion']:.1%}"
            diff = res["avg_completion"] - base_res["avg_completion"]
            adv = f"{diff*100:+.1f}%"
        else:
            base_comp = "--"
            adv = "--"

        print(f"{env_name:<18} {str(nodes):<7} {ours_comp:<14} {ours_rew:<14} {base_comp:<14} {adv:<12}")

    print("=" * 95)


def print_detailed_results(all_results: list):
    """Print detailed per-seed results."""
    print("\nDETAILED RESULTS (Per Seed):")
    print("=" * 75)

    for result in all_results:
        print(f"\n{result['env_name']}:")
        print(f"  {'Seed':<6} {'Completed':<12} {'Failed':<10} {'Completion':<15} {'Reward':<10}")
        print("-" * 60)

        for ep in result["episodes"]:
            seed = ep["seed"]
            completed = ep["completed"]
            failed = ep["failed"]
            completion_rate = ep["completion_rate"]
            reward = ep["total_reward"]
            print(f"  {seed:<6} {completed:<12} {failed:<10} {completion_rate:>6.1%}{' '*7} {reward:>7.1f}")


def main():
    print("=" * 80)
    print("  Cluster Agent - Full Environment Test Suite (MM26AI02)")
    print("=" * 80)

    seeds = [0, 1, 2]
    environments = [
        (make_light_env, "LIGHT (4 nodes)"),
        (make_sandbox_env, "SANDBOX (6 nodes)"),
        (make_medium_env, "MEDIUM (8 nodes)"),
        (make_hard_env, "HARD (12 nodes)"),
        (make_extreme_env, "EXTREME (16 nodes)"),
    ]

    agent_results = []
    baseline_results = []

    print("\n[1/2] Running HealthAwareAgent across all 5 environments (seeds: 0, 1, 2)...")
    for env_func, name in environments:
        res = run_agent_on_env(env_func, HealthAwareAgent, name, seeds=seeds, debug=False)
        agent_results.append(res)
        print(f"  [OK] {name:<20} -> Avg Completion: {res['avg_completion']:.1%} | Avg Reward: {res['avg_reward']:+.1f}")

    print("\n[2/2] Running Round-Robin Baseline across all 5 environments (for delta comparison)...")
    for env_func, name in environments:
        res = run_agent_on_env(env_func, RoundRobinNoHealthCheck, name, seeds=seeds, debug=False)
        baseline_results.append(res)
        print(f"  [OK] {name:<20} -> Avg Completion: {res['avg_completion']:.1%} | Avg Reward: {res['avg_reward']:+.1f}")

    # Print summary tables
    print_results_table(agent_results, baseline_results)
    print_detailed_results(agent_results)

    avg_across_all = sum(r["avg_completion"] for r in agent_results) / len(agent_results)
    print("\n" + "=" * 80)
    print(f"OVERALL SUMMARY: Average Completion Rate Across All Environments = {avg_across_all:.1%}")
    print("=" * 80)

    if avg_across_all > 0.75:
        print("[OK] EXCELLENT GENERALIZATION! Agent seamlessly adapts across all cluster topologies.")
    elif avg_across_all > 0.65:
        print("[WARN] GOOD GENERALIZATION! Solid performance across standard and stress environments.")
    else:
        print("[FAIL] Agent requires further threshold tuning.")


if __name__ == "__main__":
    main()
