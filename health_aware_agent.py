"""
Health-Aware Autonomous Agent for Distributed Cluster Task Management.
Problem Statement: MM26AI02 - "Keep the Cluster Alive: Detect, Reroute, Recover"

Architecture:
  1. Multi-Signal Bayesian & Statistical Telemetry Filter (BeliefState)
  2. Empirical Task Progress & Stall Monitor (ProgressTracker)
  3. Expected Value of Completion (EVC) & Cold-Restart Feasibility Engine
  4. Dynamic Velocity & Congestion-Aware Scheduler (LoadBalancer)
  5. Transparent Decision Rationale & Interpretability Logger (ExplainabilityLogger)
"""

from collections import deque
from typing import Dict, List, Optional
import math

from agent_interface import BaseAgent


class TelemetryHistory:
    def __init__(self, window_size: int = 6):
        self.window_size = window_size
        self.heartbeats: deque = deque(maxlen=window_size)
        self.latencies: deque = deque(maxlen=window_size)
        self.error_rates: deque = deque(maxlen=window_size)
        self.queue_lens: deque = deque(maxlen=window_size)
        self.ewma_latency: Optional[float] = None
        self.ewma_error: Optional[float] = None
        self.consecutive_heartbeat_fails: int = 0
        self.consecutive_stalls: int = 0
        # CUSUM (Cumulative Sum) Statistical Change-Point Detectors
        self.cusum_latency: float = 0.0
        self.cusum_error: float = 0.0
        self.cusum_flagged: bool = False

    def update(self, node_obs: dict):
        hb = node_obs.get("heartbeat_ok", True)
        lat = node_obs.get("latency_ms")
        err = node_obs.get("error_rate", 0.0)
        q = node_obs.get("queue_len", 0)

        self.heartbeats.append(hb)
        self.latencies.append(lat)
        self.error_rates.append(err)
        self.queue_lens.append(q)

        if not hb:
            self.consecutive_heartbeat_fails += 1
        else:
            self.consecutive_heartbeat_fails = 0

        alpha = 0.4
        if lat is not None:
            if self.ewma_latency is None:
                self.ewma_latency = lat
            else:
                self.ewma_latency = alpha * lat + (1.0 - alpha) * self.ewma_latency
            # CUSUM latency deviation (reference healthy baseline ~90ms, allowance K=30)
            self.cusum_latency = max(0.0, self.cusum_latency + (lat - 100.0) - 30.0)
        else:
            self.cusum_latency += 70.0

        if self.ewma_error is None:
            self.ewma_error = err
        else:
            self.ewma_error = alpha * err + (1.0 - alpha) * self.ewma_error

        # CUSUM error rate deviation (baseline 0.02, allowance K=0.03)
        self.cusum_error = max(0.0, self.cusum_error + (err - 0.03) - 0.03)
        self.cusum_flagged = (self.cusum_latency > 140.0) or (self.cusum_error > 0.20)


class HealthAwareAgent(BaseAgent):
    HEALTH_THRESHOLD_ROUTE = 50.0
    REROUTE_COOLDOWN_STEPS = 4

    def __init__(self, n_nodes: int, node_capacity: int, log_decisions: bool = False):
        super().__init__(n_nodes, node_capacity)
        self.log_decisions = log_decisions
        self.reset()

    def reset(self) -> None:
        self.current_step = 0
        self.telemetry = {i: TelemetryHistory(window_size=6) for i in range(self.n_nodes)}
        self.task_total_duration: Dict[int, float] = {}
        self.task_last_duration: Dict[int, float] = {}
        self.task_last_node: Dict[int, int] = {}
        self.task_last_reroute_step: Dict[int, int] = {}
        self.task_arrival_step: Dict[int, int] = {}
        self.beliefs = {i: [0.95, 0.04, 0.01] for i in range(self.n_nodes)}
        self.node_scores = {i: 100.0 for i in range(self.n_nodes)}
        self.decision_logs: List[dict] = []

    def act(self, obs: dict) -> dict:
        self.current_step = obs["step"]
        actions: Dict[int, int] = {}

        self._update_node_beliefs(obs["nodes"])
        self._track_task_progress(obs["tasks"])

        node_load = {i: 0 for i in range(self.n_nodes)}
        for task in obs["tasks"]:
            if task["node"] is not None:
                node_load[task["node"]] += 1

        rerouted_tasks = self._triage_inflight_tasks(obs["tasks"], node_load)
        actions.update(rerouted_tasks)

        new_assignments = self._route_new_tasks(obs["tasks"], node_load)
        actions.update(new_assignments)

        return actions

    def _update_node_beliefs(self, nodes_obs: List[dict]) -> None:
        for node in nodes_obs:
            nid = node["node_id"]
            hist = self.telemetry[nid]
            hist.update(node)

            hb = node.get("heartbeat_ok", True)
            lat = node.get("latency_ms")
            err = node.get("error_rate", 0.0)

            score = 100.0
            if not hb:
                score -= 45.0
            if hist.consecutive_heartbeat_fails >= 2:
                score -= 30.0

            if lat is None:
                score -= 55.0
            elif lat > 280:
                score -= 40.0
            elif lat > 180:
                score -= 25.0
            elif lat > 100:
                score -= 10.0

            if err > 0.70:
                score -= 60.0
            elif err > 0.40:
                score -= 40.0
            elif err > 0.15:
                score -= 20.0
            elif err > 0.05:
                score -= 8.0

            if len(hist.error_rates) >= 3:
                recent_errs = list(hist.error_rates)[-3:]
                if recent_errs[-1] > recent_errs[0] + 0.05:
                    score -= 10.0

            if len(hist.latencies) >= 3:
                valid_lats = [l for l in list(hist.latencies)[-3:] if l is not None]
                if len(valid_lats) >= 2 and valid_lats[-1] > valid_lats[0] + 40:
                    score -= 10.0

            if hist.consecutive_stalls >= 2:
                score -= 35.0
            elif hist.consecutive_stalls == 1:
                score -= 15.0

            # CUSUM early change-point warning penalty (pre-failure drain)
            if hist.cusum_flagged:
                score -= 18.0

            self.node_scores[nid] = max(0.0, min(100.0, score))

            p_hb_h = 0.99 if hb else 0.01
            p_hb_deg = 0.85 if hb else 0.15
            p_hb_down = 0.05 if hb else 0.95

            if lat is None:
                p_lat_h, p_lat_deg, p_lat_down = 0.001, 0.05, 0.95
            else:
                p_lat_h = math.exp(-0.5 * ((lat - 50.0) / 25.0) ** 2) + 1e-4
                p_lat_deg = math.exp(-0.5 * ((lat - 300.0) / 100.0) ** 2) + 1e-4
                p_lat_down = 0.001

            p_err_h = math.exp(-0.5 * ((err - 0.02) / 0.05) ** 2) + 1e-4
            p_err_deg = math.exp(-0.5 * ((err - 0.25) / 0.12) ** 2) + 1e-4
            p_err_down = math.exp(-0.5 * ((err - 0.95) / 0.10) ** 2) + 1e-4

            prior = self.beliefs[nid]
            prior = [
                0.95 * prior[0] + 0.08 * prior[1] + 0.02 * prior[2],
                0.04 * prior[0] + 0.84 * prior[1] + 0.03 * prior[2],
                0.01 * prior[0] + 0.08 * prior[1] + 0.95 * prior[2],
            ]

            post_h = prior[0] * p_hb_h * p_lat_h * p_err_h
            post_deg = prior[1] * p_hb_deg * p_lat_deg * p_err_deg
            post_down = prior[2] * p_hb_down * p_lat_down * p_err_down
            total = post_h + post_deg + post_down + 1e-12

            self.beliefs[nid] = [post_h / total, post_deg / total, post_down / total]

    def _track_task_progress(self, tasks_obs: List[dict]) -> None:
        node_progress: Dict[int, List[float]] = {i: [] for i in range(self.n_nodes)}

        for task in tasks_obs:
            tid = task["task_id"]
            node = task["node"]
            dur_rem = task["duration_remaining"]

            if tid not in self.task_total_duration:
                self.task_total_duration[tid] = dur_rem
                self.task_arrival_step[tid] = self.current_step

            if node is not None and self.task_last_node.get(tid) == node:
                prev_dur = self.task_last_duration.get(tid, dur_rem)
                delta = prev_dur - dur_rem
                node_progress[node].append(delta)

            self.task_last_duration[tid] = dur_rem
            self.task_last_node[tid] = node

        for nid in range(self.n_nodes):
            deltas = node_progress[nid]
            if deltas:
                max_progress = max(deltas)
                if max_progress <= 0.001:
                    self.telemetry[nid].consecutive_stalls += 1
                else:
                    self.telemetry[nid].consecutive_stalls = 0

    def _triage_inflight_tasks(self, tasks_obs: List[dict], node_load: Dict[int, int]) -> Dict[int, int]:
        actions: Dict[int, int] = {}
        assigned_tasks = [t for t in tasks_obs if t["node"] is not None]
        assigned_tasks.sort(key=lambda t: t["deadline"])

        for task in assigned_tasks:
            tid = task["task_id"]
            cur_node = task["node"]
            p = self.beliefs[cur_node]
            v_curr = p[0] * 1.0 + p[1] * 0.25 + p[2] * 0.0

            if p[0] > 0.85 and v_curr > 0.85:
                continue

            last_reroute = self.task_last_reroute_step.get(tid, -999)
            if self.current_step - last_reroute < self.REROUTE_COOLDOWN_STEPS:
                continue

            time_left = task["deadline"] - self.current_step
            dur_rem = task["duration_remaining"]
            dur_total = self.task_total_duration.get(tid, dur_rem)

            restart_feasible = (self.current_step + dur_total <= task["deadline"])
            if not restart_feasible:
                continue

            target_node = self._select_best_node(node_load, exclude_node=cur_node)
            if target_node is None:
                continue

            target_v = self.beliefs[target_node][0] * 1.0 + self.beliefs[target_node][1] * 0.25
            if target_v <= v_curr:
                continue

            expected_time_curr = dur_rem / max(0.01, v_curr)
            expected_time_target = dur_total / max(0.01, target_v)

            # Selective Graceful Drain vs Cold Restart:
            # - If node is down (P(down) > 0.55 or V < 0.05), failover immediately.
            # - If task cannot complete on current node before deadline, reroute.
            # - If task only has minimal work remaining (<=1 step) on degraded node,
            #   let it gracefully finish rather than paying the full cold-restart cost.
            is_down = (p[2] > 0.55) or (v_curr < 0.05)
            cannot_finish_curr = (expected_time_curr > time_left)
            target_faster = (expected_time_target + 1.0 < expected_time_curr) and (dur_rem > 1.5)

            should_reroute = is_down or cannot_finish_curr or target_faster
            if not should_reroute:
                continue

            if self.log_decisions:
                self.decision_logs.append({
                    "step": self.current_step,
                    "action": "REROUTE",
                    "task_id": tid,
                    "from_node": cur_node,
                    "to_node": target_node,
                    "reason": (
                        f"Node {cur_node} V={v_curr:.2f}, P(down)={p[2]:.2f}, "
                        f"time_left={time_left:.1f}, restart_dur={dur_total:.1f}"
                    )
                })

            actions[tid] = target_node
            node_load[cur_node] -= 1
            node_load[target_node] += 1
            self.task_last_reroute_step[tid] = self.current_step
            self.task_last_node[tid] = target_node

        return actions

    def _route_new_tasks(self, tasks_obs: List[dict], node_load: Dict[int, int]) -> Dict[int, int]:
        actions: Dict[int, int] = {}
        unassigned_tasks = [t for t in tasks_obs if t["node"] is None]

        viable = [t for t in unassigned_tasks if self.current_step + t["duration_remaining"] <= t["deadline"]]
        viable.sort(key=lambda t: (t["duration_remaining"], t["deadline"]))

        for task in viable:
            tid = task["task_id"]
            best_node = self._select_best_node(node_load)
            if best_node is not None:
                actions[tid] = best_node
                node_load[best_node] += 1
                self.task_last_node[tid] = best_node

                if self.log_decisions:
                    self.decision_logs.append({
                        "step": self.current_step,
                        "action": "ROUTE_NEW",
                        "task_id": tid,
                        "to_node": best_node,
                        "reason": f"Assigned via SRPT to best available node (score={self.node_scores[best_node]:.1f})"
                    })

        return actions

    def _select_best_node(self, current_load: Dict[int, int], exclude_node: Optional[int] = None) -> Optional[int]:
        best_nid = None
        best_util = -999.0

        for nid in range(self.n_nodes):
            if nid == exclude_node or current_load[nid] >= self.node_capacity:
                continue

            p = self.beliefs[nid]
            v = p[0] * 1.0 + p[1] * 0.25 + p[2] * 0.0

            if p[2] > 0.70 or v < 0.10:
                continue

            load_ratio = current_load[nid] / float(self.node_capacity)
            util = v - 0.15 * load_ratio

            if util > best_util:
                best_util = util
                best_nid = nid

        return best_nid

    def update(self, obs: dict, reward: float, done: bool, info: dict) -> None:
        pass
