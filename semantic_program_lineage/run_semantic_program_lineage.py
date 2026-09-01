#!/usr/bin/env python3
"""Semantic program lineage toy inspired by SUN.

This experiment isolates one mechanism from SUN-style semantic unification: when
stage guards, rewards, demonstration labels, and success checks are hand-copied,
small inconsistencies can accumulate into measurable policy drift and misleading
reported success. A single typed task specification avoids that duplication by
compiling all interfaces from one source.

The package is deliberately lightweight and synthetic. It operates on a 2-D
pick-carry-release toy, stage-conditioned linear policies, and deterministic
scripted demonstrations. It does not implement Kuafu, MPC, RL training, DP3,
vision-language models, or real robots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HOME = np.array([-0.82, -0.74], dtype=float)
MAX_SPEED = 0.115
GRASP_CAPTURE_RADIUS = 0.055
GOAL_SETTLE_RADIUS = 0.070
STAGE_NAMES = (
    "approach_object",
    "close_gripper",
    "carry_to_goal",
    "open_at_goal",
    "retreat",
)
DISPLAY = {
    "compiled_spec": "Compiled spec",
    "copied_demo_labels": "Copied demo labels",
    "copied_stage_guards": "Copied stage guards",
    "copied_rewards": "Copied rewards",
    "copied_success_check": "Copied success check",
    "copied_all": "Copied all",
}
COLORS = {
    "compiled_spec": "#1b9e77",
    "copied_demo_labels": "#d95f02",
    "copied_stage_guards": "#7570b3",
    "copied_rewards": "#e7298a",
    "copied_success_check": "#66a61e",
    "copied_all": "#c0392b",
}


@dataclass(frozen=True)
class Config:
    seed: int = 13
    train_episodes: int = 48
    val_episodes: int = 20
    test_episodes: int = 36
    max_steps: int = 44
    ridge_alpha: float = 0.010
    reward_gains: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75)
    stage_bonus: float = 0.90
    label_shift_steps: tuple[int, ...] = (2, 2, 2, 1)
    speed_scale: tuple[float, ...] = (1.00, 0.82, 0.92, 0.72, 0.88)


@dataclass(frozen=True)
class StageSpec:
    name: str
    target_entity: str
    tolerance: float
    grip_target: float
    require_holding: bool | None
    dwell: int
    reward_weight: float
    speed: float


@dataclass(frozen=True)
class Lineage:
    key: str
    drift_demo: bool = False
    drift_guard: bool = False
    drift_reward: bool = False
    drift_success: bool = False


@dataclass
class SimState:
    eef: np.ndarray
    obj: np.ndarray
    goal: np.ndarray
    retreat: np.ndarray
    gripper: float
    holding: bool
    delivered: bool

    def copy(self) -> "SimState":
        return SimState(
            eef=self.eef.copy(),
            obj=self.obj.copy(),
            goal=self.goal.copy(),
            retreat=self.retreat.copy(),
            gripper=float(self.gripper),
            holding=bool(self.holding),
            delivered=bool(self.delivered),
        )


@dataclass
class Episode:
    episode_id: str
    states: list[SimState]
    actions: np.ndarray
    stage_labels: np.ndarray


@dataclass
class RolloutResult:
    lineage: str
    split: str
    episode_id: str
    reward_gain: float
    lineage_return: float
    compiled_return: float
    true_success: float
    reported_success: float
    stage_mismatch_rate: float
    final_object_error: float
    min_goal_error: float
    steps: int
    transition_steps: list[int]
    reported_transition_steps: list[int]
    executed_stages: np.ndarray
    states: list[SimState]
    actions: np.ndarray


STAGES = (
    StageSpec("approach_object", "object", 0.055, -1.0, False, 2, 1.20, 0.95),
    StageSpec("close_gripper", "object", 0.042, +1.0, True, 2, 1.45, 0.55),
    StageSpec("carry_to_goal", "goal", 0.060, +1.0, True, 2, 1.60, 1.00),
    StageSpec("open_at_goal", "goal", 0.065, -1.0, False, 2, 1.35, 0.58),
    StageSpec("retreat", "retreat", 0.082, -1.0, False, 2, 0.95, 0.88),
)
LINEAGES = (
    Lineage("compiled_spec"),
    Lineage("copied_demo_labels", drift_demo=True),
    Lineage("copied_stage_guards", drift_guard=True),
    Lineage("copied_rewards", drift_reward=True),
    Lineage("copied_success_check", drift_success=True),
    Lineage("copied_all", drift_demo=True, drift_guard=True, drift_reward=True, drift_success=True),
)


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32)


def stage_index(name: str) -> int:
    return STAGE_NAMES.index(name)


def clip_action(action: np.ndarray) -> np.ndarray:
    out = np.asarray(action, dtype=float).copy()
    out[:2] = np.clip(out[:2], -MAX_SPEED, MAX_SPEED)
    out[2] = float(np.clip(out[2], -1.0, 1.0))
    return out


def sample_geometry(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    obj = np.array([-0.42, -0.18]) + rng.normal(0.0, [0.11, 0.09])
    goal = np.array([0.47, 0.31]) + rng.normal(0.0, [0.10, 0.08])
    goal = np.clip(goal, -0.78, 0.78)
    retreat = np.clip(goal + np.array([0.18, 0.16]), -0.88, 0.88)
    return obj, goal, retreat


def make_initial_state(seed: int, obj: np.ndarray, goal: np.ndarray, retreat: np.ndarray) -> SimState:
    rng = np.random.default_rng(seed)
    eef = HOME + rng.normal(0.0, [0.025, 0.020])
    return SimState(eef=eef, obj=obj.copy(), goal=goal.copy(), retreat=retreat.copy(), gripper=-1.0, holding=False, delivered=False)


def entity_position(state: SimState, entity: str) -> np.ndarray:
    if entity == "eef":
        return state.eef
    if entity == "object":
        return state.obj
    if entity == "goal":
        return state.goal
    if entity == "retreat":
        return state.retreat
    raise ValueError(entity)


def compiled_target(stage_idx: int, state: SimState) -> np.ndarray:
    return entity_position(state, STAGES[stage_idx].target_entity)


def reward_target(lineage: Lineage, stage_idx: int, state: SimState) -> np.ndarray:
    target = compiled_target(stage_idx, state).copy()
    if not lineage.drift_reward:
        return target
    if stage_idx == 0:
        return 0.92 * state.obj + 0.08 * state.goal
    if stage_idx == 1:
        return 0.88 * state.obj + 0.12 * state.goal + np.array([0.010, -0.008])
    if stage_idx == 2:
        return state.goal + np.array([-0.062, 0.044])
    if stage_idx == 3:
        return state.goal + np.array([-0.045, 0.035])
    if stage_idx == 4:
        return state.retreat + np.array([0.028, -0.020])
    raise ValueError(stage_idx)


def reward_grip_target(lineage: Lineage, stage_idx: int) -> float:
    if not lineage.drift_reward:
        return STAGES[stage_idx].grip_target
    if stage_idx == 2:
        return 0.88
    return STAGES[stage_idx].grip_target


def features(state: SimState) -> np.ndarray:
    obj_rel = state.obj - state.eef
    goal_rel = state.goal - state.eef
    retreat_rel = state.retreat - state.eef
    goal_obj = state.goal - state.obj
    return np.array(
        [
            1.0,
            state.eef[0],
            state.eef[1],
            state.obj[0],
            state.obj[1],
            state.goal[0],
            state.goal[1],
            state.retreat[0],
            state.retreat[1],
            obj_rel[0],
            obj_rel[1],
            goal_rel[0],
            goal_rel[1],
            retreat_rel[0],
            retreat_rel[1],
            goal_obj[0],
            goal_obj[1],
            float(state.holding),
            state.gripper,
            np.linalg.norm(obj_rel),
            np.linalg.norm(goal_rel),
            np.linalg.norm(retreat_rel),
        ],
        dtype=float,
    )


def stage_distance(stage_idx: int, state: SimState) -> float:
    spec = STAGES[stage_idx]
    if spec.name == "carry_to_goal":
        return float(np.linalg.norm(state.obj - state.goal))
    return float(np.linalg.norm(state.eef - compiled_target(stage_idx, state)))


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.zeros_like(vec)
    return vec / norm


def reward_action(lineage: Lineage, stage_idx: int, state: SimState) -> np.ndarray:
    spec = STAGES[stage_idx]
    target = reward_target(lineage, stage_idx, state)
    delta = target - state.eef
    # Bounded proportional motion avoids the two-point oscillation that a
    # saturated tanh controller creates around tight stage tolerances.
    vel = np.clip(0.72 * spec.speed * delta, -MAX_SPEED, MAX_SPEED)
    if stage_idx in (1, 3):
        vel *= 0.70
    return clip_action(np.array([vel[0], vel[1], reward_grip_target(lineage, stage_idx)], dtype=float))


def expert_action(stage_idx: int, state: SimState, stage_start_eef: np.ndarray) -> np.ndarray:
    base = reward_action(Lineage("compiled_spec"), stage_idx, state)
    target = compiled_target(stage_idx, state)
    base_vec = target - stage_start_eef
    travel = max(1e-6, float(np.linalg.norm(base_vec)))
    progress = float(np.clip(1.0 - np.linalg.norm(target - state.eef) / travel, 0.0, 1.0))
    curve_dir = normalize(np.array([-(state.goal - state.obj)[1], (state.goal - state.obj)[0]], dtype=float))
    # Dampen the lateral shaping near the target so the scripted expert does
    # not orbit just outside a stage tolerance on unlucky geometries.
    target_error = float(np.linalg.norm(target - state.eef))
    curve_scale = 0.036 * math.sin(math.pi * progress) * min(1.0, target_error / 0.16)
    if stage_idx in (0, 2, 4):
        base[:2] += curve_scale * curve_dir
    if stage_idx == 1:
        base[:2] += 0.012 * normalize(state.obj - state.eef)
    if stage_idx == 3:
        base[:2] -= 0.010 * normalize(state.goal - state.eef)
    base[2] = STAGES[stage_idx].grip_target
    return clip_action(base)


def step_env(state: SimState, action: np.ndarray) -> SimState:
    act = clip_action(action)
    next_state = state.copy()
    next_state.eef = np.clip(state.eef + act[:2], -0.92, 0.92)
    next_state.gripper = float(np.clip(0.48 * state.gripper + 0.52 * act[2], -1.0, 1.0))

    if state.holding:
        next_state.obj = next_state.eef.copy()
        next_state.holding = True
        if next_state.gripper < -0.10:
            next_state.holding = False
            next_state.obj = next_state.eef.copy()
    else:
        next_state.holding = False
        next_state.obj = state.obj.copy()
        if state.gripper < 0.10 and next_state.gripper > 0.36 and np.linalg.norm(next_state.eef - state.obj) <= GRASP_CAPTURE_RADIUS:
            next_state.holding = True
            next_state.obj = next_state.eef.copy()

    next_state.delivered = bool(state.delivered or (not next_state.holding and np.linalg.norm(next_state.obj - next_state.goal) <= GOAL_SETTLE_RADIUS))
    return next_state


def compiled_completion(stage_idx: int, state: SimState) -> bool:
    spec = STAGES[stage_idx]
    if spec.name == "approach_object":
        return bool(np.linalg.norm(state.eef - state.obj) <= spec.tolerance and state.gripper < -0.25)
    if spec.name == "close_gripper":
        return bool(state.holding and np.linalg.norm(state.eef - state.obj) <= spec.tolerance)
    if spec.name == "carry_to_goal":
        return bool(state.holding and np.linalg.norm(state.obj - state.goal) <= spec.tolerance)
    if spec.name == "open_at_goal":
        return bool((not state.holding) and np.linalg.norm(state.obj - state.goal) <= spec.tolerance and state.gripper < -0.20)
    if spec.name == "retreat":
        return bool((not state.holding) and np.linalg.norm(state.obj - state.goal) <= 0.065 and np.linalg.norm(state.eef - state.retreat) <= spec.tolerance and state.gripper < -0.30)
    raise ValueError(stage_idx)


def guard_completion(lineage: Lineage, stage_idx: int, state: SimState) -> bool:
    if not lineage.drift_guard:
        return compiled_completion(stage_idx, state)
    spec = STAGES[stage_idx]
    if spec.name == "approach_object":
        return bool(np.linalg.norm(state.eef - state.obj) <= 0.074 and state.gripper < 0.05)
    if spec.name == "close_gripper":
        return bool(state.gripper > 0.32 and np.linalg.norm(state.eef - state.obj) <= 0.070)
    if spec.name == "carry_to_goal":
        return bool(np.linalg.norm(state.eef - state.goal) <= 0.088 and state.gripper > 0.20)
    if spec.name == "open_at_goal":
        return bool(state.gripper < -0.05 and np.linalg.norm(state.eef - state.goal) <= 0.092)
    if spec.name == "retreat":
        return bool(np.linalg.norm(state.eef - state.retreat) <= 0.103 and state.gripper < 0.00)
    raise ValueError(stage_idx)


def reward_completion(lineage: Lineage, stage_idx: int, state: SimState) -> bool:
    if not lineage.drift_reward:
        return compiled_completion(stage_idx, state)
    if stage_idx == 0:
        return bool(np.linalg.norm(state.eef - state.obj) <= 0.072)
    if stage_idx == 1:
        return bool(state.gripper > 0.30 and np.linalg.norm(state.eef - state.obj) <= 0.062)
    if stage_idx == 2:
        return bool(np.linalg.norm(state.eef - reward_target(lineage, stage_idx, state)) <= 0.086 and state.gripper > 0.15)
    if stage_idx == 3:
        return bool(state.gripper < -0.05 and np.linalg.norm(state.eef - reward_target(lineage, stage_idx, state)) <= 0.090)
    if stage_idx == 4:
        return bool(np.linalg.norm(state.eef - reward_target(lineage, stage_idx, state)) <= 0.102)
    raise ValueError(stage_idx)


def success_check(lineage: Lineage, states: list[SimState]) -> bool:
    last = states[-1]
    if not lineage.drift_success:
        return compiled_completion(stage_index("retreat"), last)
    return bool(
        (not last.holding)
        and np.linalg.norm(last.obj - last.goal) <= 0.110
        and np.linalg.norm(last.eef - last.retreat) <= 0.105
        and last.gripper < -0.05
    )


def compiled_success(states: list[SimState]) -> bool:
    return compiled_completion(stage_index("retreat"), states[-1])


def stage_dwell(lineage: Lineage, stage_idx: int) -> int:
    if lineage.drift_guard:
        return max(1, STAGES[stage_idx].dwell - (1 if stage_idx in (0, 2, 3, 4) else 0))
    return STAGES[stage_idx].dwell


def reward_step(lineage: Lineage, stage_idx: int, next_state: SimState) -> float:
    spec = STAGES[stage_idx]
    target = reward_target(lineage, stage_idx, next_state)
    primary = next_state.obj if spec.name == "carry_to_goal" and next_state.holding else next_state.eef
    dist = float(np.linalg.norm(primary - target))
    grip_err = abs(next_state.gripper - reward_grip_target(lineage, stage_idx))
    hold_penalty = 0.0
    if spec.require_holding is True and not next_state.holding:
        hold_penalty = 0.55
    if spec.require_holding is False and next_state.holding:
        hold_penalty = 0.20
    base = -spec.reward_weight * dist - 0.18 * grip_err - hold_penalty
    if spec.name in ("open_at_goal", "retreat"):
        base -= 0.65 * float(np.linalg.norm(next_state.obj - next_state.goal))
    if reward_completion(lineage, stage_idx, next_state):
        base += 0.90
    if next_state.delivered:
        base += 0.18
    return base


def boundary_shift_labels(labels: np.ndarray, cfg: Config) -> np.ndarray:
    out = labels.copy()
    boundaries = []
    for s in range(len(STAGES) - 1):
        hits = np.where((labels[:-1] == s) & (labels[1:] == s + 1))[0]
        boundaries.append(int(hits[0] + 1) if len(hits) else None)
    for idx, shift in enumerate(cfg.label_shift_steps):
        boundary = boundaries[idx]
        if boundary is None:
            continue
        lo = max(0, boundary - shift)
        out[lo:boundary] = idx + 1
    return out


def rollout_expert(cfg: Config, episode_id: str, seed: int) -> Episode:
    obj, goal, retreat = sample_geometry(seed)
    state = make_initial_state(stable_seed(seed, "init"), obj, goal, retreat)
    stage = 0
    dwell = 0
    stage_start = state.eef.copy()
    states: list[SimState] = []
    actions: list[np.ndarray] = []
    labels: list[int] = []
    for _ in range(cfg.max_steps):
        states.append(state.copy())
        labels.append(stage)
        act = expert_action(stage, state, stage_start)
        actions.append(act)
        next_state = step_env(state, act)
        if stage < len(STAGES) - 1:
            if compiled_completion(stage, next_state):
                dwell += 1
                if dwell >= STAGES[stage].dwell:
                    stage += 1
                    dwell = 0
                    stage_start = next_state.eef.copy()
            else:
                dwell = 0
        state = next_state
        if compiled_success([state]):
            states.append(state.copy())
            break
    return Episode(episode_id=episode_id, states=states, actions=np.asarray(actions, dtype=float), stage_labels=np.asarray(labels, dtype=int))


def build_dataset(cfg: Config, split: str, n_episodes: int) -> list[Episode]:
    return [
        rollout_expert(cfg, f"{split}_{i:03d}", stable_seed(cfg.seed, split, i))
        for i in range(n_episodes)
    ]


def fit_stagewise_policy(episodes: Iterable[Episode], lineage: Lineage, cfg: Config) -> np.ndarray:
    xs: list[list[np.ndarray]] = [[] for _ in STAGES]
    ys: list[list[np.ndarray]] = [[] for _ in STAGES]
    for ep in episodes:
        labels = boundary_shift_labels(ep.stage_labels, cfg) if lineage.drift_demo else ep.stage_labels
        for t, action in enumerate(ep.actions):
            stage = int(labels[t])
            xs[stage].append(features(ep.states[t]))
            ys[stage].append(action)
    weights = np.zeros((len(STAGES), len(features(episodes.__iter__().__next__().states[0])), 3), dtype=float)
    for stage in range(len(STAGES)):
        x = np.vstack(xs[stage])
        y = np.vstack(ys[stage])
        lhs = (x.T @ x) / len(x) + cfg.ridge_alpha * np.eye(x.shape[1])
        rhs = (x.T @ y) / len(x)
        weights[stage] = np.linalg.solve(lhs, rhs)
    return weights


def policy_action(weights: np.ndarray, lineage: Lineage, stage_idx: int, state: SimState, gain: float) -> np.ndarray:
    bc = features(state) @ weights[stage_idx]
    reward_ctrl = reward_action(lineage, stage_idx, state)
    motion = (1.0 - gain) * bc[:2] + gain * reward_ctrl[:2]
    grip = (1.0 - gain) * bc[2] + gain * reward_ctrl[2]
    return clip_action(np.array([motion[0], motion[1], grip], dtype=float))


def compiled_labels_for_states(states: list[SimState]) -> tuple[np.ndarray, list[int]]:
    stage = 0
    dwell = 0
    labels: list[int] = []
    transitions: list[int] = []
    for t, state in enumerate(states):
        labels.append(stage)
        if stage < len(STAGES) - 1:
            if compiled_completion(stage, state):
                dwell += 1
                if dwell >= STAGES[stage].dwell:
                    transitions.append(t)
                    stage += 1
                    dwell = 0
            else:
                dwell = 0
    while len(transitions) < len(STAGES) - 1:
        transitions.append(len(states) - 1)
    return np.asarray(labels, dtype=int), transitions


def rollout_policy(weights: np.ndarray, lineage: Lineage, cfg: Config, split: str, episode_id: str, seed: int, gain: float) -> RolloutResult:
    obj, goal, retreat = sample_geometry(seed)
    state = make_initial_state(stable_seed(seed, "init"), obj, goal, retreat)
    stage = 0
    dwell = 0
    states: list[SimState] = []
    actions: list[np.ndarray] = []
    executed_stages: list[int] = []
    lineage_return = 0.0
    compiled_return_total = 0.0
    transitions: list[int] = []

    for t in range(cfg.max_steps):
        states.append(state.copy())
        executed_stages.append(stage)
        act = policy_action(weights, lineage, stage, state, gain)
        actions.append(act)
        next_state = step_env(state, act)
        lineage_return += reward_step(lineage, stage, next_state)
        compiled_return_total += reward_step(Lineage("compiled_spec"), stage, next_state)
        if stage < len(STAGES) - 1:
            if guard_completion(lineage, stage, next_state):
                dwell += 1
                if dwell >= stage_dwell(lineage, stage):
                    transitions.append(t + 1)
                    stage += 1
                    dwell = 0
            else:
                dwell = 0
        state = next_state
    states.append(state.copy())
    compiled_labels, compiled_transitions = compiled_labels_for_states(states[:-1])
    exec_labels = np.asarray(executed_stages, dtype=int)
    stage_mismatch = float(np.mean(exec_labels != compiled_labels[: len(exec_labels)]))
    final_obj_error = float(np.linalg.norm(states[-1].obj - states[-1].goal))
    min_goal_error = float(min(np.linalg.norm(s.obj - s.goal) for s in states))
    while len(transitions) < len(STAGES) - 1:
        transitions.append(cfg.max_steps)
    return RolloutResult(
        lineage=lineage.key,
        split=split,
        episode_id=episode_id,
        reward_gain=float(gain),
        lineage_return=float(lineage_return),
        compiled_return=float(compiled_return_total),
        true_success=float(compiled_success(states)),
        reported_success=float(success_check(lineage, states)),
        stage_mismatch_rate=stage_mismatch,
        final_object_error=final_obj_error,
        min_goal_error=min_goal_error,
        steps=cfg.max_steps,
        transition_steps=compiled_transitions,
        reported_transition_steps=transitions,
        executed_stages=exec_labels,
        states=states,
        actions=np.asarray(actions, dtype=float),
    )


def summarize_rollouts(results: list[RolloutResult]) -> dict[str, float]:
    return {
        "episodes": len(results),
        "reward_gain": float(np.mean([r.reward_gain for r in results])),
        "true_success": float(np.mean([r.true_success for r in results])),
        "reported_success": float(np.mean([r.reported_success for r in results])),
        "overclaim": float(np.mean([r.reported_success - r.true_success for r in results])),
        "stage_mismatch_rate": float(np.mean([r.stage_mismatch_rate for r in results])),
        "compiled_return": float(np.mean([r.compiled_return for r in results])),
        "lineage_return": float(np.mean([r.lineage_return for r in results])),
        "final_object_error": float(np.mean([r.final_object_error for r in results])),
        "min_goal_error": float(np.mean([r.min_goal_error for r in results])),
    }


def compute_interface_metrics(train_eps: list[Episode], val_eps: list[Episode], lineage: Lineage, cfg: Config) -> dict[str, float]:
    label_errors = []
    for ep in train_eps:
        labels = boundary_shift_labels(ep.stage_labels, cfg) if lineage.drift_demo else ep.stage_labels
        label_errors.append(np.mean(labels != ep.stage_labels))

    guard_errors = []
    reward_drifts = []
    for ep in val_eps:
        stage = 0
        dwell = 0
        predicted_transitions: list[int] = []
        for t, state in enumerate(ep.states[:-1]):
            if stage < len(STAGES) - 1:
                if guard_completion(lineage, stage, state):
                    dwell += 1
                    if dwell >= stage_dwell(lineage, stage):
                        predicted_transitions.append(t)
                        stage += 1
                        dwell = 0
                else:
                    dwell = 0
            compiled_stage = int(ep.stage_labels[min(t, len(ep.stage_labels) - 1)])
            reward_drifts.append(float(np.linalg.norm(reward_target(lineage, compiled_stage, state) - compiled_target(compiled_stage, state))))
        compiled_transitions = []
        for s in range(len(STAGES) - 1):
            hits = np.where((ep.stage_labels[:-1] == s) & (ep.stage_labels[1:] == s + 1))[0]
            compiled_transitions.append(int(hits[0] + 1) if len(hits) else len(ep.stage_labels) - 1)
        while len(predicted_transitions) < len(STAGES) - 1:
            predicted_transitions.append(len(ep.stage_labels) - 1)
        guard_errors.extend(abs(a - b) for a, b in zip(predicted_transitions, compiled_transitions))

    crafted = [
        SimState(np.array([0.69, 0.47]), np.array([0.415, 0.245]), np.array([0.50, 0.30]), np.array([0.68, 0.46]), -0.9, False, False),
        SimState(np.array([0.70, 0.47]), np.array([0.46, 0.27]), np.array([0.50, 0.30]), np.array([0.68, 0.46]), -0.4, False, False),
        SimState(np.array([0.66, 0.45]), np.array([0.39, 0.23]), np.array([0.50, 0.30]), np.array([0.68, 0.46]), -0.8, False, False),
    ]
    false_positive = float(np.mean([success_check(lineage, [s]) and not compiled_success([s]) for s in crafted]))
    return {
        "label_error_rate": float(np.mean(label_errors)),
        "guard_transition_abs_error": float(np.mean(guard_errors)),
        "reward_target_drift": float(np.mean(reward_drifts)),
        "success_false_positive_rate": false_positive,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_headline(summary_rows: list[dict[str, object]], path: Path) -> None:
    keys = [row["lineage"] for row in summary_rows]
    labels = [DISPLAY[k] for k in keys]
    true_success = np.array([row["true_success"] for row in summary_rows], dtype=float)
    reported = np.array([row["reported_success"] for row in summary_rows], dtype=float)
    mismatch = np.array([row["stage_mismatch_rate"] for row in summary_rows], dtype=float)

    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    axes[0].bar(x, 100.0 * true_success, color=[COLORS[k] for k in keys])
    axes[0].set_title("True semantic success")
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylim(0, 105)

    width = 0.38
    axes[1].bar(x - width / 2, 100.0 * true_success, width=width, label="true", color="#4c78a8")
    axes[1].bar(x + width / 2, 100.0 * reported, width=width, label="reported", color="#f58518")
    axes[1].set_title("Reported vs. true success")
    axes[1].set_ylim(0, 115)
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].legend(frameon=False)

    axes[2].bar(x, 100.0 * mismatch, color=[COLORS[k] for k in keys])
    axes[2].set_title("Execution stage mismatch")
    axes[2].set_ylabel("Rate (%)")
    axes[2].set_ylim(0, 105)
    axes[2].set_xticks(x, labels, rotation=25, ha="right")

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_interface_drift(interface_rows: list[dict[str, object]], path: Path) -> None:
    keys = [row["lineage"] for row in interface_rows]
    labels = [DISPLAY[k] for k in keys]
    label_err = np.array([row["label_error_rate"] for row in interface_rows])
    guard_err = np.array([row["guard_transition_abs_error"] for row in interface_rows])
    reward_err = np.array([row["reward_target_drift"] for row in interface_rows])
    false_pos = np.array([row["success_false_positive_rate"] for row in interface_rows])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    metrics = [
        (label_err * 100.0, "Demo label drift (%)"),
        (guard_err, "Guard transition error (steps)"),
        (reward_err, "Reward-target drift"),
        (false_pos * 100.0, "Success false positives (%)"),
    ]
    for ax, (vals, title) in zip(axes.flat, metrics):
        ax.bar(np.arange(len(keys)), vals, color=[COLORS[k] for k in keys])
        ax.set_title(title)
        ax.set_xticks(np.arange(len(keys)), labels, rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_reward_selection(selection_rows: list[dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for key in ("compiled_spec", "copied_rewards", "copied_all"):
        subset = [row for row in selection_rows if row["lineage"] == key]
        gains = [row["reward_gain"] for row in subset]
        returns = [row["mean_validation_return"] for row in subset]
        ax.plot(gains, returns, marker="o", linewidth=2.0, color=COLORS[key], label=DISPLAY[key])
    ax.set_title("Validation model selection under each reward interface")
    ax.set_xlabel("Reward residual gain")
    ax.set_ylabel("Mean validation return")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(compiled_rollout: RolloutResult, copied_rollout: RolloutResult, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, rollout, title in zip(axes, [compiled_rollout, copied_rollout], ["Compiled spec", "Copied all"]):
        eef = np.array([s.eef for s in rollout.states])
        obj = np.array([s.obj for s in rollout.states])
        ax.plot(eef[:, 0], eef[:, 1], color="#1f77b4", linewidth=2.2, label="eef path")
        ax.plot(obj[:, 0], obj[:, 1], color="#d62728", linewidth=2.2, label="object path")
        ax.scatter(rollout.states[0].obj[0], rollout.states[0].obj[1], marker="o", s=90, color="#444444", label="object")
        ax.scatter(rollout.states[0].goal[0], rollout.states[0].goal[1], marker="*", s=180, color="#f2c14e", label="goal")
        ax.scatter(rollout.states[0].retreat[0], rollout.states[0].retreat[1], marker="X", s=110, color="#2a9d8f", label="retreat")
        ax.scatter(rollout.states[0].eef[0], rollout.states[0].eef[1], marker="s", s=80, color="#1f77b4", alpha=0.8)
        ax.set_title(f"{title}\ntrue={rollout.true_success:.0f}, reported={rollout.reported_success:.0f}")
        ax.set_xlim(-0.95, 0.95)
        ax.set_ylim(-0.92, 0.92)
        ax.set_aspect("equal")
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_report_tex(cfg: Config, summary_rows: list[dict[str, object]], interface_rows: list[dict[str, object]], output_dir: Path, sanity: dict[str, object]) -> None:
    docs_dir = output_dir.parent / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    by_key = {row["lineage"]: row for row in summary_rows}
    iface = {row["lineage"]: row for row in interface_rows}
    compiled = by_key["compiled_spec"]
    copied_all = by_key["copied_all"]
    copied_success = by_key["copied_success_check"]
    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{lmodern}}
\usepackage{{microtype}}
\usepackage{{booktabs}}
\usepackage{{amsmath,amssymb}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{enumitem}}
\hypersetup{{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black,citecolor=blue!50!black,hypertexnames=false}}
\title{{Semantic Program Lineage Toy\\\large Hand-Copied Interfaces vs. a Single Typed Specification}}
\author{{VLA Ideas}}
\date{{September 1, 2026}}
\begin{{document}}
\maketitle
\begin{{abstract}}
This report studies one narrow mechanism inspired by SUN: semantic interfaces that are defined once and compiled together versus hand-copied interfaces that drift apart. A deterministic pick-carry-release toy compiles stage guards, reward shaping, demonstration labels, and success checks from one typed task program, then compares that baseline against small copied inconsistencies in each interface. With the checked configuration, the compiled lineage attains {compiled['true_success']*100:.1f}\% true success with zero rollout overclaim, while the fully copied lineage drops to {copied_all['true_success']*100:.1f}\% true success and reaches a {copied_all['stage_mismatch_rate']*100:.1f}\% execution-stage mismatch rate. The copied success checker does not change nominal behavior, but a crafted near-miss probe exposes a {iface['copied_success_check']['success_false_positive_rate']*100:.1f}\% false-positive rate on the three diagnostic states. The package is a transparent lineage stress test, not a SUN, Kuafu, MPC, RL, or robot reproduction.
\end{{abstract}}

\section{{Source mapping}}
The SUN paper introduces typed executables whose semantics are written once and compiled into aligned MPC costs, satisfaction predicates, RL rewards, transition guards, and diagnostics. This toy isolates only that semantic-lineage idea. Instead of language parsing, MPC, or robot learning, it uses a scripted 2-D task and stage-conditioned linear controllers so that interface drift is easy to inspect.

\section{{Toy construction}}
The typed task program contains five stages: approach object, close gripper, carry to goal, open at goal, and retreat. Each stage specifies a target entity, tolerance, gripper target, holding requirement, dwell time, and reward weight. The compiled lineage derives all of the following from that one program:
\begin{{itemize}}[leftmargin=1.5em]
  \item demonstration stage labels for supervised training,
  \item execution-time transition guards,
  \item dense reward targets used to choose a residual gain on validation rollouts,
  \item terminal success checks and diagnostics.
\end{{itemize}}

The copied baselines manually perturb those interfaces with small mistakes: demonstration labels advance 1--2 frames early, guards use slightly looser thresholds and sometimes the wrong entity, reward targets are offset from the true goal/object relations, and the copied success checker allows a short drop that the compiled checker rejects.

\section{{Checked configuration}}
The full run used {cfg.train_episodes} training, {cfg.val_episodes} validation, and {cfg.test_episodes} test geometries; a horizon of {cfg.max_steps} steps; ridge-regularized stagewise regression; and reward residual gains in $\{{{', '.join(f'{g:.2f}' for g in cfg.reward_gains)}\}}$.

\begin{{table}}[H]
\centering
\begin{{tabular}}{{lrrrr}}
\toprule
Lineage & True success & Reported success & Overclaim & Stage mismatch \\
\midrule
Compiled spec & {compiled['true_success']*100:.1f}\% & {compiled['reported_success']*100:.1f}\% & {compiled['overclaim']*100:.1f} pp & {compiled['stage_mismatch_rate']*100:.1f}\% \\
Copied demo labels & {by_key['copied_demo_labels']['true_success']*100:.1f}\% & {by_key['copied_demo_labels']['reported_success']*100:.1f}\% & {by_key['copied_demo_labels']['overclaim']*100:.1f} pp & {by_key['copied_demo_labels']['stage_mismatch_rate']*100:.1f}\% \\
Copied stage guards & {by_key['copied_stage_guards']['true_success']*100:.1f}\% & {by_key['copied_stage_guards']['reported_success']*100:.1f}\% & {by_key['copied_stage_guards']['overclaim']*100:.1f} pp & {by_key['copied_stage_guards']['stage_mismatch_rate']*100:.1f}\% \\
Copied rewards & {by_key['copied_rewards']['true_success']*100:.1f}\% & {by_key['copied_rewards']['reported_success']*100:.1f}\% & {by_key['copied_rewards']['overclaim']*100:.1f} pp & {by_key['copied_rewards']['stage_mismatch_rate']*100:.1f}\% \\
Copied success check & {by_key['copied_success_check']['true_success']*100:.1f}\% & {by_key['copied_success_check']['reported_success']*100:.1f}\% & {by_key['copied_success_check']['overclaim']*100:.1f} pp & {by_key['copied_success_check']['stage_mismatch_rate']*100:.1f}\% \\
Copied all & {copied_all['true_success']*100:.1f}\% & {copied_all['reported_success']*100:.1f}\% & {copied_all['overclaim']*100:.1f} pp & {copied_all['stage_mismatch_rate']*100:.1f}\% \\
\bottomrule
\end{{tabular}}
\caption{{Aggregate test metrics. Overclaim is reported success minus compiled semantic success.}}
\end{{table}}

\begin{{table}}[H]
\centering
\begin{{tabular}}{{lrrrr}}
\toprule
Lineage & Label drift & Guard error & Reward drift & Success false positives \\
\midrule
Compiled spec & {iface['compiled_spec']['label_error_rate']*100:.1f}\% & {iface['compiled_spec']['guard_transition_abs_error']:.2f} & {iface['compiled_spec']['reward_target_drift']:.3f} & {iface['compiled_spec']['success_false_positive_rate']*100:.1f}\% \\
Copied demo labels & {iface['copied_demo_labels']['label_error_rate']*100:.1f}\% & {iface['copied_demo_labels']['guard_transition_abs_error']:.2f} & {iface['copied_demo_labels']['reward_target_drift']:.3f} & {iface['copied_demo_labels']['success_false_positive_rate']*100:.1f}\% \\
Copied stage guards & {iface['copied_stage_guards']['label_error_rate']*100:.1f}\% & {iface['copied_stage_guards']['guard_transition_abs_error']:.2f} & {iface['copied_stage_guards']['reward_target_drift']:.3f} & {iface['copied_stage_guards']['success_false_positive_rate']*100:.1f}\% \\
Copied rewards & {iface['copied_rewards']['label_error_rate']*100:.1f}\% & {iface['copied_rewards']['guard_transition_abs_error']:.2f} & {iface['copied_rewards']['reward_target_drift']:.3f} & {iface['copied_rewards']['success_false_positive_rate']*100:.1f}\% \\
Copied success check & {iface['copied_success_check']['label_error_rate']*100:.1f}\% & {iface['copied_success_check']['guard_transition_abs_error']:.2f} & {iface['copied_success_check']['reward_target_drift']:.3f} & {iface['copied_success_check']['success_false_positive_rate']*100:.1f}\% \\
Copied all & {iface['copied_all']['label_error_rate']*100:.1f}\% & {iface['copied_all']['guard_transition_abs_error']:.2f} & {iface['copied_all']['reward_target_drift']:.3f} & {iface['copied_all']['success_false_positive_rate']*100:.1f}\% \\
\bottomrule
\end{{tabular}}
\caption{{Direct interface-drift diagnostics computed against the compiled source program.}}
\end{{table}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=\textwidth]{{../outputs/headline_metrics.png}}
  \caption{{True success, reported-vs-true success, and stage mismatch by lineage.}}
\end{{figure}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=\textwidth]{{../outputs/interface_drift.png}}
  \caption{{How much each copied interface departs from the compiled source program.}}
\end{{figure}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.82\textwidth]{{../outputs/reward_selection.png}}
  \caption{{Validation model selection under compiled and copied reward definitions.}}
\end{{figure}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=\textwidth]{{../outputs/trajectory_examples.png}}
  \caption{{Representative compiled-spec and copied-all rollouts on the same held-out geometry.}}
\end{{figure}}

\section{{Interpretation}}
Three bounded lessons appear. First, semantic lineage matters even in a toy: the copied-all pipeline loses true success relative to the compiled lineage because training labels, runtime guards, and reward-guided corrections no longer describe the same task. Second, evaluation drift can exist even when nominal rollouts do not expose it: the copied success checker leaves the controller unchanged and matches nominal rollout outcomes, yet accepts one of three crafted near-miss terminal states that the compiled checker rejects. Third, guard and reward drift interact. Once the policy switches stages early, an offset copied reward further reinforces the wrong behavior.

\section{{Sanity checks}}
The checked run passed all built-in assertions, including a successful compiled expert rollout, nonzero copied-interface drift, and a crafted short-drop false positive for the copied success checker.

\section{{Limitations}}
This experiment has no language parser, typed planner, MPC, reinforcement learning, diffusion policy, vision encoder, domain randomization, or robot embodiment. Reward shaping is only a validation-time residual selector, not policy optimization. Demonstrations are scripted and deterministic. The copied bugs are hand-designed and observable. Results should therefore be read as mechanism evidence for semantic-lineage preservation, not as evidence about SUN's benchmark numbers or deployment performance.

\end{{document}}
"""
    (docs_dir / "semantic_program_lineage_report.tex").write_text(tex, encoding="utf-8")


def run_experiment(cfg: Config, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir.parent / "docs").mkdir(parents=True, exist_ok=True)
    train_eps = build_dataset(cfg, "train", cfg.train_episodes)
    val_eps = build_dataset(cfg, "val", cfg.val_episodes)
    test_eps = build_dataset(cfg, "test", cfg.test_episodes)

    sanity = {
        "compiled_expert_success": all(compiled_success(ep.states) for ep in (train_eps[:3] + val_eps[:3] + test_eps[:3])),
        "copied_label_shift_nonzero": float(np.mean([np.mean(boundary_shift_labels(ep.stage_labels, cfg) != ep.stage_labels) for ep in train_eps])) > 0.02,
        "copied_success_false_positive_exists": bool(success_check(Lineage("copied_success_check", drift_success=True), [SimState(np.array([0.69, 0.47]), np.array([0.415, 0.245]), np.array([0.50, 0.30]), np.array([0.68, 0.46]), -0.6, False, False)]) and not compiled_success([SimState(np.array([0.69, 0.47]), np.array([0.415, 0.245]), np.array([0.50, 0.30]), np.array([0.68, 0.46]), -0.6, False, False)])),
    }
    assert all(sanity.values()), sanity

    summary_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    interface_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    metrics_payload: dict[str, object] = {
        "config": asdict(cfg),
        "stages": [asdict(stage) for stage in STAGES],
        "lineages": [asdict(lineage) for lineage in LINEAGES],
        "source_mapping": {
            "paper": "Persistent Programs For Language-Grounded Control-to-Learning-to-Real Policies",
            "arxiv": "https://arxiv.org/abs/2608.31167",
            "scope": "typed semantics compiled once into aligned interfaces",
        },
        "sanity_check": sanity,
        "summaries": {},
        "interfaces": {},
    }

    selected_rollouts: dict[str, list[RolloutResult]] = {}
    representative: dict[str, RolloutResult] = {}

    for lineage in LINEAGES:
        weights = fit_stagewise_policy(train_eps, lineage, cfg)
        interface_metrics = compute_interface_metrics(train_eps, val_eps, lineage, cfg)
        interface_row = {"lineage": lineage.key, **interface_metrics}
        interface_rows.append(interface_row)
        metrics_payload["interfaces"][lineage.key] = interface_metrics

        best_gain = None
        best_return = None
        for gain in cfg.reward_gains:
            val_rollouts = [
                rollout_policy(weights, lineage, cfg, "val", ep.episode_id, stable_seed(cfg.seed, "val", idx), gain)
                for idx, ep in enumerate(val_eps)
            ]
            mean_return = float(np.mean([r.lineage_return for r in val_rollouts]))
            selection_rows.append(
                {
                    "lineage": lineage.key,
                    "reward_gain": gain,
                    "mean_validation_return": mean_return,
                    "mean_true_success": float(np.mean([r.true_success for r in val_rollouts])),
                }
            )
            if best_return is None or mean_return > best_return + 1e-12 or (abs(mean_return - best_return) <= 1e-12 and gain < best_gain):
                best_return = mean_return
                best_gain = gain
        assert best_gain is not None

        test_rollouts = [
            rollout_policy(weights, lineage, cfg, "test", ep.episode_id, stable_seed(cfg.seed, "test", idx), best_gain)
            for idx, ep in enumerate(test_eps)
        ]
        selected_rollouts[lineage.key] = test_rollouts
        representative[lineage.key] = min(test_rollouts, key=lambda r: (r.true_success, r.reported_success, r.final_object_error))

        summary = summarize_rollouts(test_rollouts)
        summary_row = {"lineage": lineage.key, **summary, **interface_metrics}
        summary_rows.append(summary_row)
        metrics_payload["summaries"][lineage.key] = summary_row

        for rollout in test_rollouts:
            episode_rows.append(
                {
                    "lineage": rollout.lineage,
                    "episode_id": rollout.episode_id,
                    "reward_gain": rollout.reward_gain,
                    "true_success": rollout.true_success,
                    "reported_success": rollout.reported_success,
                    "stage_mismatch_rate": rollout.stage_mismatch_rate,
                    "compiled_return": rollout.compiled_return,
                    "lineage_return": rollout.lineage_return,
                    "final_object_error": rollout.final_object_error,
                    "min_goal_error": rollout.min_goal_error,
                    "transition_approach": rollout.reported_transition_steps[0],
                    "transition_grasp": rollout.reported_transition_steps[1],
                    "transition_carry": rollout.reported_transition_steps[2],
                    "transition_open": rollout.reported_transition_steps[3],
                }
            )

    summary_rows = sorted(summary_rows, key=lambda row: list(DISPLAY).index(row["lineage"]))
    interface_rows = sorted(interface_rows, key=lambda row: list(DISPLAY).index(row["lineage"]))
    selection_rows = sorted(selection_rows, key=lambda row: (list(DISPLAY).index(row["lineage"]), row["reward_gain"]))

    write_csv(output_dir / "summary_metrics.csv", summary_rows)
    write_csv(output_dir / "episode_metrics.csv", episode_rows)
    write_csv(output_dir / "interface_consistency.csv", interface_rows)
    write_csv(output_dir / "selection_metrics.csv", selection_rows)
    (output_dir / "sanity_check.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")

    plot_headline(summary_rows, output_dir / "headline_metrics.png")
    plot_interface_drift(interface_rows, output_dir / "interface_drift.png")
    plot_reward_selection(selection_rows, output_dir / "reward_selection.png")
    plot_trajectories(representative["compiled_spec"], representative["copied_all"], output_dir / "trajectory_examples.png")

    render_report_tex(cfg, summary_rows, interface_rows, output_dir, sanity)
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--train-episodes", type=int, default=Config.train_episodes)
    parser.add_argument("--val-episodes", type=int, default=Config.val_episodes)
    parser.add_argument("--test-episodes", type=int, default=Config.test_episodes)
    parser.add_argument("--max-steps", type=int, default=Config.max_steps)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true", help="Run a smaller quick check")
    args = parser.parse_args()

    cfg = Config(
        seed=args.seed,
        train_episodes=12 if args.smoke else args.train_episodes,
        val_episodes=8 if args.smoke else args.val_episodes,
        test_episodes=10 if args.smoke else args.test_episodes,
        max_steps=32 if args.smoke else args.max_steps,
    )
    run_experiment(cfg, args.output_dir)


if __name__ == "__main__":
    main()
