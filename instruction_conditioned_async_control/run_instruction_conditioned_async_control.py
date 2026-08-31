#!/usr/bin/env python3
"""Deterministic planner/control-decoupling mechanism toy.

Inspired by Instruct-to-Act (Tang et al., COLM 2026), but deliberately not a
Dreamer, VLM, VLA, or paper reproduction. A sparse configured-latency planner
issues text station instructions while a transparent ridge-regression
controller acts at every environment tick in a partially observed sequential
navigation task.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import warnings
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable

BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_DOCS_DIR = BASE_DIR / "docs"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = [
    "controller_only",
    "direct_blocking",
    "sync_instruction",
    "async_online",
    "oracle",
]
LABELS = {
    "controller_only": "Controller only",
    "direct_blocking": "Direct planner actions",
    "sync_instruction": "Synchronous instructions",
    "async_online": "Asynchronous online",
    "oracle": "Oracle planner/controller",
}
COLORS = {
    "controller_only": "#9c755f",
    "direct_blocking": "#e45756",
    "sync_instruction": "#4c78a8",
    "async_online": "#54a24b",
    "oracle": "#7a5195",
}
STATIONS = {
    "amber": np.array([-0.82, 0.72], dtype=np.float64),
    "cobalt": np.array([0.78, 0.70], dtype=np.float64),
    "mint": np.array([0.80, -0.72], dtype=np.float64),
    "depot": np.array([-0.80, -0.74], dtype=np.float64),
}
STATION_ORDER = list(STATIONS)
ROUTES = [
    ("amber", "cobalt", "depot"),
    ("cobalt", "mint", "depot"),
    ("mint", "amber", "depot"),
    ("amber", "mint", "cobalt", "depot"),
    ("cobalt", "amber", "mint", "depot"),
    ("mint", "cobalt", "amber", "depot"),
]
DIALECTS = {
    "canonical": {
        "amber": "go to the amber station",
        "cobalt": "go to the cobalt station",
        "mint": "go to the mint station",
        "depot": "deliver at the depot",
    },
    "gemma": {
        "amber": "head toward the amber beacon",
        "cobalt": "approach the cobalt beacon",
        "mint": "move to the mint terminal",
        "depot": "take the payload to the depot",
    },
    "qwen": {
        "amber": "navigate to orange amber",
        "cobalt": "travel to blue cobalt",
        "mint": "proceed to green mint",
        "depot": "finish at the delivery bay",
    },
    "gpt": {
        "amber": "visit the warm amber marker",
        "cobalt": "reach the cool cobalt marker",
        "mint": "seek the mint-green marker",
        "depot": "return to the dropoff depot",
    },
    "heldout": {
        "amber": "steer toward the amber-colored waypoint",
        "cobalt": "make for the cobalt-blue waypoint",
        "mint": "continue toward the mint waypoint",
        "depot": "complete the job at the depot",
    },
}
ALIASES = {
    "amber": ("amber", "orange", "warm"),
    "cobalt": ("cobalt", "blue", "cool"),
    "mint": ("mint", "green"),
    "depot": ("depot", "delivery", "dropoff"),
}


@dataclass(frozen=True)
class Config:
    seed: int = 41
    trials: int = 64
    episode_steps: int = 260
    dt: float = 0.08
    max_accel: float = 3.4
    max_speed: float = 1.35
    damping: float = 0.46
    station_radius: float = 0.135
    sensor_radius: float = 0.58
    hazard_radius: float = 0.145
    planner_latency: int = 10
    planner_cadence: int = 16
    transport_delay: int = 4
    direct_chunk: int = 14
    direct_action_error: float = 0.85
    direct_target_error_bonus: float = 0.18
    planner_error_rate: float = 0.055
    controller_noise: float = 0.055
    wall_limit: float = 1.05
    train_episodes: int = 150
    ridge: float = 2e-3


@dataclass(frozen=True)
class PlannerProfile:
    name: str
    dialect: str
    latency: int
    error_rate: float
    description: str


PLANNER_PROFILES = {
    "balanced": PlannerProfile("balanced", "qwen", 10, 0.055, "moderate latency and error"),
    "strong_slow": PlannerProfile("strong_slow", "gpt", 18, 0.015, "more accurate but slower"),
    "fast_terse": PlannerProfile("fast_terse", "canonical", 5, 0.12, "fast but more error-prone"),
    "heldout_synonyms": PlannerProfile("heldout_synonyms", "heldout", 10, 0.055, "unseen phrasing with recognizable semantics"),
    "cipher_interface": PlannerProfile("cipher_interface", "cipher", 8, 0.02, "interface-breaking opaque labels"),
}


@dataclass
class Trace:
    route: tuple[str, ...]
    start: np.ndarray
    hazard_phase: np.ndarray
    wind: np.ndarray


@dataclass
class EnvState:
    pos: np.ndarray
    vel: np.ndarray
    stage: int
    step: int = 0
    safety_violations: int = 0


@dataclass
class LearnedController:
    capacity: str
    weights: np.ndarray
    feature_names: list[str]
    train_mse: float
    test_mse: float
    samples: int


@dataclass
class PlannerResult:
    intended_stage: int
    target: str
    text: str
    query_step: int
    ready_step: int
    call_index: int


@dataclass
class PendingPlan:
    result: PlannerResult
    remaining: int


@dataclass
class RolloutResult:
    metrics: dict[str, Any]
    trace: dict[str, Any] | None = None


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def clip_norm(vec: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec if norm <= limit or norm == 0 else vec * (limit / norm)


def make_trace(cfg: Config, trial: int) -> Trace:
    rng = np.random.default_rng(stable_seed(cfg.seed, trial, "environment"))
    route = ROUTES[int(rng.integers(0, len(ROUTES)))]
    start = rng.uniform(-0.22, 0.22, size=2)
    hazard_phase = rng.uniform(-math.pi, math.pi, size=6)
    wind = rng.normal(0.0, 0.055, size=(cfg.episode_steps + 64, 2))
    for k in range(1, len(wind)):
        wind[k] = 0.82 * wind[k - 1] + 0.18 * wind[k]
    return Trace(route=route, start=start, hazard_phase=hazard_phase, wind=wind)


def hazard_positions(step: int, trace: Trace, cfg: Config) -> np.ndarray:
    t = step * cfg.dt
    p = trace.hazard_phase
    return np.array(
        [
            [0.10 + 0.48 * math.sin(0.72 * t + p[0]), 0.10 + 0.23 * math.sin(1.11 * t + p[1])],
            [-0.12 + 0.28 * math.sin(0.93 * t + p[2]), -0.12 + 0.49 * math.sin(0.61 * t + p[3])],
            [0.50 * math.sin(0.43 * t + p[4]), 0.48 * math.cos(0.47 * t + p[5])],
        ],
        dtype=np.float64,
    )


def local_avoidance(pos: np.ndarray, hazards: np.ndarray, cfg: Config) -> tuple[np.ndarray, float]:
    avoid = np.zeros(2, dtype=np.float64)
    nearest = 99.0
    for h in hazards:
        delta = pos - h
        dist = float(np.linalg.norm(delta))
        nearest = min(nearest, dist)
        if 1e-8 < dist < cfg.sensor_radius:
            strength = (1.0 / max(dist, 0.08) - 1.0 / cfg.sensor_radius) / max(dist, 0.08)
            avoid += 0.10 * strength * delta
    return clip_norm(avoid, 2.4), nearest


def expert_action(pos: np.ndarray, vel: np.ndarray, target: str, hazards: np.ndarray, cfg: Config) -> np.ndarray:
    delta = STATIONS[target] - pos
    avoid, _ = local_avoidance(pos, hazards, cfg)
    action = 3.05 * delta - 1.62 * vel + 1.05 * avoid
    if np.linalg.norm(delta) < 0.25:
        action -= 0.65 * vel
    return clip_norm(action, cfg.max_accel)


def parse_instruction(text: str | None) -> str | None:
    if not text:
        return None
    lower = text.lower()
    for station, aliases in ALIASES.items():
        if any(alias in lower for alias in aliases):
            return station
    return None


def feature_vector(
    pos: np.ndarray,
    vel: np.ndarray,
    hazards: np.ndarray,
    instruction: str,
    capacity: str,
    cfg: Config,
) -> tuple[np.ndarray, list[str]]:
    target = parse_instruction(instruction)
    delta = STATIONS[target] - pos if target in STATIONS else np.zeros(2)
    avoid, nearest = local_avoidance(pos, hazards, cfg)
    distance = float(np.linalg.norm(delta))
    base = [1.0, delta[0], delta[1], vel[0], vel[1]]
    names = ["bias", "target_dx", "target_dy", "vel_x", "vel_y"]
    if capacity in ("medium", "large"):
        base.extend([avoid[0], avoid[1], distance, delta[0] * distance, delta[1] * distance])
        names.extend(["avoid_x", "avoid_y", "target_distance", "dx_distance", "dy_distance"])
    if capacity == "large":
        base.extend([vel[0] * distance, vel[1] * distance, 1.0 / max(nearest, 0.08) if nearest < cfg.sensor_radius else 0.0])
        names.extend(["vx_distance", "vy_distance", "inverse_near_hazard"])
    return np.asarray(base, dtype=np.float64), names


def demo_segments(cfg: Config) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    dialects = ["canonical", "gemma", "qwen", "gpt"]
    for episode in range(cfg.train_episodes):
        tr = make_trace(replace(cfg, seed=cfg.seed + 777), episode)
        state = EnvState(pos=tr.start.copy(), vel=np.zeros(2), stage=0)
        seg_start = 0
        segment_rows: list[dict[str, Any]] = []
        seg_actions: list[np.ndarray] = []
        dialect = dialects[episode % len(dialects)]
        max_demo_steps = min(cfg.episode_steps, 220)
        while state.step < max_demo_steps and state.stage < len(tr.route):
            target = tr.route[state.stage]
            hazards = hazard_positions(state.step, tr, cfg)
            action = expert_action(state.pos, state.vel, target, hazards, cfg)
            # Collect an autonomous segment first.  The instruction is attached
            # only after the segment boundary is observed, matching the claimed
            # post-hoc relabeling order rather than leaking it into collection.
            segment_rows.append(
                {
                    "episode": episode,
                    "stage": state.stage,
                    "target": target,
                    "pos": state.pos.copy(),
                    "vel": state.vel.copy(),
                    "hazards": hazards.copy(),
                    "action": action.copy(),
                }
            )
            seg_actions.append(action.copy())
            old_stage = state.stage
            step_environment(state, action, tr, cfg)
            if state.stage != old_stage:
                instruction = DIALECTS[dialect][target]
                for row in segment_rows:
                    row["instruction"] = instruction
                rows.extend(segment_rows)
                segments.append(
                    {
                        "episode": episode,
                        "stage": old_stage,
                        "target": target,
                        "dialect": dialect,
                        "instruction": instruction,
                        "start_step": seg_start,
                        "end_step": state.step - 1,
                        "length": len(seg_actions),
                        "mean_ax": float(np.mean([a[0] for a in seg_actions])),
                        "mean_ay": float(np.mean([a[1] for a in seg_actions])),
                    }
                )
                seg_start = state.step
                segment_rows = []
                seg_actions = []
    return rows, segments


def train_controllers(cfg: Config) -> tuple[dict[str, LearnedController], list[dict[str, Any]]]:
    rows, segments = demo_segments(cfg)
    rng = np.random.default_rng(stable_seed(cfg.seed, "train_split"))
    order = rng.permutation(len(rows))
    split = int(0.82 * len(rows))
    train_idx, test_idx = order[:split], order[split:]
    sample_limits = {"small": min(900, len(train_idx)), "medium": min(4200, len(train_idx)), "large": len(train_idx)}
    models: dict[str, LearnedController] = {}
    for capacity in ("small", "medium", "large"):
        selected = train_idx[: sample_limits[capacity]]
        x_list, y_list = [], []
        names: list[str] = []
        for idx in selected:
            row = rows[int(idx)]
            feat, names = feature_vector(row["pos"], row["vel"], row["hazards"], row["instruction"], capacity, cfg)
            x_list.append(feat)
            y_list.append(row["action"])
        x = np.asarray(x_list)
        y = np.asarray(y_list)
        reg = cfg.ridge * np.eye(x.shape[1])
        reg[0, 0] = cfg.ridge * 0.1
        weights = np.linalg.solve(x.T @ x + reg, x.T @ y)
        train_pred = x @ weights
        test_x, test_y = [], []
        for idx in test_idx:
            row = rows[int(idx)]
            feat, _ = feature_vector(row["pos"], row["vel"], row["hazards"], row["instruction"], capacity, cfg)
            test_x.append(feat)
            test_y.append(row["action"])
        test_xa = np.asarray(test_x)
        test_ya = np.asarray(test_y)
        models[capacity] = LearnedController(
            capacity=capacity,
            weights=weights,
            feature_names=names,
            train_mse=float(np.mean((train_pred - y) ** 2)),
            test_mse=float(np.mean((test_xa @ weights - test_ya) ** 2)),
            samples=len(selected),
        )
    return models, segments


def learned_action(
    model: LearnedController,
    state: EnvState,
    instruction: str,
    hazards: np.ndarray,
    cfg: Config,
    noise_std: float,
    noise_key: tuple[Any, ...],
) -> np.ndarray:
    feat, _ = feature_vector(state.pos, state.vel, hazards, instruction, model.capacity, cfg)
    action = feat @ model.weights
    if noise_std > 0:
        rng = np.random.default_rng(stable_seed(*noise_key))
        action = action + rng.normal(0.0, noise_std, size=2)
    return clip_norm(np.asarray(action, dtype=np.float64), cfg.max_accel)


def step_environment(state: EnvState, action: np.ndarray, trace: Trace, cfg: Config) -> dict[str, Any]:
    action = clip_norm(action, cfg.max_accel)
    hazards = hazard_positions(state.step, trace, cfg)
    accel = action - cfg.damping * state.vel + trace.wind[min(state.step, len(trace.wind) - 1)]
    state.vel = clip_norm(state.vel + cfg.dt * accel, cfg.max_speed)
    state.pos = state.pos + cfg.dt * state.vel
    wall_hit = bool(np.any(np.abs(state.pos) > cfg.wall_limit))
    if wall_hit:
        state.pos = np.clip(state.pos, -cfg.wall_limit, cfg.wall_limit)
        state.vel *= 0.25
    collision = any(float(np.linalg.norm(state.pos - h)) < cfg.hazard_radius for h in hazards)
    if collision:
        state.safety_violations += 1
        nearest = min(hazards, key=lambda h: float(np.linalg.norm(state.pos - h)))
        push = state.pos - nearest
        if np.linalg.norm(push) < 1e-8:
            push = np.array([1.0, 0.0])
        state.pos = nearest + clip_norm(push, cfg.hazard_radius + 0.015)
        state.vel *= 0.35
    if wall_hit:
        state.safety_violations += 1
    completed = False
    if state.stage < len(trace.route):
        target = trace.route[state.stage]
        if float(np.linalg.norm(state.pos - STATIONS[target])) <= cfg.station_radius and float(np.linalg.norm(state.vel)) <= 0.78:
            state.stage += 1
            completed = True
    state.step += 1
    return {"collision": collision, "wall_hit": wall_hit, "completed_stage": completed}


def planner_instruction(profile: PlannerProfile, target: str, call_index: int) -> str:
    if profile.dialect == "cipher":
        return f"execute token {chr(65 + (call_index % 20))}{(call_index * 7) % 97:02d}"
    return DIALECTS[profile.dialect][target]


def make_plan_result(
    trace: Trace,
    intended_stage: int,
    profile: PlannerProfile,
    query_step: int,
    ready_step: int,
    call_index: int,
    seed_key: tuple[Any, ...],
) -> PlannerResult:
    intended_stage = max(0, min(intended_stage, len(trace.route) - 1))
    target = trace.route[intended_stage]
    rng = np.random.default_rng(stable_seed(*seed_key, call_index, intended_stage, "planner"))
    if rng.random() < profile.error_rate:
        target = STATION_ORDER[int(rng.integers(0, len(STATION_ORDER)))]
        if target == trace.route[intended_stage]:
            target = STATION_ORDER[(STATION_ORDER.index(target) + 1) % len(STATION_ORDER)]
    text = planner_instruction(profile, target, call_index)
    return PlannerResult(intended_stage, target, text, query_step, ready_step, call_index)


def direct_open_loop_chunk(state: EnvState, target: str, trace: Trace, cfg: Config) -> list[np.ndarray]:
    pos = state.pos.copy()
    vel = state.vel.copy()
    snapshot_hazards = hazard_positions(state.step, trace, cfg)
    out: list[np.ndarray] = []
    for k in range(cfg.direct_chunk):
        delta = STATIONS[target] - pos
        avoid, _ = local_avoidance(pos, snapshot_hazards, cfg)
        action = 2.75 * delta - 1.10 * vel + 0.55 * avoid
        # Deterministic low-level token/precision error: the slow high-level
        # planner is a poor high-rate actuator even when its subgoal is sound.
        phase = 1.17 * (state.step + k) + 0.31 * state.stage
        action = action + cfg.direct_action_error * np.array([math.sin(phase), math.cos(1.43 * phase)])
        action = clip_norm(action, cfg.max_accel)
        out.append(action)
        accel = action - cfg.damping * vel
        vel = clip_norm(vel + cfg.dt * accel, cfg.max_speed)
        pos = pos + cfg.dt * vel
    return out


def run_rollout(
    method: str,
    cfg: Config,
    trial: int,
    models: dict[str, LearnedController],
    *,
    profile: PlannerProfile | None = None,
    controller_capacity: str = "medium",
    controller_noise: float | None = None,
    planner_latency: int | None = None,
    planner_cadence: int | None = None,
    transport_delay: int | None = None,
    record: bool = False,
) -> RolloutResult:
    trace = make_trace(cfg, trial)
    state = EnvState(pos=trace.start.copy(), vel=np.zeros(2), stage=0)
    profile = profile or PLANNER_PROFILES["balanced"]
    latency = profile.latency if planner_latency is None else planner_latency
    cadence = cfg.planner_cadence if planner_cadence is None else planner_cadence
    extra_delay = cfg.transport_delay if transport_delay is None else transport_delay
    noise_std = cfg.controller_noise if controller_noise is None else controller_noise
    model = models[controller_capacity]
    call_index = 0
    idle_steps = 0
    stale_errors = 0
    wrong_instruction_errors = 0
    parse_failures = 0
    instruction_switches = 0
    path_length = 0.0
    stage_completions = 0
    active_instruction: PlannerResult | None = None
    pending: PendingPlan | None = None
    drafts: dict[int, PlannerResult] = {}
    direct_queue: list[np.ndarray] = []
    direct_target: str | None = None
    direct_intended_stage: int | None = None
    self_route = ROUTES[0]
    self_stage = 0
    last_action = np.zeros(2)
    action_changes: list[float] = []
    positions = [state.pos.copy()]
    actions: list[np.ndarray] = []
    stages = [state.stage]
    instr_targets: list[str | None] = []
    idle_flags: list[bool] = []
    next_normal_request_step = cadence

    def start_plan(intended_stage: int, plan_latency: int, add_transport: bool) -> PendingPlan:
        nonlocal call_index
        call_index += 1
        total = max(0, int(plan_latency)) + (max(0, int(extra_delay)) if add_transport else 0)
        plan_profile = profile
        if method == "direct_blocking":
            plan_profile = replace(profile, error_rate=min(0.45, profile.error_rate + cfg.direct_target_error_bonus))
        result = make_plan_result(
            trace,
            intended_stage,
            plan_profile,
            state.step,
            state.step + total,
            call_index,
            # Common random numbers: matched planner/controller methods see the
            # same planner-error draw for the same trial, call, and stage.  The
            # direct-action stress foil may still use a larger disclosed error
            # threshold, but it does not receive an unrelated random sequence.
            (cfg.seed, trial, "planner_common_random_numbers"),
        )
        return PendingPlan(result=result, remaining=total)

    # All non-controller-only systems pay or receive an initial plan. Async has
    # the same one-time cold start as synchronous operation.
    if method in ("sync_instruction", "async_online"):
        pending = start_plan(0, latency, False)
    elif method == "oracle":
        call_index = 1
        active_instruction = PlannerResult(0, trace.route[0], DIALECTS["canonical"][trace.route[0]], 0, 0, 1)

    while state.step < cfg.episode_steps and state.stage < len(trace.route):
        stage_before = state.stage
        action = np.zeros(2)
        idle = False
        active_target: str | None = None
        active_intended_stage: int | None = None

        # Advance planner wall-clock concurrently or while blocked.
        if pending is not None:
            if pending.remaining > 0:
                pending.remaining -= 1
            if pending.remaining <= 0:
                result = pending.result
                pending = None
                if method == "async_online" and result.intended_stage > state.stage:
                    drafts[result.intended_stage] = result
                else:
                    active_instruction = result
                    instruction_switches += 1

        hazards = hazard_positions(state.step, trace, cfg)
        if method == "controller_only":
            target = self_route[self_stage % len(self_route)]
            text = DIALECTS["canonical"][target]
            action = learned_action(
                model, state, text, hazards, cfg, noise_std,
                (cfg.seed, trial, "learned_controller", controller_capacity, state.step),
            )
            active_target = target
        elif method == "direct_blocking":
            if not direct_queue and pending is None and active_instruction is None:
                pending = start_plan(state.stage, latency, False)
            if pending is not None:
                idle = True
            elif not direct_queue and active_instruction is not None:
                direct_target = active_instruction.target
                direct_intended_stage = active_instruction.intended_stage
                direct_queue = direct_open_loop_chunk(state, direct_target, trace, cfg)
                active_instruction = None
            if direct_queue:
                action = direct_queue.pop(0)
                active_target = direct_target
                active_intended_stage = direct_intended_stage
        elif method == "sync_instruction":
            if active_instruction is None and pending is None:
                pending = start_plan(state.stage, latency, False)
            if active_instruction is None:
                idle = True
            else:
                active_target = parse_instruction(active_instruction.text)
                active_intended_stage = active_instruction.intended_stage
                if active_target is None:
                    parse_failures += 1
                action = learned_action(
                    model, state, active_instruction.text, hazards, cfg, noise_std,
                    (cfg.seed, trial, "learned_controller", controller_capacity, state.step),
                )
        elif method == "async_online":
            # If the current stage has a completed draft, atomically swap it in.
            if state.stage in drafts and (active_instruction is None or active_instruction.intended_stage != state.stage):
                active_instruction = drafts.pop(state.stage)
                instruction_switches += 1
            if active_instruction is None:
                idle = True
            else:
                active_target = parse_instruction(active_instruction.text)
                active_intended_stage = active_instruction.intended_stage
                if active_target is None:
                    parse_failures += 1
                action = learned_action(
                    model, state, active_instruction.text, hazards, cfg, noise_std,
                    (cfg.seed, trial, "learned_controller", controller_capacity, state.step),
                )
                # Sparse online lookahead: request exactly one draft for the next
                # stage once the cadence threshold is reached.
                next_stage = state.stage + 1
                if (
                    next_stage < len(trace.route)
                    and pending is None
                    and next_stage not in drafts
                    and state.step >= next_normal_request_step
                ):
                    pending = start_plan(next_stage, latency, True)
                    next_normal_request_step = state.step + cadence
        elif method == "oracle":
            target = trace.route[state.stage]
            active_target = target
            active_intended_stage = state.stage
            action = expert_action(state.pos, state.vel, target, hazards, cfg)
        else:
            raise ValueError(f"unknown method {method}")

        if idle:
            idle_steps += 1
            action = np.zeros(2)
        if state.stage < len(trace.route) and not idle:
            if active_target != trace.route[state.stage]:
                wrong_instruction_errors += 1
            if active_intended_stage is not None and active_intended_stage < state.stage:
                stale_errors += 1

        old_pos = state.pos.copy()
        event = step_environment(state, action, trace, cfg)
        path_length += float(np.linalg.norm(state.pos - old_pos))
        action_changes.append(float(np.linalg.norm(action - last_action)))
        last_action = action.copy()
        actions.append(action.copy())
        positions.append(state.pos.copy())
        idle_flags.append(idle)
        instr_targets.append(active_target)
        stages.append(state.stage)

        # Controller-only advances its internal route when it reaches its own target.
        if method == "controller_only" and float(np.linalg.norm(state.pos - STATIONS[self_route[self_stage % len(self_route)]])) <= cfg.station_radius:
            self_stage = (self_stage + 1) % len(self_route)
        if event["completed_stage"]:
            stage_completions += 1
            if method == "sync_instruction":
                active_instruction = None
            elif method == "async_online":
                if state.stage in drafts:
                    active_instruction = drafts.pop(state.stage)
                    instruction_switches += 1
                elif state.stage < len(trace.route):
                    # Keep executing the old instruction while an urgent current-
                    # stage plan arrives; this exposes staleness rather than idle.
                    if pending is None or pending.result.intended_stage != state.stage:
                        pending = start_plan(state.stage, latency, True)
                    next_normal_request_step = state.step + cadence
            elif method == "oracle" and state.stage < len(trace.route):
                call_index += 1
                active_instruction = PlannerResult(
                    state.stage,
                    trace.route[state.stage],
                    DIALECTS["canonical"][trace.route[state.stage]],
                    state.step,
                    state.step,
                    call_index,
                )
        if state.stage < stage_before:
            raise AssertionError("stage must be monotonic")

    success = state.stage >= len(trace.route)
    steps = state.step
    metrics = {
        "trial": trial,
        "method": method,
        "label": LABELS[method],
        "route": ">".join(trace.route),
        "success": int(success),
        "completion_steps": steps,
        "completion_time_s": steps * cfg.dt,
        "active_steps": steps - idle_steps,
        "idle_steps": idle_steps,
        "idle_fraction": idle_steps / max(steps, 1),
        "stale_instruction_errors": stale_errors,
        "stale_error_fraction": stale_errors / max(steps - idle_steps, 1),
        "wrong_instruction_errors": wrong_instruction_errors,
        "wrong_instruction_fraction": wrong_instruction_errors / max(steps - idle_steps, 1),
        "control_smoothness": float(np.sum(action_changes) / max(steps - idle_steps, 1)) if action_changes else 0.0,
        "safety_violations": state.safety_violations,
        "planner_calls": call_index,
        "instruction_switches": instruction_switches,
        "instruction_parse_failures": parse_failures,
        "stage_completions": stage_completions,
        "route_stages": len(trace.route),
        "path_length": path_length,
        "planner_profile": profile.name,
        "planner_latency": latency,
        "planner_cadence": cadence,
        "transport_delay": extra_delay,
        "controller_capacity": controller_capacity,
        "controller_noise": noise_std,
    }
    trace_out = None
    if record:
        trace_out = {
            "positions": np.asarray(positions),
            "actions": np.asarray(actions),
            "stages": np.asarray(stages),
            "instruction_targets": instr_targets,
            "idle": np.asarray(idle_flags),
            "route": trace.route,
            "hazards": np.asarray([hazard_positions(k, trace, cfg) for k in range(steps + 1)]),
        }
    return RolloutResult(metrics=metrics, trace=trace_out)


def summarize(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)
    metric_keys = [
        "success",
        "completion_steps",
        "completion_time_s",
        "idle_steps",
        "idle_fraction",
        "stale_instruction_errors",
        "stale_error_fraction",
        "wrong_instruction_errors",
        "wrong_instruction_fraction",
        "control_smoothness",
        "safety_violations",
        "planner_calls",
        "path_length",
    ]
    out: list[dict[str, Any]] = []
    for key, vals in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        row = {k: v for k, v in zip(group_keys, key)}
        row["n"] = len(vals)
        for metric in metric_keys:
            arr = np.asarray([float(v[metric]) for v in vals], dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(arr))
            row[f"{metric}_sem"] = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
        out.append(row)
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def method_summary_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["method"]: row for row in rows}


def plot_methods(summary_rows: list[dict[str, Any]], output: pathlib.Path) -> None:
    smap = method_summary_map(summary_rows)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.7))
    metrics = [
        ("success_mean", "Success", lambda x: 100 * x, "%"),
        ("completion_steps_mean", "Completion / horizon steps", lambda x: x, "steps"),
        ("idle_fraction_mean", "Idle fraction", lambda x: 100 * x, "%"),
        ("stale_instruction_errors_mean", "Stale-instruction ticks", lambda x: x, "ticks"),
        ("control_smoothness_mean", "Action change (lower smoother)", lambda x: x, "L2/tick"),
        ("safety_violations_mean", "Safety violations", lambda x: x, "count"),
    ]
    xs = np.arange(len(METHOD_ORDER))
    labels = ["Controller\nonly", "Direct\nblocking", "Sync\ninstructions", "Async\nonline", "Oracle"]
    for ax, (metric, title, transform, ylabel) in zip(axes.flat, metrics):
        vals = [transform(smap[m][metric]) for m in METHOD_ORDER]
        sem_metric = metric.replace("_mean", "_sem")
        errs = [transform(smap[m][sem_metric]) for m in METHOD_ORDER]
        ax.bar(xs, vals, yerr=errs, color=[COLORS[m] for m in METHOD_ORDER], capsize=3)
        ax.set_xticks(xs, labels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Planner/control decoupling: paired default evaluation", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sweeps(sweep_summary: list[dict[str, Any]], output: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.3))
    specs = [
        ("planner_latency", "latency", "Planner latency (ticks)"),
        ("planner_cadence", "cadence", "Planning cadence (ticks)"),
        ("transport_delay", "staleness", "Extra instruction delay (ticks)"),
        ("controller_noise", "capacity_noise", "Controller noise"),
        ("planner_profile", "planner_swap", "Planner swap"),
    ]
    for ax, (xkey, sweep, title) in zip(axes.flat[:5], specs):
        rows = [r for r in sweep_summary if r["sweep"] == sweep]
        if sweep == "latency":
            for method in ("direct_blocking", "sync_instruction", "async_online"):
                mr = sorted([r for r in rows if r["method"] == method], key=lambda r: float(r[xkey]))
                ax.plot([float(r[xkey]) for r in mr], [100 * r["success_mean"] for r in mr], marker="o", label=LABELS[method])
        elif sweep == "capacity_noise":
            cap_colors = {"small": "#e45756", "medium": "#4c78a8", "large": "#54a24b"}
            twin = ax.twinx()
            for cap in ("small", "medium", "large"):
                mr = sorted([r for r in rows if r["controller_capacity"] == cap], key=lambda r: float(r[xkey]))
                xs_cap = [float(r[xkey]) for r in mr]
                ax.plot(xs_cap, [100 * r["success_mean"] for r in mr], marker="o", color=cap_colors[cap], label=cap)
                twin.plot(xs_cap, [r["safety_violations_mean"] for r in mr], linestyle="--", alpha=0.55, color=cap_colors[cap])
            twin.set_ylabel("safety violations (dashed)")
        elif sweep == "planner_swap":
            rows = sorted(rows, key=lambda r: str(r[xkey]))
            ax.bar(np.arange(len(rows)), [100 * r["success_mean"] for r in rows], color="#54a24b")
            ax.set_xticks(np.arange(len(rows)), [str(r[xkey]).replace("_", "\n") for r in rows], fontsize=8)
        else:
            rows = sorted(rows, key=lambda r: float(r[xkey]))
            ax.plot([float(r[xkey]) for r in rows], [100 * r["success_mean"] for r in rows], marker="o", color="#54a24b")
            twin = ax.twinx()
            twin.plot([float(r[xkey]) for r in rows], [r["stale_instruction_errors_mean"] for r in rows], marker="s", color="#e45756", alpha=0.75)
            twin.set_ylabel("stale ticks", color="#e45756")
        ax.set_title(title)
        ax.set_ylabel("success (%)")
        ax.grid(alpha=0.25)
        if sweep in ("latency", "capacity_noise"):
            ax.legend(fontsize=8)
    ax = axes.flat[5]
    rows = [r for r in sweep_summary if r["sweep"] == "planner_swap"]
    rows = sorted(rows, key=lambda r: str(r["planner_profile"]))
    ax.bar(np.arange(len(rows)), [r["planner_calls_mean"] for r in rows], color="#4c78a8", label="planner calls")
    ax.plot(np.arange(len(rows)), [r["wrong_instruction_errors_mean"] for r in rows], color="#e45756", marker="o", label="wrong-target ticks")
    ax.set_xticks(np.arange(len(rows)), [str(r["planner_profile"]).replace("_", "\n") for r in rows], fontsize=8)
    ax.set_title("Planner interface/work trade-off")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Mechanism sweeps (paired deterministic trials)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_representative(traces: dict[str, dict[str, Any]], output: pathlib.Path, cfg: Config) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.0))
    for ax, method in zip(axes.flat[:5], METHOD_ORDER):
        tr = traces[method]
        pos = tr["positions"]
        ax.plot(pos[:, 0], pos[:, 1], color=COLORS[method], lw=1.8)
        ax.scatter(pos[0, 0], pos[0, 1], marker="o", color="black", s=25, label="start")
        for name, station in STATIONS.items():
            ax.scatter(station[0], station[1], marker="*", s=90)
            ax.text(station[0] + 0.03, station[1] + 0.03, name, fontsize=8)
        for h in tr["hazards"][:: max(1, len(tr["hazards"]) // 20), 0]:
            ax.add_patch(plt.Circle(h, cfg.hazard_radius, color="#e45756", alpha=0.035))
        ax.set_xlim(-1.12, 1.12)
        ax.set_ylim(-1.12, 1.12)
        ax.set_aspect("equal")
        ax.set_title(LABELS[method])
        ax.grid(alpha=0.18)
    ax = axes.flat[5]
    for method in METHOD_ORDER:
        tr = traces[method]
        acts = tr["actions"]
        if len(acts):
            ax.plot(np.linalg.norm(acts, axis=1), label=LABELS[method], color=COLORS[method], alpha=0.85)
    ax.set_title("Action magnitude")
    ax.set_xlabel("environment tick")
    ax.set_ylabel("acceleration norm")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.suptitle(f"Representative paired route: {' → '.join(traces['oracle']['route'])}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_").replace("%", "\\%")


def generate_report(
    path: pathlib.Path,
    cfg: Config,
    summary_rows: list[dict[str, Any]],
    sweep_summary: list[dict[str, Any]],
    training: dict[str, LearnedController],
    command: str,
) -> None:
    smap = method_summary_map(summary_rows)
    table_lines = []
    for method in METHOD_ORDER:
        row = smap[method]
        table_lines.append(
            f"{latex_escape(LABELS[method])} & {100*row['success_mean']:.1f} & {row['completion_steps_mean']:.1f} & "
            f"{row['idle_steps_mean']:.1f} & {row['stale_instruction_errors_mean']:.1f} & "
            f"{row['control_smoothness_mean']:.3f} & {row['safety_violations_mean']:.2f} & {row['planner_calls_mean']:.1f} \\\\"
        )
    async_row = smap["async_online"]
    sync_row = smap["sync_instruction"]
    direct_row = smap["direct_blocking"]
    oracle_row = smap["oracle"]
    text = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{lmodern}}
\usepackage{{microtype}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{xcolor}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{enumitem}}
\hypersetup{{hypertexnames=false,colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black}}
\title{{Instruction-Conditioned Asynchronous Control:\\A Deterministic Planner--Controller Decoupling Toy}}
\author{{Andy Park\\\href{{mailto:andypark.purdue@gmail.com}}{{andypark.purdue@gmail.com}}}}
\date{{August 31, 2026}}
\begin{{document}}
\maketitle
\begin{{abstract}}
This report isolates one mechanism from Instruct-to-Act: a slow sparse instruction planner can run separately from a fast language-conditioned controller, so planning latency need not become low-level control idle time. The environment is a partially observed sequential continuous-control toy, and the controller is transparent ridge regression trained from post-hoc instruction-relabeled expert segments. It is emphatically \textbf{{not}} a Dreamer, VLM, VLA, or paper reproduction.
\end{{abstract}}

\section{{Source claim and bounded question}}
Tang et al. propose a plug-and-play VLM planner that emits sparse text instructions to a language-conditioned world-model controller running at control frequency \cite{{paper,project,code}}. Their controller is trained with post-hoc instruction annotation, and their online inference lets planning proceed concurrently with execution. The local question is narrower: under an explicit tick-level latency model, does separating planner and controller reduce blocking while retaining enough semantic guidance to solve hidden sequential routes?

\section{{Toy setup}}
A point agent must visit a hidden ordered route of three or four named stations. The controller observes proprioception, all station-relative vectors, and only nearby moving hazards; it does not observe the route or future hazard motion. A planner can observe route progress and emits a station instruction. Actions are two-dimensional accelerations at every {cfg.dt:.2f}-s tick. This makes route state and hazard futures partially observed while keeping the scheduling mechanism inspectable.

We first generate autonomous expert rollouts, segment them at station-completion boundaries, relabel each segment with one of four natural-language dialects, and fit ridge controllers. The medium controller uses {training['medium'].samples} labeled action steps and has held-out action MSE {training['medium'].test_mse:.4f}. This is a transparent behavior-cloning analogue of post-hoc instruction supervision, not world-model RL.

\section{{Compared systems}}
\begin{{itemize}}[leftmargin=1.5em]
\item \textbf{{Controller only:}} a learned self-operating policy follows the modal route prior without planner calls.
\item \textbf{{Direct/blocking planner actions:}} every open-loop action chunk is generated after a blocking planner delay. The direct-as-actuator foil has an explicit deterministic action perturbation of {cfg.direct_action_error:.2f} and a +{cfg.direct_target_error_bonus:.2f} target-error probability; these are disclosed toy knobs, not measured VLM errors.
\item \textbf{{Synchronous instruction controller:}} the learned controller acts at high frequency, but idles at every instruction boundary while the planner responds.
\item \textbf{{Asynchronous online planning:}} the controller continues acting while the planner prepares the next instruction; if it is late, the last instruction remains active and creates measurable staleness rather than idle time.
\item \textbf{{Oracle:}} perfect zero-latency stage instructions and the analytical expert controller.
\end{{itemize}}

The central scheduling comparison is synchronous versus asynchronous instructions: both use the same learned controller, planner profile, environment trace, planner-error random draws, and per-tick controller-noise draws. They differ in whether planning blocks control. The asynchronous default also pays a disclosed four-tick transport delay. The direct-action row is an intentionally perturbed foil rather than an Instruct-to-Act baseline or a capacity-matched learned system; the oracle is an analytical upper bound.

\section{{Relation to other experiments in this repository}}
\begin{{itemize}}[leftmargin=1.5em]
\item \texttt{{streaming\_action\_denoising}} (FlashVLA) pipelines staggered refinement inside a low-level action decoder and couples cleaner chunks to noisier future chunks. This report instead separates a sparse semantic planner from a high-rate controller; it has no denoising buffer or decoder-level causal attention.
\item \texttt{{async\_chunking\_compare}} and \texttt{{anticipatory\_context\_chunking}} compensate prediction-to-execution delay by estimating the handoff-time robot state and, in the latter, the changing observation/environment latent. Here current toy state is observed every control tick; the delayed object is semantic intent.
\item \texttt{{context\_chunk\_tradeoff}} and \texttt{{openvla\_oft\_systems\_toy}} vary observation context, action horizon, parallel chunk availability, or refresh cadence. This toy emits one action each tick and varies sparse instruction scheduling; instruction dwell is not an open-loop action horizon.
\item \texttt{{turbo\_vla\_direct\_control}} removes a large execution-time language bottleneck with a compact direct V+L$\rightarrow$A path. The present design deliberately retains a slow high-level language planner and tests whether a separate controller absorbs its latency.
\item \texttt{{prefix\_rl\_chunking}} regularizes copied committed action prefixes under RL. This toy has neither action-prefix conditioning nor RL; its continuity error is stale semantic intent rather than prefix drift.
\item \path{{demo_prompted_policy}} and \path{{video_prompt_shortcut_resistance}} test learned task specification and shortcut resistance. The station-word parser here is only a narrow semantic-interface check, not prompt grounding.
\end{{itemize}}

\section{{Metrics}}
Success is completion of the ordered route before the horizon. Completion steps/time include planner-blocking ticks. Idle counts zero-action planning ticks. A stale-instruction error is an active tick still executing an instruction intended for an already completed stage. CSVs separately report wrong-target ticks, which also include planner mistakes or interface parse failures. Smoothness is action-change accumulated per active tick, $\sum_t\lVert a_t-a_{{t-1}}\rVert_2/N_{{active}}$. Safety violations count moving-hazard contacts and wall hits. Planner calls expose sparse-instruction versus direct-action workload.

\section{{Default paired evaluation}}
The exact generation command was:
\begin{{verbatim}}
{command}
\end{{verbatim}}
Each method saw the same {cfg.trials} routes, starts, moving hazards, and disturbance traces.

\begin{{table}}[H]
\centering
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrr}}
\toprule
Method & Success (\%) & Steps & Idle & Stale ticks & Smoothness & Safety & Planner calls \\
\midrule
{chr(10).join(table_lines)}
\bottomrule
\end{{tabular}}}}
\caption{{Default paired results. Failed episodes contribute the full horizon to completion steps.}}
\end{{table}}

\begin{{figure}}[H]\centering
\includegraphics[width=\textwidth]{{../outputs/method_comparison.png}}
\caption{{Reliability, completion, blocking, staleness, smoothness, and safety.}}
\end{{figure}}

The asynchronous system achieves {100*async_row['success_mean']:.1f}\% success with {async_row['idle_steps_mean']:.1f} idle ticks, compared with {100*sync_row['success_mean']:.1f}\% and {sync_row['idle_steps_mean']:.1f} for synchronous instructions. Direct action generation uses {direct_row['planner_calls_mean']:.1f} calls and idles {direct_row['idle_steps_mean']:.1f} ticks on average, versus {async_row['planner_calls_mean']:.1f} calls and {async_row['idle_steps_mean']:.1f} idle ticks for asynchronous instructions. The oracle reaches {100*oracle_row['success_mean']:.1f}\%, retaining a visible upper-bound gap from controller approximation, planner errors, and stale instructions.

\section{{Sweeps and planner swaps}}
\begin{{figure}}[H]\centering
\includegraphics[width=\textwidth]{{../outputs/mechanism_sweeps.png}}
\caption{{Planner latency, planning cadence, transport staleness, controller capacity/noise, and swappable planner profiles. Red secondary curves show stale-instruction ticks where applicable.}}
\end{{figure}}

The sweeps vary planner latency, asynchronous lookahead cadence, extra instruction transport delay, controller training capacity plus action noise, and planner latency/error/dialect profiles. Familiar and held-out semantic synonyms share one language interface; the cipher planner deliberately breaks that interface and is a negative control rather than a realistic planner.

\begin{{figure}}[H]\centering
\includegraphics[width=\textwidth]{{../outputs/representative_rollout.png}}
\caption{{One paired route. Dynamic hazard circles are lightly overlaid; the final panel shows action magnitudes and blocking discontinuities.}}
\end{{figure}}

\section{{Interpretation}}
The mechanism toy supports a bounded scheduling claim: sparse instructions let a high-frequency controller absorb planner latency that would otherwise become direct blocking. It also exposes the trade-off rather than hiding it: asynchrony converts some idle time into stale-instruction execution, so planner latency, request cadence, and controller recovery must be co-designed. Semantic relabeling makes compatible planner swaps possible, while opaque labels fail at the interface.

\section{{Limitations}}
\begin{{itemize}}[leftmargin=1.5em]
\item This is not a Dreamer/VLM reproduction: there are no pixels, RSSM, latent imagination, actor--critic learning, pretrained language encoder, or API/model timing measurements.
\item The planner is scripted and latency is configured in environment ticks. Planner ``quality'' is a controlled instruction-error probability.
\item The controller's semantic parser and station-relative features are much easier than real language grounding and visual representation learning.
\item The direct-action row contains explicit extra action and target errors, so its rank is assumption-sensitive. Controller ``capacity'' also changes both features and sample count.
\item The task is single-agent and navigation-like; it omits long-horizon crafting, contact-rich robotics, communication, and open-ended instructions.
\item Safety and smoothness are toy proxies. Rankings depend on the dynamics, route distribution, thresholds, and planner/controller budgets in the CSV outputs.
\end{{itemize}}

\begin{{thebibliography}}{{9}}
\bibitem{{paper}} Z. Tang, K. R. Allen, S. van Steenkiste, I. Dasgupta, and A. Suhr. \emph{{Decoupling Planning and Control for Instructable Agents}}. COLM 2026, arXiv:2608.26788. \url{{https://arxiv.org/abs/2608.26788}}
\bibitem{{project}} Instruct-to-Act project page. \url{{https://zinengtang.github.io/instruct-to-act/}}
\bibitem{{code}} Official implementation. \url{{https://github.com/zinengtang/instruct-to-act-code}}
\end{{thebibliography}}
\end{{document}}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sanity_checks(
    cfg: Config,
    models: dict[str, LearnedController],
    default_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    sweep_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    model = models["medium"]
    state = EnvState(pos=np.array([0.0, 0.0]), vel=np.zeros(2), stage=0)
    hazards = np.array([[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]])
    target_checks = {}
    for target in STATION_ORDER:
        text = DIALECTS["heldout"][target]
        action = learned_action(model, state, text, hazards, cfg, 0.0, ("sanity", target))
        target_checks[target] = float(np.dot(action, STATIONS[target])) > 0
    dialect_ok = all(parse_instruction(DIALECTS[d][s]) == s for d in DIALECTS for s in STATION_ORDER)
    smap = method_summary_map(summary_rows)
    deterministic_a = run_rollout("async_online", cfg, 0, models).metrics
    deterministic_b = run_rollout("async_online", cfg, 0, models).metrics
    deterministic_ok = deterministic_a == deterministic_b
    stale_rows = sorted([r for r in sweep_summary if r["sweep"] == "staleness"], key=lambda r: float(r["transport_delay"]))
    stale_trend = stale_rows[-1]["stale_instruction_errors_mean"] >= stale_rows[0]["stale_instruction_errors_mean"]
    checks = {
        "target_conditioning_all_directions": all(target_checks.values()),
        "target_conditioning_detail": target_checks,
        "all_semantic_dialects_parse": dialect_ok,
        "deterministic_repeat": deterministic_ok,
        "oracle_success_at_least_90pct": smap["oracle"]["success_mean"] >= 0.90,
        "async_less_idle_than_sync": smap["async_online"]["idle_steps_mean"] < smap["sync_instruction"]["idle_steps_mean"],
        "direct_more_planner_calls_than_async": smap["direct_blocking"]["planner_calls_mean"] > smap["async_online"]["planner_calls_mean"],
        "extra_delay_not_less_stale": stale_trend,
        "trial_rows": len(default_rows),
    }
    required = [v for k, v in checks.items() if isinstance(v, bool)]
    checks["all_passed"] = all(required)
    if not checks["all_passed"]:
        failed = [k for k, v in checks.items() if isinstance(v, bool) and not v]
        raise AssertionError(f"sanity checks failed: {failed}")
    return checks


def run_experiment(cfg: Config, output_dir: pathlib.Path, write_report: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models, segments = train_controllers(cfg)
    training_payload = {
        cap: {
            "capacity": model.capacity,
            "samples": model.samples,
            "feature_names": model.feature_names,
            "weights": model.weights.tolist(),
            "train_mse": model.train_mse,
            "test_mse": model.test_mse,
        }
        for cap, model in models.items()
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(training_payload, indent=2), encoding="utf-8")
    write_csv(output_dir / "instruction_relabeling_segments.csv", segments[: min(600, len(segments))])

    default_rows: list[dict[str, Any]] = []
    for trial in range(cfg.trials):
        for method in METHOD_ORDER:
            default_rows.append(run_rollout(method, cfg, trial, models).metrics)
    summary_rows = summarize(default_rows, ["method", "label"])
    write_csv(output_dir / "trial_metrics.csv", default_rows)
    write_csv(output_dir / "summary_metrics.csv", summary_rows)

    sweep_rows: list[dict[str, Any]] = []
    sweep_trials = max(4, cfg.trials // 2)
    latency_values = [0, 6, 12, 24] if cfg.trials > 6 else [0, 12]
    for latency in latency_values:
        for trial in range(sweep_trials):
            for method in ("direct_blocking", "sync_instruction", "async_online"):
                row = run_rollout(method, cfg, trial, models, planner_latency=latency).metrics
                row["sweep"] = "latency"
                sweep_rows.append(row)
    cadence_values = [4, 8, 16, 32] if cfg.trials > 6 else [4, 24]
    for cadence in cadence_values:
        for trial in range(sweep_trials):
            row = run_rollout("async_online", cfg, trial, models, planner_cadence=cadence).metrics
            row["sweep"] = "cadence"
            sweep_rows.append(row)
    delay_values = [0, 4, 8, 16] if cfg.trials > 6 else [0, 12]
    for delay in delay_values:
        for trial in range(sweep_trials):
            row = run_rollout("async_online", cfg, trial, models, transport_delay=delay).metrics
            row["sweep"] = "staleness"
            sweep_rows.append(row)
    capacities = ("small", "medium", "large")
    noise_values = [0.0, 0.07, 0.16] if cfg.trials > 6 else [0.0, 0.14]
    for capacity in capacities:
        for noise in noise_values:
            for trial in range(sweep_trials):
                row = run_rollout(
                    "async_online", cfg, trial, models, controller_capacity=capacity, controller_noise=noise
                ).metrics
                row["sweep"] = "capacity_noise"
                sweep_rows.append(row)
    for profile in PLANNER_PROFILES.values():
        for trial in range(sweep_trials):
            row = run_rollout("async_online", cfg, trial, models, profile=profile).metrics
            row["sweep"] = "planner_swap"
            sweep_rows.append(row)
    sweep_group_keys = [
        "sweep",
        "method",
        "planner_latency",
        "planner_cadence",
        "transport_delay",
        "controller_capacity",
        "controller_noise",
        "planner_profile",
    ]
    sweep_summary = summarize(sweep_rows, sweep_group_keys)
    write_csv(output_dir / "sweep_trial_metrics.csv", sweep_rows)
    write_csv(output_dir / "sweep_summary.csv", sweep_summary)

    representative: dict[str, dict[str, Any]] = {}
    rep_trial = 3 if cfg.trials > 3 else 0
    for method in METHOD_ORDER:
        result = run_rollout(method, cfg, rep_trial, models, record=True)
        if result.trace is None:
            raise AssertionError("representative trace missing")
        representative[method] = result.trace

    plot_methods(summary_rows, output_dir / "method_comparison.png")
    plot_sweeps(sweep_summary, output_dir / "mechanism_sweeps.png")
    plot_representative(representative, output_dir / "representative_rollout.png", cfg)
    checks = sanity_checks(cfg, models, default_rows, summary_rows, sweep_summary)
    (output_dir / "sanity_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    command = (
        "python instruction_conditioned_async_control/run_instruction_conditioned_async_control.py "
        f"--seed {cfg.seed} --trials {cfg.trials} --episode-steps {cfg.episode_steps}"
    )
    report_command = (
        "cd /home/andypark/Projects/repos/vla-ideas\n"
        "PYTHONWARNINGS=error MPLBACKEND=Agg \\\n"
        "  /home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \\\n"
        "  instruction_conditioned_async_control/\\\n"
        "  run_instruction_conditioned_async_control.py \\\n"
        f"  --seed {cfg.seed} --trials {cfg.trials} --episode-steps {cfg.episode_steps}"
    )
    metrics_payload = {
        "claim_boundary": "Mechanism toy only; not a Dreamer, VLM, VLA, or paper reproduction.",
        "config": asdict(cfg),
        "exact_command": command,
        "method_summary": {row["method"]: row for row in summary_rows},
        "planner_profiles": {name: asdict(profile) for name, profile in PLANNER_PROFILES.items()},
        "training": training_payload,
        "sanity_checks": checks,
        "outputs": [
            "trial_metrics.csv",
            "summary_metrics.csv",
            "sweep_trial_metrics.csv",
            "sweep_summary.csv",
            "instruction_relabeling_segments.csv",
            "training_metrics.json",
            "sanity_checks.json",
            "method_comparison.png",
            "mechanism_sweeps.png",
            "representative_rollout.png",
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    if write_report:
        generate_report(
            DEFAULT_DOCS_DIR / "instruction_conditioned_async_control_report.tex",
            cfg,
            summary_rows,
            sweep_summary,
            models,
            report_command,
        )
    return metrics_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--episode-steps", type=int, default=260)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoke", action="store_true", help="short run with reduced sweep grids")
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 1 or args.episode_steps < 40:
        raise ValueError("trials must be >=1 and episode-steps >=40")
    cfg = Config(seed=args.seed, trials=args.trials, episode_steps=args.episode_steps)
    if args.smoke:
        cfg = replace(cfg, trials=min(args.trials, 4), episode_steps=min(args.episode_steps, 120), train_episodes=28)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        payload = run_experiment(cfg, args.output_dir.resolve(), not args.no_report and args.output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve())
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "method_summary": payload["method_summary"], "sanity": payload["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
