#!/usr/bin/env python3
"""Deterministic systems/control toy inspired by FlashVLA streaming action decoding.

This is not a VLA, flow-matching model, learned distillation method, or hardware
benchmark.  It isolates scheduling and continuity mechanisms with an analytical
2-D chunk controller and configured decoder-pass costs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
from dataclasses import asdict, dataclass, replace
from typing import Any

BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_DOCS_DIR = BASE_DIR / "docs"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = [
    "isolated_n_step",
    "few_step_distilled",
    "streaming_no_causal",
    "streaming_causal",
    "streaming_causal_compensated",
    "future_state_conditioned",
]
LABELS = {
    "isolated_n_step": "Isolated 6-step",
    "few_step_distilled": "Few-step degraded proxy",
    "streaming_no_causal": "Staggered, no coupling",
    "streaming_causal": "Staggered + causal only",
    "streaming_causal_compensated": "Staggered + causal + handoff compensation",
    "future_state_conditioned": "Future-state conditioned",
}
COLORS = {
    "isolated_n_step": "#4c78a8",
    "few_step_distilled": "#f58518",
    "streaming_no_causal": "#eeca3b",
    "streaming_causal": "#54a24b",
    "streaming_causal_compensated": "#2a9d8f",
    "future_state_conditioned": "#b279a2",
}
PLOT_LABELS = {
    "isolated_n_step": "Isolated\n6-step",
    "few_step_distilled": "Few-step\ndegraded proxy",
    "streaming_no_causal": "Staggered\nno coupling",
    "streaming_causal": "Staggered\n+ causal only",
    "streaming_causal_compensated": "Staggered\n+ causal + comp.",
    "future_state_conditioned": "Future-state\nconditioned",
}
MARKERS = {method: marker for method, marker in zip(METHOD_ORDER, ["o", "s", "^", "D", "v", "P"])}


@dataclass(frozen=True)
class Config:
    seed: int = 23
    trials: int = 48
    episode_steps: int = 240
    dt: float = 0.025
    chunk_size: int = 8
    denoise_steps: int = 6
    distilled_steps: int = 2
    pass_ms: float = 20.0
    max_accel: float = 4.8
    kp: float = 4.1
    kd: float = 2.35
    robot_drag_true: float = 0.33
    robot_drag_nominal: float = 0.20
    actuator_gain: float = 0.88
    target_accel_scale: float = 0.72
    target_noise_std: float = 0.045
    robot_noise_std: float = 0.075
    gust_accel: float = 1.8
    decoder_noise_std: float = 0.95
    refinement_gain: float = 0.56
    causal_action_blend: float = 0.58
    success_tail_steps: int = 72
    success_tail_rmse: float = 0.20
    success_final_error: float = 0.30


@dataclass
class Observation:
    robot: np.ndarray
    target: np.ndarray
    target_accel: np.ndarray


@dataclass
class Chunk:
    actions: np.ndarray
    assumed_start: np.ndarray
    planned_end: np.ndarray
    context_step: int
    chunk_id: int
    last_action: np.ndarray


@dataclass
class StreamSlot:
    actions: np.ndarray
    age: int
    chunk_id: int
    assumed_start: np.ndarray
    planned_end: np.ndarray
    context_step: int


@dataclass
class TrialTrace:
    phase: np.ndarray
    target_noise: np.ndarray
    robot_noise: np.ndarray
    gust: np.ndarray


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**32 - 1)


def method_latency_ms(method: str, cfg: Config, latency_scale: float) -> float:
    passes = cfg.distilled_steps if method == "few_step_distilled" else 1 if method.startswith("streaming") else cfg.denoise_steps
    return cfg.pass_ms * latency_scale * passes


def method_steady_passes(method: str, cfg: Config) -> int:
    if method.startswith("streaming"):
        return 1
    if method == "few_step_distilled":
        return cfg.distilled_steps
    return cfg.denoise_steps


def cold_start_passes(method: str, cfg: Config) -> int:
    if method == "few_step_distilled":
        return cfg.distilled_steps
    return cfg.denoise_steps


def systems_row(method: str, cfg: Config, latency_scale: float) -> dict[str, float | str]:
    pass_ms = cfg.pass_ms * latency_scale
    steady_passes = method_steady_passes(method, cfg)
    first_passes = cold_start_passes(method, cfg)
    causal_pairs = cfg.denoise_steps * (cfg.denoise_steps - 1) / 2 if method == "streaming_causal" else 0.0
    slot_updates = float(cfg.denoise_steps if method.startswith("streaming") else steady_passes)
    return {
        "method": method,
        "label": LABELS[method],
        "latency_scale": latency_scale,
        "configured_pass_ms": pass_ms,
        "steady_sequential_passes_per_chunk": float(steady_passes),
        "cold_start_sequential_passes": float(first_passes),
        "steady_chunk_throughput_hz": 1000.0 / (pass_ms * steady_passes),
        "steady_action_throughput_hz": cfg.chunk_size * 1000.0 / (pass_ms * steady_passes),
        "configured_first_action_latency_ms": pass_ms * first_passes,
        "chunk_slot_updates_per_emitted_chunk": slot_updates,
        "causal_pair_interactions_per_pass": causal_pairs,
    }


def make_trace(cfg: Config, trial_seed: int, disturbance_scale: float) -> TrialTrace:
    rng = np.random.default_rng(trial_seed)
    phase = rng.uniform(-np.pi, np.pi, size=4)
    target_noise = rng.normal(0.0, cfg.target_noise_std * disturbance_scale, size=(cfg.episode_steps + 32, 2))
    robot_noise = rng.normal(0.0, cfg.robot_noise_std * disturbance_scale, size=(cfg.episode_steps + 32, 2))
    gust = np.zeros((cfg.episode_steps + 32, 2), dtype=np.float64)
    for fraction, base_angle in ((0.34, 0.2), (0.61, 2.2), (0.79, -1.1)):
        center = int(fraction * cfg.episode_steps) + int(rng.integers(-4, 5))
        width = int(rng.integers(3, 7))
        angle = base_angle + rng.normal(0.0, 0.35)
        vec = cfg.gust_accel * disturbance_scale * np.array([np.cos(angle), np.sin(angle)])
        gust[center : center + width] += vec
    return TrialTrace(phase=phase, target_noise=target_noise, robot_noise=robot_noise, gust=gust)


def base_target_accel(step: int, trace: TrialTrace, cfg: Config) -> np.ndarray:
    t = step * cfg.dt
    return cfg.target_accel_scale * np.array(
        [
            0.75 * np.sin(1.05 * t + trace.phase[0]) + 0.36 * np.sin(2.7 * t + trace.phase[1]),
            0.68 * np.cos(0.82 * t + trace.phase[2]) + 0.31 * np.sin(2.2 * t + trace.phase[3]),
        ],
        dtype=np.float64,
    )


def observe(robot: np.ndarray, target: np.ndarray, step: int, trace: TrialTrace, cfg: Config) -> Observation:
    return Observation(
        robot=robot.copy(),
        target=target.copy(),
        target_accel=base_target_accel(step, trace, cfg) + trace.target_noise[step],
    )


def clip_action(action: np.ndarray, cfg: Config) -> np.ndarray:
    norm = float(np.linalg.norm(action))
    if norm > cfg.max_accel:
        return action * (cfg.max_accel / norm)
    return action


def nominal_robot_step(state: np.ndarray, action: np.ndarray, cfg: Config) -> np.ndarray:
    p, v = state[:2], state[2:]
    accel = action - cfg.robot_drag_nominal * v
    v_next = v + cfg.dt * accel
    p_next = p + cfg.dt * v + 0.5 * cfg.dt**2 * accel
    return np.concatenate([p_next, v_next])


def target_step(target: np.ndarray, accel: np.ndarray, cfg: Config) -> np.ndarray:
    p, v = target[:2], target[2:]
    v_next = v + cfg.dt * accel
    p_next = p + cfg.dt * v + 0.5 * cfg.dt**2 * accel
    return np.concatenate([p_next, v_next])


def true_robot_step(robot: np.ndarray, action: np.ndarray, disturbance: np.ndarray, cfg: Config) -> np.ndarray:
    p, v = robot[:2], robot[2:]
    accel = cfg.actuator_gain * action - cfg.robot_drag_true * v + disturbance
    v_next = v + cfg.dt * accel
    p_next = p + cfg.dt * v + 0.5 * cfg.dt**2 * accel
    return np.concatenate([p_next, v_next])


def rollout_robot(state: np.ndarray, actions: np.ndarray, cfg: Config) -> np.ndarray:
    out = state.copy()
    for action in actions:
        out = nominal_robot_step(out, action, cfg)
    return out


def rollout_target(target: np.ndarray, accel: np.ndarray, steps: int, cfg: Config) -> np.ndarray:
    out = target.copy()
    for _ in range(max(steps, 0)):
        out = target_step(out, accel, cfg)
    return out


def plan_chunk(
    robot_start: np.ndarray,
    target_start: np.ndarray,
    target_accel: np.ndarray,
    cfg: Config,
    anchor_action: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    robot = robot_start.copy()
    target = target_start.copy()
    actions: list[np.ndarray] = []
    for k in range(cfg.chunk_size):
        raw = cfg.kp * (target[:2] - robot[:2]) + cfg.kd * (target[2:] - robot[2:])
        action = clip_action(raw, cfg)
        if anchor_action is not None:
            weight = cfg.causal_action_blend * math.exp(-k / 1.7)
            action = clip_action((1.0 - weight) * action + weight * anchor_action, cfg)
        actions.append(action)
        robot = nominal_robot_step(robot, action, cfg)
        target = target_step(target, target_accel, cfg)
    return np.asarray(actions), robot


def decoder_noise(trial_seed: int, chunk_id: int, family: str, cfg: Config) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(trial_seed, chunk_id, family, "decoder"))
    raw = rng.normal(0.0, cfg.decoder_noise_std, size=(cfg.chunk_size, 2))
    # Correlated residual resembles a coarse chunk-level decoding error rather than actuator white noise.
    for k in range(1, cfg.chunk_size):
        raw[k] = 0.72 * raw[k - 1] + 0.28 * raw[k]
    return raw


def isolated_decode(
    method: str,
    obs: Observation,
    committed_suffix: np.ndarray,
    trial_seed: int,
    chunk_id: int,
    context_step: int,
    cfg: Config,
) -> Chunk:
    if method == "future_state_conditioned":
        assumed_robot = rollout_robot(obs.robot, committed_suffix, cfg)
        assumed_target = rollout_target(obs.target, obs.target_accel, len(committed_suffix), cfg)
    else:
        assumed_robot = obs.robot.copy()
        assumed_target = obs.target.copy()

    candidate, planned_end = plan_chunk(assumed_robot, assumed_target, obs.target_accel, cfg)
    family = "isolated" if method in ("isolated_n_step", "future_state_conditioned") else "distilled"
    noise = decoder_noise(trial_seed, chunk_id, family, cfg)
    steps = cfg.distilled_steps if method == "few_step_distilled" else cfg.denoise_steps
    residual = (1.0 - cfg.refinement_gain) ** steps
    actions = candidate + residual * noise
    if method == "few_step_distilled":
        # A deterministic quality/capacity proxy: fewer steps under-react to rapid within-chunk changes.
        smooth = actions.copy()
        for k in range(1, cfg.chunk_size):
            smooth[k] = 0.60 * smooth[k - 1] + 0.40 * smooth[k]
        actions = 0.94 * smooth
    actions = np.asarray([clip_action(a, cfg) for a in actions])
    planned_end = rollout_robot(assumed_robot, actions, cfg)
    return Chunk(
        actions=actions,
        assumed_start=assumed_robot,
        planned_end=planned_end,
        context_step=context_step,
        chunk_id=chunk_id,
        last_action=actions[-1].copy(),
    )


class StreamingDecoder:
    def __init__(self, causal: bool, trial_seed: int, cfg: Config, compensate_handoff: bool = False):
        self.causal = causal
        self.compensate_handoff = compensate_handoff
        self.trial_seed = trial_seed
        self.cfg = cfg
        self.slots: list[StreamSlot] = []
        self.next_chunk_id = 0
        self.slot_updates = 0
        self.coupling_pairs = 0

    def _new_slot(self, obs: Observation, context_step: int) -> StreamSlot:
        chunk_id = self.next_chunk_id
        self.next_chunk_id += 1
        noise = decoder_noise(self.trial_seed, chunk_id, "stream", self.cfg)
        return StreamSlot(
            actions=noise.copy(),
            age=0,
            chunk_id=chunk_id,
            assumed_start=obs.robot.copy(),
            planned_end=obs.robot.copy(),
            context_step=context_step,
        )

    def step(
        self,
        obs: Observation,
        context_step: int,
        predecessor_end: np.ndarray | None,
        predecessor_action: np.ndarray | None,
        steps_to_boundary: int,
    ) -> Chunk | None:
        self.slots.append(self._new_slot(obs, context_step))
        # Keep decoder-level continuation separate from observation-delay
        # compensation.  The causal-only row starts from the cleaner
        # predecessor's endpoint.  A separate diagnostic row blends that
        # endpoint with a simple rollout of the latest observation.
        if self.compensate_handoff and predecessor_action is not None:
            fresh_boundary = rollout_robot(
                obs.robot,
                np.tile(predecessor_action[None, :], (max(steps_to_boundary, 1), 1)),
                self.cfg,
            )
            causal_start = (
                0.85 * fresh_boundary + 0.15 * predecessor_end
                if predecessor_end is not None
                else fresh_boundary
            )
        else:
            causal_start = predecessor_end.copy() if predecessor_end is not None else obs.robot.copy()
        causal_anchor = predecessor_action.copy() if predecessor_action is not None else None

        for index, slot in enumerate(self.slots):
            offset_steps = max(steps_to_boundary, 0) + index * self.cfg.chunk_size
            target_at_slot = rollout_target(obs.target, obs.target_accel, offset_steps, self.cfg)
            if self.causal:
                assumed_start = causal_start.copy()
                candidate, _ = plan_chunk(
                    assumed_start,
                    target_at_slot,
                    obs.target_accel,
                    self.cfg,
                    anchor_action=causal_anchor,
                )
                self.coupling_pairs += index + (1 if predecessor_end is not None else 0)
            else:
                # Independent slots know their time index but not the trajectory produced by cleaner chunks.
                assumed_start = obs.robot.copy()
                candidate, _ = plan_chunk(assumed_start, target_at_slot, obs.target_accel, self.cfg)

            slot.actions = slot.actions + self.cfg.refinement_gain * (candidate - slot.actions)
            slot.actions = np.asarray([clip_action(a, self.cfg) for a in slot.actions])
            slot.age += 1
            slot.assumed_start = assumed_start
            slot.planned_end = rollout_robot(assumed_start, slot.actions, self.cfg)
            slot.context_step = context_step
            self.slot_updates += 1
            if self.causal:
                causal_start = slot.planned_end.copy()
                causal_anchor = slot.actions[-1].copy()

        if self.slots and self.slots[0].age >= self.cfg.denoise_steps:
            slot = self.slots.pop(0)
            return Chunk(
                actions=slot.actions.copy(),
                assumed_start=slot.assumed_start.copy(),
                planned_end=slot.planned_end.copy(),
                context_step=slot.context_step,
                chunk_id=slot.chunk_id,
                last_action=slot.actions[-1].copy(),
            )
        return None

    def cold_start(self, obs: Observation, context_step: int) -> Chunk:
        emitted = None
        for _ in range(self.cfg.denoise_steps):
            emitted = self.step(obs, context_step, None, None, 0)
        assert emitted is not None
        return emitted


def simulate_trial(
    method: str,
    cfg: Config,
    trial_seed: int,
    latency_scale: float,
    disturbance_scale: float,
    keep_trajectory: bool = False,
) -> tuple[dict[str, float | int | str], dict[str, np.ndarray] | None]:
    trace = make_trace(cfg, trial_seed, disturbance_scale)
    robot = np.array([0.0, -0.15, 0.0, 0.0], dtype=np.float64)
    target = np.array([1.15, 0.35, 0.18, -0.06], dtype=np.float64)
    stream = (
        StreamingDecoder(
            method in ("streaming_causal", "streaming_causal_compensated"),
            trial_seed,
            cfg,
            compensate_handoff=method == "streaming_causal_compensated",
        )
        if method.startswith("streaming")
        else None
    )

    latency_ms = method_latency_ms(method, cfg, latency_scale)
    latency_steps = max(1, int(math.ceil(latency_ms / (1000.0 * cfg.dt))))
    pending: tuple[int, Chunk] | None = None
    ready: Chunk | None = None
    active: Chunk | None = None
    active_index = 0
    previous_action = np.zeros(2, dtype=np.float64)
    previous_chunk_last_action: np.ndarray | None = None
    first_action_step: int | None = None
    next_chunk_id = 0
    deadline_miss_steps = 0
    launched_chunks = 0
    emitted_chunks = 0
    decoder_passes = 0
    boundary_jumps: list[float] = []
    handoff_staleness: list[float] = []
    context_age_steps: list[int] = []

    robot_hist = []
    target_hist = []
    action_hist = []
    error_hist = []
    boundary_hist = []
    idle_hist = []

    def launch(step: int, obs: Observation, suffix: np.ndarray, steps_to_boundary: int) -> tuple[int, Chunk]:
        nonlocal next_chunk_id, launched_chunks, decoder_passes
        launched_chunks += 1
        if method.startswith("streaming"):
            assert stream is not None
            predecessor_end = active.planned_end if active is not None else None
            predecessor_action = active.last_action if active is not None else previous_chunk_last_action
            if emitted_chunks == 0 and not stream.slots:
                chunk = stream.cold_start(obs, step)
                decoder_passes += cfg.denoise_steps
                ready_at = step + max(1, int(math.ceil(cfg.denoise_steps * cfg.pass_ms * latency_scale / (1000.0 * cfg.dt))))
            else:
                chunk = stream.step(obs, step, predecessor_end, predecessor_action, steps_to_boundary)
                if chunk is None:
                    raise RuntimeError("steady streaming pass did not emit a mature chunk")
                decoder_passes += 1
                ready_at = step + latency_steps
        else:
            chunk = isolated_decode(method, obs, suffix, trial_seed, next_chunk_id, step, cfg)
            next_chunk_id += 1
            decoder_passes += method_steady_passes(method, cfg)
            ready_at = step + latency_steps
        return ready_at, chunk

    for step in range(cfg.episode_steps):
        if pending is not None and step >= pending[0]:
            ready = pending[1]
            pending = None

        if active is not None and active_index >= len(active.actions):
            previous_chunk_last_action = active.actions[-1].copy()
            active = None
            active_index = 0

        if active is None and ready is not None:
            active = ready
            ready = None
            active_index = 0
            emitted_chunks += 1
            boundary_hist.append(step)
            if first_action_step is None:
                first_action_step = step
            if previous_chunk_last_action is not None:
                boundary_jumps.append(float(np.linalg.norm(active.actions[0] - previous_chunk_last_action)))
            handoff_staleness.append(float(np.linalg.norm(active.assumed_start[:2] - robot[:2])))
            context_age_steps.append(max(0, step - active.context_step))

        if active is None:
            suffix = np.empty((0, 2), dtype=np.float64)
            remaining = 0
        else:
            suffix = active.actions[active_index:].copy()
            remaining = len(suffix)

        if pending is None and ready is None:
            should_launch = False
            if first_action_step is None and active is None:
                should_launch = step == 0
            elif active is not None and remaining <= latency_steps:
                should_launch = True
            elif active is None and first_action_step is not None:
                should_launch = True
            if should_launch:
                obs = observe(robot, target, step, trace, cfg)
                pending = launch(step, obs, suffix, remaining)

        if active is not None:
            action = active.actions[active_index].copy()
            active_index += 1
            idle = 0.0
        else:
            action = previous_action.copy() if first_action_step is not None else np.zeros(2, dtype=np.float64)
            idle = 1.0
            if first_action_step is not None:
                deadline_miss_steps += 1

        target_accel = base_target_accel(step, trace, cfg) + trace.target_noise[step]
        target = target_step(target, target_accel, cfg)
        disturbance = trace.robot_noise[step] + trace.gust[step]
        robot = true_robot_step(robot, action, disturbance, cfg)
        previous_action = action.copy()

        err = float(np.linalg.norm(robot[:2] - target[:2]))
        robot_hist.append(robot.copy())
        target_hist.append(target.copy())
        action_hist.append(action.copy())
        error_hist.append(err)
        idle_hist.append(idle)

    errors = np.asarray(error_hist)
    actions = np.asarray(action_hist)
    tail = errors[-min(cfg.success_tail_steps, len(errors)) :]
    first_latency_sim_ms = float("nan") if first_action_step is None else first_action_step * cfg.dt * 1000.0
    jerk = np.diff(actions, axis=0) / cfg.dt if len(actions) > 1 else np.zeros((0, 2))
    metric: dict[str, float | int | str] = {
        "method": method,
        "trial_seed": trial_seed,
        "latency_scale": latency_scale,
        "disturbance_scale": disturbance_scale,
        "success": float(np.sqrt(np.mean(tail**2)) < cfg.success_tail_rmse and errors[-1] < cfg.success_final_error),
        "tracking_rmse": float(np.sqrt(np.mean(errors**2))),
        "tail_tracking_rmse": float(np.sqrt(np.mean(tail**2))),
        "final_error": float(errors[-1]),
        "boundary_discontinuity": float(np.mean(boundary_jumps)) if boundary_jumps else 0.0,
        "p95_boundary_discontinuity": float(np.percentile(boundary_jumps, 95)) if boundary_jumps else 0.0,
        "handoff_staleness": float(np.mean(handoff_staleness)) if handoff_staleness else float("nan"),
        "context_age_ms": float(np.mean(context_age_steps) * cfg.dt * 1000.0) if context_age_steps else float("nan"),
        "rms_jerk": float(np.sqrt(np.mean(np.sum(jerk**2, axis=1)))) if len(jerk) else 0.0,
        "idle_fraction": float(np.mean(idle_hist)),
        "deadline_miss_steps": deadline_miss_steps,
        "simulated_first_action_latency_ms": first_latency_sim_ms,
        "configured_first_action_latency_ms": cfg.pass_ms * latency_scale * cold_start_passes(method, cfg),
        "emitted_chunks": emitted_chunks,
        "decoder_launches": launched_chunks,
        "decoder_sequential_passes": decoder_passes,
        "decoder_slot_updates": stream.slot_updates if stream is not None else decoder_passes,
        "causal_pair_interactions": stream.coupling_pairs if stream is not None else 0,
    }
    trajectory = None
    if keep_trajectory:
        trajectory = {
            "robot": np.asarray(robot_hist),
            "target": np.asarray(target_hist),
            "action": actions,
            "error": errors,
            "idle": np.asarray(idle_hist),
            "boundaries": np.asarray(boundary_hist, dtype=int),
        }
    return metric, trajectory


def aggregate(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    groups: dict[tuple[str, float, float], list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = (str(row["method"]), float(row["latency_scale"]), float(row["disturbance_scale"]))
        groups.setdefault(key, []).append(row)
    metric_names = [
        "success",
        "tracking_rmse",
        "tail_tracking_rmse",
        "final_error",
        "boundary_discontinuity",
        "p95_boundary_discontinuity",
        "handoff_staleness",
        "context_age_ms",
        "rms_jerk",
        "idle_fraction",
        "deadline_miss_steps",
        "simulated_first_action_latency_ms",
        "emitted_chunks",
        "decoder_sequential_passes",
        "decoder_slot_updates",
        "causal_pair_interactions",
    ]
    out: list[dict[str, float | str]] = []
    for (method, latency_scale, disturbance_scale), trials in sorted(groups.items(), key=lambda x: (METHOD_ORDER.index(x[0][0]), x[0][1], x[0][2])):
        row: dict[str, float | str] = {
            "method": method,
            "label": LABELS[method],
            "latency_scale": latency_scale,
            "disturbance_scale": disturbance_scale,
            "trials": float(len(trials)),
        }
        for name in metric_names:
            vals = np.asarray([float(t[name]) for t in trials], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            row[f"{name}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"{name}_sem"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def get_summary(summary: list[dict[str, float | str]], method: str, latency: float, disturbance: float) -> dict[str, float | str]:
    for row in summary:
        if row["method"] == method and math.isclose(float(row["latency_scale"]), latency) and math.isclose(float(row["disturbance_scale"]), disturbance):
            return row
    raise KeyError((method, latency, disturbance))


def run_sanity_checks(cfg: Config) -> dict[str, Any]:
    obs = Observation(
        robot=np.array([0.0, 0.0, 0.25, -0.1]),
        target=np.array([1.0, 0.4, 0.15, 0.05]),
        target_accel=np.array([0.2, -0.1]),
    )
    suffix = np.tile(np.array([[0.8, -0.2]]), (4, 1))
    stale_error = float(np.linalg.norm(obs.robot[:2] - rollout_robot(obs.robot, suffix, cfg)[:2]))
    future = rollout_robot(obs.robot, suffix, cfg)
    future_error = float(np.linalg.norm(future[:2] - rollout_robot(obs.robot, suffix, cfg)[:2]))

    candidate, _ = plan_chunk(obs.robot, obs.target, obs.target_accel, cfg)
    noise = decoder_noise(1, 0, "check", cfg)
    full_error = float(np.sqrt(np.mean(((1 - cfg.refinement_gain) ** cfg.denoise_steps * noise) ** 2)))
    few_error = float(np.sqrt(np.mean(((1 - cfg.refinement_gain) ** cfg.distilled_steps * noise) ** 2)))

    no_causal = StreamingDecoder(False, 101, cfg)
    causal = StreamingDecoder(True, 101, cfg)
    no_chunks, causal_chunks = [], []
    pred_end = obs.robot.copy()
    pred_action = np.zeros(2)
    for p in range(cfg.denoise_steps + 2):
        nc = no_causal.step(obs, p, pred_end, pred_action, 1)
        cc = causal.step(obs, p, pred_end, pred_action, 1)
        if nc is not None:
            no_chunks.append(nc)
        if cc is not None:
            causal_chunks.append(cc)
            pred_end, pred_action = cc.planned_end, cc.last_action
    no_jump = float(np.linalg.norm(no_chunks[1].actions[0] - no_chunks[0].actions[-1]))
    causal_jump = float(np.linalg.norm(causal_chunks[1].actions[0] - causal_chunks[0].actions[-1]))

    metric_a, traj_a = simulate_trial("streaming_causal", replace(cfg, episode_steps=80, success_tail_steps=24), 777, 1.0, 1.0, True)
    metric_b, traj_b = simulate_trial("streaming_causal", replace(cfg, episode_steps=80, success_tail_steps=24), 777, 1.0, 1.0, True)
    compensated_metric, _ = simulate_trial(
        "streaming_causal_compensated",
        replace(cfg, episode_steps=80, success_tail_steps=24),
        777,
        1.0,
        1.0,
        False,
    )
    assert traj_a is not None and traj_b is not None
    deterministic_delta = float(np.max(np.abs(traj_a["robot"] - traj_b["robot"])))

    checks = {
        "full_refinement_beats_few_step": {"pass": full_error < few_error, "full_rmse": full_error, "few_rmse": few_error},
        "future_projection_removes_nominal_staleness": {"pass": future_error < stale_error, "stale_error": stale_error, "future_error": future_error},
        "stream_first_emit_after_n_passes": {"pass": causal_chunks[0].chunk_id == 0 and causal_chunks[1].chunk_id == 1, "first_ids": [c.chunk_id for c in causal_chunks[:2]]},
        "causal_surrogate_reduces_simple_boundary_jump": {"pass": causal_jump < no_jump, "causal_jump": causal_jump, "no_causal_jump": no_jump},
        "handoff_compensation_is_separate_and_reduces_state_error": {
            "pass": float(compensated_metric["handoff_staleness"]) < float(metric_a["handoff_staleness"]),
            "causal_only_handoff_error": float(metric_a["handoff_staleness"]),
            "compensated_handoff_error": float(compensated_metric["handoff_staleness"]),
        },
        "streaming_steady_throughput_exceeds_isolated": {
            "pass": systems_row("streaming_causal", cfg, 1.0)["steady_chunk_throughput_hz"] > systems_row("isolated_n_step", cfg, 1.0)["steady_chunk_throughput_hz"],
            "streaming_hz": systems_row("streaming_causal", cfg, 1.0)["steady_chunk_throughput_hz"],
            "isolated_hz": systems_row("isolated_n_step", cfg, 1.0)["steady_chunk_throughput_hz"],
        },
        "cold_start_not_hidden": {
            "pass": systems_row("streaming_causal", cfg, 1.0)["configured_first_action_latency_ms"] == systems_row("isolated_n_step", cfg, 1.0)["configured_first_action_latency_ms"],
            "streaming_ms": systems_row("streaming_causal", cfg, 1.0)["configured_first_action_latency_ms"],
            "isolated_ms": systems_row("isolated_n_step", cfg, 1.0)["configured_first_action_latency_ms"],
        },
        "deterministic_replay": {"pass": deterministic_delta == 0.0 and metric_a == metric_b, "max_state_delta": deterministic_delta},
    }
    checks["all_pass"] = all(bool(v["pass"]) for v in checks.values() if isinstance(v, dict) and "pass" in v)
    return checks


def make_plots(
    cfg: Config,
    summary: list[dict[str, float | str]],
    systems: list[dict[str, float | str]],
    trajectories: dict[str, dict[str, np.ndarray]],
    output_dir: pathlib.Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    base_rows = [get_summary(summary, m, 1.0, 1.0) for m in METHOD_ORDER]
    x = np.arange(len(METHOD_ORDER))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    panels = [
        ("success_mean", "Success rate", 100.0),
        ("tracking_rmse_mean", "Full-rollout tracking RMSE", 1.0),
        ("boundary_discontinuity_mean", "Mean boundary action jump", 1.0),
        ("handoff_staleness_mean", "Handoff state error", 1.0),
    ]
    for ax, (key, title, scale) in zip(axes.flat, panels):
        vals = [float(r[key]) * scale for r in base_rows]
        sems = [float(r[key.replace("_mean", "_sem")]) * scale for r in base_rows]
        ax.bar(x, vals, yerr=sems, color=[COLORS[m] for m in METHOD_ORDER], capsize=3)
        ax.set_title(title)
        ax.set_xticks(x, [PLOT_LABELS[m] for m in METHOD_ORDER])
        if key == "success_mean":
            ax.set_ylim(0, 105)
            ax.set_ylabel("percent")
    fig.suptitle(f"Default condition: {cfg.trials} paired trials, latency scale 1.0, disturbance scale 1.0")
    fig.tight_layout()
    fig.savefig(output_dir / "control_quality.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    latency_values = sorted({float(r["latency_scale"]) for r in summary})
    disturbance_values = sorted({float(r["disturbance_scale"]) for r in summary})
    for method in METHOD_ORDER:
        y = [float(get_summary(summary, method, lat, 1.0)["success_mean"]) * 100 for lat in latency_values]
        axes[0].plot(latency_values, y, marker=MARKERS[method], color=COLORS[method], label=LABELS[method])
        y2 = [float(get_summary(summary, method, 1.0, dist)["tail_tracking_rmse_mean"]) for dist in disturbance_values]
        axes[1].plot(disturbance_values, y2, marker=MARKERS[method], color=COLORS[method], label=LABELS[method])
    axes[0].set(xlabel="Decoder latency scale", ylabel="Success (%)", title="Compute-delay robustness (disturbance = 1.0)", ylim=(-2, 102))
    axes[1].set(xlabel="Disturbance scale", ylabel="Tail-window tracking RMSE", title="Disturbance robustness (latency = 1.0)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(output_dir / "robustness_sweeps.png", dpi=180)
    plt.close(fig)

    sys_base = [next(r for r in systems if r["method"] == m and math.isclose(float(r["latency_scale"]), 1.0)) for m in METHOD_ORDER]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].bar(x, [float(r["steady_chunk_throughput_hz"]) for r in sys_base], color=[COLORS[m] for m in METHOD_ORDER])
    axes[0].set_title("Configured steady chunk throughput")
    axes[0].set_ylabel("chunks/s")
    axes[1].bar(x, [float(r["configured_first_action_latency_ms"]) for r in sys_base], color=[COLORS[m] for m in METHOD_ORDER])
    axes[1].set_title("Configured cold-start latency")
    axes[1].set_ylabel("ms")
    axes[2].bar(x, [float(r["chunk_slot_updates_per_emitted_chunk"]) for r in sys_base], color=[COLORS[m] for m in METHOD_ORDER])
    axes[2].set_title("Chunk-slot update proxy")
    for ax in axes:
        ax.set_xticks(x, [PLOT_LABELS[m] for m in METHOD_ORDER])
    fig.suptitle("Configured timing model; not measured hardware latency")
    fig.tight_layout()
    fig.savefig(output_dir / "systems_proxies.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8), sharex=True)
    t = np.arange(cfg.episode_steps) * cfg.dt
    for method in METHOD_ORDER:
        traj = trajectories[method]
        axes[0].plot(t, traj["error"], color=COLORS[method], label=LABELS[method], alpha=0.92)
        action_norm = np.linalg.norm(traj["action"], axis=1)
        axes[1].plot(t, action_norm, color=COLORS[method], label=LABELS[method], alpha=0.92)
    axes[0].set(ylabel="Position error", title="Representative paired rollout (default condition)")
    axes[1].set(xlabel="Time (s)", ylabel="Action magnitude")
    axes[0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "representative_rollout.png", dpi=180)
    plt.close(fig)


def latex_escape(text: str) -> str:
    for old, new in [("_", r"\_"), ("%", r"\%"), ("&", r"\&")]:
        text = text.replace(old, new)
    return text


def write_tex_report(cfg: Config, summary: list[dict[str, float | str]], systems: list[dict[str, float | str]], docs_dir: pathlib.Path, command: str) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in METHOD_ORDER:
        q = get_summary(summary, method, 1.0, 1.0)
        s = next(r for r in systems if r["method"] == method and math.isclose(float(r["latency_scale"]), 1.0))
        rows.append(
            f"{latex_escape(LABELS[method])} & {100*float(q['success_mean']):.1f} & {float(q['tracking_rmse_mean']):.3f} & "
            f"{float(q['boundary_discontinuity_mean']):.3f} & {float(q['handoff_staleness_mean']):.3f} & "
            f"{float(s['steady_chunk_throughput_hz']):.2f} & {float(s['configured_first_action_latency_ms']):.0f} \\\\"
        )
    causal = get_summary(summary, "streaming_causal", 1.0, 1.0)
    compensated = get_summary(summary, "streaming_causal_compensated", 1.0, 1.0)
    no_causal = get_summary(summary, "streaming_no_causal", 1.0, 1.0)
    future = get_summary(summary, "future_state_conditioned", 1.0, 1.0)
    tex = rf"""\documentclass[11pt]{{article}}

\usepackage[margin=1in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{lmodern}}
\usepackage{{microtype}}
\usepackage{{amsmath, amssymb}}
\usepackage{{booktabs}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{enumitem}}
\usepackage{{graphicx}}
\usepackage{{float}}

\hypersetup{{hypertexnames=false,colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black,citecolor=blue!50!black}}
\title{{Streaming Action Denoising:\\A Deterministic Throughput--Continuity Control Toy}}
\author{{Andy Park\\\href{{mailto:andypark.purdue@gmail.com}}{{andypark.purdue@gmail.com}}}}
\date{{August 31, 2026}}

\begin{{document}}
\setlength{{\emergencystretch}}{{3em}}
\maketitle

\begin{{abstract}}
This note tests a narrow systems/control question inspired by FlashVLA: can staggered chunk refinement separate steady-state decoder throughput from per-chunk denoising depth, and can a causal chunk-continuation surrogate reduce asynchronous boundary mismatch? In this configured deterministic toy, streaming emits one chunk per pass after warm-up and causal continuation improves boundary smoothness, but causal-only endpoint chaining accumulates handoff-state drift. A separately labeled compensation surrogate repairs much of that drift. This is not a VLA reproduction, learned flow matching, or a hardware benchmark.
\end{{abstract}}

\section{{Motivation}}
Isolated iterative decoding concentrates all denoising passes before one chunk becomes available. FlashVLA instead keeps chunks at staggered noise levels and advances them jointly under chunk-wise causal attention \cite{{primary,code}}. The toy separates three questions: configured sequential throughput, one-time cold start, and closed-loop behavior when decoded chunks arrive after the observation that launched them.

\section{{References Checked}}
Primary references were the FlashVLA paper \cite{{primary}} and official repository \cite{{code}}. The paper reports measured VLA/GPU results, including one emitted action chunk per steady streaming step, causal attention ablations, and real/simulated task results. None of those reported numbers are treated as local measurements here.

\section{{Toy Setup}}
A two-dimensional disturbed point mass tracks a maneuvering target with an analytical PD chunk planner. Each chunk contains {cfg.chunk_size} low-level accelerations. The configured action-expert pass cost is {cfg.pass_ms:.0f} ms, the isolated depth is {cfg.denoise_steps}, and the few-step proxy uses {cfg.distilled_steps} passes. Six methods are compared:
\begin{{itemize}}[leftmargin=1.5em]
\item isolated {cfg.denoise_steps}-step chunks;
\item a {cfg.distilled_steps}-step quality/capacity proxy;
\item a staggered buffer with independent chunk refinement;
\item the same buffer with analytical cleaner-to-noisier continuation coupling only;
\item causal continuation plus a separately labeled handoff-state compensation surrogate; and
\item isolated decoding conditioned on a nominal rollout of committed actions.
\end{{itemize}}
The causal update is only a mechanism surrogate: it chains predicted endpoints and blends boundary actions. It is not transformer attention. The fixed pass time assumes one joint streaming pass costs the same wall-clock time as one isolated pass; the script therefore calls throughput and work values \emph{{proxies}}, not benchmarks.

\section{{Relation to other experiments in this repository}}
\begin{{itemize}}[leftmargin=1.5em]
\item \texttt{{instruction\_conditioned\_async\_control}} separates a slow semantic planner from a high-rate controller. FlashVLA instead pipelines refinement inside the action decoder; there is no sparse instruction handoff here.
\item \texttt{{async\_chunking\_compare}} and \texttt{{anticipatory\_context\_chunking}} estimate execution-time robot and observation context after inference delay. This toy isolates decoder scheduling and cleaner-to-noisier continuation; \texttt{{future\_state\_conditioned}} is the separate observation-delay-compensation control.
\item \path{{context_chunk_tradeoff}} varies temporal context and action horizon, and \path{{openvla_oft_systems_toy}} varies parallel chunk availability and refresh cadence. Here chunk size is fixed while denoising depth, slot staggering, and configured decoder latency vary.
\item \texttt{{turbo\_vla\_direct\_control}} changes the execution architecture to a compact direct V+L$\rightarrow$A path. FlashVLA changes iterative action decoding and can in principle sit on either a direct or larger VLA backbone.
\item \texttt{{prefix\_rl\_chunking}} preserves committed action prefixes during RL training. The causal surrogate here supplies predicted endpoint/action context between denoising slots at inference and has no prefix-copy loss.
\item \path{{demo_prompted_policy}} and \path{{video_prompt_shortcut_resistance}} study task grounding and shortcut resistance. This package has no prompt input and makes no prompt-reliance claim.
\end{{itemize}}

\paragraph{{Baseline fairness.}}
The clean decoder-only ablation is staggered decoding without versus with causal continuation: scheduler, slot count, refinement rule, decoder noise, launch policy, and environment traces are shared. The compensated streaming row then adds handoff-state correction as a separate factor. The isolated, few-step, and future-state rows are diagnostic controls, not training- or architecture-matched learned baselines. The few-step row includes hand-coded smoothing and attenuation. Reported chunks/s is configured back-to-back decoder service capacity under the equal-cost joint-pass assumption, not measured GPU throughput or the simulator's consumed chunk rate.

\section{{Metrics}}
The experiment reports success from terminal tracking thresholds, full and tail tracking RMSE, handoff state error, action jump at chunk boundaries, jerk, idle/deadline-miss time, configured steady-state throughput, configured and tick-quantized first-action latency, sequential-pass counts, chunk-slot updates, and causal pair-interaction counts.

\section{{Results}}
The generated result used {cfg.trials} paired trials for each method at each point of a $3\times3$ latency/disturbance sweep:
\begin{{verbatim}}
{command}
\end{{verbatim}}

\begin{{table}}[H]
\centering
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrr}}
\toprule
Method & Success (\%) & Track RMSE & Boundary jump & Handoff error & Chunks/s & Cold start (ms) \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}%
}}
\caption{{Default condition (latency scale 1.0, disturbance scale 1.0). Throughput and cold-start columns come from the configured timing model; control metrics are trial aggregates.}}
\label{{tab:results}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{../outputs/control_quality.png}}
\caption{{Default-condition control metrics. Error bars are standard errors over paired trials.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{../outputs/robustness_sweeps.png}}
\caption{{Compute-delay and disturbance sweeps. The two sweeps hold the other factor at 1.0.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{../outputs/systems_proxies.png}}
\caption{{Configured systems proxies. Staggering improves back-to-back decoder service capacity but does not erase chunk-slot refinement work or causal interaction cost.}}
\end{{figure}}

\section{{Interpretation}}
At the default condition, causal-only streaming changes boundary discontinuity from {float(no_causal['boundary_discontinuity_mean']):.3f} to {float(causal['boundary_discontinuity_mean']):.3f}, but success changes from {100*float(no_causal['success_mean']):.1f}\% to {100*float(causal['success_mean']):.1f}\% because chaining predicted endpoints accumulates handoff-state drift. Adding the separately labeled handoff-compensation surrogate raises success to {100*float(compensated['success_mean']):.1f}\%. Future-state conditioning reaches {100*float(future['success_mean']):.1f}\% success but retains isolated decoding's configured steady cadence. Thus staggering, inter-chunk continuity, and observation-delay compensation are distinct mechanisms: the first changes service cadence, the second can smooth boundaries, and the third is needed to control accumulated state-age error in this toy. None of these rows establishes real-VLA superiority.

\section{{Limitations and Follow-ups}}
\begin{{itemize}}[leftmargin=1.5em]
\item Timing is configured, not measured on a GPU. Joint-pass cost, memory traffic, compilation, and kernel-launch effects are omitted.
\item The decoder is an analytical contraction toward a PD plan. There is no learned flow field, image/language context, training, or distribution shift.
\item The causal mechanism is endpoint/action chaining, not chunk-wise transformer attention. It deliberately excludes the explicit observation rollout used by the future-state baseline.
\item Future-state conditioning uses the same nominal dynamics as the planner, with plant mismatch but no learned predictor error.
\item Success thresholds are toy-specific. Follow-ups should measure an actual denoiser, vary buffer span independently, and evaluate contact-rich or constrained dynamics.
\end{{itemize}}

\begin{{thebibliography}}{{9}}
\bibitem{{primary}} Z. Li, J. Tang, and Z. Liu. \emph{{Streaming Action Decoding for Fast and Asynchronous VLA Inference}}. arXiv:2608.27384, 2026. \url{{https://arxiv.org/abs/2608.27384}}
\bibitem{{code}} Z-Lab. \emph{{FlashVLA official implementation}}, inspected at commit \texttt{{5227b039ebd4}}. \url{{https://github.com/z-lab/flashvla}}
\end{{thebibliography}}
\end{{document}}
"""
    (docs_dir / "streaming_action_denoising_report.tex").write_text(tex, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--trials", type=int, default=48)
    parser.add_argument("--episode-steps", type=int, default=240)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--docs-dir", type=pathlib.Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--smoke", action="store_true", help="Run a short one-condition verification without writing the report.")
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(seed=args.seed, trials=args.trials, episode_steps=args.episode_steps)
    if args.smoke:
        cfg = replace(cfg, trials=min(args.trials, 3), episode_steps=min(args.episode_steps, 96), success_tail_steps=32)
        latency_scales = [1.0]
        disturbance_scales = [1.0]
    else:
        latency_scales = [1.0, 1.5, 2.0]
        disturbance_scales = [0.5, 1.0, 1.8]

    output_dir = args.output_dir.resolve()
    docs_dir = args.docs_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    sanity = run_sanity_checks(cfg)
    if not sanity["all_pass"]:
        raise RuntimeError(f"sanity checks failed: {sanity}")

    trial_rows: list[dict[str, float | int | str]] = []
    representative: dict[str, dict[str, np.ndarray]] = {}
    for latency_scale in latency_scales:
        for disturbance_scale in disturbance_scales:
            for trial in range(cfg.trials):
                trial_seed = stable_seed(cfg.seed, trial, latency_scale, disturbance_scale)
                for method in METHOD_ORDER:
                    keep = latency_scale == 1.0 and disturbance_scale == 1.0 and trial == 0
                    metric, trajectory = simulate_trial(method, cfg, trial_seed, latency_scale, disturbance_scale, keep)
                    trial_rows.append(metric)
                    if keep and trajectory is not None:
                        representative[method] = trajectory

    summary = aggregate(trial_rows)
    systems = [systems_row(m, cfg, scale) for scale in latency_scales for m in METHOD_ORDER]
    write_csv(output_dir / "trial_metrics.csv", trial_rows)
    write_csv(output_dir / "summary_metrics.csv", summary)
    write_csv(output_dir / "systems_metrics.csv", systems)
    (output_dir / "sanity_checks.json").write_text(json.dumps(sanity, indent=2, sort_keys=True), encoding="utf-8")

    if set(representative) == set(METHOD_ORDER):
        make_plots(cfg, summary, systems, representative, output_dir)

    command = (
        "/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python "
        "streaming_action_denoising/run_streaming_action_denoising.py "
        f"--seed {cfg.seed} --trials {cfg.trials} --episode-steps {cfg.episode_steps}"
    )
    report_command = (
        "cd /home/andypark/Projects/repos/vla-ideas\n"
        "/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \\\n"
        "  streaming_action_denoising/run_streaming_action_denoising.py \\\n"
        f"  --seed {cfg.seed} --trials {cfg.trials} --episode-steps {cfg.episode_steps}"
    )
    metrics_json = {
        "experiment": "streaming_action_denoising",
        "generated_date": "2026-08-31",
        "command": command,
        "config": asdict(cfg),
        "latency_scales": latency_scales,
        "disturbance_scales": disturbance_scales,
        "method_labels": LABELS,
        "default_condition": {m: get_summary(summary, m, 1.0, 1.0) for m in METHOD_ORDER},
        "stress_condition": ({m: get_summary(summary, m, 2.0, 1.8) for m in METHOD_ORDER} if not args.smoke else None),
        "systems_default": {m: next(r for r in systems if r["method"] == m and math.isclose(float(r["latency_scale"]), 1.0)) for m in METHOD_ORDER},
        "claims_boundary": {
            "timing": "Configured fixed-cost scheduling proxy; not wall-clock hardware measurement.",
            "causal": "Endpoint/action continuation surrogate; not learned chunk-wise attention.",
            "handoff_compensation": "Separate analytical rollout/blend diagnostic; not attributed to FlashVLA's decoder mechanism.",
            "distillation": "Deterministic few-refinement quality proxy; not a trained distilled policy.",
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2, sort_keys=True), encoding="utf-8")

    if not args.smoke and not args.no_report:
        write_tex_report(cfg, summary, systems, docs_dir, report_command)

    print(f"wrote {len(trial_rows)} trial rows and {len(summary)} aggregate rows to {output_dir}")
    for method in METHOD_ORDER:
        row = get_summary(summary, method, 1.0, 1.0)
        sysrow = next(r for r in systems if r["method"] == method and math.isclose(float(r["latency_scale"]), 1.0))
        print(
            f"{method:26s} success={100*float(row['success_mean']):5.1f}% "
            f"track={float(row['tracking_rmse_mean']):.3f} boundary={float(row['boundary_discontinuity_mean']):.3f} "
            f"stale={float(row['handoff_staleness_mean']):.3f} throughput={float(sysrow['steady_chunk_throughput_hz']):.2f} chunks/s"
        )


if __name__ == "__main__":
    main()
