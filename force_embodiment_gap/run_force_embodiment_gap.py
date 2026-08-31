#!/usr/bin/env python3
"""Deterministic behavior-cloning toy for force and embodiment matching.

Two contact-rich tasks intentionally alias visually while contact force changes:
a stuck fastener and a tight insertion. Five BC conditions compare current
vision, short temporal history, current force, force demonstrations collected
with a mismatched handheld gripper, and a matched/co-calibrated multimodal
interface. This is a low-dimensional mechanism probe, not a robot benchmark or
a reproduction of RAI Institute's LBMs/Koala system.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

TASKS = ("fastener", "insertion")
METHODS = (
    "vision_only",
    "temporal_history",
    "force",
    "mismatched_gripper_force",
    "matched_multimodal_bc",
)
LABELS = {
    "vision_only": "Vision only",
    "temporal_history": "Temporal history",
    "force": "Vision + force",
    "mismatched_gripper_force": "Mismatched-gripper force",
    "matched_multimodal_bc": "Matched multimodal BC",
}
COLORS = {
    "vision_only": "#9c755f",
    "temporal_history": "#f2cf5b",
    "force": "#4c78a8",
    "mismatched_gripper_force": "#e45756",
    "matched_multimodal_bc": "#54a24b",
}


@dataclass(frozen=True)
class Config:
    seed: int = 31
    train_episodes: int = 180
    validation_episodes: int = 45
    eval_episodes: int = 120
    sweep_episodes: int = 32
    max_steps: int = 64
    history_len: int = 4
    n_estimators: int = 48
    max_depth: int = 16
    min_samples_leaf: int = 3
    command_axial_limit: float = 1.75
    command_lateral_limit: float = 1.25
    approach_start: float = -0.24
    success_progress: float = 1.0
    fastener_damage_margin: float = 0.44
    insertion_damage_margin: float = 0.38


@dataclass(frozen=True)
class Hardware:
    name: str
    axial_gain: float
    lateral_gain: float
    sensor_matrix: tuple[tuple[float, float], tuple[float, float]]
    sensor_bias: tuple[float, float]

    def sense(self, physical_force: np.ndarray) -> np.ndarray:
        return np.asarray(self.sensor_matrix, dtype=float) @ physical_force + np.asarray(
            self.sensor_bias, dtype=float
        )

    def calibrate(self, raw_force: np.ndarray) -> np.ndarray:
        return np.linalg.solve(
            np.asarray(self.sensor_matrix, dtype=float),
            raw_force - np.asarray(self.sensor_bias, dtype=float),
        )

    def motor_to_physical(self, command: np.ndarray) -> np.ndarray:
        return command * np.array([self.axial_gain, self.lateral_gain], dtype=float)

    def physical_to_motor(self, effort: np.ndarray) -> np.ndarray:
        return effort / np.array([self.axial_gain, self.lateral_gain], dtype=float)


@dataclass(frozen=True)
class Scenario:
    task: str
    threshold: float
    initial_alignment: float
    seed: int


@dataclass
class State:
    progress: float
    alignment: float
    axial_force: float
    lateral_force: float
    released: bool = False
    damaged: bool = False
    success: bool = False


@dataclass
class Sample:
    visual: np.ndarray
    raw_force: np.ndarray
    calibrated_force: np.ndarray
    previous_motor: np.ndarray
    previous_physical: np.ndarray
    history: np.ndarray
    motor_target: np.ndarray
    physical_target: np.ndarray


@dataclass
class EpisodeResult:
    trial: int
    task: str
    method: str
    success: bool
    damaged: bool
    stuck: bool
    steps: int
    final_progress: float
    final_alignment: float
    peak_axial_force: float
    peak_force_norm: float
    force_overshoot: float
    contact_steps: int
    command_energy: float
    action_smoothness: float
    trace: dict[str, list[float]] | None = None


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def nominal_hardware() -> Hardware:
    return Hardware("matched_target", 1.0, 1.0, ((1.0, 0.0), (0.0, 1.0)), (0.0, 0.0))


def source_handheld_gripper() -> Hardware:
    # Deliberately different lever arm, lateral transmission, force frame, and zero.
    return Hardware(
        "mismatched_handheld_source",
        0.72,
        1.28,
        ((0.76, 0.18), (-0.11, 1.24)),
        (0.08, -0.055),
    )


def calibration_hardware(scale: float) -> Hardware:
    cross = 0.14 * (scale - 1.0)
    lateral_scale = 1.0 + 0.55 * (scale - 1.0)
    return Hardware(
        f"calibration_{scale:.2f}",
        1.0,
        1.0,
        ((scale, cross), (-0.75 * cross, lateral_scale)),
        (0.10 * (scale - 1.0), -0.07 * (scale - 1.0)),
    )


def morphology_hardware(scale: float) -> Hardware:
    return Hardware(
        f"morphology_{scale:.2f}",
        scale,
        1.0 / scale,
        ((1.0, 0.0), (0.0, 1.0)),
        (0.0, 0.0),
    )


def scenario_for(task: str, seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    if task == "fastener":
        return Scenario(task, float(rng.uniform(0.72, 1.12)), 0.0, seed)
    sign = float(rng.choice([-1.0, 1.0]))
    alignment = sign * float(rng.uniform(0.13, 0.29))
    return Scenario(task, float(rng.uniform(0.58, 0.88)), alignment, seed)


def initial_state(sc: Scenario, cfg: Config) -> State:
    return State(cfg.approach_start, sc.initial_alignment, 0.0, 0.0)


def visual_observation(state: State, task: str, previous_progress: float) -> np.ndarray:
    # Sign is hidden in insertion and contact progress is exactly flat while stuck.
    progress = float(np.clip(state.progress, -0.25, 1.0))
    alignment_magnitude = round(abs(state.alignment) / 0.04) * 0.04
    moving = float(state.progress > previous_progress + 1e-8)
    return np.array(
        [
            float(task == "fastener"),
            float(task == "insertion"),
            progress,
            alignment_magnitude,
            moving,
            float(progress >= 0.0),
        ],
        dtype=float,
    )


def expert_physical_action(state: State, sc: Scenario) -> np.ndarray:
    if state.progress < 0.0:
        return np.array([0.78, -2.2 * state.lateral_force if sc.task == "insertion" else 0.0])
    if state.released:
        lateral = -1.8 * state.lateral_force if sc.task == "insertion" else 0.0
        return np.array([0.68, float(np.clip(lateral, -0.9, 0.9))])

    if sc.task == "fastener":
        axial = max(0.38, state.axial_force + 0.145)
        return np.array([min(axial, 1.38), 0.0])

    lateral = float(np.clip(-2.9 * state.lateral_force, -1.05, 1.05))
    if abs(state.lateral_force) > 0.16:
        axial = max(0.34, min(0.58, state.axial_force + 0.06))
    else:
        axial = max(0.38, state.axial_force + 0.125)
    return np.array([min(axial, 1.25), lateral])


def step_dynamics(
    state: State, command: np.ndarray, sc: Scenario, hw: Hardware, cfg: Config
) -> State:
    command = np.array(
        [
            np.clip(command[0], 0.0, cfg.command_axial_limit),
            np.clip(command[1], -cfg.command_lateral_limit, cfg.command_lateral_limit),
        ],
        dtype=float,
    )
    effort = hw.motor_to_physical(command)
    nxt = replace(state)

    if state.progress < 0.0:
        nxt.progress = min(0.0, state.progress + 0.082 * effort[0])
        nxt.alignment = float(np.clip(state.alignment + 0.035 * effort[1], -0.6, 0.6))
        nxt.axial_force = 0.0
        nxt.lateral_force = 1.7 * nxt.alignment if sc.task == "insertion" else 0.0
        return nxt

    if not state.released:
        nxt.alignment = float(np.clip(state.alignment + 0.052 * effort[1], -0.6, 0.6))
        contact_load = 0.10 * abs(nxt.alignment) if sc.task == "insertion" else 0.0
        nxt.axial_force = 0.68 * state.axial_force + 0.32 * max(0.0, effort[0] + contact_load)
        if sc.task == "insertion":
            nxt.lateral_force = 0.38 * state.lateral_force + 0.62 * (
                2.55 * nxt.alignment + 0.10 * effort[1]
            )
            effective_threshold = sc.threshold + 0.75 * abs(nxt.alignment)
            aligned = abs(nxt.alignment) < 0.058
            margin = cfg.insertion_damage_margin
        else:
            nxt.lateral_force = 0.0
            effective_threshold = sc.threshold
            aligned = True
            margin = cfg.fastener_damage_margin

        if nxt.axial_force > effective_threshold + margin:
            nxt.damaged = True
        elif aligned and nxt.axial_force >= effective_threshold:
            nxt.released = True
        return nxt

    nxt.axial_force = 0.64 * state.axial_force + 0.36 * (0.30 * effort[0])
    if sc.task == "insertion":
        nxt.alignment = float(np.clip(state.alignment + 0.035 * effort[1], -0.6, 0.6))
        nxt.lateral_force = 0.45 * state.lateral_force + 0.55 * (
            1.8 * nxt.alignment + 0.08 * effort[1]
        )
        speed = 0.076 * effort[0] * max(0.15, 1.0 - 2.2 * abs(nxt.alignment))
    else:
        nxt.lateral_force = 0.0
        speed = 0.080 * effort[0]
    nxt.progress = state.progress + speed
    if nxt.progress >= cfg.success_progress:
        nxt.progress = cfg.success_progress
        nxt.success = True
    return nxt


def temporal_feature(
    visual: np.ndarray, history_visual: list[np.ndarray], history_motor: list[np.ndarray], history_len: int
) -> np.ndarray:
    entries: list[np.ndarray] = []
    missing = history_len - len(history_visual)
    for _ in range(missing):
        entries.append(np.zeros(8, dtype=float))
    for vis, action in zip(history_visual[-history_len:], history_motor[-history_len:]):
        entries.append(np.concatenate([vis, action]))
    return np.concatenate([visual, *entries])


def sample_features(sample: Sample, method: str) -> np.ndarray:
    if method == "vision_only":
        return sample.visual
    if method == "temporal_history":
        return sample.history
    if method == "force":
        return np.concatenate([sample.visual, sample.raw_force])
    if method == "mismatched_gripper_force":
        return np.concatenate([sample.visual, sample.raw_force, sample.previous_motor])
    if method == "matched_multimodal_bc":
        return np.concatenate(
            [sample.visual, sample.calibrated_force, sample.previous_physical]
        )
    raise ValueError(method)


def collect_expert_episode(sc: Scenario, hw: Hardware, cfg: Config) -> list[Sample]:
    state = initial_state(sc, cfg)
    previous_progress = state.progress
    previous_motor = np.zeros(2, dtype=float)
    previous_physical = np.zeros(2, dtype=float)
    history_visual: list[np.ndarray] = []
    history_motor: list[np.ndarray] = []
    samples: list[Sample] = []
    for _ in range(cfg.max_steps):
        visual = visual_observation(state, sc.task, previous_progress)
        physical_force = np.array([state.axial_force, state.lateral_force], dtype=float)
        raw_force = hw.sense(physical_force)
        physical_target = expert_physical_action(state, sc)
        motor_target = hw.physical_to_motor(physical_target)
        hist = temporal_feature(visual, history_visual, history_motor, cfg.history_len)
        samples.append(
            Sample(
                visual,
                raw_force,
                hw.calibrate(raw_force),
                previous_motor.copy(),
                previous_physical.copy(),
                hist,
                motor_target,
                physical_target,
            )
        )
        history_visual.append(visual.copy())
        history_motor.append(motor_target.copy())
        previous_progress = state.progress
        previous_motor = motor_target.copy()
        previous_physical = physical_target.copy()
        state = step_dynamics(state, motor_target, sc, hw, cfg)
        if state.success or state.damaged:
            break
    return samples


def collect_dataset(
    hw: Hardware, cfg: Config, episodes: int, split: str
) -> list[Sample]:
    data: list[Sample] = []
    for task in TASKS:
        for i in range(episodes):
            sc = scenario_for(task, stable_seed(cfg.seed, split, hw.name, task, i))
            data.extend(collect_expert_episode(sc, hw, cfg))
    return data


class BCPolicy:
    def __init__(self, method: str, cfg: Config) -> None:
        self.method = method
        self.cfg = cfg
        self.model = ExtraTreesRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            max_features=1.0,
            random_state=stable_seed(cfg.seed, method),
            n_jobs=1,
        )
        self.validation_rmse = math.nan

    def fit(self, training: list[Sample], validation: list[Sample]) -> None:
        x = np.asarray([sample_features(s, self.method) for s in training])
        if self.method == "matched_multimodal_bc":
            y = np.asarray([s.physical_target for s in training])
        else:
            y = np.asarray([s.motor_target for s in training])
        self.model.fit(x, y)
        vx = np.asarray([sample_features(s, self.method) for s in validation])
        if self.method == "matched_multimodal_bc":
            vy = np.asarray([s.physical_target for s in validation])
        else:
            vy = np.asarray([s.motor_target for s in validation])
        prediction = self.model.predict(vx)
        self.validation_rmse = float(np.sqrt(np.mean((prediction - vy) ** 2)))

    def command(self, sample: Sample, hw: Hardware) -> np.ndarray:
        value = np.asarray(self.model.predict(sample_features(sample, self.method)[None, :])[0])
        if self.method == "matched_multimodal_bc":
            return hw.physical_to_motor(value)
        return value


def fit_policies(cfg: Config) -> dict[str, BCPolicy]:
    target = nominal_hardware()
    source = source_handheld_gripper()
    nominal_train = collect_dataset(target, cfg, cfg.train_episodes, "train")
    nominal_val = collect_dataset(target, cfg, cfg.validation_episodes, "validation")
    mismatch_train = collect_dataset(source, cfg, cfg.train_episodes, "train")
    mismatch_val = collect_dataset(source, cfg, cfg.validation_episodes, "validation")
    policies: dict[str, BCPolicy] = {}
    for method in METHODS:
        policy = BCPolicy(method, cfg)
        if method == "mismatched_gripper_force":
            policy.fit(mismatch_train, mismatch_val)
        else:
            policy.fit(nominal_train, nominal_val)
        policies[method] = policy
    return policies


def rollout(
    policy: BCPolicy,
    sc: Scenario,
    hw: Hardware,
    cfg: Config,
    trial: int,
    keep_trace: bool = False,
) -> EpisodeResult:
    state = initial_state(sc, cfg)
    previous_progress = state.progress
    previous_motor = np.zeros(2, dtype=float)
    previous_physical = np.zeros(2, dtype=float)
    history_visual: list[np.ndarray] = []
    history_motor: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    peak_axial = 0.0
    peak_norm = 0.0
    max_overshoot = 0.0
    contact_steps = 0
    trace: dict[str, list[float]] | None = None
    if keep_trace:
        trace = {k: [] for k in ("step", "progress", "alignment", "axial_force", "lateral_force", "axial_command", "lateral_command")}

    steps = 0
    for step in range(cfg.max_steps):
        visual = visual_observation(state, sc.task, previous_progress)
        physical_force = np.array([state.axial_force, state.lateral_force], dtype=float)
        raw_force = hw.sense(physical_force)
        hist = temporal_feature(visual, history_visual, history_motor, cfg.history_len)
        sample = Sample(
            visual,
            raw_force,
            hw.calibrate(raw_force),
            previous_motor.copy(),
            previous_physical.copy(),
            hist,
            np.zeros(2),
            np.zeros(2),
        )
        command = policy.command(sample, hw)
        command = np.array(
            [
                np.clip(command[0], 0.0, cfg.command_axial_limit),
                np.clip(command[1], -cfg.command_lateral_limit, cfg.command_lateral_limit),
            ]
        )
        actions.append(command)
        if state.progress >= 0.0 and not state.released:
            contact_steps += 1
        effective_threshold = sc.threshold + (
            0.75 * abs(state.alignment) if sc.task == "insertion" else 0.0
        )
        peak_axial = max(peak_axial, state.axial_force)
        peak_norm = max(peak_norm, float(np.hypot(state.axial_force, state.lateral_force)))
        max_overshoot = max(max_overshoot, state.axial_force - effective_threshold)
        if trace is not None:
            trace["step"].append(float(step))
            trace["progress"].append(float(state.progress))
            trace["alignment"].append(float(state.alignment))
            trace["axial_force"].append(float(state.axial_force))
            trace["lateral_force"].append(float(state.lateral_force))
            trace["axial_command"].append(float(command[0]))
            trace["lateral_command"].append(float(command[1]))
        history_visual.append(visual.copy())
        history_motor.append(command.copy())
        previous_progress = state.progress
        previous_motor = command.copy()
        previous_physical = hw.motor_to_physical(command)
        state = step_dynamics(state, command, sc, hw, cfg)
        # Include the state produced by the final command.  Damage/release can
        # occur on that transition, so sampling force only before stepping
        # understated peak force and threshold overshoot on terminal failures.
        terminal_threshold = sc.threshold + (
            0.75 * abs(state.alignment) if sc.task == "insertion" else 0.0
        )
        peak_axial = max(peak_axial, state.axial_force)
        peak_norm = max(peak_norm, float(np.hypot(state.axial_force, state.lateral_force)))
        max_overshoot = max(max_overshoot, state.axial_force - terminal_threshold)
        steps = step + 1
        if state.success or state.damaged:
            break

    if actions:
        arr = np.asarray(actions)
        energy = float(np.mean(np.sum(arr**2, axis=1)))
        smoothness = float(np.mean(np.sum(np.diff(arr, axis=0) ** 2, axis=1))) if len(arr) > 1 else 0.0
    else:
        energy = smoothness = 0.0
    stuck = not state.success and not state.damaged
    return EpisodeResult(
        trial,
        sc.task,
        policy.method,
        state.success,
        state.damaged,
        stuck,
        steps,
        float(state.progress),
        float(state.alignment),
        float(peak_axial),
        float(peak_norm),
        float(max(0.0, max_overshoot)),
        contact_steps,
        energy,
        smoothness,
        trace,
    )


def evaluate(
    policies: dict[str, BCPolicy], hw: Hardware, cfg: Config, episodes: int, tag: str
) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    for task in TASKS:
        for trial in range(episodes):
            sc = scenario_for(task, stable_seed(cfg.seed, "eval", tag, task, trial))
            for method in METHODS:
                results.append(rollout(policies[method], sc, hw, cfg, trial))
    return results


def aggregate(rows: list[EpisodeResult]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[EpisodeResult]]] = []
    for task in (*TASKS, "overall"):
        for method in METHODS:
            subset = [r for r in rows if r.method == method and (task == "overall" or r.task == task)]
            groups.append((task, method, subset))
    for task, method, subset in groups:
        success_steps = [r.steps for r in subset if r.success]
        output.append(
            {
                "task": task,
                "method": method,
                "episodes": len(subset),
                "success_rate": float(np.mean([r.success for r in subset])),
                "damage_rate": float(np.mean([r.damaged for r in subset])),
                "stuck_rate": float(np.mean([r.stuck for r in subset])),
                "mean_steps": float(np.mean([r.steps for r in subset])),
                "mean_steps_success": float(np.mean(success_steps)) if success_steps else None,
                "mean_final_progress": float(np.mean([r.final_progress for r in subset])),
                "mean_abs_final_alignment": float(np.mean([abs(r.final_alignment) for r in subset])),
                "mean_peak_axial_force": float(np.mean([r.peak_axial_force for r in subset])),
                "p95_peak_axial_force": float(np.percentile([r.peak_axial_force for r in subset], 95)),
                "mean_force_overshoot": float(np.mean([r.force_overshoot for r in subset])),
                "mean_contact_steps": float(np.mean([r.contact_steps for r in subset])),
                "mean_command_energy": float(np.mean([r.command_energy for r in subset])),
                "mean_action_smoothness": float(np.mean([r.action_smoothness for r in subset])),
            }
        )
    return output


def sweep(
    policies: dict[str, BCPolicy], cfg: Config, kind: str, values: Iterable[float]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in values:
        hw = calibration_hardware(value) if kind == "calibration" else morphology_hardware(value)
        episodes = evaluate(policies, hw, cfg, cfg.sweep_episodes, f"{kind}_{value:.3f}")
        for item in aggregate(episodes):
            if item["task"] == "overall":
                rows.append(
                    {
                        "sweep": kind,
                        "value": value,
                        "method": item["method"],
                        "episodes": item["episodes"],
                        "success_rate": item["success_rate"],
                        "damage_rate": item["damage_rate"],
                        "stuck_rate": item["stuck_rate"],
                        "mean_final_progress": item["mean_final_progress"],
                        "mean_peak_axial_force": item["mean_peak_axial_force"],
                        "mean_force_overshoot": item["mean_force_overshoot"],
                    }
                )
    return rows


def sanity_checks(policies: dict[str, BCPolicy], cfg: Config) -> dict[str, Any]:
    sc = Scenario("fastener", 0.95, 0.0, 0)
    low = State(0.0, 0.0, 0.25, 0.0)
    high = State(0.0, 0.0, 0.82, 0.0)
    vis_low = visual_observation(low, "fastener", 0.0)
    vis_high = visual_observation(high, "fastener", 0.0)
    act_low = expert_physical_action(low, sc)
    act_high = expert_physical_action(high, sc)
    alias_equal = bool(np.array_equal(vis_low, vis_high))
    action_gap = float(np.linalg.norm(act_high - act_low))

    test_force = np.array([0.83, -0.27])
    hw = source_handheld_gripper()
    recovered = hw.calibrate(hw.sense(test_force))
    calibration_error = float(np.max(np.abs(recovered - test_force)))

    desired = np.array([0.81, -0.34])
    physical_errors = []
    for scale in (0.7, 1.0, 1.3):
        mhw = morphology_hardware(scale)
        executed = mhw.motor_to_physical(mhw.physical_to_motor(desired))
        physical_errors.append(float(np.max(np.abs(executed - desired))))

    replay_sc = scenario_for("insertion", stable_seed(cfg.seed, "determinism"))
    first = rollout(policies["matched_multimodal_bc"], replay_sc, nominal_hardware(), cfg, 0, True)
    second = rollout(policies["matched_multimodal_bc"], replay_sc, nominal_hardware(), cfg, 0, True)
    deterministic = first.trace == second.trace and asdict(first) == asdict(second)

    checks = {
        "contact_alias": {
            "pass": alias_equal and action_gap > 0.4,
            "identical_visual_observation": alias_equal,
            "required_action_l2_gap": action_gap,
        },
        "force_calibration_inverse": {
            "pass": calibration_error < 1e-12,
            "max_abs_error": calibration_error,
        },
        "matched_action_coordinates": {
            "pass": max(physical_errors) < 1e-12,
            "max_physical_action_error": max(physical_errors),
        },
        "deterministic_replay": {"pass": deterministic},
    }
    checks["all_pass"] = all(v["pass"] for v in checks.values())
    return checks


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def episode_dict(row: EpisodeResult) -> dict[str, Any]:
    data = asdict(row)
    data.pop("trace")
    return data


def plot_summary(summary: list[dict[str, Any]]) -> None:
    overall = {r["method"]: r for r in summary if r["task"] == "overall"}
    x = np.arange(len(METHODS))
    colors = [COLORS[m] for m in METHODS]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].bar(x, [100 * overall[m]["success_rate"] for m in METHODS], color=colors)
    axes[0].set_ylabel("Success (%)")
    axes[0].set_ylim(0, 105)
    axes[1].bar(x, [100 * overall[m]["damage_rate"] for m in METHODS], color=colors)
    axes[1].set_ylabel("Damage failures (%)")
    axes[1].set_ylim(0, max(10, 110 * max(overall[m]["damage_rate"] for m in METHODS)))
    axes[2].bar(x, [overall[m]["mean_final_progress"] for m in METHODS], color=colors)
    axes[2].set_ylabel("Mean final progress")
    axes[2].axhline(1.0, color="black", lw=1, ls="--")
    for ax in axes:
        ax.set_xticks(x, [LABELS[m] for m in METHODS], rotation=28, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Matched force/action embodiment resolves contact aliasing in this toy")
    fig.tight_layout()
    fig.savefig(OUT / "method_summary.png", dpi=190)
    plt.close(fig)


def plot_sweep(rows: list[dict[str, Any]], kind: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for method in METHODS:
        subset = sorted([r for r in rows if r["method"] == method], key=lambda r: r["value"])
        values = [r["value"] for r in subset]
        axes[0].plot(values, [100 * r["success_rate"] for r in subset], marker="o", color=COLORS[method], label=LABELS[method])
        axes[1].plot(values, [100 * r["damage_rate"] for r in subset], marker="o", color=COLORS[method], label=LABELS[method])
    axes[0].set_ylabel("Success (%)")
    axes[1].set_ylabel("Damage failures (%)")
    for ax in axes:
        ax.set_xlabel("Sensor calibration scale" if kind == "calibration" else "Axial morphology gain (lateral gain = inverse)")
        ax.axvline(1.0, color="black", lw=1, ls="--")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2, loc="best")
    fig.suptitle(f"{kind.capitalize()} sweep")
    fig.tight_layout()
    fig.savefig(OUT / f"{kind}_sweep.png", dpi=190)
    plt.close(fig)


def plot_representative(policies: dict[str, BCPolicy], cfg: Config) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 8.5), sharex="col")
    hw = nominal_hardware()
    for col, task in enumerate(TASKS):
        sc = scenario_for(task, stable_seed(cfg.seed, "representative", task))
        for method in METHODS:
            result = rollout(policies[method], sc, hw, cfg, 0, True)
            assert result.trace is not None
            t = result.trace["step"]
            axes[0, col].plot(t, result.trace["progress"], color=COLORS[method], label=LABELS[method])
            axes[1, col].plot(t, result.trace["axial_force"], color=COLORS[method])
            quantity = result.trace["alignment"] if task == "insertion" else result.trace["axial_command"]
            axes[2, col].plot(t, quantity, color=COLORS[method])
        axes[0, col].set_title("Stuck fastener" if task == "fastener" else "Tight insertion")
        axes[0, col].set_ylabel("Progress")
        axes[1, col].set_ylabel("Axial force")
        axes[2, col].set_ylabel("Axial command" if task == "fastener" else "Signed alignment")
        axes[2, col].set_xlabel("Step")
        for row in range(3):
            axes[row, col].grid(alpha=0.22)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Representative paired rollouts")
    fig.tight_layout()
    fig.savefig(OUT / "representative_rollouts.png", dpi=190)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--train-episodes", type=int, default=180)
    parser.add_argument("--validation-episodes", type=int, default=45)
    parser.add_argument("--eval-episodes", type=int, default=120)
    parser.add_argument("--sweep-episodes", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        seed=args.seed,
        train_episodes=36 if args.smoke else args.train_episodes,
        validation_episodes=12 if args.smoke else args.validation_episodes,
        eval_episodes=14 if args.smoke else args.eval_episodes,
        sweep_episodes=8 if args.smoke else args.sweep_episodes,
        n_estimators=20 if args.smoke else Config.n_estimators,
        max_depth=13 if args.smoke else Config.max_depth,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    policies = fit_policies(cfg)
    nominal_rows = evaluate(policies, nominal_hardware(), cfg, cfg.eval_episodes, "nominal")
    summary = aggregate(nominal_rows)
    calibration_values = (0.70, 0.85, 1.00, 1.15, 1.30)
    morphology_values = (0.70, 0.85, 1.00, 1.15, 1.30)
    calibration_rows = sweep(policies, cfg, "calibration", calibration_values)
    morphology_rows = sweep(policies, cfg, "morphology", morphology_values)
    checks = sanity_checks(policies, cfg)
    if not checks["all_pass"]:
        raise RuntimeError(f"sanity check failure: {checks}")

    for row in summary:
        row["validation_action_rmse"] = policies[row["method"]].validation_rmse
    write_csv(OUT / "summary_metrics.csv", summary)
    write_csv(OUT / "trial_metrics.csv", [episode_dict(r) for r in nominal_rows])
    write_csv(OUT / "calibration_sweep.csv", calibration_rows)
    write_csv(OUT / "morphology_sweep.csv", morphology_rows)
    with (OUT / "sanity_checks.json").open("w") as f:
        json.dump(checks, f, indent=2, sort_keys=True)
    payload = {
        "config": asdict(cfg),
        "run_mode": "smoke" if args.smoke else "full",
        "method_definitions": {
            "vision_only": "current quantized visual state; raw motor-action BC",
            "temporal_history": "current visual state plus four previous visual/action pairs",
            "force": "current visual state plus raw two-axis force; matched nominal demonstrations",
            "mismatched_gripper_force": "force/proprio BC trained on a different handheld gripper and transferred without calibration",
            "matched_multimodal_bc": "visual, calibrated force, and previous physical action in a shared matched action frame",
        },
        "validation_action_rmse": {m: policies[m].validation_rmse for m in METHODS},
        "nominal_summary": summary,
        "calibration_sweep": calibration_rows,
        "morphology_sweep": morphology_rows,
        "sanity_checks": checks,
    }
    with (OUT / "metrics.json").open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)

    plot_summary(summary)
    plot_sweep(calibration_rows, "calibration")
    plot_sweep(morphology_rows, "morphology")
    plot_representative(policies, cfg)

    overall = {r["method"]: r for r in summary if r["task"] == "overall"}
    print("Force embodiment gap toy complete")
    for method in METHODS:
        print(
            f"{method:28s} success={overall[method]['success_rate']:.3f} "
            f"damage={overall[method]['damage_rate']:.3f} "
            f"progress={overall[method]['mean_final_progress']:.3f} "
            f"val_rmse={policies[method].validation_rmse:.4f}"
        )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
