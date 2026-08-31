#!/usr/bin/env python3
"""Matched force-feedback demonstration quality toy.

This deterministic simulation is inspired by PHABS, but it is not a
reproduction of the device, pilot, or planned downstream policy study. It asks
a prospective mechanism question: if the same demonstrators perform the same
bimanual fragile-object task with haptic feedback on versus off, can cleaner
force targets improve a simple imitation policy under object-property shift?
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Iterable

BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    "pose_only_visual",
    "force_annotated_visual",
    "haptic_force_visual_motion",
    "passive_noisy_force",
    "haptic_force_rich",
)
LABELS = {
    "pose_only_visual": "Pose-only / visual demos",
    "force_annotated_visual": "Force-annotated / visual demos",
    "haptic_force_visual_motion": "Haptic force, visual motion",
    "passive_noisy_force": "Noisy passive-force labels",
    "haptic_force_rich": "Haptic force-rich demos",
}
COLORS = {
    "pose_only_visual": "#7f7f7f",
    "force_annotated_visual": "#f58518",
    "haptic_force_visual_motion": "#eeca3b",
    "passive_noisy_force": "#b279a2",
    "haptic_force_rich": "#54a24b",
}
SHIFT_LEVELS = (0.0, 0.33, 0.67, 1.0)


@dataclass(frozen=True)
class Config:
    seed: int = 37
    demo_scenarios: int = 72
    demo_repeats: int = 2
    eval_trials_per_shift: int = 180
    horizon: int = 84
    ridge: float = 0.08
    max_retries: int = 3
    shift_levels: tuple[float, ...] = SHIFT_LEVELS


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    mass: float
    friction: float
    fragility: float
    tightness: float
    alignment: float
    visual_mass: float
    visual_friction: float
    visual_fragility: float
    visual_tightness: float


@dataclass
class Demo:
    scenario: Scenario
    repeat: int
    feedback: str
    phase: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    oracle_force: np.ndarray
    rollout: dict[str, float]


class RidgePolicy:
    """One identical multi-output ridge regressor for every data condition."""

    def __init__(self, ridge: float):
        self.ridge = ridge
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgePolicy":
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0) + 1.0e-6
        z = (x - self.mean) / self.scale
        z = np.column_stack([np.ones(len(z)), z])
        penalty = np.eye(z.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        self.weights = np.linalg.solve(z.T @ z + penalty, z.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None or self.weights is None:
            raise RuntimeError("Policy must be fitted before prediction")
        z = (x - self.mean) / self.scale
        z = np.column_stack([np.ones(len(z)), z])
        return z @ self.weights


def stable_seed(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32)


def smoothstep(x: np.ndarray | float, lo: float, hi: float) -> np.ndarray:
    u = np.clip((np.asarray(x, dtype=float) - lo) / (hi - lo), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def load_shares(phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nominal bimanual handoff/insertion load allocation."""
    p = np.asarray(phase)
    left = np.zeros_like(p)
    right = np.zeros_like(p)

    left += ((p >= 0.14) & (p < 0.36)).astype(float)
    handoff = (p >= 0.36) & (p < 0.58)
    h = smoothstep(p[handoff], 0.36, 0.58)
    left[handoff] = 1.0 - 0.86 * h
    right[handoff] = 0.12 + 0.88 * h
    right += ((p >= 0.58) & (p < 0.75)).astype(float)
    insert = (p >= 0.75) & (p < 0.92)
    left[insert] = 0.34
    right[insert] = 0.66
    release = (p >= 0.92) & (p <= 1.0)
    r = 1.0 - smoothstep(p[release], 0.92, 0.985)
    left[release] = 0.34 * r
    right[release] = 0.66 * r
    return left, right


def contact_mask(phase: np.ndarray) -> np.ndarray:
    left, right = load_shares(phase)
    return np.column_stack([left > 0.01, right > 0.01])


def force_requirements(s: Scenario, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_share, right_share = load_shares(phase)
    shares = np.column_stack([left_share, right_share])
    # A compact normal-force proxy: heavier and lower-friction objects need more
    # squeeze; contact floor keeps a hand attached during low-load overlap.
    base = 0.52 * s.mass * 9.81 / max(s.friction, 0.25)
    required = base * shares + 0.22 * (shares > 0.01)
    insertion = ((phase >= 0.75) & (phase < 0.92)).astype(float)[:, None]
    safe = s.fragility * (1.0 - insertion * (0.10 + 0.15 * s.tightness))
    safe = np.repeat(safe, 2, axis=1)
    target = required + 0.22 + 0.10 * s.tightness
    # When no feasible no-slip/no-damage force exists, use the least-bad midpoint.
    target = np.minimum(target, 0.82 * safe)
    target *= shares > 0.01
    return required, safe, target


def scenario_from_rng(scenario_id: int, shift: float, rng: np.random.Generator) -> Scenario:
    # Shift simultaneously raises mass/tightness, lowers friction/fragility, and
    # increases visual property-estimation error.
    mass = rng.uniform(0.36 + 0.10 * shift, 0.62 + 0.18 * shift)
    friction = rng.uniform(0.58 - 0.17 * shift, 0.86 - 0.10 * shift)
    fragility = rng.uniform(7.2 - 2.0 * shift, 10.4 - 1.5 * shift)
    tightness = rng.uniform(0.22 + 0.20 * shift, 0.64 + 0.28 * shift)
    alignment = rng.normal(0.0, 0.035 + 0.075 * shift)
    noise = 0.025 + 0.085 * shift
    return Scenario(
        scenario_id=scenario_id,
        mass=float(mass),
        friction=float(friction),
        fragility=float(fragility),
        tightness=float(tightness),
        alignment=float(alignment),
        visual_mass=float(mass * (1.0 + rng.normal(0.0, noise))),
        visual_friction=float(friction * (1.0 + rng.normal(0.0, noise * 1.15))),
        visual_fragility=float(fragility * (1.0 + rng.normal(0.0, noise * 1.25))),
        visual_tightness=float(np.clip(tightness + rng.normal(0.0, noise), 0.0, 1.25)),
    )


def make_scenarios(count: int, shift: float, seed: int, split: str) -> list[Scenario]:
    return [
        scenario_from_rng(
            i,
            shift,
            np.random.default_rng(stable_seed(seed, split, shift, i)),
        )
        for i in range(count)
    ]


def observation_features(s: Scenario, phase: np.ndarray) -> np.ndarray:
    p = np.asarray(phase)
    left, right = load_shares(p)
    insertion = ((p >= 0.75) & (p < 0.92)).astype(float)
    handoff = ((p >= 0.36) & (p < 0.58)).astype(float)
    return np.column_stack(
        [
            p,
            p**2,
            p**3,
            np.sin(np.pi * p),
            np.cos(np.pi * p),
            np.sin(2.0 * np.pi * p),
            np.cos(2.0 * np.pi * p),
            left,
            right,
            handoff,
            insertion,
            np.full_like(p, s.visual_mass),
            np.full_like(p, s.visual_friction),
            np.full_like(p, s.visual_fragility),
            np.full_like(p, s.visual_tightness),
            np.full_like(p, s.alignment),
            left * s.visual_mass / max(s.visual_friction, 0.2),
            right * s.visual_mass / max(s.visual_friction, 0.2),
            insertion * s.visual_tightness,
            insertion * abs(s.alignment),
        ]
    )


def default_pose_force(phase: np.ndarray) -> np.ndarray:
    left, right = load_shares(phase)
    # A common force-blind controller: one generic squeeze level gated by pose.
    return 5.35 * np.column_stack([left > 0.01, right > 0.01])


def rollout_metrics(
    s: Scenario,
    phase: np.ndarray,
    actions: np.ndarray,
    rng: np.random.Generator,
    max_retries: int,
) -> dict[str, float]:
    speed = np.clip(actions[:, 0], 0.30, 1.35)
    force = np.clip(actions[:, 1:3], 0.0, 12.5)
    balance = np.clip(actions[:, 3], -1.0, 1.0)
    required, safe, oracle = force_requirements(s, phase)
    active = contact_mask(phase)
    n_active = max(int(active.sum()), 1)

    deficit = np.maximum(required - force, 0.0) * active
    excess = np.maximum(force - safe, 0.0) * active
    force_error = (force - oracle) * active
    force_rmse = float(np.sqrt(np.sum(force_error**2) / n_active))
    force_error_variance = float(np.var(force_error[active]))
    excess_force = float(np.sum(excess) / n_active)

    handoff = (phase >= 0.36) & (phase < 0.58)
    insertion = (phase >= 0.75) & (phase < 0.92)
    left, right = load_shares(phase)
    desired_balance = right - left
    balance_error = np.abs(balance - desired_balance)
    handoff_balance_error = float(np.mean(balance_error[handoff]))

    deficit_ratio = float(np.sum(deficit) / (np.sum(required * active) + 1.0e-6))
    low_force_run = 0
    longest_low_force = 0
    for bad in np.any(deficit > (0.18 + 0.08 * s.mass), axis=1):
        low_force_run = low_force_run + 1 if bad else 0
        longest_low_force = max(longest_low_force, low_force_run)

    disturbance = float(rng.normal(0.0, 0.05))
    slip_score = (
        4.8 * deficit_ratio
        + 0.95 * handoff_balance_error
        + 0.055 * longest_low_force
        + 0.22 * max(0.0, -disturbance)
        - 0.62
    )
    slip_probability = float(1.0 / (1.0 + math.exp(-4.0 * slip_score)))
    slip_drop = float(rng.random() < slip_probability)

    excess_peak = float(np.max(excess))
    damage_load = 0.095 * float(np.sum(excess)) + 0.42 * excess_peak
    damage_probability = float(1.0 - math.exp(-max(damage_load - 0.10, 0.0)))
    object_damage = float(rng.random() < damage_probability)

    insertion_speed = float(np.mean(speed[insertion]))
    insertion_balance = float(np.mean(balance_error[insertion]))
    force_insert_error = float(np.sqrt(np.mean(force_error[insertion] ** 2)))
    ideal_speed = 0.88 - 0.25 * s.tightness
    alignment_residual = abs(s.alignment) + 0.14 * insertion_balance
    retry_score = (
        2.5 * max(insertion_speed - ideal_speed, 0.0)
        + 1.8 * alignment_residual
        + 0.20 * force_insert_error
        + 0.36 * handoff_balance_error
    )
    retries = int(np.clip(math.floor(retry_score + rng.uniform(0.0, 0.85)), 0, max_retries + 1))
    jam_probability = float(np.clip(0.10 + 0.34 * retry_score + 0.10 * s.tightness, 0.0, 0.94))
    insertion_failure = float(rng.random() < jam_probability and retries > max_retries - 1)
    success = float(not slip_drop and not object_damage and not insertion_failure and retries <= max_retries)

    return {
        "success": success,
        "object_damage": object_damage,
        "slip_drop": slip_drop,
        "insertion_failure": insertion_failure,
        "excess_force": excess_force,
        "force_profile_rmse": force_rmse,
        "force_error_variance": force_error_variance,
        "retries": float(retries),
        "handoff_balance_error": handoff_balance_error,
        "mean_speed": float(np.mean(speed)),
        "insertion_speed": insertion_speed,
        "slip_probability": slip_probability,
        "damage_probability": damage_probability,
    }


def generate_demo(s: Scenario, repeat: int, feedback: str, cfg: Config) -> Demo:
    # Common random numbers pair the same synthetic operator/repeat across
    # feedback conditions; only the feedback model changes its response.
    rng = np.random.default_rng(stable_seed(cfg.seed, "demo", s.scenario_id, repeat, "matched_operator"))
    phase = np.linspace(0.0, 1.0, cfg.horizon)
    x = observation_features(s, phase)
    required, safe, oracle = force_requirements(s, phase)
    active = contact_mask(phase)
    insertion = (phase >= 0.75) & (phase < 0.92)
    handoff = (phase >= 0.36) & (phase < 0.58)
    left, right = load_shares(phase)
    desired_balance = right - left

    if feedback == "haptic_on":
        target = oracle + 0.30 * active
        noise_scale = 0.17
        lag = 0.25
        overgrip = 0.04
        speed = 1.00 - 0.35 * insertion * s.tightness
        speed -= 0.04 * handoff
        balance_noise = 0.035
    elif feedback == "haptic_off":
        # Visual-only operators estimate load imperfectly, adopt a larger generic
        # safety margin, and correct contact later with larger oscillations.
        est_base = 0.52 * s.visual_mass * 9.81 / max(s.visual_friction, 0.25)
        shares = np.column_stack([left, right])
        target = est_base * shares + 0.92 * active
        target = np.minimum(target, 8.8) * active
        noise_scale = 0.48
        lag = 0.63
        overgrip = 0.48
        speed = 0.90 - 0.20 * insertion * s.visual_tightness
        speed -= 0.12 * handoff
        balance_noise = 0.12
    else:
        raise ValueError(feedback)

    force = np.zeros((cfg.horizon, 2), dtype=float)
    ar = rng.normal(0.0, noise_scale, size=2)
    for t in range(1, cfg.horizon):
        ar = 0.74 * ar + rng.normal(0.0, noise_scale * 0.55, size=2)
        contact_pulse = 0.0
        if feedback == "haptic_off" and (14 <= t <= 22 or 31 <= t <= 49):
            contact_pulse = 0.28 * math.sin(0.92 * t + repeat)
        desired = target[t] + overgrip * active[t] + ar + contact_pulse
        # Haptic feedback corrects from the actual required/safe envelope.
        if feedback == "haptic_on":
            desired += 0.38 * (oracle[t] - force[t - 1])
            desired = np.minimum(desired, 0.94 * safe[t])
        force[t] = lag * force[t - 1] + (1.0 - lag) * desired
        force[t] *= active[t]
    force = np.clip(force, 0.0, 12.5)

    speed += rng.normal(0.0, 0.025 if feedback == "haptic_on" else 0.065, cfg.horizon)
    speed = np.clip(speed, 0.35, 1.25)
    balance = desired_balance + rng.normal(0.0, balance_noise, cfg.horizon)
    actions = np.column_stack([speed, force, balance])
    rollout = rollout_metrics(
        s,
        phase,
        actions,
        np.random.default_rng(stable_seed(cfg.seed, "demo_rollout", s.scenario_id, repeat, "matched_operator")),
        cfg.max_retries,
    )
    return Demo(s, repeat, feedback, phase, x, actions, oracle, rollout)


def passive_force_labels(force: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logged = np.roll(force, 1, axis=0)
    logged[:1] = 0.0
    bias = rng.normal(0.0, 0.26, size=(1, 2))
    noise = rng.normal(0.0, 0.48, size=force.shape)
    logged = 0.94 * logged + bias + noise
    dropout = rng.random(force.shape) < 0.030
    logged[dropout] = 0.0
    return np.clip(logged, 0.0, 12.5)


def training_arrays(demos: list[Demo], method: str, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    selected_feedback = "haptic_on" if method == "haptic_force_rich" else "haptic_off"
    selected = [d for d in demos if d.feedback == selected_feedback]
    haptic_pairs = {
        (d.scenario.scenario_id, d.repeat): d
        for d in demos
        if d.feedback == "haptic_on"
    }
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for d in selected:
        action = d.actions.copy()
        if method == "pose_only_visual":
            action[:, 1:3] = default_pose_force(d.phase)
        elif method == "haptic_force_visual_motion":
            # Crossed ablation: preserve the visual-only speed/balance targets and
            # observations, but substitute force from the exactly matched haptic
            # rollout. This isolates force-target quality from motion timing.
            pair = haptic_pairs[(d.scenario.scenario_id, d.repeat)]
            if not np.array_equal(d.observations, pair.observations):
                raise AssertionError("Matched visual/haptic observations must be identical")
            action[:, 1:3] = pair.actions[:, 1:3]
        elif method == "passive_noisy_force":
            action[:, 1:3] = passive_force_labels(
                action[:, 1:3], stable_seed(cfg.seed, "passive", d.scenario.scenario_id, d.repeat)
            )
        xs.append(d.observations)
        ys.append(action)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def evaluate_policy(
    policy: RidgePolicy,
    method: str,
    scenarios: list[Scenario],
    shift: float,
    cfg: Config,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    representative: dict[str, np.ndarray] = {}
    phase = np.linspace(0.0, 1.0, cfg.horizon)
    for i, s in enumerate(scenarios):
        actions = policy.predict(observation_features(s, phase))
        actions[:, 0] = np.clip(actions[:, 0], 0.30, 1.35)
        actions[:, 1:3] = np.clip(actions[:, 1:3], 0.0, 12.5)
        actions[:, 3] = np.clip(actions[:, 3], -1.0, 1.0)
        metrics = rollout_metrics(
            s,
            phase,
            actions,
            np.random.default_rng(stable_seed(cfg.seed, "eval", shift, i)),
            cfg.max_retries,
        )
        rows.append(
            {
                "method": method,
                "shift": shift,
                "trial": i,
                "mass": s.mass,
                "friction": s.friction,
                "fragility": s.fragility,
                "tightness": s.tightness,
                **metrics,
            }
        )
        if i == min(7, len(scenarios) - 1):
            required, safe, oracle = force_requirements(s, phase)
            representative = {
                "phase": phase,
                "actions": actions,
                "required": required,
                "safe": safe,
                "oracle": oracle,
            }
    return rows, representative


def mean_dict(rows: Iterable[dict[str, Any]], keys: Iterable[str]) -> dict[str, float]:
    items = list(rows)
    return {key: float(np.mean([float(r[key]) for r in items])) for key in keys}


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_sanity(cfg: Config, output_dir: pathlib.Path) -> dict[str, Any]:
    scenarios = make_scenarios(10, 0.0, cfg.seed, "sanity")
    off = [generate_demo(s, 0, "haptic_off", cfg) for s in scenarios]
    on = [generate_demo(s, 0, "haptic_on", cfg) for s in scenarios]
    duplicate = generate_demo(scenarios[0], 0, "haptic_on", cfg)
    off_rmse = float(np.mean([d.rollout["force_profile_rmse"] for d in off]))
    on_rmse = float(np.mean([d.rollout["force_profile_rmse"] for d in on]))
    off_var = float(np.mean([d.rollout["force_error_variance"] for d in off]))
    on_var = float(np.mean([d.rollout["force_error_variance"] for d in on]))
    checks = {
        "matched_scenario_count": bool(len(off) == len(on) == len(scenarios)),
        "matched_parameters_exact": bool(all(a.scenario == b.scenario for a, b in zip(off, on))),
        "matched_observations_exact": bool(
            all(np.array_equal(a.observations, b.observations) for a, b in zip(off, on))
        ),
        "deterministic_generation": bool(np.array_equal(on[0].actions, duplicate.actions)),
        "haptic_demo_force_rmse_lower": bool(on_rmse < 0.65 * off_rmse),
        "haptic_demo_force_variance_lower": bool(on_var < 0.60 * off_var),
        "haptic_demo_damage_not_higher": bool(
            np.mean([d.rollout["object_damage"] for d in on])
            <= np.mean([d.rollout["object_damage"] for d in off])
        ),
    }
    result = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "values": {
            "visual_force_rmse": off_rmse,
            "haptic_force_rmse": on_rmse,
            "visual_force_error_variance": off_var,
            "haptic_force_error_variance": on_var,
            "visual_mean_retries": float(np.mean([d.rollout["retries"] for d in off])),
            "haptic_mean_retries": float(np.mean([d.rollout["retries"] for d in on])),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sanity_check.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        failed = [k for k, v in checks.items() if not v]
        raise AssertionError(f"Sanity check failed: {failed}")
    return result


def demo_rows(demos: list[Demo]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d in demos:
        rows.append(
            {
                "scenario_id": d.scenario.scenario_id,
                "repeat": d.repeat,
                "feedback": d.feedback,
                "mass": d.scenario.mass,
                "friction": d.scenario.friction,
                "fragility": d.scenario.fragility,
                "tightness": d.scenario.tightness,
                **d.rollout,
            }
        )
    return rows


def aggregate_results(trials: list[dict[str, Any]], cfg: Config) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_keys = [
        "success",
        "object_damage",
        "slip_drop",
        "insertion_failure",
        "excess_force",
        "force_profile_rmse",
        "force_error_variance",
        "retries",
        "handoff_balance_error",
        "mean_speed",
    ]
    shift_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for method in METHODS:
        by_method = [r for r in trials if r["method"] == method]
        per_shift: list[dict[str, Any]] = []
        for shift in cfg.shift_levels:
            cell = [r for r in by_method if abs(float(r["shift"]) - shift) < 1.0e-9]
            row = {"method": method, "label": LABELS[method], "shift": shift, **mean_dict(cell, metric_keys)}
            shift_rows.append(row)
            per_shift.append(row)
        in_dist = per_shift[0]
        hard = per_shift[-1]
        all_means = mean_dict(by_method, metric_keys)
        summary_rows.append(
            {
                "method": method,
                "label": LABELS[method],
                "success_all": all_means["success"],
                "success_in_distribution": in_dist["success"],
                "robust_success_hard_shift": hard["success"],
                "robustness_auc": float(np.trapezoid([r["success"] for r in per_shift], cfg.shift_levels)),
                "object_damage_all": all_means["object_damage"],
                "slip_drop_all": all_means["slip_drop"],
                "excess_force_all": all_means["excess_force"],
                "force_profile_rmse_all": all_means["force_profile_rmse"],
                "force_error_variance_all": all_means["force_error_variance"],
                "retries_all": all_means["retries"],
                "handoff_balance_error_all": all_means["handoff_balance_error"],
                "mean_speed_all": all_means["mean_speed"],
            }
        )
    return shift_rows, summary_rows


def plot_demo_profiles(demos: list[Demo], output_dir: pathlib.Path) -> None:
    scenario_id = 5
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), sharex=True)
    for col, feedback in enumerate(("haptic_off", "haptic_on")):
        selected = [d for d in demos if d.scenario.scenario_id == scenario_id and d.feedback == feedback]
        for d in selected:
            axes[0, col].plot(d.phase, d.actions[:, 1], alpha=0.72, lw=1.5)
            axes[1, col].plot(d.phase, d.actions[:, 2], alpha=0.72, lw=1.5)
        d0 = selected[0]
        axes[0, col].plot(d0.phase, d0.oracle_force[:, 0], "k--", lw=2.0, label="oracle target")
        axes[1, col].plot(d0.phase, d0.oracle_force[:, 1], "k--", lw=2.0)
        axes[0, col].set_title("Visual only" if feedback == "haptic_off" else "Haptic feedback")
        axes[1, col].set_xlabel("Task phase")
    axes[0, 0].set_ylabel("Left grip force [N]")
    axes[1, 0].set_ylabel("Right grip force [N]")
    axes[0, 0].legend(frameon=False)
    for ax in axes.flat:
        ax.grid(alpha=0.22)
    fig.suptitle("Matched demonstration force profiles (same object, two repeats)")
    fig.tight_layout()
    fig.savefig(output_dir / "matched_demo_force_profiles.png", dpi=180)
    plt.close(fig)


def plot_summary(summary: list[dict[str, Any]], output_dir: pathlib.Path) -> None:
    metrics = [
        ("success_all", "Success ↑", (0, 1)),
        ("object_damage_all", "Object damage ↓", (0, 1)),
        ("slip_drop_all", "Slip / drop ↓", (0, 1)),
        ("force_profile_rmse_all", "Force-profile RMSE ↓", None),
        ("retries_all", "Retries ↓", None),
        ("robust_success_hard_shift", "Hard-shift success ↑", (0, 1)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4))
    x = np.arange(len(METHODS))
    for ax, (key, title, ylim) in zip(axes.flat, metrics):
        vals = [next(r[key] for r in summary if r["method"] == m) for m in METHODS]
        ax.bar(x, vals, color=[COLORS[m] for m in METHODS])
        ax.set_title(title)
        ax.set_xticks(
            x,
            ["Pose", "Visual+F", "Haptic F\nvisual motion", "Passive", "Haptic+F"],
            rotation=18,
        )
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Identical ridge imitation policies trained from different demonstration signals")
    fig.tight_layout()
    fig.savefig(output_dir / "policy_summary.png", dpi=180)
    plt.close(fig)


def plot_robustness(shift_rows: list[dict[str, Any]], output_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1))
    for method in METHODS:
        rows = [r for r in shift_rows if r["method"] == method]
        x = [r["shift"] for r in rows]
        axes[0].plot(x, [r["success"] for r in rows], marker="o", color=COLORS[method], label=LABELS[method])
        axes[1].plot(x, [r["object_damage"] for r in rows], marker="o", color=COLORS[method])
        axes[2].plot(x, [r["force_profile_rmse"] for r in rows], marker="o", color=COLORS[method])
    axes[0].set_title("Success")
    axes[1].set_title("Object damage")
    axes[2].set_title("Force-profile RMSE")
    for ax in axes:
        ax.set_xlabel("Distribution-shift severity")
        ax.grid(alpha=0.24)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_sweep.png", dpi=180)
    plt.close(fig)


def plot_representative(reps: dict[str, dict[str, np.ndarray]], output_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.0), sharex=True)
    for method in METHODS:
        rep = reps[method]
        total_force = rep["actions"][:, 1] + rep["actions"][:, 2]
        axes[0].plot(rep["phase"], total_force, color=COLORS[method], lw=1.7, label=LABELS[method])
        error = np.sqrt(np.mean((rep["actions"][:, 1:3] - rep["oracle"]) ** 2, axis=1))
        axes[1].plot(rep["phase"], error, color=COLORS[method], lw=1.7)
    ref = reps["haptic_force_rich"]
    axes[0].plot(ref["phase"], ref["oracle"].sum(axis=1), "k--", lw=2.0, label="oracle target")
    axes[0].set_ylabel("Total grip command [N]")
    axes[1].set_ylabel("Instantaneous force RMSE [N]")
    axes[1].set_xlabel("Task phase")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.axvspan(0.36, 0.58, color="#dddddd", alpha=0.25)
        ax.axvspan(0.75, 0.92, color="#ffe6cc", alpha=0.25)
    fig.suptitle("Representative hard-shift handoff and insertion")
    fig.tight_layout()
    fig.savefig(output_dir / "representative_policy_rollout.png", dpi=180)
    plt.close(fig)


def run(cfg: Config, output_dir: pathlib.Path, sanity_only: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sanity = run_sanity(cfg, output_dir)
    if sanity_only:
        return {"sanity": sanity}

    train_scenarios = make_scenarios(cfg.demo_scenarios, 0.0, cfg.seed, "train")
    demos = [
        generate_demo(s, repeat, feedback, cfg)
        for s in train_scenarios
        for repeat in range(cfg.demo_repeats)
        for feedback in ("haptic_off", "haptic_on")
    ]
    demo_table = demo_rows(demos)
    write_csv(output_dir / "demonstration_metrics.csv", demo_table)

    policies: dict[str, RidgePolicy] = {}
    train_diagnostics: dict[str, dict[str, float]] = {}
    for method in METHODS:
        x, y = training_arrays(demos, method, cfg)
        policy = RidgePolicy(cfg.ridge).fit(x, y)
        pred = policy.predict(x)
        policies[method] = policy
        train_diagnostics[method] = {
            "samples": float(len(x)),
            "action_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            "force_rmse": float(np.sqrt(np.mean((pred[:, 1:3] - y[:, 1:3]) ** 2))),
        }

    all_trials: list[dict[str, Any]] = []
    representative: dict[str, dict[str, np.ndarray]] = {}
    for shift in cfg.shift_levels:
        eval_scenarios = make_scenarios(cfg.eval_trials_per_shift, shift, cfg.seed, "eval")
        for method in METHODS:
            rows, rep = evaluate_policy(policies[method], method, eval_scenarios, shift, cfg)
            all_trials.extend(rows)
            if shift == cfg.shift_levels[-1]:
                representative[method] = rep

    shift_rows, summary_rows = aggregate_results(all_trials, cfg)
    write_csv(output_dir / "policy_trials.csv", all_trials)
    write_csv(output_dir / "shift_sweep.csv", shift_rows)
    write_csv(output_dir / "summary_metrics.csv", summary_rows)

    demo_summary = []
    demo_metric_keys = [
        "success",
        "object_damage",
        "slip_drop",
        "excess_force",
        "force_profile_rmse",
        "force_error_variance",
        "retries",
        "handoff_balance_error",
    ]
    for feedback in ("haptic_off", "haptic_on"):
        rows = [r for r in demo_table if r["feedback"] == feedback]
        demo_summary.append({"feedback": feedback, **mean_dict(rows, demo_metric_keys)})

    metrics = {
        "scope": "prospective synthetic mechanism test; not a PHABS reproduction or source result",
        "config": asdict(cfg),
        "sanity": sanity,
        "demonstration_summary": demo_summary,
        "training_diagnostics": train_diagnostics,
        "policy_summary": summary_rows,
        "shift_sweep": shift_rows,
    }
    by_method = {row["method"]: row for row in summary_rows}
    visual_force = by_method["force_annotated_visual"]
    crossed_force = by_method["haptic_force_visual_motion"]
    full_haptic = by_method["haptic_force_rich"]
    metrics["crossed_ablation"] = {
        "interpretation": (
            "Haptic force with visual-only speed/balance isolates force-target quality; "
            "the remaining gap to the full haptic condition reflects other generated motion targets."
        ),
        "success_gain_haptic_force_at_fixed_visual_motion_pp": 100.0
        * (crossed_force["success_all"] - visual_force["success_all"]),
        "force_rmse_reduction_haptic_force_at_fixed_visual_motion_fraction": 1.0
        - crossed_force["force_profile_rmse_all"] / visual_force["force_profile_rmse_all"],
        "success_gain_full_haptic_vs_crossed_pp": 100.0
        * (full_haptic["success_all"] - crossed_force["success_all"]),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    plot_demo_profiles(demos, output_dir)
    plot_summary(summary_rows, output_dir)
    plot_robustness(shift_rows, output_dir)
    plot_representative(representative, output_dir)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--demo-scenarios", type=int, default=Config.demo_scenarios)
    parser.add_argument("--demo-repeats", type=int, default=Config.demo_repeats)
    parser.add_argument("--eval-trials-per-shift", type=int, default=Config.eval_trials_per_shift)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Small smoke sweep after the same deterministic sanity check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        seed=args.seed,
        demo_scenarios=16 if args.quick else args.demo_scenarios,
        demo_repeats=1 if args.quick else args.demo_repeats,
        eval_trials_per_shift=30 if args.quick else args.eval_trials_per_shift,
    )
    metrics = run(cfg, args.output_dir, sanity_only=args.sanity_only)
    if args.sanity_only:
        print(json.dumps(metrics["sanity"], indent=2))
        return
    print(f"Wrote outputs to {args.output_dir}")
    print("method, success_all, hard_shift_success, damage, slip_drop, force_rmse, retries")
    for row in metrics["policy_summary"]:
        print(
            f"{row['method']}, {row['success_all']:.3f}, {row['robust_success_hard_shift']:.3f}, "
            f"{row['object_damage_all']:.3f}, {row['slip_drop_all']:.3f}, "
            f"{row['force_profile_rmse_all']:.3f}, {row['retries_all']:.3f}"
        )


if __name__ == "__main__":
    main()
