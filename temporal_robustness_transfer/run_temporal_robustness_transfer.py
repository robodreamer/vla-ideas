#!/usr/bin/env python3
"""Toy test of whether imitation preserves an expert's temporal robustness.

This is a deterministic NumPy/Matplotlib mechanism study inspired by
ParcelStow (Enwerem, Baras, and Belta, 2026). It is not a reproduction of
Isaac Lab, ACT, the released checkpoints, or the paper's numerical results.

A compact insertion task is demonstrated at multiple execution rates. Identical
ridge regressors are then given different temporal representations or runtime
mechanisms: raw elapsed time, normalized phase, explicit speed conditioning,
force feedback, and dynamics-aware augmentation. Evaluation uses matched
initial conditions at demonstrated anchors, held-out interpolation rates, and
extrapolation rates, with ordered stage and insertion diagnostics.
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

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np

TRAIN_RATES = (0.5, 1.0, 1.5, 2.0)
EVAL_RATES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0)
METHODS = (
    "scripted_expert",
    "raw_time_scaling",
    "phase_normalization",
    "speed_conditioned",
    "force_feedback",
    "dynamics_augmentation",
)
LEARNED_METHODS = METHODS[1:]
LABELS = {
    "scripted_expert": "Scripted expert",
    "raw_time_scaling": "Raw elapsed time",
    "phase_normalization": "Normalized phase",
    "speed_conditioned": "Speed-conditioned",
    "force_feedback": "Speed + force feedback",
    "dynamics_augmentation": "Dynamics augmentation",
}
COLORS = {
    "scripted_expert": "#222222",
    "raw_time_scaling": "#9b9b9b",
    "phase_normalization": "#f28e2b",
    "speed_conditioned": "#4e79a7",
    "force_feedback": "#59a14f",
    "dynamics_augmentation": "#b07aa1",
}
STAGES = ("acquisition", "lift", "reorientation", "preinsert", "insertion", "release", "settling")
FAILURES = (
    "success",
    "no_force_closure",
    "lift_loss",
    "reorientation",
    "insertion_misalignment",
    "insertion_jam",
    "release",
    "settling",
)


@dataclass(frozen=True)
class Config:
    seed: int = 53
    mode: str = "full"
    demos_per_rate: int = 48
    eval_episodes_per_rate: int = 180
    phase_steps: int = 81
    ridge: float = 0.035
    train_rates: tuple[float, ...] = TRAIN_RATES
    eval_rates: tuple[float, ...] = EVAL_RATES


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    lateral_mm: float
    yaw_deg: float
    friction: float
    compliance: float
    servo_gain: float
    sensor_bias_n: float


class RidgePolicy:
    def __init__(self, ridge: float):
        self.ridge = ridge
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgePolicy":
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0) + 1.0e-7
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
    z = np.clip((np.asarray(x, dtype=float) - lo) / (hi - lo), 0.0, 1.0)
    return z * z * (3.0 - 2.0 * z)


def physical_time(phase: np.ndarray, rate: float) -> np.ndarray:
    """ParcelStow-inspired schedule: 6.3 s fixed + 7.8/r s scaled."""
    p = np.asarray(phase, dtype=float)
    out = np.empty_like(p)
    fixed = p <= 0.30
    out[fixed] = 6.3 * p[fixed] / 0.30
    out[~fixed] = 6.3 + (7.8 / rate) * (p[~fixed] - 0.30) / 0.70
    return out


def cycle_duration(rate: float) -> float:
    return 6.3 + 7.8 / rate


def rate_regime(rate: float) -> str:
    if any(abs(rate - x) < 1.0e-9 for x in TRAIN_RATES):
        return "demonstrated_anchor"
    if min(TRAIN_RATES) < rate < max(TRAIN_RATES):
        return "interpolation"
    return "extrapolation"


def scenario_from_seed(scenario_id: int, seed: int, split: str, augmented: bool = False) -> Scenario:
    rng = np.random.default_rng(stable_seed(seed, split, scenario_id))
    scale = 1.35 if augmented else 1.0
    return Scenario(
        scenario_id=scenario_id,
        lateral_mm=float(rng.normal(0.0, 4.4 * scale)),
        yaw_deg=float(rng.normal(0.0, 3.8 * scale)),
        friction=float(rng.uniform(0.43 - 0.05 * augmented, 0.64 + 0.04 * augmented)),
        compliance=float(rng.uniform(0.84 - 0.10 * augmented, 1.16 + 0.12 * augmented)),
        servo_gain=float(rng.uniform(0.93 - 0.05 * augmented, 1.05 + 0.04 * augmented)),
        sensor_bias_n=float(rng.normal(0.0, 0.10)),
    )


def expert_requirements(s: Scenario, rate: float) -> dict[str, float]:
    q = rate - 1.0
    # The required cumulative corrections include speed-dependent dynamic lead.
    align = -(
        s.lateral_mm
        + 0.62 * q * s.yaw_deg
        + 1.15 * q * q / max(s.compliance, 0.65)
        + 0.85 * q * (1.0 - s.servo_gain) * 10.0
    )
    rotate = -(
        s.yaw_deg
        + 1.55 * q
        + 1.30 * q * q / max(s.compliance, 0.65)
        + 5.0 * q * (1.0 - s.servo_gain)
    )
    insertion_speed = rate * (0.94 + 0.08 * rate) / max(s.compliance, 0.72)
    grip_acquire = 1.55 + 1.02 / max(s.friction, 0.30)
    grip_dynamic = grip_acquire + 0.19 * rate * rate / max(s.compliance, 0.72)
    return {
        "align": align,
        "rotate": rotate,
        "insertion_speed": insertion_speed,
        "grip_acquire": grip_acquire,
        "grip_dynamic": grip_dynamic,
    }


def expert_actions(
    phase: np.ndarray,
    s: Scenario,
    rate: float,
    rng: np.random.Generator | None = None,
    noisy: bool = False,
) -> np.ndarray:
    p = np.asarray(phase, dtype=float)
    req = expert_requirements(s, rate)
    align = req["align"] * smoothstep(p, 0.46, 0.77)
    rotate = req["rotate"] * smoothstep(p, 0.31, 0.68)
    insertion = req["insertion_speed"] * np.sin(
        np.pi * np.clip((p - 0.755) / 0.155, 0.0, 1.0)
    ) ** 1.25
    acquire_gate = smoothstep(p, 0.13, 0.22)
    release_gate = 1.0 - smoothstep(p, 0.915, 0.965)
    dynamic_gate = smoothstep(p, 0.29, 0.39)
    grip = (
        req["grip_acquire"]
        + dynamic_gate * (req["grip_dynamic"] - req["grip_acquire"])
    ) * acquire_gate * release_gate
    actions = np.column_stack([align, rotate, insertion, grip])
    if noisy:
        if rng is None:
            raise ValueError("noisy demonstrations require an RNG")
        phase_noise = np.column_stack(
            [
                rng.normal(0.0, 0.14, len(p)),
                rng.normal(0.0, 0.12, len(p)),
                rng.normal(0.0, 0.025, len(p)),
                rng.normal(0.0, 0.035, len(p)),
            ]
        )
        phase_noise[:, :2] *= (0.35 + smoothstep(p, 0.25, 0.80))[:, None]
        actions = actions + phase_noise
    return actions


def temporal_basis(coord: np.ndarray, centers: int = 13, width: float = 0.095) -> np.ndarray:
    x = np.asarray(coord, dtype=float)
    c = np.linspace(0.0, 1.0, centers)
    rbf = np.exp(-0.5 * ((x[:, None] - c[None, :]) / width) ** 2)
    return np.column_stack(
        [
            x,
            x**2,
            x**3,
            np.sin(np.pi * x),
            np.cos(np.pi * x),
            np.sin(2.0 * np.pi * x),
            np.cos(2.0 * np.pi * x),
            rbf,
        ]
    )


def scenario_context(s: Scenario) -> np.ndarray:
    return np.array(
        [
            s.lateral_mm / 8.0,
            s.yaw_deg / 8.0,
            (s.friction - 0.53) / 0.12,
            (s.compliance - 1.0) / 0.18,
            (s.servo_gain - 1.0) / 0.09,
        ],
        dtype=float,
    )


def feature_matrix(method: str, phase: np.ndarray, s: Scenario, rate: float) -> np.ndarray:
    p = np.asarray(phase, dtype=float)
    if method == "raw_time_scaling":
        # Mechanical whole-trajectory time scaling: elapsed seconds are divided
        # by that rollout's total duration. Because fixed and speed-scaled phases
        # occupy different fractions of the cycle, this is not task phase.
        coord = physical_time(p, rate) / cycle_duration(rate)
    else:
        coord = p
    basis = temporal_basis(coord)
    context = scenario_context(s)
    ctx = np.repeat(context[None, :], len(p), axis=0)
    interactions = np.einsum("ni,nj->nij", basis, ctx).reshape(len(p), -1)
    pieces = [basis, ctx, interactions]

    if method in ("speed_conditioned", "force_feedback"):
        # Intentionally first-order in rate: interpolation is supported, while
        # quadratic contact/dynamic demand remains an extrapolation challenge.
        r = np.full((len(p), 1), rate - 1.0)
        pieces.extend([r, basis * r])
    elif method == "dynamics_augmentation":
        q = rate - 1.0
        dyn = np.column_stack(
            [
                np.full(len(p), q),
                np.full(len(p), q * q),
                np.full(len(p), rate * rate / max(s.compliance, 0.65)),
                np.full(len(p), q * (1.0 - s.servo_gain)),
                np.full(len(p), rate / max(s.friction, 0.30)),
            ]
        )
        pieces.extend([dyn, np.einsum("ni,nj->nij", basis, dyn).reshape(len(p), -1)])
    return np.column_stack(pieces)


def build_training_set(cfg: Config, method: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    rates = list(cfg.train_rates)
    demos_per_rate = cfg.demos_per_rate
    augmented_count = 0

    for rate in rates:
        for i in range(demos_per_rate):
            sid = int(round(rate * 1000)) * 10000 + i
            s = scenario_from_seed(sid, cfg.seed, "train")
            phase = np.linspace(0.0, 1.0, cfg.phase_steps)
            rng = np.random.default_rng(stable_seed(cfg.seed, "demo-noise", method, rate, i))
            xs.append(feature_matrix(method, phase, s, rate))
            ys.append(expert_actions(phase, s, rate, rng=rng, noisy=True))

    if method == "dynamics_augmentation":
        # Explicitly synthetic augmentation: broader rates and dynamics than the
        # expert demonstration range. This is an ablation, not free real data.
        aug_rates = (0.35, 0.65, 0.85, 1.25, 1.75, 2.15, 2.45, 2.8, 3.2)
        per_rate = max(5, demos_per_rate // 3)
        for rate in aug_rates:
            for i in range(per_rate):
                sid = 7000000 + int(round(rate * 1000)) * 1000 + i
                s = scenario_from_seed(sid, cfg.seed, "augmented-train", augmented=True)
                phase = np.linspace(0.0, 1.0, cfg.phase_steps)
                rng = np.random.default_rng(stable_seed(cfg.seed, "aug-noise", rate, i))
                xs.append(feature_matrix(method, phase, s, rate))
                ys.append(expert_actions(phase, s, rate, rng=rng, noisy=True))
                augmented_count += 1

    x = np.vstack(xs)
    y = np.vstack(ys)
    return x, y, {
        "demonstrations": len(rates) * demos_per_rate + augmented_count,
        "base_demonstrations": len(rates) * demos_per_rate,
        "augmented_demonstrations": augmented_count,
        "samples": len(x),
        "features": x.shape[1],
    }


def value_at(phase: np.ndarray, actions: np.ndarray, query: float, column: int) -> float:
    return float(np.interp(query, phase, actions[:, column]))


def feedback_corrected_actions(
    phase: np.ndarray,
    predicted: np.ndarray,
    s: Scenario,
    rate: float,
) -> np.ndarray:
    """Toy contact feedback residual used only by the force-feedback policy."""
    out = predicted.copy()
    req = expert_actions(phase, s, rate, noisy=False)
    acquire = smoothstep(phase, 0.16, 0.25) * (1.0 - smoothstep(phase, 0.91, 0.96))
    insertion = smoothstep(phase, 0.72, 0.80) * (1.0 - smoothstep(phase, 0.90, 0.95))
    # Grip regulation is available after contact; lateral/orientation correction
    # is available only near insertion from signed receptacle force.
    out[:, 3] += 0.82 * acquire * (req[:, 3] - out[:, 3] + s.sensor_bias_n)
    out[:, 0] += 0.62 * insertion * (req[:, 0] - out[:, 0])
    out[:, 1] += 0.48 * insertion * (req[:, 1] - out[:, 1])
    out[:, 2] += 0.52 * insertion * (req[:, 2] - out[:, 2])
    return out


def rollout(
    cfg: Config,
    method: str,
    rate: float,
    episode: int,
    policy: RidgePolicy | None,
) -> dict[str, Any]:
    s = scenario_from_seed(episode, cfg.seed, f"eval-{rate}")
    phase = np.linspace(0.0, 1.0, cfg.phase_steps)
    required_actions = expert_actions(phase, s, rate, noisy=False)
    if method == "scripted_expert":
        predicted = required_actions.copy()
    else:
        assert policy is not None
        predicted = policy.predict(feature_matrix(method, phase, s, rate))
        if method == "force_feedback":
            predicted = feedback_corrected_actions(phase, predicted, s, rate)

    req = expert_requirements(s, rate)
    rng = np.random.default_rng(stable_seed(cfg.seed, "eval-disturbance", rate, episode))
    disturbance_align = float(rng.normal(0.0, 0.72 + 0.22 * max(rate - 2.0, 0.0)))
    disturbance_rot = float(rng.normal(0.0, 0.65 + 0.25 * max(rate - 2.0, 0.0)))
    disturbance_force = float(rng.normal(0.0, 0.28))

    grip_acq = value_at(phase, predicted, 0.265, 3)
    grip_dyn = value_at(phase, predicted, 0.70, 3)
    align_cmd = value_at(phase, predicted, 0.785, 0)
    rotate_cmd = value_at(phase, predicted, 0.715, 1)
    insert_speed = value_at(phase, predicted, 0.835, 2)
    release_force = max(value_at(phase, predicted, 0.975, 3), 0.0)

    # The expert command carries a 0.28 N acquisition reserve above the toy's
    # physical closure threshold; the learned command may or may not preserve it.
    force_closure_margin = grip_acq - (req["grip_acquire"] - 0.28) - 0.10 * abs(s.lateral_mm) / 8.0
    dynamic_grip_margin = grip_dyn - req["grip_dynamic"]
    alignment_signed = (align_cmd - req["align"]) * s.servo_gain + disturbance_align
    orientation_signed = (rotate_cmd - req["rotate"]) * s.servo_gain + disturbance_rot
    alignment_mm = abs(alignment_signed)
    orientation_deg = abs(orientation_signed)
    speed_error = abs(insert_speed - req["insertion_speed"])

    unavoidable = max(rate - 2.0, 0.0)
    peak_contact_force_n = (
        1.35
        + 0.61 * rate * rate / max(s.compliance, 0.72)
        + 0.47 * alignment_mm
        + 0.105 * orientation_deg
        + 1.15 * speed_error
        + 0.80 * max(-dynamic_grip_margin, 0.0)
        + 0.22 * max(dynamic_grip_margin - 0.45, 0.0)
        + 1.35 * unavoidable * unavoidable
        + disturbance_force
    )
    peak_contact_force_n = max(float(peak_contact_force_n), 0.0)
    insertion_depth_mm = (
        60.0
        - 1.20 * alignment_mm
        - 0.34 * orientation_deg
        - 1.35 * max(peak_contact_force_n - 4.2, 0.0)
        - 2.10 * speed_error
    )
    insertion_depth_mm = float(np.clip(insertion_depth_mm, 0.0, 62.0))
    settle_error_deg = orientation_deg + 0.36 * alignment_mm + 0.42 * max(peak_contact_force_n - 5.0, 0.0)

    acquired = force_closure_margin > 0.0
    lifted = acquired and dynamic_grip_margin > -0.48
    reoriented = lifted and orientation_deg <= 15.0
    preinsert = reoriented and orientation_deg <= 13.5
    aligned = alignment_mm <= 9.6 and orientation_deg <= 10.5
    jammed = peak_contact_force_n >= 8.35 or insertion_depth_mm < 50.0
    inserted = preinsert and aligned and not jammed
    released = inserted and release_force < 1.15
    settled = released and settle_error_deg <= 10.0

    if not acquired:
        failure = "no_force_closure"
    elif not lifted:
        failure = "lift_loss"
    elif not reoriented:
        failure = "reorientation"
    elif not aligned:
        failure = "insertion_misalignment"
    elif jammed:
        failure = "insertion_jam"
    elif not released:
        failure = "release"
    elif not settled:
        failure = "settling"
    else:
        failure = "success"

    stage_pass = {
        "acquisition": acquired,
        "lift": lifted,
        "reorientation": reoriented,
        "preinsert": preinsert,
        "insertion": inserted,
        "release": released,
        "settling": settled,
    }
    action_rmse = float(np.sqrt(np.mean((predicted - required_actions) ** 2)))
    return {
        "method": method,
        "label": LABELS[method],
        "rate": rate,
        "regime": rate_regime(rate),
        "episode": episode,
        "success": int(failure == "success"),
        "failure": failure,
        "cycle_duration_s": cycle_duration(rate),
        "force_closure_margin": force_closure_margin,
        "force_closure": int(force_closure_margin > 0.0),
        "dynamic_grip_margin": dynamic_grip_margin,
        "alignment_mm": alignment_mm,
        "orientation_deg": orientation_deg,
        "speed_error": speed_error,
        "peak_contact_force_n": peak_contact_force_n,
        "insertion_depth_mm": insertion_depth_mm,
        "settle_error_deg": settle_error_deg,
        "release_force_n": release_force,
        "action_rmse": action_rmse,
        **{f"stage_{stage}": int(stage_pass[stage]) for stage in STAGES},
    }


def mean(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else float("nan")


def trapz_auc(points: list[tuple[float, float]], lo: float, hi: float) -> float:
    selected = sorted((x, y) for x, y in points if lo <= x <= hi)
    if len(selected) < 2:
        return float("nan")
    x = np.array([p[0] for p in selected])
    y = np.array([p[1] for p in selected])
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def aggregate(cfg: Config, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    speed_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for rate in cfg.eval_rates:
            subset = [r for r in rows if r["method"] == method and abs(r["rate"] - rate) < 1e-9]
            speed_rows.append(
                {
                    "method": method,
                    "label": LABELS[method],
                    "rate": rate,
                    "regime": rate_regime(rate),
                    "episodes": len(subset),
                    "success_rate": mean(subset, "success"),
                    "force_closure_rate": mean(subset, "force_closure"),
                    "alignment_mm": mean(subset, "alignment_mm"),
                    "orientation_deg": mean(subset, "orientation_deg"),
                    "peak_contact_force_n": mean(subset, "peak_contact_force_n"),
                    "insertion_depth_mm": mean(subset, "insertion_depth_mm"),
                    "action_rmse": mean(subset, "action_rmse"),
                    **{f"stage_{stage}_rate": mean(subset, f"stage_{stage}") for stage in STAGES},
                    **{
                        f"failure_{failure}_rate": float(np.mean([r["failure"] == failure for r in subset]))
                        for failure in FAILURES
                    },
                }
            )

    summary: list[dict[str, Any]] = []
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        by_rate = [r for r in speed_rows if r["method"] == method]
        rate_success = [(float(r["rate"]), float(r["success_rate"])) for r in by_rate]
        nominal = next(r["success_rate"] for r in by_rate if abs(r["rate"] - 1.0) < 1e-9)
        at_two = next(r["success_rate"] for r in by_rate if abs(r["rate"] - 2.0) < 1e-9)
        interpolation = [r for r in subset if r["regime"] == "interpolation"]
        extrapolation = [r for r in subset if r["regime"] == "extrapolation"]
        summary.append(
            {
                "method": method,
                "label": LABELS[method],
                "episodes": len(subset),
                "success_rate": mean(subset, "success"),
                "nominal_success": nominal,
                "r2_success": at_two,
                "temporal_drop_r1_to_r2": nominal - at_two,
                "interpolation_success": mean(interpolation, "success"),
                "extrapolation_success": mean(extrapolation, "success"),
                "demonstrated_range_auc": trapz_auc(rate_success, 0.5, 2.0),
                "extrapolation_auc": trapz_auc(rate_success, 2.0, 3.0),
                "force_closure_failure_rate": 1.0 - mean(subset, "force_closure"),
                "insertion_misalignment_rate": float(np.mean([r["failure"] == "insertion_misalignment" for r in subset])),
                "insertion_jam_rate": float(np.mean([r["failure"] == "insertion_jam" for r in subset])),
                "mean_alignment_mm": mean(subset, "alignment_mm"),
                "mean_peak_contact_force_n": mean(subset, "peak_contact_force_n"),
                "mean_insertion_depth_mm": mean(subset, "insertion_depth_mm"),
                "action_rmse": mean(subset, "action_rmse"),
            }
        )

    stage_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for regime in ("demonstrated_anchor", "interpolation", "extrapolation"):
            subset = [r for r in rows if r["method"] == method and r["regime"] == regime]
            for stage in STAGES:
                stage_rows.append(
                    {
                        "method": method,
                        "label": LABELS[method],
                        "regime": regime,
                        "stage": stage,
                        "pass_rate": mean(subset, f"stage_{stage}"),
                        "episodes": len(subset),
                    }
                )
    return speed_rows, summary, stage_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plot_operating_envelope(cfg: Config, speed_rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    for method in METHODS:
        rows = [r for r in speed_rows if r["method"] == method]
        ax.plot(
            [r["rate"] for r in rows],
            [100.0 * r["success_rate"] for r in rows],
            marker="o",
            linewidth=2.4 if method != "scripted_expert" else 2.8,
            color=COLORS[method],
            label=LABELS[method],
        )
    ax.axvspan(0.5, 2.0, color="#d7ebf7", alpha=0.32, label="demonstrated range")
    for r in TRAIN_RATES:
        ax.axvline(r, color="#8aa6b8", alpha=0.18, linewidth=0.9)
    ax.axvline(2.0, color="#4f6d7a", linestyle="--", linewidth=1.2)
    ax.text(2.03, 3.0, "extrapolation", color="#4f6d7a", fontsize=9)
    ax.set(xlabel="Execution speedup factor r", ylabel="Success rate (%)", ylim=(-2, 103))
    ax.set_title("Temporal robustness operating envelope")
    ax.grid(alpha=0.24)
    ax.legend(ncol=2, fontsize=8.7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_stage_diagnostics(stage_rows: list[dict[str, Any]], out: Path) -> None:
    methods = LEARNED_METHODS
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True)
    for ax, regime in zip(axes, ("interpolation", "extrapolation")):
        x = np.arange(len(STAGES))
        for method in methods:
            rows = [r for r in stage_rows if r["method"] == method and r["regime"] == regime]
            values = [next(r["pass_rate"] for r in rows if r["stage"] == stage) for stage in STAGES]
            ax.plot(x, np.array(values) * 100.0, marker="o", linewidth=2.0, color=COLORS[method], label=LABELS[method])
        ax.set_xticks(x, [s.replace("reorientation", "reorient").replace("acquisition", "acquire") for s in STAGES], rotation=35, ha="right")
        ax.set_title(regime.capitalize())
        ax.grid(alpha=0.22)
        ax.set_ylim(-2, 103)
    axes[0].set_ylabel("Episodes passing ordered stage (%)")
    axes[1].legend(fontsize=8.2, loc="lower left")
    fig.suptitle("Where temporal transfer fails")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_insertion_proxies(cfg: Config, speed_rows: list[dict[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5))
    specs = (
        ("alignment_mm", "Mean pre-insertion\nmisalignment (mm)"),
        ("peak_contact_force_n", "Peak receptacle-force\nproxy (N)"),
        ("force_closure_rate", "Acquisitions with\nforce closure (%)"),
    )
    for ax, (key, ylabel) in zip(axes, specs):
        for method in LEARNED_METHODS:
            rows = [r for r in speed_rows if r["method"] == method]
            values = [r[key] * (100.0 if key == "force_closure_rate" else 1.0) for r in rows]
            ax.plot([r["rate"] for r in rows], values, marker="o", linewidth=1.9, color=COLORS[method], label=LABELS[method])
        ax.axvline(2.0, color="#555555", linestyle="--", alpha=0.6)
        ax.set(xlabel="Speedup r", ylabel=ylabel)
        ax.grid(alpha=0.22)
    axes[0].axhline(9.6, color="#c44e52", linestyle=":", linewidth=1.2, label="toy limit")
    axes[1].axhline(8.35, color="#c44e52", linestyle=":", linewidth=1.2)
    axes[2].legend(fontsize=7.7, loc="lower left")
    fig.suptitle("Insertion and acquisition diagnostics")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_representative(cfg: Config, policies: dict[str, RidgePolicy], out: Path) -> None:
    rate = 2.5
    s = scenario_from_seed(7, cfg.seed, f"eval-{rate}")
    phase = np.linspace(0.0, 1.0, cfg.phase_steps)
    req = expert_actions(phase, s, rate, noisy=False)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0), sharex=True)
    titles = ("Lateral correction (mm)", "Yaw correction (deg)", "Insertion-speed command", "Grip command (N)")
    for j, ax in enumerate(axes.flat):
        ax.plot(phase, req[:, j], color=COLORS["scripted_expert"], linewidth=2.8, label="expert target")
        for method in LEARNED_METHODS:
            pred = policies[method].predict(feature_matrix(method, phase, s, rate))
            if method == "force_feedback":
                pred = feedback_corrected_actions(phase, pred, s, rate)
            ax.plot(phase, pred[:, j], color=COLORS[method], linewidth=1.7, label=LABELS[method])
        ax.axvspan(0.755, 0.91, color="#f6bd60", alpha=0.16)
        ax.set_title(titles[j])
        ax.grid(alpha=0.22)
    for ax in axes[-1]:
        ax.set_xlabel("Normalized task phase")
    axes[0, 0].legend(fontsize=7.8, ncol=2)
    fig.suptitle("Representative extrapolation trajectory at r = 2.5")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def sanity_suite(cfg: Config) -> dict[str, Any]:
    phase = np.linspace(0.0, 1.0, cfg.phase_steps)
    s1 = scenario_from_seed(11, cfg.seed, "sanity")
    s2 = scenario_from_seed(11, cfg.seed, "sanity")
    assert s1 == s2
    a1 = expert_actions(phase, s1, 1.0, noisy=False)
    a2 = expert_actions(phase, s2, 1.0, noisy=False)
    assert np.array_equal(a1, a2)
    assert math.isclose(cycle_duration(1.0), 14.1, abs_tol=1e-9)
    assert math.isclose(cycle_duration(2.0), 10.2, abs_tol=1e-9)
    assert expert_requirements(s1, 2.0)["grip_dynamic"] > expert_requirements(s1, 1.0)["grip_dynamic"]

    deliberately_bad = a1.copy()
    deliberately_bad[:, 0] += 6.0
    deliberately_bad[:, 1] += 5.0
    deliberately_bad[:, 2] *= 1.55
    corrected = feedback_corrected_actions(phase, deliberately_bad, s1, 1.0)
    key = int(np.argmin(np.abs(phase - 0.835)))
    before = float(np.linalg.norm(deliberately_bad[key, :3] - a1[key, :3]))
    after = float(np.linalg.norm(corrected[key, :3] - a1[key, :3]))
    assert after < before
    return {
        "passed": True,
        "deterministic_scenario": True,
        "deterministic_expert": True,
        "nominal_cycle_duration_s": cycle_duration(1.0),
        "r2_cycle_duration_s": cycle_duration(2.0),
        "higher_speed_requires_more_dynamic_grip": True,
        "force_feedback_reduces_stress_action_error": True,
        "stress_error_before": before,
        "stress_error_after": after,
    }


def configuration_for_mode(args: argparse.Namespace) -> Config:
    cfg = Config(seed=args.seed, mode=args.mode)
    if args.mode == "sanity":
        cfg = replace(cfg, demos_per_rate=4, eval_episodes_per_rate=8, phase_steps=51)
    elif args.mode == "quick":
        cfg = replace(cfg, demos_per_rate=12, eval_episodes_per_rate=40, phase_steps=61)
    if args.demos_per_rate is not None:
        cfg = replace(cfg, demos_per_rate=args.demos_per_rate)
    if args.eval_episodes_per_rate is not None:
        cfg = replace(cfg, eval_episodes_per_rate=args.eval_episodes_per_rate)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sanity", "quick", "full"), default="full")
    parser.add_argument("--quick", action="store_true", help="alias for --mode quick")
    parser.add_argument("--sanity-only", action="store_true", help="alias for --mode sanity")
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--demos-per-rate", type=int)
    parser.add_argument("--eval-episodes-per-rate", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.quick:
        args.mode = "quick"
    if args.sanity_only:
        args.mode = "sanity"
    return args


def main() -> None:
    args = parse_args()
    cfg = configuration_for_mode(args)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    sanity = sanity_suite(cfg)
    policies: dict[str, RidgePolicy] = {}
    training: dict[str, Any] = {}
    for method in LEARNED_METHODS:
        x, y, info = build_training_set(cfg, method)
        policies[method] = RidgePolicy(cfg.ridge).fit(x, y)
        training[method] = info

    episode_rows: list[dict[str, Any]] = []
    for method in METHODS:
        policy = None if method == "scripted_expert" else policies[method]
        for rate in cfg.eval_rates:
            for episode in range(cfg.eval_episodes_per_rate):
                episode_rows.append(rollout(cfg, method, rate, episode, policy))

    speed_rows, summary, stage_rows = aggregate(cfg, episode_rows)
    summary_by_method = {row["method"]: row for row in summary}
    expert_r2 = summary_by_method["scripted_expert"]["r2_success"]
    comparisons = {
        method: {
            "r2_gap_to_expert": expert_r2 - summary_by_method[method]["r2_success"],
            "temporal_drop_excess_over_expert": summary_by_method[method]["temporal_drop_r1_to_r2"]
            - summary_by_method["scripted_expert"]["temporal_drop_r1_to_r2"],
            "extrapolation_success_gain_over_speed_conditioned": summary_by_method[method]["extrapolation_success"]
            - summary_by_method["speed_conditioned"]["extrapolation_success"],
        }
        for method in LEARNED_METHODS
    }

    write_json(out / "sanity_check.json", sanity)
    write_csv(out / "episode_metrics.csv", episode_rows)
    write_csv(out / "speed_sweep.csv", speed_rows)
    write_csv(out / "stage_diagnostics.csv", stage_rows)
    write_csv(out / "summary_metrics.csv", summary)
    write_json(
        out / "metrics.json",
        {
            "claim_boundary": "Synthetic mechanism study; not ParcelStow/ACT reproduction or real-robot evidence.",
            "config": asdict(cfg),
            "sanity": sanity,
            "training": training,
            "summary": summary,
            "comparisons": comparisons,
        },
    )

    plot_operating_envelope(cfg, speed_rows, out / "operating_envelope.png")
    plot_stage_diagnostics(stage_rows, out / "stage_diagnostics.png")
    plot_insertion_proxies(cfg, speed_rows, out / "insertion_proxies.png")
    plot_representative(cfg, policies, out / "representative_trajectory.png")

    print(f"mode={cfg.mode} seed={cfg.seed} output={out}")
    print("method                         success   r=1    r=2   interp  extra   drop")
    for row in summary:
        print(
            f"{row['method']:<30} {row['success_rate']:.3f}  "
            f"{row['nominal_success']:.3f}  {row['r2_success']:.3f}  "
            f"{row['interpolation_success']:.3f}  {row['extrapolation_success']:.3f}  "
            f"{row['temporal_drop_r1_to_r2']:.3f}"
        )


if __name__ == "__main__":
    main()
