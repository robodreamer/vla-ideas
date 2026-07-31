#!/usr/bin/env python3
"""Toy B-spline action parameterization experiment.

This is a deliberately small analogue of B-spline Policy: a chunked policy emits a
low-rate action trajectory, while the controller executes at a much higher rate.
The toy compares discrete waypoint chunks against compact cubic B-spline chunks
under temporal speed-up and actuator limits.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE_DIR, ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

METHOD_ORDER = [
    "discrete_safe_1x",
    "discrete_fast_3x",
    "discrete_adaptive_3x",
    "bspline_fast_3x",
    "bspline_adaptive_3x",
]

LABELS = {
    "discrete_safe_1x": "Discrete chunks, 1x",
    "discrete_fast_3x": "Discrete chunks, 3x",
    "discrete_adaptive_3x": "Discrete + curvature time-law, 3x",
    "bspline_fast_3x": "B-spline chunks, 3x",
    "bspline_adaptive_3x": "B-spline + curvature time-law, 3x",
}

COLORS = {
    "discrete_safe_1x": "#4c78a8",
    "discrete_fast_3x": "#f58518",
    "discrete_adaptive_3x": "#b279a2",
    "bspline_fast_3x": "#54a24b",
    "bspline_adaptive_3x": "#e45756",
}


@dataclass
class Config:
    seed: int = 7
    trials: int = 160
    dt: float = 0.005  # 200 Hz low-level execution
    demo_duration: float = 6.0
    speedup: float = 3.0
    waypoint_hz: float = 10.0
    waypoint_noise: float = 0.020
    bspline_controls: int = 14
    degree: int = 3
    max_speed: float = 3.5
    max_accel: float = 24.0
    kp: float = 120.0
    kd: float = 20.0
    settle_time: float = 0.75
    ood_radius: float = 0.08
    goal_tolerance: float = 0.09


def expert_path(s: np.ndarray) -> np.ndarray:
    """Smooth 2D demonstration manifold with mixed curvature."""
    s = np.asarray(s, dtype=float)
    x = 2.8 * s
    y = 0.34 * np.sin(2.4 * np.pi * s) + 0.18 * np.sin(6.0 * np.pi * s + 0.35)
    # A small lateral hook near the end makes aggressive time-scaling expose lag.
    y += 0.22 * np.exp(-((s - 0.78) / 0.09) ** 2)
    return np.stack([x, y], axis=-1)


def path_derivatives(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s, dtype=float)
    dx = np.full_like(s, 2.8)
    dy = (
        0.34 * 2.4 * np.pi * np.cos(2.4 * np.pi * s)
        + 0.18 * 6.0 * np.pi * np.cos(6.0 * np.pi * s + 0.35)
        + 0.22 * np.exp(-((s - 0.78) / 0.09) ** 2) * (-2.0 * (s - 0.78) / 0.09**2)
    )
    ddx = np.zeros_like(s)
    ddy = (
        -0.34 * (2.4 * np.pi) ** 2 * np.sin(2.4 * np.pi * s)
        - 0.18 * (6.0 * np.pi) ** 2 * np.sin(6.0 * np.pi * s + 0.35)
    )
    g = np.exp(-((s - 0.78) / 0.09) ** 2)
    ddy += 0.22 * g * (4.0 * (s - 0.78) ** 2 / 0.09**4 - 2.0 / 0.09**2)
    return np.stack([dx, dy], axis=-1), np.stack([ddx, ddy], axis=-1)


def curvature_from_derivatives(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    den = np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-9)
    return num / den


def curvature(s: np.ndarray) -> np.ndarray:
    d1, d2 = path_derivatives(s)
    return curvature_from_derivatives(d1, d2)


def bspline_curvature(spline: BSpline, s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    d1 = spline.derivative(1)(s)
    d2 = spline.derivative(2)(s)
    return curvature_from_derivatives(d1, d2)


def waypoint_curvature(s_way: np.ndarray, waypoints: np.ndarray) -> np.ndarray:
    d1 = np.gradient(waypoints, s_way, axis=0)
    d2 = np.gradient(d1, s_way, axis=0)
    kappa = curvature_from_derivatives(d1, d2)
    return np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)


def make_noisy_waypoints(cfg: Config, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    n = int(cfg.demo_duration * cfg.waypoint_hz) + 1
    s = np.linspace(0.0, 1.0, n)
    pts = expert_path(s)
    # Keep the task endpoints reliable; perturb only the interior chunk commands.
    noise = rng.normal(scale=cfg.waypoint_noise, size=pts.shape)
    taper = np.sin(np.pi * s)[:, None]
    pts = pts + noise * taper
    pts[0] = expert_path(np.array([0.0]))[0]
    pts[-1] = expert_path(np.array([1.0]))[0]
    return s, pts


def open_uniform_knots(n_controls: int, degree: int) -> np.ndarray:
    if n_controls <= degree:
        raise ValueError("n_controls must exceed degree")
    n_internal = n_controls - degree - 1
    internal = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
    return np.concatenate([np.zeros(degree + 1), internal, np.ones(degree + 1)])


def fit_bspline_actions(s_way: np.ndarray, waypoints: np.ndarray, cfg: Config) -> BSpline:
    knots = open_uniform_knots(cfg.bspline_controls, cfg.degree)
    return make_lsq_spline(s_way, waypoints, knots, k=cfg.degree)


def make_adaptive_s_values(
    cfg: Config,
    curvature_at_s,
    slowdown_gain: float = 0.22,
) -> Tuple[np.ndarray, float]:
    dt = cfg.dt
    s_values = [0.0]
    v_s = cfg.speedup / cfg.demo_duration
    t = 0.0
    while s_values[-1] < 1.0 and t < cfg.demo_duration:
        s_now = s_values[-1]
        kappa = float(curvature_at_s(s_now))
        target_v_s = cfg.speedup / cfg.demo_duration / (1.0 + slowdown_gain * kappa)
        v_s += np.clip(target_v_s - v_s, -1.8 * dt, 1.8 * dt)
        s_values.append(min(1.0, s_now + v_s * dt))
        t += dt
    s = np.asarray(s_values)
    duration = (len(s) - 1) * cfg.dt
    return s, duration


def dense_waypoint_command(s: np.ndarray, s_way: np.ndarray, waypoints: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.interp(s, s_way, waypoints[:, 0]),
        np.interp(s, s_way, waypoints[:, 1]),
    ])


def make_command(
    method: str,
    s_way: np.ndarray,
    waypoints: np.ndarray,
    spline: BSpline,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    if method == "discrete_safe_1x":
        duration = cfg.demo_duration
        n = int(duration / cfg.dt) + 1
        s = np.linspace(0.0, 1.0, n)
        command = dense_waypoint_command(s, s_way, waypoints)
        return s, command, {"command_duration": duration, "nominal_speedup": 1.0}

    if method == "discrete_fast_3x":
        duration = cfg.demo_duration / cfg.speedup
        n = int(duration / cfg.dt) + 1
        s = np.linspace(0.0, 1.0, n)
        command = dense_waypoint_command(s, s_way, waypoints)
        return s, command, {"command_duration": duration, "nominal_speedup": cfg.speedup}

    if method == "discrete_adaptive_3x":
        # Fairer time-law baseline for dense waypoints: estimate curvature from the
        # emitted waypoint polyline, not from the hidden expert path. The geometry
        # remains the piecewise-linear dense command.
        kappa_way = waypoint_curvature(s_way, waypoints)
        s, duration = make_adaptive_s_values(
            cfg, lambda ss: np.interp(ss, s_way, kappa_way), slowdown_gain=0.22
        )
        command = dense_waypoint_command(s, s_way, waypoints)
        return s, command, {"command_duration": duration, "nominal_speedup": cfg.demo_duration / duration}

    if method == "bspline_fast_3x":
        duration = cfg.demo_duration / cfg.speedup
        n = int(duration / cfg.dt) + 1
        s = np.linspace(0.0, 1.0, n)
        command = spline(s)
        return s, command, {"command_duration": duration, "nominal_speedup": cfg.speedup}

    if method == "bspline_adaptive_3x":
        # Preserve the same B-spline geometry, but slow down in high-curvature spans.
        # Curvature is computed from the fitted action curve, not the hidden expert.
        s, duration = make_adaptive_s_values(
            cfg, lambda ss: bspline_curvature(spline, np.array([ss]))[0], slowdown_gain=0.22
        )
        command = spline(s)
        return s, command, {"command_duration": duration, "nominal_speedup": cfg.demo_duration / duration}

    raise KeyError(method)


def simulate_plant(command: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    pos = command[0].copy()
    vel = np.zeros(2)
    positions = []
    velocities = []
    accels = []
    settle_targets = np.repeat(command[-1:], int(cfg.settle_time / cfg.dt), axis=0)
    for target in np.concatenate([command, settle_targets], axis=0):
        err = target - pos
        desired_accel = cfg.kp * err - cfg.kd * vel
        norm = float(np.linalg.norm(desired_accel))
        if norm > cfg.max_accel:
            desired_accel *= cfg.max_accel / norm
        vel = vel + desired_accel * cfg.dt
        speed = float(np.linalg.norm(vel))
        if speed > cfg.max_speed:
            vel *= cfg.max_speed / speed
        pos = pos + vel * cfg.dt
        positions.append(pos.copy())
        velocities.append(vel.copy())
        accels.append(desired_accel.copy())
    eval_duration = (len(positions) - 1) * cfg.dt
    return np.asarray(positions), np.asarray(velocities), np.asarray(accels), eval_duration


def nearest_path_errors(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    # Small arrays, so an explicit dense nearest-neighbor distance is fine here.
    diff = points[:, None, :] - reference[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1)).min(axis=1)


def command_smoothness(command: np.ndarray, dt: float) -> Tuple[float, float]:
    vel = np.gradient(command, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    rms_acc = float(np.sqrt(np.mean(np.sum(acc * acc, axis=1))))
    rms_jerk = float(np.sqrt(np.mean(np.sum(jerk * jerk, axis=1))))
    return rms_acc, rms_jerk


def plant_jerk(accels: np.ndarray, dt: float) -> float:
    jerk = np.gradient(accels, dt, axis=0)
    return float(np.sqrt(np.mean(np.sum(jerk * jerk, axis=1))))


def predicted_scalar_count(method: str, waypoints: np.ndarray, spline: BSpline, cfg: Config) -> float:
    if method.startswith("bspline"):
        # The toy uses fixed open-uniform knots. A learned policy would only need to
        # predict control points; the full decodable representation also carries knots.
        return float(cfg.bspline_controls * waypoints.shape[1])
    return float(waypoints.size)


def full_repr_scalar_count(method: str, waypoints: np.ndarray, spline: BSpline, cfg: Config) -> float:
    if method.startswith("bspline"):
        return float(cfg.bspline_controls * waypoints.shape[1] + len(spline.t))
    return float(waypoints.size)


def run_trial(trial: int, cfg: Config, rng: np.random.Generator, keep_trace: bool = False):
    s_way, waypoints = make_noisy_waypoints(cfg, rng)
    spline = fit_bspline_actions(s_way, waypoints, cfg)
    ref = expert_path(np.linspace(0.0, 1.0, 600))
    rows = []
    traces = {}
    for method in METHOD_ORDER:
        s_cmd, command, meta = make_command(method, s_way, waypoints, spline, cfg)
        pos, vel, acc, eval_duration = simulate_plant(command, cfg)
        errors = nearest_path_errors(pos, ref)
        final_error = float(np.linalg.norm(pos[-1] - ref[-1]))
        collision = bool(np.max(errors) > cfg.ood_radius)
        success = bool((not collision) and final_error < cfg.goal_tolerance)
        rms_command_accel, rms_command_jerk = command_smoothness(command, cfg.dt)
        rows.append(
            {
                "trial": trial,
                "method": method,
                "success": int(success),
                "collision_or_ood": int(collision),
                "command_duration_s": float(meta["command_duration"]),
                "eval_duration_s": float(eval_duration),
                "effective_speedup": float(cfg.demo_duration / eval_duration),
                "nominal_command_speedup": float(meta["nominal_speedup"]),
                "max_path_error": float(np.max(errors)),
                "mean_path_error": float(np.mean(errors)),
                "final_error": final_error,
                "rms_command_accel": rms_command_accel,
                "rms_command_jerk": rms_command_jerk,
                "rms_plant_accel": float(np.sqrt(np.mean(np.sum(acc * acc, axis=1)))),
                "rms_plant_jerk": plant_jerk(acc, cfg.dt),
                "predicted_params_scalar_count": predicted_scalar_count(method, waypoints, spline, cfg),
                "full_repr_scalar_count": full_repr_scalar_count(method, waypoints, spline, cfg),
            }
        )
        if keep_trace:
            traces[method] = {
                "s": s_cmd,
                "command": command,
                "pos": pos,
                "errors": errors,
                "command_duration": meta["command_duration"],
                "eval_duration": eval_duration,
            }
    return rows, traces, waypoints, spline


def summarize(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for method in METHOD_ORDER:
        subset = [r for r in rows if r["method"] == method]
        summary[method] = {
            "success_rate": float(np.mean([r["success"] for r in subset])),
            "collision_or_ood_rate": float(np.mean([r["collision_or_ood"] for r in subset])),
            "mean_command_duration_s": float(np.mean([r["command_duration_s"] for r in subset])),
            "mean_eval_duration_s": float(np.mean([r["eval_duration_s"] for r in subset])),
            "mean_effective_speedup": float(np.mean([r["effective_speedup"] for r in subset])),
            "mean_nominal_command_speedup": float(np.mean([r["nominal_command_speedup"] for r in subset])),
            "mean_max_path_error": float(np.mean([r["max_path_error"] for r in subset])),
            "mean_final_error": float(np.mean([r["final_error"] for r in subset])),
            "mean_rms_command_accel": float(np.mean([r["rms_command_accel"] for r in subset])),
            "mean_rms_command_jerk": float(np.mean([r["rms_command_jerk"] for r in subset])),
            "mean_rms_plant_jerk": float(np.mean([r["rms_plant_jerk"] for r in subset])),
            "mean_predicted_params_scalar_count": float(np.mean([r["predicted_params_scalar_count"] for r in subset])),
            "mean_full_repr_scalar_count": float(np.mean([r["full_repr_scalar_count"] for r in subset])),
        }
    return summary


def write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_trace(traces, waypoints, cfg: Config, path: str) -> None:
    ref_s = np.linspace(0.0, 1.0, 600)
    ref = expert_path(ref_s)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.plot(ref[:, 0], ref[:, 1], "k--", lw=2, label="expert path")
    ax.scatter(waypoints[:, 0], waypoints[:, 1], s=10, color="0.7", label="noisy 10 Hz actions")
    for method in METHOD_ORDER:
        pos = traces[method]["pos"]
        ax.plot(pos[:, 0], pos[:, 1], color=COLORS[method], lw=2, label=LABELS[method])
    ax.set_title("Executed trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8)

    ax = axes[1]
    for method in METHOD_ORDER:
        tr = traces[method]
        t = np.arange(len(tr["errors"])) * cfg.dt
        ax.plot(t, tr["errors"], color=COLORS[method], lw=2, label=LABELS[method])
    ax.axhline(cfg.ood_radius, color="k", lw=1, ls="--", label="OOD/collision tube")
    ax.set_title("Distance from demonstration manifold")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("nearest-path error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(summary: Dict[str, Dict[str, float]], path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))
    metrics = [
        ("success_rate", "Success rate", 100.0),
        ("collision_or_ood_rate", "Collision/OOD rate", 100.0),
        ("mean_eval_duration_s", "Mean eval duration (s)", 1.0),
        ("mean_rms_plant_jerk", "Plant RMS jerk", 1.0),
    ]
    xs = np.arange(len(METHOD_ORDER))
    for ax, (key, title, scale) in zip(axes.flat, metrics):
        vals = [summary[m][key] * scale for m in METHOD_ORDER]
        ax.bar(xs, vals, color=[COLORS[m] for m in METHOD_ORDER])
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[m] for m in METHOD_ORDER], rotation=25, ha="right")
        if "rate" in key:
            ax.set_ylim(0, 105)
            ax.set_ylabel("%")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(summary: Dict[str, Dict[str, float]], cfg: Config, path: str) -> None:
    def pct(x):
        return f"{100*x:.1f}%"

    lines = [
        "# B-spline Action Parameterization Toy Report",
        "",
        "This run tests the B-spline Policy intuition in a tiny 2D setting: a chunked policy emits noisy low-rate waypoints, while the plant executes at 200 Hz with velocity/acceleration limits.",
        "",
        "The toy is intentionally not a diffusion model or robot benchmark. It isolates whether replacing a dense discrete action chunk with cubic B-spline knots/control points gives a smoother high-frequency command when the same geometric behavior is temporally scaled.",
        "",
        "## Latest verified run",
        "",
        f"- Trials: `{cfg.trials}`",
        f"- Low-level execution: `{1/cfg.dt:.0f} Hz`",
        f"- Policy waypoint rate: `{cfg.waypoint_hz:.0f} Hz`",
        f"- Requested speed-up: `{cfg.speedup:.1f}x`",
        f"- B-spline controls: `{cfg.bspline_controls}` per axis, degree `{cfg.degree}`",
        "",
        "| Method | Success | Collision/OOD | Eval duration | Max path error | Cmd jerk | Plant jerk | Predicted scalars | Full scalars |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in METHOD_ORDER:
        s = summary[m]
        lines.append(
            f"| {LABELS[m]} | {pct(s['success_rate'])} | {pct(s['collision_or_ood_rate'])} | "
            f"{s['mean_eval_duration_s']:.2f}s | {s['mean_max_path_error']:.3f} | "
            f"{s['mean_rms_command_jerk']:.1f} | {s['mean_rms_plant_jerk']:.1f} | "
            f"{s['mean_predicted_params_scalar_count']:.0f} | {s['mean_full_repr_scalar_count']:.0f} |"
        )
    lines += [
        "",
        "## What the toy suggests",
        "",
        "- Naively executing the same low-rate discrete action waypoints faster is brittle: it shortens nominal duration, but the limited plant lags and leaves the demonstration tube more often.",
        "- B-spline chunks preserve a continuous action curve. The fast B-spline variant uses fewer predicted scalars than the dense waypoint chunk and produces lower command jerk under the same speed-up.",
        "- Both adaptive variants test retiming rather than replanning. The B-spline version computes curvature from the fitted action curve and shows why a continuous representation is convenient for local time scaling; the discrete adaptive baseline is included to avoid attributing all recovery uniquely to B-splines.",
        "",
        "## Mapping to B-spline Policy",
        "",
        "| Toy component | B-spline Policy analogue | Simplification |",
        "| --- | --- | --- |",
        "| Noisy 10 Hz waypoints | Discrete-time action chunk from a VLA/diffusion policy | The policy is analytic plus noise, not learned. |",
        "| Cubic `BSpline(t, c, k)` | Predicted knot vector plus control points | Knots are fixed open-uniform; only control points are fit. |",
        "| 200 Hz PD plant | Low-level robot controller | Simple point-mass limits, no manipulator dynamics. |",
        "| 3x execution and curvature slowdown | Temporal scaling of continuous B-spline actions | No real scheduler, replanning, perception, or contact. |",
        "| Demo-tube OOD metric | Staying near the demonstrated action manifold | Geometric proxy, not visual/proprioceptive distribution shift. |",
        "",
        "## Generated artifacts",
        "",
        "- `outputs/bspline_action_metrics.csv`",
        "- `outputs/bspline_action_summary.json`",
        "- `outputs/bspline_action_rollout.png`",
        "- `outputs/bspline_action_monte_carlo.png`",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="B-spline action representation toy")
    parser.add_argument("--trials", type=int, default=160)
    parser.add_argument("--quick", action="store_true", help="Run a small smoke test")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = Config(seed=args.seed, trials=12 if args.quick else args.trials)
    rng = np.random.default_rng(cfg.seed)
    rows: List[Dict[str, float]] = []
    trace_payload = None
    for trial in range(cfg.trials):
        trial_rows, traces, waypoints, spline = run_trial(
            trial, cfg, rng, keep_trace=(trial == 0)
        )
        rows.extend(trial_rows)
        if trial == 0:
            trace_payload = (traces, waypoints)

    summary = summarize(rows)
    write_csv(os.path.join(OUTPUT_DIR, "bspline_action_metrics.csv"), rows)
    with open(os.path.join(OUTPUT_DIR, "bspline_action_summary.json"), "w") as f:
        json.dump({"config": cfg.__dict__, "summary": summary}, f, indent=2)
    assert trace_payload is not None
    plot_trace(*trace_payload, cfg, os.path.join(OUTPUT_DIR, "bspline_action_rollout.png"))
    plot_summary(summary, os.path.join(OUTPUT_DIR, "bspline_action_monte_carlo.png"))
    write_report(summary, cfg, os.path.join(DOCS_DIR, "bspline_action_toy_report.md"))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
