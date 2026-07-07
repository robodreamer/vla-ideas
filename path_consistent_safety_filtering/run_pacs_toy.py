#!/usr/bin/env python3
"""Toy PACS-style path-consistent safety filtering experiment.

This is not a PACS or Ruckig implementation. It isolates the key mechanism:
keep the diffusion-policy chunk geometry fixed, and alter only the scalar time law
(progress speed) when a dynamic obstacle makes the nominal trajectory unsafe.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


@dataclass(frozen=True)
class Scenario:
    name: str
    x_cross: float
    y_offset: float
    t_cross: float
    obstacle_speed: float
    obstacle_radius: float = 0.105
    robot_radius: float = 0.045
    seed: int = 0


@dataclass
class Rollout:
    method: str
    scenario: Scenario
    t: np.ndarray
    p: np.ndarray
    s: np.ndarray
    v_s: np.ndarray
    obstacle: np.ndarray
    min_clearance: np.ndarray
    ood: np.ndarray
    speed_scale: np.ndarray
    success: bool
    collision: bool
    timed_out: bool
    path_error_mean: float
    path_error_max: float
    jerk_proxy: float
    duration: float


def build_path(n: int = 1200) -> Dict[str, np.ndarray]:
    # Smooth path with enough curvature that sideways CBF corrections visibly leave
    # the demonstration tube.
    s = np.linspace(0.0, 1.0, n)
    x = s
    y = 0.18 * np.sin(2 * np.pi * s) + 0.055 * np.sin(5 * np.pi * s)
    pts = np.stack([x, y], axis=1)
    d = np.gradient(pts, axis=0)
    tangents = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    return {"s": s, "pts": pts, "tangents": tangents, "normals": normals}


PATH = build_path()
DEMO_TUBE_RADIUS = 0.035
SAFETY_MARGIN = 0.21
HARD_MARGIN = 0.155
DT = 0.035
MAX_T = 7.0
NOMINAL_SPEED = 0.23  # scalar progress per second, not Euclidean speed
MAX_ACCEL = 0.95
MAX_JERK = 10.0


def path_at(s_value: float) -> np.ndarray:
    s_value = float(np.clip(s_value, 0.0, 1.0))
    return np.array([
        np.interp(s_value, PATH["s"], PATH["pts"][:, 0]),
        np.interp(s_value, PATH["s"], PATH["pts"][:, 1]),
    ])


def tangent_at(s_value: float) -> np.ndarray:
    s_value = float(np.clip(s_value, 0.0, 1.0))
    tx = np.interp(s_value, PATH["s"], PATH["tangents"][:, 0])
    ty = np.interp(s_value, PATH["s"], PATH["tangents"][:, 1])
    t = np.array([tx, ty])
    return t / (np.linalg.norm(t) + 1e-9)


def nearest_path(p: np.ndarray) -> Tuple[float, float, np.ndarray]:
    d = PATH["pts"] - p[None, :]
    idx = int(np.argmin(np.sum(d * d, axis=1)))
    nearest = PATH["pts"][idx]
    return float(PATH["s"][idx]), float(np.linalg.norm(p - nearest)), nearest


def obstacle_at(t: float, sc: Scenario) -> np.ndarray:
    # Obstacle crosses vertically through/near the path at x_cross around t_cross.
    return np.array([sc.x_cross, sc.y_offset + sc.obstacle_speed * (t - sc.t_cross)])


def min_future_clearance(s: float, t: float, v_guess: float, sc: Scenario, horizon: float = 1.35) -> float:
    checks = np.linspace(0.0, horizon, 34)
    min_c = 10.0
    for tau in checks:
        p = path_at(s + v_guess * tau)
        o = obstacle_at(t + tau, sc)
        clearance = np.linalg.norm(p - o) - (sc.obstacle_radius + sc.robot_radius)
        min_c = min(min_c, float(clearance))
    return min_c


def update_scalar_speed(v: float, a: float, target_v: float) -> Tuple[float, float]:
    # A small 1D jerk/accel-limited OTG stand-in. Ruckig would solve this exactly;
    # this Euler version is enough to demonstrate that slowdown is a time-law change.
    desired_a = np.clip((target_v - v) / DT, -MAX_ACCEL, MAX_ACCEL)
    da = np.clip(desired_a - a, -MAX_JERK * DT, MAX_JERK * DT)
    a = float(np.clip(a + da, -MAX_ACCEL, MAX_ACCEL))
    v = float(np.clip(v + a * DT, 0.0, NOMINAL_SPEED))
    return v, a


def rollout(method: str, sc: Scenario) -> Rollout:
    rng = np.random.default_rng(sc.seed)
    t_values: List[float] = []
    p_values: List[np.ndarray] = []
    s_values: List[float] = []
    v_values: List[float] = []
    o_values: List[np.ndarray] = []
    clearances: List[float] = []
    oods: List[float] = []
    scales: List[float] = []

    s = 0.0
    p = path_at(0.0).copy()
    v_s = NOMINAL_SPEED
    accel = 0.0
    collision = False

    for k in range(int(MAX_T / DT)):
        t = k * DT
        obs = obstacle_at(t, sc)
        nearest_s, path_err, nearest_p = nearest_path(p)
        if method == "nominal":
            target_v = NOMINAL_SPEED
            v_s = target_v
            s = min(1.0, s + v_s * DT)
            p = path_at(s)
            speed_scale = 1.0
        elif method == "pacs_time_law":
            # Keep the waypoint/path geometry fixed; only slow scalar progress when
            # reachable occupancies along the future path threaten the obstacle tube.
            future = min_future_clearance(s, t, NOMINAL_SPEED, sc)
            if future <= HARD_MARGIN:
                speed_scale = 0.0
            elif future < SAFETY_MARGIN:
                speed_scale = (future - HARD_MARGIN) / (SAFETY_MARGIN - HARD_MARGIN)
            else:
                speed_scale = 1.0
            target_v = NOMINAL_SPEED * float(np.clip(speed_scale, 0.0, 1.0))
            v_s, accel = update_scalar_speed(v_s, accel, target_v)
            s = min(1.0, s + v_s * DT)
            p = path_at(s)
        elif method == "reactive_cbf_like":
            # A deliberately simple reactive shield: add sideways velocity away from
            # the obstacle, then let the policy try to recover to the nearest path.
            # This often remains safe, but creates OOD configurations.
            s = max(s, nearest_s)
            tangent = tangent_at(s)
            desired = NOMINAL_SPEED * tangent
            vec = p - obs
            dist = np.linalg.norm(vec) + 1e-9
            influence = 0.42
            if dist < influence:
                repel = (vec / dist) * 0.56 * ((influence - dist) / influence) ** 2
            else:
                repel = np.zeros(2)
            # Recovery weakens once outside the demo tube, modeling policy degradation.
            recovery_gain = 1.6 if path_err < DEMO_TUBE_RADIUS else 0.45
            recovery = recovery_gain * (nearest_p - p)
            noise = rng.normal(0.0, 0.008, size=2) if path_err > DEMO_TUBE_RADIUS else 0.0
            vel = desired + repel + recovery + noise
            speed_cap = 0.48
            vel = vel / max(1.0, np.linalg.norm(vel) / speed_cap)
            p = p + vel * DT
            s = max(0.0, nearest_path(p)[0])
            v_s = float(np.dot(vel, tangent_at(s)))
            speed_scale = 1.0
        else:
            raise ValueError(method)

        clearance = float(np.linalg.norm(p - obs) - (sc.obstacle_radius + sc.robot_radius))
        if clearance < 0.0:
            collision = True

        _, path_err, _ = nearest_path(p)
        t_values.append(t)
        p_values.append(p.copy())
        s_values.append(s)
        v_values.append(v_s)
        o_values.append(obs.copy())
        clearances.append(clearance)
        oods.append(float(path_err > DEMO_TUBE_RADIUS))
        scales.append(float(speed_scale))

        if s >= 0.999:
            break

    t_arr = np.array(t_values)
    p_arr = np.vstack(p_values)
    s_arr = np.array(s_values)
    v_arr = np.array(v_values)
    o_arr = np.vstack(o_values)
    c_arr = np.array(clearances)
    ood_arr = np.array(oods)
    scale_arr = np.array(scales)
    path_errors = np.array([nearest_path(pp)[1] for pp in p_arr])
    if len(v_arr) >= 3:
        jerk_proxy = float(np.mean(np.abs(np.diff(v_arr, n=2))) / (DT * DT))
    else:
        jerk_proxy = 0.0
    success = bool(s_arr[-1] >= 0.999 and not collision and np.mean(ood_arr) < 0.35)
    return Rollout(
        method=method,
        scenario=sc,
        t=t_arr,
        p=p_arr,
        s=s_arr,
        v_s=v_arr,
        obstacle=o_arr,
        min_clearance=c_arr,
        ood=ood_arr,
        speed_scale=scale_arr,
        success=success,
        collision=collision,
        timed_out=bool(s_arr[-1] < 0.999),
        path_error_mean=float(path_errors.mean()),
        path_error_max=float(path_errors.max()),
        jerk_proxy=jerk_proxy,
        duration=float(t_arr[-1]) if len(t_arr) else 0.0,
    )


def make_scenarios(n: int, seed: int, quick: bool) -> List[Scenario]:
    rng = np.random.default_rng(seed)
    if quick:
        n = min(n, 48)
    scenarios: List[Scenario] = []
    for i in range(n):
        x = float(rng.uniform(0.25, 0.78))
        y_path = path_at(x)[1]
        speed = float(rng.uniform(0.28, 0.55)) * (1 if rng.random() > 0.5 else -1)
        # Put the crossing near nominal arrival time, but jitter enough to create
        # easy, hard, and irrelevant cases.
        t_nom = x / NOMINAL_SPEED
        t_cross = float(np.clip(t_nom + rng.normal(0.0, 0.42), 0.6, 4.8))
        # obstacle_at(t_cross) == [x_cross, y_offset], so y_offset is the
        # crossing coordinate on/near the policy path.
        y_offset = float(y_path + rng.normal(0.0, 0.018))
        radius = float(rng.uniform(0.085, 0.12))
        scenarios.append(
            Scenario(
                name=f"trial_{i:03d}",
                x_cross=x,
                y_offset=y_offset,
                t_cross=t_cross,
                obstacle_speed=speed,
                obstacle_radius=radius,
                seed=seed + i,
            )
        )
    return scenarios


def summarize(rows: List[Rollout]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for method in sorted(set(r.method for r in rows)):
        rr = [r for r in rows if r.method == method]
        out[method] = {
            "success_rate": float(np.mean([r.success for r in rr])),
            "collision_rate": float(np.mean([r.collision for r in rr])),
            "timeout_rate": float(np.mean([r.timed_out for r in rr])),
            "ood_rate": float(np.mean([np.mean(r.ood) for r in rr])),
            "mean_duration_s": float(np.mean([r.duration for r in rr])),
            "mean_min_clearance": float(np.mean([np.min(r.min_clearance) for r in rr])),
            "mean_path_error": float(np.mean([r.path_error_mean for r in rr])),
            "max_path_error": float(np.max([r.path_error_max for r in rr])),
            "jerk_proxy": float(np.mean([r.jerk_proxy for r in rr])),
        }
    return out


def write_metrics(rows: List[Rollout], summary: Dict[str, Dict[str, float]]) -> None:
    OUT.mkdir(exist_ok=True)
    with (OUT / "pacs_toy_metrics.csv").open("w", newline="") as f:
        fieldnames = [
            "trial", "method", "success", "collision", "timed_out", "duration_s",
            "min_clearance", "ood_fraction", "path_error_mean", "path_error_max", "jerk_proxy",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "trial": r.scenario.name,
                "method": r.method,
                "success": int(r.success),
                "collision": int(r.collision),
                "timed_out": int(r.timed_out),
                "duration_s": f"{r.duration:.3f}",
                "min_clearance": f"{np.min(r.min_clearance):.4f}",
                "ood_fraction": f"{np.mean(r.ood):.4f}",
                "path_error_mean": f"{r.path_error_mean:.4f}",
                "path_error_max": f"{r.path_error_max:.4f}",
                "jerk_proxy": f"{r.jerk_proxy:.4f}",
            })
    with (OUT / "pacs_toy_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


def plot_rollout(example: Dict[str, Rollout]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    pts = PATH["pts"]
    ax.plot(pts[:, 0], pts[:, 1], "k--", lw=1.4, label="DP chunk path")
    ax.fill_between(pts[:, 0], pts[:, 1] - DEMO_TUBE_RADIUS, pts[:, 1] + DEMO_TUBE_RADIUS, color="0.8", alpha=0.35, label="demo tube (proxy)")
    colors = {"nominal": "tab:gray", "reactive_cbf_like": "tab:red", "pacs_time_law": "tab:blue"}
    labels = {"nominal": "raw policy", "reactive_cbf_like": "reactive CBF-like", "pacs_time_law": "PACS time-law"}
    for method, r in example.items():
        ax.plot(r.p[:, 0], r.p[:, 1], color=colors[method], lw=2.2, label=labels[method])
    # obstacle path from the example scenario
    rr = next(iter(example.values()))
    ax.plot(rr.obstacle[:, 0], rr.obstacle[:, 1], color="tab:orange", lw=1.2, label="moving obstacle")
    for j in np.linspace(0, len(rr.obstacle) - 1, 4, dtype=int):
        circ = plt.Circle(rr.obstacle[j], rr.scenario.obstacle_radius, color="tab:orange", alpha=0.18)
        ax.add_patch(circ)
    ax.set_title("Geometry: PACS changes speed, not path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    for method, r in example.items():
        ax.plot(r.t, r.v_s, color=colors[method], lw=2, label=labels[method])
    ax.axhline(NOMINAL_SPEED, color="k", ls="--", lw=1, alpha=0.4)
    ax.set_title("Scalar progress speed")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("path progress speed ds/dt")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "pacs_toy_rollout.png", dpi=180)
    plt.close(fig)


def plot_summary(summary: Dict[str, Dict[str, float]]) -> None:
    methods = ["nominal", "reactive_cbf_like", "pacs_time_law"]
    names = ["raw", "reactive", "PACS"]
    metrics = ["success_rate", "collision_rate", "ood_rate", "mean_duration_s"]
    titles = ["success", "collision", "OOD time", "duration [s]"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, metric, title in zip(axes, metrics, titles):
        vals = [summary[m][metric] for m in methods]
        ax.bar(names, vals, color=["tab:gray", "tab:red", "tab:blue"])
        ax.set_title(title)
        if metric.endswith("rate") or metric == "ood_rate":
            ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Monte Carlo dynamic obstacle sweep")
    fig.tight_layout()
    fig.savefig(OUT / "pacs_toy_monte_carlo.png", dpi=180)
    plt.close(fig)


def action_chunk_sanity() -> Dict[str, float]:
    # Action chunk/waypoint sanity check: identical waypoints, different time law.
    way_s = np.linspace(0.05, 0.55, 12)
    waypoints = np.array([path_at(s) for s in way_s])
    dense_nom = np.array([path_at(s) for s in np.linspace(way_s[0], way_s[-1], 200)])
    # Slowdown samples fewer progress values by the same wall-clock time but remains
    # a subset of the same geometric curve.
    dense_slow = np.array([path_at(s) for s in np.linspace(way_s[0], way_s[-1] * 0.72, 200)])
    nearest_errors = []
    for p in dense_slow:
        nearest_errors.append(nearest_path(p)[1])
    return {
        "waypoint_count": int(len(waypoints)),
        "nominal_path_points": int(len(dense_nom)),
        "slowdown_path_points": int(len(dense_slow)),
        "slowdown_max_geometry_error": float(np.max(nearest_errors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    scenarios = make_scenarios(args.trials, args.seed, args.quick)
    methods = ["nominal", "reactive_cbf_like", "pacs_time_law"]
    rows: List[Rollout] = []
    for sc in scenarios:
        for method in methods:
            rows.append(rollout(method, sc))
    summary = summarize(rows)
    sanity = action_chunk_sanity()
    summary["action_chunk_sanity"] = sanity  # type: ignore[assignment]
    write_metrics(rows, summary)

    # Pick a hard representative where PACS succeeds and the reactive shield leaves tube.
    example_scenario = None
    for sc in scenarios:
        rr = {r.method: r for r in rows if r.scenario.name == sc.name}
        if rr["pacs_time_law"].success and np.mean(rr["reactive_cbf_like"].ood) > 0.12:
            example_scenario = sc
            break
    if example_scenario is None:
        example_scenario = scenarios[len(scenarios) // 2]
    example = {r.method: r for r in rows if r.scenario.name == example_scenario.name}
    plot_rollout(example)
    plot_summary(summary)  # type: ignore[arg-type]

    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT / 'pacs_toy_metrics.csv'}")
    print(f"wrote {OUT / 'pacs_toy_summary.json'}")
    print(f"wrote {OUT / 'pacs_toy_rollout.png'}")
    print(f"wrote {OUT / 'pacs_toy_monte_carlo.png'}")


if __name__ == "__main__":
    main()
