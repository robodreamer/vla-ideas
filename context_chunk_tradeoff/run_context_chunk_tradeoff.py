#!/usr/bin/env python3
"""Paired context-versus-chunking toy for hidden-mode tracking.

The target moves in a persistent hidden signed-velocity regime.  The evaluated
analytical controller sees only noisy target positions, estimates the regime
from recent temporal displacements, and executes an H-step action chunk open
loop.  Longer context can denoise the regime estimate; when context is held at
C16 in this toy, shorter chunks react sooner after switches or robot impulses.

The generated demonstration CSV contains real closed-loop oracle rollouts, but
the evaluated controller is analytical and is not trained from that CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
DOCS = BASE / "docs"
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".mplconfig"))
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Config:
    steps: int = 180
    dt: float = 0.10
    contexts: tuple[int, ...] = (1, 2, 4, 8, 16)
    horizons: tuple[int, ...] = (1, 2, 4, 8, 16)
    observation_noise: float = 0.30
    target_speed: float = 0.82
    target_velocity_noise: float = 0.10
    velocity_noise_memory: float = 0.72
    robot_drag: float = 0.32
    max_action: float = 3.2
    kp: float = 1.85
    kd: float = 1.05
    max_estimated_speed: float = 1.35
    switch_probability: float = 0.018
    disturbance_probability: float = 0.015
    disturbance_scale: float = 1.65
    success_tail_error: float = 0.68
    success_fraction: float = 0.73
    recovery_pre_window: int = 6
    recovery_impact_window: int = 10
    recovery_min_rise: float = 0.10
    recovery_fraction: float = 0.25
    recovery_hold: int = 2


def clamp(x: float, limit: float) -> float:
    return float(np.clip(x, -limit, limit))


def infer_mode(
    observations: list[float], context: int, dt: float, max_speed: float
) -> tuple[float, float, int]:
    """Estimate current target position and signed drift from recent transitions.

    ``context`` counts temporal displacements, so C1 uses the latest two noisy
    positions and is a genuine (high-variance) one-transition estimate rather
    than a forced zero.  Ck fits a line to the latest k+1 observations.  The
    estimate is clipped to keep early/partial-window rollouts bounded.
    """
    sample_count = min(len(observations), context + 1)
    ys = np.asarray(observations[-sample_count:], dtype=float)
    if sample_count < 2:
        return float(ys[-1]), 0.0, 0
    xs = np.arange(sample_count, dtype=float) * dt
    x_centered = xs - xs.mean()
    slope = float(
        np.dot(x_centered, ys - ys.mean())
        / (np.dot(x_centered, x_centered) + 1e-12)
    )
    slope = clamp(slope, max_speed)
    fitted_current = float(ys.mean() + slope * (xs[-1] - xs.mean()))
    return fitted_current, slope, sample_count - 1


def plan_chunk(
    robot_p: float,
    robot_v: float,
    estimated_target_p: float,
    estimated_target_v: float,
    horizon: int,
    cfg: Config,
) -> np.ndarray:
    """Roll the analytical PD tracker through its nominal dynamics."""
    actions: list[float] = []
    p, v = robot_p, robot_v
    for k in range(horizon):
        desired = estimated_target_p + estimated_target_v * (k + 1) * cfg.dt
        action = clamp(
            cfg.kp * (desired - p) + cfg.kd * (estimated_target_v - v),
            cfg.max_action,
        )
        actions.append(action)
        v += cfg.dt * (action - cfg.robot_drag * v)
        p += cfg.dt * v
    return np.asarray(actions, dtype=float)


def exogenous_episode(
    seed: int, cfg: Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate one controller-independent episode reused by every C/H cell."""
    rng = np.random.default_rng(seed)
    modes = np.empty(cfg.steps, dtype=float)
    modes[0] = rng.choice([-1.0, 1.0])
    for t in range(1, cfg.steps):
        modes[t] = (
            -modes[t - 1]
            if rng.random() < cfg.switch_probability
            else modes[t - 1]
        )

    velocity_noise = np.empty(cfg.steps, dtype=float)
    velocity_noise[0] = rng.normal(0.0, cfg.target_velocity_noise)
    innovation_scale = cfg.target_velocity_noise * math.sqrt(
        1.0 - cfg.velocity_noise_memory**2
    )
    for t in range(1, cfg.steps):
        velocity_noise[t] = (
            cfg.velocity_noise_memory * velocity_noise[t - 1]
            + rng.normal(0.0, innovation_scale)
        )

    impulses = np.zeros(cfg.steps, dtype=float)
    hits = rng.random(cfg.steps) < cfg.disturbance_probability
    impulses[hits] = rng.normal(0.0, cfg.disturbance_scale, int(hits.sum()))
    obs_noise = rng.normal(0.0, cfg.observation_noise, cfg.steps)
    return modes, velocity_noise, impulses, obs_noise


def event_times_from(modes: np.ndarray, impulses: np.ndarray) -> list[int]:
    switches = np.flatnonzero(np.r_[False, modes[1:] != modes[:-1]])
    disturbances = np.flatnonzero(impulses != 0.0)
    return sorted(set(int(t) for t in np.r_[switches, disturbances]))


def evaluate_recovery(
    errors: np.ndarray, event_times: list[int], cfg: Config
) -> tuple[dict[str, float], list[dict[str, float | int | bool]]]:
    """Measure recovery only after an event produces a demonstrated error rise.

    For each event, the pre-event baseline is the median of the preceding
    ``recovery_pre_window`` errors.  The event window ends at the next event or
    episode end.  An event enters the recovery metric only if the error rises
    by ``recovery_min_rise`` within the impact window.  The threshold search
    starts strictly after that post-event peak, where the recovery condition is
    necessarily unsatisfied, and recovery requires ``recovery_hold``
    consecutive samples below baseline plus ``recovery_fraction`` of the
    observed rise.  The primary restricted delay is measured from the event;
    peak-to-recovery delay is retained as a secondary diagnostic.

    Unrecovered impactful events contribute their next-event/episode censoring
    limit to the restricted mean delay, rather than disappearing from the mean.
    """
    records: list[dict[str, float | int | bool]] = []
    evaluable = impactful = recovered = 0
    restricted_delay_sum = 0.0
    peak_to_recovery_sum = 0.0

    for i, event in enumerate(event_times):
        window_end = event_times[i + 1] if i + 1 < len(event_times) else len(errors)
        enough_history = event >= cfg.recovery_pre_window
        enough_response = window_end - event >= 2
        record: dict[str, float | int | bool] = {
            "event": event,
            "window_end": window_end,
            "evaluable": bool(enough_history and enough_response),
            "impactful": False,
            "recovered": False,
        }
        if not record["evaluable"]:
            records.append(record)
            continue

        evaluable += 1
        baseline = float(
            np.median(errors[event - cfg.recovery_pre_window : event])
        )
        impact_end = min(window_end, event + cfg.recovery_impact_window)
        response = errors[event:impact_end]
        peak_offset = int(np.argmax(response))
        peak_time = event + peak_offset
        peak = float(response[peak_offset])
        rise = peak - baseline
        record.update(baseline=baseline, peak=peak, peak_time=peak_time, rise=rise)
        if rise < cfg.recovery_min_rise:
            records.append(record)
            continue

        impactful += 1
        threshold = baseline + cfg.recovery_fraction * rise
        record.update(impactful=True, threshold=threshold)
        recovery_time: int | None = None
        first_candidate = peak_time + 1
        last_start = window_end - cfg.recovery_hold
        for t in range(first_candidate, last_start + 1):
            if np.all(errors[t : t + cfg.recovery_hold] <= threshold):
                recovery_time = t
                break

        censor_delay = float(window_end - event)
        if recovery_time is None:
            restricted_delay_sum += censor_delay
            peak_to_recovery_sum += float(window_end - peak_time)
            record.update(
                recovered=False,
                restricted_delay=censor_delay,
                peak_to_recovery=float(window_end - peak_time),
            )
        else:
            recovered += 1
            event_delay = float(recovery_time - event)
            peak_delay = float(recovery_time - peak_time)
            restricted_delay_sum += event_delay
            peak_to_recovery_sum += peak_delay
            record.update(
                recovered=True,
                recovery_time=recovery_time,
                restricted_delay=event_delay,
                peak_to_recovery=peak_delay,
            )
        records.append(record)

    metrics = {
        "event_count": float(len(event_times)),
        "event_evaluable": float(evaluable),
        "event_impactful": float(impactful),
        "event_coverage": float(impactful / evaluable) if evaluable else 0.0,
        "recovery_recovered": float(recovered),
        "recovery_unrecovered": float(impactful - recovered),
        "recovery_rate": float(recovered / impactful) if impactful else 0.0,
        "recovery_delay_sum": float(restricted_delay_sum),
        "recovery_steps": (
            float(restricted_delay_sum / impactful) if impactful else float("nan")
        ),
        "peak_to_recovery_delay_sum": float(peak_to_recovery_sum),
        "peak_to_recovery_steps": (
            float(peak_to_recovery_sum / impactful) if impactful else float("nan")
        ),
    }
    return metrics, records


def rollout(
    seed: int, context: int, horizon: int, cfg: Config, record: bool = False
) -> dict:
    modes, velocity_noise, impulses, obs_noise = exogenous_episode(seed, cfg)
    target_p = 0.0
    robot_p, robot_v = -0.7, 0.0
    observations: list[float] = []
    errors: list[float] = []
    actions: list[float] = []
    estimates: list[float] = []
    fitted_targets: list[float] = []
    target_positions: list[float] = []
    target_velocities: list[float] = []
    robot_positions: list[float] = []
    mode_correct: list[float] = []
    persistent_mode_correct: list[float] = []
    replans = 0
    chunk = np.zeros(horizon, dtype=float)
    estimate = 0.0
    fitted_target = 0.0
    last_switch = 0

    for t in range(cfg.steps):
        if t > 0 and modes[t] != modes[t - 1]:
            last_switch = t
        observed_target = target_p + obs_noise[t]
        observations.append(float(observed_target))

        if t % horizon == 0:
            fitted_target, estimate, transitions = infer_mode(
                observations, context, cfg.dt, cfg.max_estimated_speed
            )
            chunk = plan_chunk(
                robot_p, robot_v, fitted_target, estimate, horizon, cfg
            )
            replans += 1
            if transitions >= context:
                correct = float(np.sign(estimate) == modes[t])
                mode_correct.append(correct)
                if t - last_switch >= max(cfg.contexts):
                    persistent_mode_correct.append(correct)

        action = float(chunk[t % horizon])

        # Apply the hidden regime and robot impulse before recording post-event
        # error.  Thus errors[t] is the first response sample for an event at t.
        target_v = float(modes[t] * cfg.target_speed + velocity_noise[t])
        target_p += cfg.dt * target_v
        robot_v += cfg.dt * (action - cfg.robot_drag * robot_v) + impulses[t]
        robot_p += cfg.dt * robot_v

        errors.append(abs(target_p - robot_p))
        actions.append(action)
        estimates.append(estimate)
        fitted_targets.append(fitted_target)
        target_positions.append(target_p)
        target_velocities.append(target_v)
        robot_positions.append(robot_p)

    errors_a = np.asarray(errors, dtype=float)
    actions_a = np.asarray(actions, dtype=float)
    event_times = event_times_from(modes, impulses)
    recovery_metrics, recovery_records = evaluate_recovery(
        errors_a, event_times, cfg
    )
    result = {
        "success": float(
            np.mean(errors_a[-45:] < cfg.success_tail_error)
            >= cfg.success_fraction
        ),
        "tracking_error": float(np.mean(errors_a)),
        "tail_error": float(np.mean(errors_a[-45:])),
        "jerk": float(np.mean(np.diff(actions_a) ** 2) / cfg.dt**2),
        "planning_calls": float(replans),
        "mode_accuracy": float(np.mean(mode_correct)) if mode_correct else float("nan"),
        "persistent_mode_accuracy": (
            float(np.mean(persistent_mode_correct))
            if persistent_mode_correct
            else float("nan")
        ),
        **recovery_metrics,
    }
    if record:
        result["trace"] = {
            "error": errors_a.tolist(),
            "action": actions_a.tolist(),
            "mode": modes.tolist(),
            "estimate": estimates,
            "impulse": impulses.tolist(),
            "target_position": target_positions,
            "target_velocity": target_velocities,
            "fitted_target": fitted_targets,
            "robot_position": robot_positions,
            "event_times": event_times,
            "recovery_records": recovery_records,
        }
    return result


def expert_demonstration(seed: int, cfg: Config) -> list[dict]:
    """Generate a real closed-loop oracle rollout; hidden state is analysis-only."""
    modes, velocity_noise, impulses, obs_noise = exogenous_episode(seed, cfg)
    target_p = 0.0
    robot_p, robot_v = -0.7, 0.0
    rows: list[dict] = []
    for step in range(cfg.steps):
        target_v = float(modes[step] * cfg.target_speed + velocity_noise[step])
        observed_target = target_p + obs_noise[step]
        expert_action = float(
            plan_chunk(robot_p, robot_v, target_p, target_v, 1, cfg)[0]
        )
        target_p_next = target_p + cfg.dt * target_v
        robot_v_next = (
            robot_v
            + cfg.dt * (expert_action - cfg.robot_drag * robot_v)
            + impulses[step]
        )
        robot_p_next = robot_p + cfg.dt * robot_v_next
        rows.append(
            {
                "demo_seed": seed,
                "step": step,
                "noisy_target_observation": observed_target,
                "hidden_mode": modes[step],
                "target_position": target_p,
                "target_velocity": target_v,
                "robot_position": robot_p,
                "robot_velocity": robot_v,
                "expert_action": expert_action,
                "robot_impulse": impulses[step],
                "next_target_position": target_p_next,
                "next_robot_position": robot_p_next,
            }
        )
        target_p = target_p_next
        robot_p, robot_v = robot_p_next, robot_v_next
    return rows


def finite_values(values: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(values), dtype=float)
    return a[np.isfinite(a)]


def mean_sem(values: Iterable[float]) -> tuple[float, float]:
    a = finite_values(values)
    if len(a) == 0:
        return float("nan"), float("nan")
    sem = float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0
    return float(a.mean()), sem


def aggregate_cell(cell: list[dict], context: int, horizon: int) -> dict:
    aggregate: dict[str, float | int] = {"context": context, "horizon": horizon}
    mean_keys = (
        "success",
        "tracking_error",
        "tail_error",
        "jerk",
        "planning_calls",
        "mode_accuracy",
        "persistent_mode_accuracy",
    )
    for key in mean_keys:
        aggregate[key], aggregate[key + "_sem"] = mean_sem(r[key] for r in cell)

    for key in (
        "event_count",
        "event_evaluable",
        "event_impactful",
        "recovery_recovered",
        "recovery_unrecovered",
        "recovery_delay_sum",
        "peak_to_recovery_delay_sum",
    ):
        aggregate[key] = float(sum(r[key] for r in cell))
    aggregate["event_coverage"] = (
        aggregate["event_impactful"] / aggregate["event_evaluable"]
        if aggregate["event_evaluable"]
        else 0.0
    )
    aggregate["recovery_rate"] = (
        aggregate["recovery_recovered"] / aggregate["event_impactful"]
        if aggregate["event_impactful"]
        else 0.0
    )
    aggregate["recovery_steps"] = (
        aggregate["recovery_delay_sum"] / aggregate["event_impactful"]
        if aggregate["event_impactful"]
        else float("nan")
    )
    aggregate["peak_to_recovery_steps"] = (
        aggregate["peak_to_recovery_delay_sum"] / aggregate["event_impactful"]
        if aggregate["event_impactful"]
        else float("nan")
    )
    aggregate["recovery_steps_sem"] = mean_sem(
        r["recovery_steps"] for r in cell
    )[1]
    aggregate["event_coverage_sem"] = mean_sem(
        r["event_coverage"] for r in cell if r["event_evaluable"] > 0
    )[1]
    aggregate["recovery_rate_sem"] = mean_sem(
        r["recovery_rate"] for r in cell if r["event_impactful"] > 0
    )[1]
    return aggregate


def recovery_unit_checks(cfg: Config) -> dict[str, bool]:
    """Hand-computed checks for event ordering, impact gating, and censoring."""
    errors = np.asarray(
        [0.20] * 6
        + [0.35, 0.70, 0.48, 0.29, 0.23, 0.22]
        + [0.21, 0.20],
        dtype=float,
    )
    metrics, records = evaluate_recovery(errors, [6], cfg)
    threshold_after_peak = bool(
        records[0]["impactful"]
        and records[0]["peak_time"] == 7
        and records[0]["recovery_time"] == 9
        and metrics["recovery_steps"] == 3.0
    )

    no_rise_errors = np.asarray([0.20] * 14, dtype=float)
    no_rise, _ = evaluate_recovery(no_rise_errors, [6], cfg)
    no_rise_excluded = bool(no_rise["event_impactful"] == 0.0)

    censored_errors = np.asarray(
        [0.20] * 6 + [0.45, 0.75, 0.72, 0.70, 0.69, 0.68, 0.67, 0.66],
        dtype=float,
    )
    censored, censored_records = evaluate_recovery(censored_errors, [6, 10], cfg)
    next_event_censors = bool(
        censored_records[0]["impactful"]
        and not censored_records[0]["recovered"]
        and censored_records[0]["window_end"] == 10
        and censored_records[0]["restricted_delay"] == 4.0
    )
    return {
        "recovery_search_starts_after_post_event_peak": threshold_after_peak,
        "events_without_error_rise_are_excluded": no_rise_excluded,
        "next_event_censors_recovery_window": next_event_censors,
    }


def sanity_checks(cfg: Config) -> dict:
    persistent_cfg = replace(
        cfg, switch_probability=0.0, disturbance_probability=0.0
    )
    short = [
        rollout(1000 + i, 1, 1, persistent_cfg)["persistent_mode_accuracy"]
        for i in range(32)
    ]
    long = [
        rollout(1000 + i, 16, 1, persistent_cfg)["persistent_mode_accuracy"]
        for i in range(32)
    ]
    reactive = [rollout(2000 + i, 8, 1, cfg) for i in range(48)]
    open_loop = [rollout(2000 + i, 8, 16, cfg) for i in range(48)]

    def pooled_recovery(rows: list[dict]) -> float:
        n = sum(r["event_impactful"] for r in rows)
        return float(sum(r["recovery_delay_sum"] for r in rows) / n)

    unit = recovery_unit_checks(cfg)
    short_accuracy = float(np.nanmean(short))
    long_accuracy = float(np.nanmean(long))
    short_recovery = pooled_recovery(reactive)
    long_recovery = pooled_recovery(open_loop)
    checks: dict[str, float | bool] = {
        "c1_persistent_mode_accuracy": short_accuracy,
        "c16_persistent_mode_accuracy": long_accuracy,
        "c1_is_meaningful": bool(short_accuracy > 0.52),
        "long_context_improves_persistent_mode_inference": bool(
            long_accuracy > short_accuracy + 0.15
        ),
        "horizon_1_restricted_recovery_steps": short_recovery,
        "horizon_16_restricted_recovery_steps": long_recovery,
        "fixed_schedule_h1_lower_recovery": bool(short_recovery < long_recovery),
        "fixed_schedule_recovery_gap_h16_minus_h1": float(
            long_recovery - short_recovery
        ),
        **unit,
    }
    checks["passed"] = bool(
        checks["c1_is_meaningful"]
        and checks["long_context_improves_persistent_mode_inference"]
        and all(unit.values())
    )
    return checks


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def paired_recovery_comparison(
    trial_rows: list[dict], context: int = 16, bootstrap_samples: int = 20_000
) -> dict[str, float | int]:
    """Compare H16 against H1 on paired seeds with a deterministic bootstrap."""
    by_key = {
        (int(row["seed"]), int(row["context"]), int(row["horizon"])): float(
            row["recovery_steps"]
        )
        for row in trial_rows
    }
    seeds = sorted(
        {
            int(row["seed"])
            for row in trial_rows
            if int(row["context"]) == context
        }
    )
    differences = np.asarray(
        [by_key[(seed, context, 16)] - by_key[(seed, context, 1)] for seed in seeds],
        dtype=float,
    )
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        raise RuntimeError("paired recovery comparison has no finite seed pairs")
    rng = np.random.default_rng(20260818)
    draws = rng.integers(
        0, len(differences), size=(bootstrap_samples, len(differences))
    )
    bootstrap_means = differences[draws].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "context": context,
        "short_horizon": 1,
        "long_horizon": 16,
        "paired_seed_count": int(len(differences)),
        "mean_h16_minus_h1_steps": float(differences.mean()),
        "bootstrap_95_low": float(low),
        "bootstrap_95_high": float(high),
        "bootstrap_samples": bootstrap_samples,
    }


def annotation_color(im, value: float) -> str:
    r, g, b, _ = im.cmap(im.norm(value))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.58 else "white"


def plot_heatmaps(summary: list[dict], cfg: Config) -> None:
    metrics = [
        ("success", "Success rate", "viridis", ".2f"),
        ("tracking_error", "Mean tracking error", "magma_r", ".2f"),
        (
            "persistent_mode_accuracy",
            "Persistent-mode accuracy",
            "viridis",
            ".2f",
        ),
        ("recovery_steps", "Restricted recovery (steps)", "magma_r", ".1f"),
        ("jerk", "Jerk proxy", "magma_r", ".1f"),
        ("planning_calls", "Planning calls / episode", "cividis", ".0f"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11, 12), constrained_layout=True)
    for ax, (key, title, cmap, fmt) in zip(axes.flat, metrics):
        data = np.array(
            [
                [
                    next(
                        r[key]
                        for r in summary
                        if r["context"] == c and r["horizon"] == h
                    )
                    for h in cfg.horizons
                ]
                for c in cfg.contexts
            ],
            dtype=float,
        )
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("Open-loop chunk horizon H")
        ax.set_ylabel("Temporal context C (transitions)")
        ax.set_xticks(range(len(cfg.horizons)), cfg.horizons)
        ax.set_yticks(range(len(cfg.contexts)), cfg.contexts)
        for i in range(len(cfg.contexts)):
            for j in range(len(cfg.horizons)):
                ax.text(
                    j,
                    i,
                    format(data[i, j], fmt),
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="semibold",
                    color=annotation_color(im, data[i, j]),
                )
        fig.colorbar(im, ax=ax, shrink=0.84)
    fig.suptitle("Context versus chunking: 25-cell paired sweep", fontsize=14)
    fig.savefig(OUT / "tradeoff_heatmaps.png", dpi=180)
    plt.close(fig)


def plot_frontier(summary: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    sizes = np.asarray([r["planning_calls"] for r in summary], dtype=float)
    sc = ax.scatter(
        [r["recovery_steps"] for r in summary],
        [r["jerk"] for r in summary],
        c=[r["horizon"] for r in summary],
        s=25 + 2.0 * sizes,
        cmap="plasma",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
    )
    for r in summary:
        key = (r["context"], r["horizon"])
        if key in {(1, 1), (16, 1), (16, 16), (8, 4)}:
            offset = (-6, -8) if key == (1, 1) else (5, 5)
            ax.annotate(
                f"C{r['context']}/H{r['horizon']}",
                (r["recovery_steps"], r["jerk"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                ha="right" if key == (1, 1) else "left",
                va="top" if key == (1, 1) else "bottom",
            )
    ax.margins(x=0.08, y=0.12)
    ax.set_xlabel("Restricted event recovery delay (lower is better)")
    ax.set_ylabel("Jerk proxy (lower is smoother)")
    ax.set_title("Reactivity–smoothness trade-off; size = planning calls")
    fig.colorbar(sc, ax=ax, label="Chunk horizon H")
    fig.savefig(OUT / "pareto_tradeoff.png", dpi=180)
    plt.close(fig)


def plot_rollout(cfg: Config, seed: int) -> None:
    short = rollout(seed, 16, 1, cfg, record=True)["trace"]
    long = rollout(seed, 16, 16, cfg, record=True)["trace"]
    t = np.arange(cfg.steps)
    fig, axes = plt.subplots(
        4, 1, sharex=True, figsize=(10, 8.8), constrained_layout=True
    )
    true_regime_velocity = np.asarray(short["mode"]) * cfg.target_speed
    axes[0].step(
        t,
        true_regime_velocity,
        where="post",
        label="hidden regime velocity",
        color="black",
        lw=1.3,
    )
    axes[0].plot(t, short["estimate"], label="C16 estimate, H1", alpha=0.85)
    axes[0].plot(t, long["estimate"], label="C16 estimate, H16", alpha=0.85)
    axes[0].set_ylim(-1.5, 1.5)
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].set_ylabel("velocity")

    axes[1].plot(t, short["error"], label="H1 reactive")
    axes[1].plot(t, long["error"], label="H16 open loop")
    axes[1].axhline(cfg.success_tail_error, color="gray", ls="--", lw=0.8)
    axes[1].legend(fontsize=8)
    axes[1].set_ylabel("absolute error")

    axes[2].plot(t, short["action"], label="H1 action")
    axes[2].plot(t, long["action"], label="H16 action")
    axes[2].legend(fontsize=8)
    axes[2].set_ylabel("action")

    impulses = np.asarray(long["impulse"], dtype=float)
    impulse_times = np.flatnonzero(impulses)
    axes[3].stem(
        impulse_times,
        impulses[impulse_times],
        linefmt="C3-",
        markerfmt="C3o",
        basefmt=" ",
        label="robot impulse",
    )
    axes[3].step(t, short["mode"], where="post", color="black", lw=1.0, label="mode")
    axes[3].legend(ncol=2, fontsize=8)
    axes[3].set_ylabel("events")
    axes[3].set_xlabel("time step")
    for ax in axes:
        for event in short["event_times"]:
            ax.axvline(event, color="0.65", lw=0.55, alpha=0.45)
    fig.savefig(OUT / "paired_rollout.png", dpi=180)
    plt.close(fig)


def command_text(trials: int, seed: int) -> str:
    return (
        "cd /home/andypark/Projects/repos/vla-ideas\n"
        "/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \\\n"
        "  context_chunk_tradeoff/run_context_chunk_tradeoff.py \\\n"
        f"  --trials {trials} --seed {seed}"
    )


def write_reports(
    summary: list[dict],
    trials: int,
    seed: int,
    cfg: Config,
    sanity: dict,
    recovery_comparison: dict,
) -> None:
    def row(c: int, h: int) -> dict:
        return next(
            r for r in summary if r["context"] == c and r["horizon"] == h
        )

    baselines = [row(1, 1), row(16, 1), row(16, 16)]
    command = command_text(trials, seed)
    md_rows = "\n".join(
        f"| C{r['context']} / H{r['horizon']} | {r['success']:.1%} | "
        f"{r['tracking_error']:.3f} | {r['persistent_mode_accuracy']:.1%} | "
        f"{r['recovery_steps']:.2f} | {r['event_coverage']:.1%} | "
        f"{int(r['recovery_unrecovered'])}/{int(r['event_impactful'])} | "
        f"{r['jerk']:.1f} | {r['planning_calls']:.0f} |"
        for r in baselines
    )
    md = f"""# Context versus Chunking Toy

## Narrow question

When a controller must infer a hidden persistent signed-velocity regime from noisy target positions, how does more temporal context trade against longer open-loop action chunks after unpredictable regime switches and robot disturbances?

This is a synthetic mechanism test, not a VLA-paper reproduction or evidence about a physical robot.

## Setup

The target moves with hidden drift $m_t v_0$, where $m_t \in {{-1,+1}}$ persists and occasionally flips. The controller observes noisy target position only. At a planning call, C1 estimates velocity from the latest observed displacement (two adjacent positions); Ck fits a bounded least-squares slope to the latest k temporal displacements. Thus C1 is noisy but meaningful, and the inferred quantity matches the declared hidden signed-velocity mode. The controller rolls an analytical PD tracker forward and commits H actions open loop.

The same mode sequence, target-velocity noise, robot impulses, and observation noise are reused across all 25 C/H configurations for each seed. The generated demonstration CSV contains real closed-loop oracle trajectories with hidden state retained only for analysis. The evaluated controller is analytical and is **not trained from the CSV**.

## Metrics

- **Success:** at least {cfg.success_fraction:.0%} of the final 45 post-transition errors are below {cfg.success_tail_error:.2f}.
- **Tracking error:** episode mean absolute target–robot separation.
- **Persistent-mode accuracy:** sign agreement between the estimated velocity and hidden mode at planning calls after a full C-transition window and at least {max(cfg.contexts)} steps since the latest mode switch.
- **Restricted event-to-recovery delay:** each event is applied before its first response error. Its baseline is the median of the prior {cfg.recovery_pre_window} errors. An event is included only if error rises by at least {cfg.recovery_min_rise:.2f} within {cfg.recovery_impact_window} steps. The threshold search begins strictly after the post-event peak, while the reported delay is measured from the event. Recovery requires {cfg.recovery_hold} consecutive errors below baseline plus {cfg.recovery_fraction:.0%} of the observed rise. The window ends at the next event or episode end; unrecovered events contribute that censoring limit.
- **Event coverage:** impactful events divided by evaluable scheduled events. **Unrecovered** is reported explicitly among impactful events.
- **Jerk proxy:** mean squared first difference of action divided by $dt^2$.
- **Planning calls:** analytical policy invocations per episode, not wall-clock compute.

## Latest generated result

```bash
{command}
```

| Controller | Success | Error | Persistent mode accuracy | Recovery | Coverage | Unrecovered | Jerk | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{md_rows}

The deterministic mechanism checks passed: persistent-mode accuracy was {sanity['c1_persistent_mode_accuracy']:.1%} for C1 and {sanity['c16_persistent_mode_accuracy']:.1%} for C16, while hand-computed event traces verify impact gating, post-peak threshold search, and next-event censoring. The small fixed surprise schedule gives {sanity['horizon_1_restricted_recovery_steps']:.2f} event-to-recovery steps for H1 and {sanity['horizon_16_restricted_recovery_steps']:.2f} for H16, but is treated as descriptive rather than a pass criterion. Across the full paired C16 sweep, the per-seed H16-minus-H1 recovery gap is {recovery_comparison['mean_h16_minus_h1_steps']:.2f} steps (deterministic bootstrap 95% interval [{recovery_comparison['bootstrap_95_low']:.2f}, {recovery_comparison['bootstrap_95_high']:.2f}]).

![Full metric sweep](../outputs/tradeoff_heatmaps.png)

![Reactivity–smoothness trade-off](../outputs/pareto_tradeoff.png)

![Paired rollout](../outputs/paired_rollout.png)

## Interpretation and limits

In this persistent signed-drift regime, longer context improves mode inference by averaging position noise. It need not improve immediately after a switch because old-regime samples remain in the window. Longer chunks sharply reduce planning calls and reduce the action-difference jerk proxy on average across context settings; at fixed C16, however, H16 is less smooth than H1, so smoothness is not claimed to improve monotonically. With context held at C16, H1 has lower restricted event-to-recovery delay than H16 because it can replace stale plans sooner; that reactivity claim is not generalized to every context setting. No single C/H cell is encoded as an automatic winner.

The result is limited to a one-dimensional analytical controller, synthetic matched distributions, hand-set dynamics, and a response-conditioned recovery metric. It omits images, language, learned policies, contact, inference latency, and hardware. Event coverage must be read alongside recovery because events without a demonstrable error rise are excluded rather than mislabeled as instant recoveries.
"""
    (DOCS / "context_chunk_tradeoff_report.md").write_text(md)

    tex_rows = "\n".join(
        f"C{r['context']} / H{r['horizon']} & {100*r['success']:.1f}\\% & "
        f"{r['tracking_error']:.3f} & {100*r['persistent_mode_accuracy']:.1f}\\% & "
        f"{r['recovery_steps']:.2f} & {100*r['event_coverage']:.1f}\\% & "
        f"{int(r['recovery_unrecovered'])}/{int(r['event_impactful'])} & "
        f"{r['jerk']:.1f} & {r['planning_calls']:.0f} \\\\"
        for r in baselines
    )
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

\hypersetup{{
  colorlinks=true,
  linkcolor=blue!50!black,
  urlcolor=blue!50!black,
  citecolor=blue!50!black
}}

\title{{Context versus Chunking:\\A Hidden Signed-Velocity Control Toy}}
\author{{Andy Park\\\href{{mailto:andypark.purdue@gmail.com}}{{andypark.purdue@gmail.com}}}}
\date{{August 18, 2026}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This note tests a narrow control mechanism: longer temporal context can denoise inference of a persistent hidden signed-velocity regime, while action-chunk horizon controls how often the policy can refresh after surprises. The final sweep contains 25 context/horizon cells and {trials} paired trials per cell. Persistent-mode accuracy rises from {100*sanity['c1_persistent_mode_accuracy']:.1f}\% for one displacement to {100*sanity['c16_persistent_mode_accuracy']:.1f}\% for 16 displacements in the fixed persistent-regime check. With context held at C16, the mean paired H16-minus-H1 restricted event-to-recovery gap is {recovery_comparison['mean_h16_minus_h1_steps']:.2f} steps, with a deterministic bootstrap 95\% interval of [{recovery_comparison['bootstrap_95_low']:.2f}, {recovery_comparison['bootstrap_95_high']:.2f}]. This is a synthetic analytical-controller experiment, not a trained VLA or a real-robot claim.
\end{{abstract}}

\section{{Motivation}}

Observation history and action chunking solve different problems. History can average observation noise and reveal motion that is ambiguous in a single displacement. Chunking amortizes planning calls and can smooth actions, but a committed open-loop chunk cannot react until the next planning boundary. This experiment places both mechanisms in one paired toy without assuming that one configuration must dominate every metric.

\section{{References and Scope}}

The experiment follows standard least-squares line fitting for the temporal estimator and standard feedback-control ideas for the analytical tracker \cite{{nist-ls,feedback}}. It is not a reproduction of either reference, a VLA paper, or a learned action-chunking system. All numerical results below are local measurements generated by the repository script.

\section{{Toy Setup}}

A one-dimensional target has hidden mode $m_t\in\{{-1,+1\}}$. During a persistent regime its velocity is
\[
  v_t^\star = m_t v_0 + \epsilon_t,
\]
where $\epsilon_t$ is correlated zero-mean process noise. The mode flips with a fixed per-step probability. The robot receives only noisy target position $y_t=p_t^\star+\eta_t$; hidden mode, target velocity, and disturbances are retained only for analysis.

At each planning call, C$k$ fits a line to the latest $k+1$ positions, i.e. the latest $k$ temporal displacements. C1 therefore uses $y_t-y_{{t-1}}$ and is a meaningful but noisy baseline. The slope $\hat v_t$ is clipped to $[-{cfg.max_estimated_speed:.2f},{cfg.max_estimated_speed:.2f}]$ to bound early partial-window estimates. A nominal PD tracker then predicts target position using $\hat v_t$ and emits an H-step action chunk. Those H actions execute open loop unless the episode ends.

The sweep crosses $C,H\in\{{1,2,4,8,16\}}$. For each trial seed, every cell receives exactly the same hidden-mode sequence, correlated target-velocity noise, observation noise, and robot impulses. Twelve separate closed-loop oracle demonstrations are also generated. Their oracle actions use true target state, but the evaluated controller is analytical and is \textbf{{not trained from the demonstration CSV}}.

\section{{Metrics}}

Let $e_t=|p_t^\star-p_t|$ denote error after the mode-dependent target transition and robot impulse at step $t$.
\begin{{itemize}}[leftmargin=1.5em]
  \item \textbf{{Success:}} at least {100*cfg.success_fraction:.0f}\% of the final 45 errors satisfy $e_t<{cfg.success_tail_error:.2f}$.
  \item \textbf{{Tracking error:}} episode mean $T^{{-1}}\sum_t e_t$.
  \item \textbf{{Persistent-mode accuracy:}} $\operatorname{{sign}}(\hat v_t)=m_t$ at planning calls with a complete C-transition window and at least {max(cfg.contexts)} steps since the most recent switch. The common persistence guard separates denoising from unavoidable post-switch window lag.
  \item \textbf{{Jerk proxy:}} $\frac{{1}}{{T-1}}\sum_{{t=1}}^{{T-1}}(u_t-u_{{t-1}})^2/\Delta t^2$. This is an action-difference proxy, not physical jerk.
  \item \textbf{{Planning calls:}} number of analytical policy invocations per episode; it is not measured wall-clock latency.
\end{{itemize}}

\subsection{{Event recovery definition}}

For an event at step $s$, the event is applied before $e_s$ is recorded. The baseline $b_s$ is the median of the preceding {cfg.recovery_pre_window} errors. The response window is $[s,n)$, where $n$ is the next event or episode end. Within its first {cfg.recovery_impact_window} samples, let $q$ be the post-event peak at step $p$. The event is \emph{{impactful}} only when $q-b_s\ge {cfg.recovery_min_rise:.2f}$. Threshold search starts strictly after $p$, where the recovery condition is necessarily false, and ends at the first later pair of consecutive errors below
\[
  b_s + {cfg.recovery_fraction:.2f}(q-b_s).
\]
The primary delay is measured from the event step $s$, not from the peak; peak-to-recovery delay is retained only as a diagnostic. If no such pair appears before $n$, the event is unrecovered and contributes the censoring delay $n-s$ to the restricted mean. Event coverage is impactful divided by evaluable scheduled events; unrecovered counts are reported explicitly. Events without enough pre-event history or response room are not evaluable, and events without a demonstrated rise are excluded rather than counted as zero-delay recoveries.

\section{{Results}}

The committed artifacts were generated with:
\begin{{verbatim}}
{command}
\end{{verbatim}}

\begin{{table}}[H]
\centering
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrrr}}
\toprule
Controller & Success & Error & Mode acc. & Recovery & Coverage & Unrec. & Jerk & Calls \\
\midrule
{tex_rows}
\bottomrule
\end{{tabular}}}}
\caption{{Selected baselines from {trials} paired trials per cell (seed {seed}). Recovery is the pooled restricted mean over impactful events. Coverage is impactful/evaluable; Unrec. is unrecovered/impactful.}}
\label{{tab:baselines}}
\end{{table}}

The deterministic persistent-regime check gives C1 mode accuracy {100*sanity['c1_persistent_mode_accuracy']:.1f}\% and C16 accuracy {100*sanity['c16_persistent_mode_accuracy']:.1f}\%. Hand-computed traces verify post-peak threshold search, no-rise exclusion, and next-event censoring. The small fixed surprise schedule gives {sanity['horizon_1_restricted_recovery_steps']:.2f} event-to-recovery steps for H1 and {sanity['horizon_16_restricted_recovery_steps']:.2f} for H16, but is descriptive rather than a pass criterion. Across the full paired C16 sweep, the per-seed H16-minus-H1 gap is {recovery_comparison['mean_h16_minus_h1_steps']:.2f} steps (deterministic bootstrap 95\% interval [{recovery_comparison['bootstrap_95_low']:.2f}, {recovery_comparison['bootstrap_95_high']:.2f}]).

\begin{{figure}}[H]
  \centering
  \includegraphics[width=\textwidth]{{../outputs/tradeoff_heatmaps.png}}
  \caption{{The full 25-cell sweep. Mode-inference and jerk evidence are shown directly alongside task, recovery, and planning metrics. Annotation color adapts to the cell luminance.}}
  \label{{fig:heatmaps}}
\end{{figure}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.88\textwidth]{{../outputs/pareto_tradeoff.png}}
  \caption{{Reactivity--smoothness trade-off. Longer chunks reduce planning calls and reduce the action-difference jerk proxy on average across contexts. With context held at C16, H1 has lower restricted event-to-recovery delay than H16.}}
  \label{{fig:tradeoff}}
\end{{figure}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=\textwidth]{{../outputs/paired_rollout.png}}
  \caption{{One paired exogenous rollout. Estimates are bounded during warm-up. Gray lines mark mode switches or impulses; the H16 controller can retain a stale plan between boundaries.}}
  \label{{fig:rollout}}
\end{{figure}}

\section{{Interpretation}}

In the intended persistent regime, longer context materially improves inference because it averages noisy position displacements. The same history can lag immediately after a switch because old-regime samples remain in the window, so context is not claimed to help monotonically at every instant. Longer action chunks retain the expected planning-call advantage and reduce the action-difference jerk proxy on average across context settings, but smoothness is not monotonic at fixed C16. With context held at C16, H1 has lower restricted event-to-recovery delay than H16 because it can replace stale plans sooner; that reactivity claim is not generalized to every context setting. These are metric trade-offs, not an encoded automatic winner.

\clearpage
\section{{Limitations and Follow-ups}}

The tracker is one-dimensional, its policy is analytical, and train/test distribution shift is absent because there is no training stage. Position noise is Gaussian; hidden modes are binary; disturbances are velocity impulses; and the controller has no images, language, contact, latency, or hardware. Recovery is conditional on a measurable error response, so coverage and unrecovered accounting are essential to interpretation. The response threshold, persistence guard, and success tolerance are local definitions rather than community benchmarks.

Useful follow-ups are to pre-register the recovery constants on a calibration set, evaluate more switch frequencies and non-Gaussian noise, separate sensing from planning latency, replace position observations with learned visual features, and repeat the paired design with a trained chunking policy.

\begin{{thebibliography}}{{9}}
\bibitem{{nist-ls}} NIST/SEMATECH. \emph{{Linear Least Squares Regression}}. \url{{https://www.itl.nist.gov/div898/handbook/pmd/section1/pmd141.htm}}
\bibitem{{feedback}} K. J. \AA str\"om and R. M. Murray. \emph{{Feedback Systems: An Introduction for Scientists and Engineers}}. Princeton University Press, 2008. \url{{https://fbswiki.org/wiki/index.php/Main_Page}}
\end{{thebibliography}}

\end{{document}}
"""
    (DOCS / "context_chunk_tradeoff_report.tex").write_text(tex)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    trials = 12 if args.smoke else args.trials
    cfg = Config()
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    sanity = sanity_checks(cfg)
    if not sanity["passed"]:
        raise RuntimeError(f"deterministic sanity checks failed: {sanity}")

    trial_rows: list[dict] = []
    summary: list[dict] = []
    for context in cfg.contexts:
        for horizon in cfg.horizons:
            cell: list[dict] = []
            for i in range(trials):
                result = rollout(args.seed + i, context, horizon, cfg)
                result.update(
                    seed=args.seed + i, context=context, horizon=horizon
                )
                trial_rows.append(result)
                cell.append(result)
            summary.append(aggregate_cell(cell, context, horizon))

    recovery_comparison = paired_recovery_comparison(trial_rows)
    write_csv(OUT / "sweep_trials.csv", trial_rows)
    write_csv(OUT / "sweep_summary.csv", summary)
    demo_rows: list[dict] = []
    for demo_seed in range(args.seed, args.seed + 12):
        demo_rows.extend(expert_demonstration(demo_seed, cfg))
    write_csv(OUT / "expert_demonstrations.csv", demo_rows)
    metrics = {
        "config": asdict(cfg),
        "seed": args.seed,
        "trials_per_cell": trials,
        "paired_trial_count": trials * len(cfg.contexts) * len(cfg.horizons),
        "sanity": sanity,
        "paired_recovery_comparison": recovery_comparison,
        "summary": summary,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (OUT / "sanity_check.json").write_text(
        json.dumps(sanity, indent=2) + "\n"
    )

    plot_heatmaps(summary, cfg)
    plot_frontier(summary)
    plot_rollout(cfg, args.seed + 5)
    write_reports(summary, trials, args.seed, cfg, sanity, recovery_comparison)

    print(
        f"wrote {len(trial_rows)} paired trials, {len(summary)} cells, "
        f"and reports to {BASE}"
    )
    for c, h in [(1, 1), (16, 1), (16, 16)]:
        r = next(
            x for x in summary if x["context"] == c and x["horizon"] == h
        )
        print(
            f"C{c}/H{h}: success={r['success']:.1%}, "
            f"error={r['tracking_error']:.3f}, "
            f"mode={r['persistent_mode_accuracy']:.1%}, "
            f"recovery={r['recovery_steps']:.2f}, "
            f"coverage={r['event_coverage']:.1%}, "
            f"unrecovered={int(r['recovery_unrecovered'])}/"
            f"{int(r['event_impactful'])}, jerk={r['jerk']:.1f}, "
            f"calls={r['planning_calls']:.0f}"
        )


if __name__ == "__main__":
    main()
