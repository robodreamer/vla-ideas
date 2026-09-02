#!/usr/bin/env python3
"""Deterministic toy for adaptive action execution from cross-attention entropy.

This is a mechanism probe, not a reproduction of arXiv:2609.00908. A point robot
tracks a staged 2-D reference with a frozen chunk policy. The policy's future
prediction becomes less grounded at a stage-dependent horizon. Compared stopping
rules use fixed lengths, rollout/state error, ensemble disagreement, cross-attention
entropy, a shuffled-attention control, or future-aware oracle action error.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import time
from dataclasses import asdict, dataclass, replace
from typing import Any

BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = [
    "fixed_3",
    "fixed_6",
    "fixed_12",
    "fixed_24",
    "state_error",
    "ensemble_uncertainty",
    "attention_entropy",
    "shuffled_attention",
    "oracle_stop",
]
LABELS = {
    "fixed_3": "Fixed 3",
    "fixed_6": "Fixed 6",
    "fixed_12": "Fixed 12",
    "fixed_24": "Fixed 24",
    "state_error": "State-error gate",
    "ensemble_uncertainty": "Ensemble uncertainty",
    "attention_entropy": "Attention entropy",
    "shuffled_attention": "Shuffled attention",
    "oracle_stop": "Oracle stop",
}
COLORS = {
    "fixed_3": "#9ecae9",
    "fixed_6": "#4292c6",
    "fixed_12": "#2171b5",
    "fixed_24": "#084594",
    "state_error": "#f28e2b",
    "ensemble_uncertainty": "#e15759",
    "attention_entropy": "#59a14f",
    "shuffled_attention": "#b07aa1",
    "oracle_stop": "#333333",
}
CONDITIONS = ["clean", "disturbance", "distractor", "combined"]


@dataclass(frozen=True)
class Config:
    dt: float = 0.08
    episode_steps: int = 168
    predicted_horizon: int = 24
    token_count: int = 48
    attention_window: int = 4
    entropy_ratio: float = 0.94
    entropy_stability: float = 0.012
    max_action: float = 5.0
    kp: float = 5.4
    kd: float = 2.9
    drag_true: float = 0.24
    drag_model: float = 0.15
    action_gain: float = 0.93
    plan_noise: float = 0.030
    late_bias: float = 1.25
    ensemble_samples: int = 8
    state_error_threshold: float = 0.55
    ensemble_threshold: float = 0.24
    oracle_error_threshold: float = 0.34
    handoff_action_scale: float = 0.72
    handoff_jitter: float = 0.20
    success_tail_steps: int = 48
    success_tail_rmse: float = 0.48
    success_final_error: float = 0.52


@dataclass(frozen=True)
class Mode:
    trials: int
    diagnostic_trials: int
    robustness_severities: tuple[float, ...]


MODES = {
    "sanity": Mode(4, 2, (0.0, 1.0)),
    "quick": Mode(32, 12, (0.0, 0.75, 1.5)),
    "full": Mode(120, 40, (0.0, 0.5, 1.0, 1.5, 2.0)),
}


def stable_seed(seed: int, *parts: Any) -> int:
    value = int(seed) & 0xFFFFFFFF
    for part in parts:
        for byte in str(part).encode("utf-8"):
            value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-9)


def rotate(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=float)


def reference_trajectory(cfg: Config, trial_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Smooth staged reference: coarse approach, curved transfer, precise insertion."""
    rng = np.random.default_rng(trial_seed)
    t = np.arange(cfg.episode_steps + cfg.predicted_horizon + 2) * cfg.dt
    phase = rng.uniform(-0.35, 0.35)
    ref = np.zeros((len(t), 2), dtype=float)
    stages = np.zeros(len(t), dtype=int)
    s1, s2 = 58, 116

    # Coarse transport: broad, slowly curving motion.
    q = t[:s1]
    ref[:s1, 0] = -1.5 + 0.70 * q
    ref[:s1, 1] = 0.45 * np.sin(0.55 * q + phase)

    # Interaction: a tighter arc around an object.
    q = t[s1:s2] - t[s1]
    anchor = ref[s1 - 1]
    ref[s1:s2, 0] = anchor[0] + 1.30 * np.sin(0.58 * q)
    ref[s1:s2, 1] = anchor[1] + 1.05 * (1.0 - np.cos(0.58 * q))
    stages[s1:s2] = 1

    # Precision: short alternating corrections converging to a socket.
    q = t[s2:] - t[s2]
    anchor = ref[s2 - 1]
    decay = np.exp(-0.20 * q)
    ref[s2:, 0] = anchor[0] + 0.55 * (1.0 - np.exp(-0.38 * q))
    ref[s2:, 1] = anchor[1] + 0.34 * decay * np.sin(2.25 * q + phase)
    stages[s2:] = 2
    return ref, stages


def scenario(cfg: Config, seed: int, trial: int, condition: str, severity: float = 1.0) -> dict[str, Any]:
    trial_seed = stable_seed(seed, trial, "world")
    rng = np.random.default_rng(trial_seed)
    ref, stages = reference_trajectory(cfg, trial_seed)
    n = len(ref)
    process_noise = rng.normal(0.0, 0.016, size=(n, 2))
    impulses = np.zeros((n, 2), dtype=float)
    has_disturbance = condition in {"disturbance", "combined"}
    if has_disturbance and severity > 0:
        for base in (43, 91, 132):
            idx = int(np.clip(base + rng.integers(-7, 8), 4, cfg.episode_steps - 4))
            direction = unit(rng.normal(size=2))
            impulses[idx] += severity * rng.uniform(1.3, 2.0) * direction
            impulses[idx + 1] += severity * rng.uniform(0.35, 0.65) * direction
    distractor = severity if condition in {"distractor", "combined"} else 0.0
    init = np.r_[ref[0] + rng.normal(0.0, 0.18, 2), rng.normal(0.0, 0.08, 2)]
    return {
        "reference": ref,
        "stages": stages,
        "process_noise": process_noise,
        "impulses": impulses,
        "distractor": distractor,
        "initial_state": init,
        "trial_seed": trial_seed,
    }


def true_step(state: np.ndarray, action: np.ndarray, noise: np.ndarray, impulse: np.ndarray, cfg: Config) -> np.ndarray:
    p, v = state[:2], state[2:]
    a = cfg.action_gain * action - cfg.drag_true * v + noise + impulse
    v2 = v + cfg.dt * a
    p2 = p + cfg.dt * v + 0.5 * cfg.dt**2 * a
    return np.r_[p2, v2]


def model_step(state: np.ndarray, action: np.ndarray, cfg: Config) -> np.ndarray:
    p, v = state[:2], state[2:]
    a = action - cfg.drag_model * v
    return np.r_[p + cfg.dt * v + 0.5 * cfg.dt**2 * a, v + cfg.dt * a]


def pd_action(state: np.ndarray, target: np.ndarray, target_v: np.ndarray, cfg: Config) -> np.ndarray:
    raw = cfg.kp * (target - state[:2]) + cfg.kd * (target_v - state[2:])
    return cfg.max_action * np.tanh(raw / cfg.max_action)


def stage_grounding_horizon(stage: int, local_speed: float) -> float:
    base = (9.0, 5.0, 3.5)[int(stage)]
    return float(np.clip(base - 0.8 * max(local_speed - 0.65, 0.0), 3.0, 14.0))


def attention_distribution(mix: float, relevant: int, tokens: int) -> np.ndarray:
    p = np.full(tokens, mix / tokens, dtype=float)
    p[relevant % tokens] += 1.0 - mix
    return p / p.sum()


def entropy_from_probs(p: np.ndarray) -> float:
    return float(-np.sum(p * np.log(np.maximum(p, 1e-12))))


def oracle_plan(state: np.ndarray, t: int, world: dict[str, Any], cfg: Config) -> np.ndarray:
    """Counterfactual ideal actions with future reference and disturbances exposed."""
    s = state.copy()
    actions = []
    ref = world["reference"]
    for j in range(cfg.predicted_horizon):
        idx = min(t + j, len(ref) - 2)
        target_v = (ref[idx + 1] - ref[idx]) / cfg.dt
        a = pd_action(s, ref[idx], target_v, cfg)
        actions.append(a)
        s = true_step(s, a, world["process_noise"][idx], world["impulses"][idx], cfg)
    return np.asarray(actions)


def make_plan(state: np.ndarray, t: int, world: dict[str, Any], cfg: Config, seed: int, trial: int, replan: int) -> dict[str, Any]:
    ref, stages = world["reference"], world["stages"]
    idx0 = min(t, len(ref) - 3)
    v0 = (ref[idx0 + 1] - ref[max(idx0 - 1, 0)]) / (2.0 * cfg.dt)
    speed = float(np.linalg.norm(v0))
    stage = int(stages[idx0])
    grounding = stage_grounding_horizon(stage, speed)
    rng = np.random.default_rng(stable_seed(seed, trial, replan, "policy"))
    drift_dir = unit(rotate(v0 if speed > 1e-5 else rng.normal(size=2), rng.choice([-1.0, 1.0]) * (0.70 + 0.18 * stage)))
    relevant = int((stage * 13 + trial * 7 + replan * 3) % cfg.token_count)

    s = state.copy()
    plan, model_states, entropies, q_values = [], [], [], []
    ensemble = np.zeros((cfg.ensemble_samples, cfg.predicted_horizon, 2), dtype=float)
    ensemble_rng = np.random.default_rng(stable_seed(seed, trial, replan, "ensemble_all"))
    for j in range(cfg.predicted_horizon):
        horizon_step = j + 1
        # Attention dispersion anticipates the point at which the common-mode plan error rises.
        # The offset also matches the paper rule's k-step evidence window and +k boundary.
        q_attention = 1.0 / (1.0 + math.exp(-(horizon_step - grounding) / 1.35))
        q_error = 1.0 / (1.0 + math.exp(-(horizon_step - (grounding + 6.0)) / 1.35))
        q_values.append(q_error)
        pred_target = ref[idx0] + horizon_step * cfg.dt * v0
        # Common-mode drift dominates late predictions; ensemble disagreement sees only part of it.
        pred_target = pred_target + cfg.late_bias * q_error * (horizon_step / cfg.predicted_horizon) * drift_dir
        pred_v = v0 + 0.35 * q_error * drift_dir
        base = pd_action(s, pred_target, pred_v, cfg)
        base += rng.normal(0.0, cfg.plan_noise * (1.0 + 2.0 * q_error), size=2)
        base = np.clip(base, -cfg.max_action, cfg.max_action)
        plan.append(base)
        s = model_step(s, base, cfg)
        model_states.append(s.copy())

        sample_scale = 0.035 + 0.17 * q_error + 0.025 * stage
        ensemble[:, j] = np.clip(
            base[None, :] + ensemble_rng.normal(0.0, sample_scale, size=(cfg.ensemble_samples, 2)),
            -cfg.max_action,
            cfg.max_action,
        )

        mix = 0.08 + 0.919 * q_attention
        if world["distractor"] > 0:
            # Irrelevant visual clutter spreads attention early and adds horizon-wise flicker.
            flicker = 0.045 * math.sin(1.73 * horizon_step + 0.4 * trial)
            clutter = np.clip(world["distractor"] * (0.10 + flicker), 0.0, 0.32)
            mix = float(np.clip(mix + clutter * (1.0 - mix), 0.0, 0.9995))
        p = attention_distribution(mix, relevant + j // 5, cfg.token_count)
        entropies.append(entropy_from_probs(p))

    return {
        "actions": np.asarray(plan),
        "model_states": np.asarray(model_states),
        "ensemble": ensemble,
        "entropy": np.asarray(entropies),
        "q": np.asarray(q_values),
        "grounding": grounding,
        "oracle_actions": oracle_plan(state, t, world, cfg),
    }


def sustained_threshold(values: np.ndarray, threshold: float, window: int, above: bool = True) -> int:
    for i in range(0, len(values) - window + 1):
        segment = values[i : i + window]
        if bool(np.all(segment >= threshold) if above else np.all(segment <= threshold)):
            return i + 1
    return len(values)


def attention_horizon(entropy: np.ndarray, cfg: Config) -> int:
    k = cfg.attention_window
    if len(entropy) < 2 * k:
        return len(entropy)
    smooth = np.convolve(entropy, np.ones(k) / k, mode="valid")
    delta = np.diff(smooth)
    high = cfg.entropy_ratio * math.log(cfg.token_count)
    # Paper-inspired rule: k consecutive high smoothed values plus stable mean change.
    for j in range(0, len(smooth) - k):
        if np.min(smooth[j : j + k]) >= high and abs(float(np.mean(delta[j : j + k]))) < cfg.entropy_stability:
            return int(np.clip(j + k + 1, 1, cfg.predicted_horizon))
    return cfg.predicted_horizon


def select_horizon(method: str, plan: dict[str, Any], state: np.ndarray, cfg: Config, seed: int, trial: int, replan: int) -> tuple[int, np.ndarray]:
    hmax = cfg.predicted_horizon
    signal = np.zeros(hmax, dtype=float)
    if method.startswith("fixed_"):
        return min(int(method.split("_")[1]), hmax), signal
    if method == "state_error":
        linear = state[:2][None, :] + cfg.dt * np.arange(1, hmax + 1)[:, None] * state[2:][None, :]
        signal = np.linalg.norm(plan["model_states"][:, :2] - linear, axis=1)
        return sustained_threshold(signal, cfg.state_error_threshold, 2), signal
    if method == "ensemble_uncertainty":
        signal = np.mean(np.std(plan["ensemble"], axis=0), axis=1)
        return sustained_threshold(signal, cfg.ensemble_threshold, 2), signal
    if method in {"attention_entropy", "shuffled_attention"}:
        signal = plan["entropy"].copy()
        if method == "shuffled_attention":
            rng = np.random.default_rng(stable_seed(seed, trial, replan, "shuffle"))
            signal = signal[rng.permutation(len(signal))]
        return attention_horizon(signal, cfg), signal
    if method == "oracle_stop":
        signal = np.linalg.norm(plan["actions"] - plan["oracle_actions"], axis=1)
        return sustained_threshold(signal, cfg.oracle_error_threshold, 2), signal
    raise ValueError(method)


def rollout(method: str, world: dict[str, Any], cfg: Config, seed: int, trial: int, keep_trace: bool = False) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    state = world["initial_state"].copy()
    errors, horizons, selected_signals, true_action_errors = [], [], [], []
    states = [state.copy()]
    t, replans = 0, 0
    prev_action = np.zeros(2, dtype=float)
    while t < cfg.episode_steps:
        plan = make_plan(state, t, world, cfg, seed, trial, replans)
        h, signal = select_horizon(method, plan, state, cfg, seed, trial, replans)
        h = int(np.clip(h, 1, min(cfg.predicted_horizon, cfg.episode_steps - t)))
        horizons.append(h)
        selected_signals.append(float(signal[min(h - 1, len(signal) - 1)]))
        true_action_errors.append(float(np.linalg.norm(plan["actions"][min(h - 1, cfg.predicted_horizon - 1)] - plan["oracle_actions"][min(h - 1, cfg.predicted_horizon - 1)])))
        for j in range(h):
            action = plan["actions"][j].copy()
            if j == 0:
                # Replanning boundaries are not free: a partial hold and small discontinuity.
                jrng = np.random.default_rng(stable_seed(seed, trial, replans, "handoff"))
                action = cfg.handoff_action_scale * action + (1.0 - cfg.handoff_action_scale) * prev_action
                action += jrng.normal(0.0, cfg.handoff_jitter, 2)
            action = np.clip(action, -cfg.max_action, cfg.max_action)
            state = true_step(state, action, world["process_noise"][t], world["impulses"][t], cfg)
            prev_action = action
            errors.append(float(np.linalg.norm(state[:2] - world["reference"][t + 1])))
            states.append(state.copy())
            t += 1
            if t >= cfg.episode_steps:
                break
        replans += 1

    tail = np.asarray(errors[-cfg.success_tail_steps :])
    final_error = errors[-1]
    tail_rmse = float(np.sqrt(np.mean(tail**2)))
    tracking_rmse = float(np.sqrt(np.mean(np.asarray(errors) ** 2)))
    success = tail_rmse < cfg.success_tail_rmse and final_error < cfg.success_final_error
    metric = {
        "method": method,
        "success": int(success),
        "tracking_rmse": tracking_rmse,
        "tail_rmse": tail_rmse,
        "final_error": float(final_error),
        "inference_calls": replans,
        "mean_horizon": float(np.mean(horizons)),
        "horizon_std": float(np.std(horizons)),
        "p10_horizon": float(np.percentile(horizons, 10)),
        "p90_horizon": float(np.percentile(horizons, 90)),
        "selected_action_error": float(np.mean(true_action_errors)),
        "compute_normalized": float(replans / cfg.episode_steps),
    }
    trace = {
        "states": np.asarray(states),
        "errors": np.asarray(errors),
        "horizons": np.asarray(horizons),
        "selected_signals": np.asarray(selected_signals),
    }
    return metric, trace if keep_trace else {}


def evaluate(seed: int, trials: int, cfg: Config, conditions: list[str], severity: float = 1.0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        for trial in range(trials):
            world = scenario(cfg, seed, trial, condition, severity)
            for method in METHOD_ORDER:
                metric, _ = rollout(method, world, cfg, seed, trial)
                rows.append({"condition": condition, "severity": severity, "trial": trial, **metric})
    return rows


def group_summary(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)
    out = []
    metrics = ["success", "tracking_rmse", "tail_rmse", "final_error", "inference_calls", "mean_horizon", "selected_action_error"]
    for key, vals in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        row = {k: v for k, v in zip(keys, key)}
        for metric in metrics:
            arr = np.asarray([float(v[metric]) for v in vals])
            row[metric] = float(np.mean(arr))
            row[f"{metric}_sem"] = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
        row["n"] = len(vals)
        out.append(row)
    return out


def signal_diagnostics(seed: int, trials: int, cfg: Config) -> list[dict[str, Any]]:
    rows = []
    for condition in CONDITIONS:
        for trial in range(trials):
            world = scenario(cfg, seed, trial, condition, 1.0)
            t = (19 * trial + 23) % (cfg.episode_steps - cfg.predicted_horizon - 1)
            state = world["initial_state"].copy()
            for u in range(t):
                target_v = (world["reference"][u + 1] - world["reference"][u]) / cfg.dt
                a = pd_action(state, world["reference"][u], target_v, cfg)
                state = true_step(state, a, world["process_noise"][u], world["impulses"][u], cfg)
            plan = make_plan(state, t, world, cfg, seed, trial, 0)
            action_error = np.linalg.norm(plan["actions"] - plan["oracle_actions"], axis=1)
            linear = state[:2][None, :] + cfg.dt * np.arange(1, cfg.predicted_horizon + 1)[:, None] * state[2:][None, :]
            signals = {
                "attention_entropy": plan["entropy"] / math.log(cfg.token_count),
                "state_error": np.linalg.norm(plan["model_states"][:, :2] - linear, axis=1),
                "ensemble_uncertainty": np.mean(np.std(plan["ensemble"], axis=0), axis=1),
            }
            for name, signal in signals.items():
                corr = float(np.corrcoef(signal, action_error)[0, 1]) if np.std(signal) > 1e-9 else 0.0
                rows.append({"condition": condition, "trial": trial, "signal": name, "pearson_action_error": corr, "mean_signal": float(np.mean(signal)), "mean_action_error": float(np.mean(action_error))})
    return rows


def robustness_sweep(seed: int, trials: int, cfg: Config, severities: tuple[float, ...]) -> list[dict[str, Any]]:
    rows = []
    for severity in severities:
        condition = "clean" if severity == 0 else "combined"
        sweep = evaluate(seed + 211, trials, cfg, [condition], severity)
        rows.extend(group_summary(sweep, ("severity", "method")))
    return rows


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def lookup(summary: list[dict[str, Any]], method: str, condition: str | None = None) -> dict[str, Any]:
    for row in summary:
        if row.get("method") == method and (condition is None or row.get("condition") == condition):
            return row
    raise KeyError((method, condition))


def plot_headline(summary: list[dict[str, Any]], out: pathlib.Path) -> None:
    clean = [lookup(summary, m, "clean") for m in METHOD_ORDER]
    combined = [lookup(summary, m, "combined") for m in METHOD_ORDER]
    x = np.arange(len(METHOD_ORDER))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(x - 0.18, [r["success"] for r in clean], 0.36, color=[COLORS[m] for m in METHOD_ORDER], alpha=0.75, label="clean")
    axes[0].bar(x + 0.18, [r["success"] for r in combined], 0.36, color=[COLORS[m] for m in METHOD_ORDER], hatch="//", label="disturbance+distractor")
    axes[0].set_ylabel("success rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(x, [r["tracking_rmse"] for r in combined], color=[COLORS[m] for m in METHOD_ORDER])
    axes[1].set_ylabel("tracking RMSE (combined)")
    axes[2].bar(x, [r["inference_calls"] for r in combined], color=[COLORS[m] for m in METHOD_ORDER])
    axes[2].set_ylabel("inference calls / episode")
    for ax in axes:
        ax.set_xticks(x, [LABELS[m] for m in METHOD_ORDER], rotation=43, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Accuracy–efficiency comparison for execution-horizon rules")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_conditions(summary: list[dict[str, Any]], out: pathlib.Path) -> None:
    selected = ["fixed_3", "fixed_6", "fixed_12", "fixed_24", "state_error", "ensemble_uncertainty", "attention_entropy", "shuffled_attention", "oracle_stop"]
    x = np.arange(len(CONDITIONS))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    for method in selected:
        vals = [lookup(summary, method, c) for c in CONDITIONS]
        axes[0].plot(x, [r["success"] for r in vals], marker="o", lw=2, color=COLORS[method], label=LABELS[method])
        axes[1].plot(x, [r["mean_horizon"] for r in vals], marker="o", lw=2, color=COLORS[method], label=LABELS[method])
    axes[0].set_ylabel("success rate")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel("mean executed horizon")
    for ax in axes:
        ax.set_xticks(x, [c.replace("_", " ") for c in CONDITIONS])
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Robustness to physical disturbances and irrelevant visual distractors")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_frontier(summary: list[dict[str, Any]], out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for condition, marker in [("clean", "o"), ("combined", "s")]:
        for method in METHOD_ORDER:
            r = lookup(summary, method, condition)
            ax.scatter(r["inference_calls"], r["tracking_rmse"], s=75, marker=marker, color=COLORS[method], edgecolor="white", linewidth=0.7)
            if condition == "combined":
                ax.annotate(LABELS[method], (r["inference_calls"], r["tracking_rmse"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("inference calls / episode (lower is cheaper)")
    ax.set_ylabel("tracking RMSE (lower is better)")
    ax.grid(alpha=0.25)
    ax.set_title("Accuracy–efficiency frontier (circles clean, squares combined)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_robustness(rows: list[dict[str, Any]], out: pathlib.Path) -> None:
    methods = ["fixed_6", "fixed_12", "fixed_24", "ensemble_uncertainty", "attention_entropy", "shuffled_attention", "oracle_stop"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    for method in methods:
        vals = sorted([r for r in rows if r["method"] == method], key=lambda r: r["severity"])
        axes[0].plot([r["severity"] for r in vals], [r["success"] for r in vals], marker="o", color=COLORS[method], label=LABELS[method])
        axes[1].plot([r["severity"] for r in vals], [r["inference_calls"] for r in vals], marker="o", color=COLORS[method], label=LABELS[method])
    axes[0].set_ylabel("success rate")
    axes[1].set_ylabel("inference calls")
    for ax in axes:
        ax.set_xlabel("combined disturbance+distractor severity")
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_example(seed: int, cfg: Config, out: pathlib.Path) -> None:
    world = scenario(cfg, seed, 3, "combined", 1.0)
    methods = ["fixed_3", "fixed_12", "fixed_24", "attention_entropy", "shuffled_attention", "oracle_stop"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ref = world["reference"][: cfg.episode_steps + 1]
    axes[0, 0].plot(ref[:, 0], ref[:, 1], "k--", lw=2, label="reference")
    for method in methods:
        _, trace = rollout(method, world, cfg, seed, 3, keep_trace=True)
        axes[0, 0].plot(trace["states"][:, 0], trace["states"][:, 1], color=COLORS[method], lw=1.5, label=LABELS[method])
        axes[0, 1].plot(trace["errors"], color=COLORS[method], lw=1.4, label=LABELS[method])
    axes[0, 0].set_title("Representative trajectories")
    axes[0, 0].axis("equal")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    axes[0, 1].set_title("Tracking error")
    axes[0, 1].set_xlabel("control step")
    axes[0, 1].set_ylabel("position error")

    t = 123
    state = np.r_[world["reference"][t] + np.array([0.12, -0.08]), [0.0, 0.0]]
    plan = make_plan(state, t, world, cfg, seed, 3, 0)
    action_err = np.linalg.norm(plan["actions"] - plan["oracle_actions"], axis=1)
    x = np.arange(1, cfg.predicted_horizon + 1)
    ax = axes[1, 0]
    ax.plot(x, plan["entropy"] / math.log(cfg.token_count), color=COLORS["attention_entropy"], lw=2, label="normalized attention entropy")
    ax.plot(x, action_err / max(action_err.max(), 1e-9), color="#444444", lw=2, label="normalized action error")
    h = attention_horizon(plan["entropy"], cfg)
    ax.axvline(h, ls="--", color=COLORS["attention_entropy"], label=f"attention stop={h}")
    ax.axhline(cfg.entropy_ratio, ls=":", color="gray", label="entropy threshold")
    ax.set_xlabel("action index in predicted chunk")
    ax.set_title("Signal at a precision-stage query")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    for method in ["fixed_3", "fixed_12", "attention_entropy", "shuffled_attention", "oracle_stop"]:
        _, trace = rollout(method, world, cfg, seed, 3, keep_trace=True)
        ax.hist(trace["horizons"], bins=np.arange(0.5, cfg.predicted_horizon + 1.5), alpha=0.45, color=COLORS[method], label=LABELS[method])
    ax.set_xlabel("executed horizon")
    ax.set_ylabel("chunk count")
    ax.set_title("Horizon distribution in the same episode")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def sanity_checks(seed: int, cfg: Config, summary: list[dict[str, Any]]) -> dict[str, Any]:
    world = scenario(cfg, seed, 0, "clean", 1.0)
    example_t = 126
    example_state = np.r_[world["reference"][example_t], [0.0, 0.0]]
    plan = make_plan(example_state, example_t, world, cfg, seed, 0, 0)
    h_attn = attention_horizon(plan["entropy"], cfg)
    shuffled = plan["entropy"][np.random.default_rng(stable_seed(seed, 0, 0, "shuffle")).permutation(cfg.predicted_horizon)]
    checks = {
        "attention_entropy_rises": bool(plan["entropy"][-1] > plan["entropy"][0] + 0.45 * math.log(cfg.token_count)),
        "attention_gate_is_nontrivial": bool(1 < h_attn < cfg.predicted_horizon),
        "shuffle_preserves_entropy_multiset": bool(np.allclose(np.sort(plan["entropy"]), np.sort(shuffled))),
        "fixed_horizons_exact": all(abs(lookup(summary, f"fixed_{h}", "clean")["mean_horizon"] - h) < 0.12 for h in (3, 6, 12, 24)),
        "oracle_not_worse_than_fixed24_combined": bool(lookup(summary, "oracle_stop", "combined")["tracking_rmse"] <= lookup(summary, "fixed_24", "combined")["tracking_rmse"]),
        "attention_beats_shuffled_combined": bool(lookup(summary, "attention_entropy", "combined")["tracking_rmse"] < lookup(summary, "shuffled_attention", "combined")["tracking_rmse"]),
        "attention_uses_fewer_calls_than_fixed3_clean": bool(lookup(summary, "attention_entropy", "clean")["inference_calls"] < lookup(summary, "fixed_3", "clean")["inference_calls"]),
    }
    return {"passed": bool(all(checks.values())), "checks": checks, "example_attention_horizon": h_attn}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="quick")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    mode = MODES[args.mode]
    trials = args.trials if args.trials is not None else mode.trials
    cfg = Config()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()

    trial_rows = evaluate(args.seed, trials, cfg, CONDITIONS, 1.0)
    summary = group_summary(trial_rows, ("condition", "method"))
    overall = group_summary(trial_rows, ("method",))
    diagnostics = signal_diagnostics(args.seed + 97, min(trials, mode.diagnostic_trials), cfg)
    diagnostic_summary = group_summary(
        [{**r, "success": 0, "tracking_rmse": r["pearson_action_error"], "tail_rmse": 0, "final_error": 0, "inference_calls": 0, "mean_horizon": 0, "selected_action_error": r["mean_action_error"]} for r in diagnostics],
        ("condition", "signal"),
    )
    robust_trials = max(8, trials // 3)
    robustness = robustness_sweep(args.seed, robust_trials, cfg, mode.robustness_severities)
    sanity = sanity_checks(args.seed, cfg, summary)

    write_csv(out / "trial_metrics.csv", trial_rows)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "overall_summary.csv", overall)
    write_csv(out / "signal_diagnostics.csv", diagnostics)
    write_csv(out / "signal_summary.csv", diagnostic_summary)
    write_csv(out / "robustness_sweep.csv", robustness)
    (out / "sanity_checks.json").write_text(json.dumps(sanity, indent=2) + "\n")

    plot_headline(summary, out / "headline_metrics.png")
    plot_conditions(summary, out / "condition_robustness.png")
    plot_frontier(summary, out / "efficiency_frontier.png")
    plot_robustness(robustness, out / "robustness_sweep.png")
    plot_example(args.seed, cfg, out / "mechanism_example.png")

    metrics = {
        "experiment": "attention_gated_chunk_horizon",
        "mode": args.mode,
        "seed": args.seed,
        "trials_per_condition": trials,
        "config": asdict(cfg),
        "source": {
            "title": "Knowing When to Stop: Adaptive Action Chunking via Internal Cross-Attention Dynamics in VLAs",
            "url": "https://arxiv.org/abs/2609.00908",
            "version": "v1, September 1, 2026",
        },
        "overall_summary": overall,
        "condition_summary": summary,
        "signal_summary": diagnostic_summary,
        "robustness_sweep": robustness,
        "sanity": sanity,
        "runtime_seconds": time.time() - start,
        "claim_boundary": "Synthetic mechanism probe only; no VLA weights, image tokens, robot data, or paper benchmark reproduction.",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"Wrote {out}")
    print(f"mode={args.mode} trials/condition={trials} runtime={metrics['runtime_seconds']:.2f}s sanity={sanity['passed']}")
    for method in METHOD_ORDER:
        row = lookup(summary, method, "combined")
        print(f"{method:22s} combined success={row['success']:.3f} rmse={row['tracking_rmse']:.3f} calls={row['inference_calls']:.1f} horizon={row['mean_horizon']:.1f}")


if __name__ == "__main__":
    main()
