#!/usr/bin/env python3
"""PredVLA-inspired predictive-state correction toy.

This deterministic NumPy experiment isolates one mechanism: a generative recurrent
state predicts robot/target motion, and sensory observations can revise that state
only through prediction errors.  Setting correction iterations to zero is therefore
an exact open-loop ablation within this toy.

This is not a PredVLA reproduction.  It uses an identified linear dynamics model,
low-dimensional observations, analytical control, and lightweight observer rules.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DOCS = ROOT / "docs"

STATE_DIM = 8
OBS_DIM = 6
ACTION_DIM = 2
STATE_NAMES = ["robot_x", "robot_y", "robot_vx", "robot_vy", "target_x", "target_y", "target_vx", "target_vy"]
METHODS = [
    "simple_recurrent",
    "finite_window_attention_like",
    "learned_robust_observer",
    "predictive_coding",
    "predictive_coding_open_loop",
]
DISPLAY = {
    "simple_recurrent": "Simple recurrent observer",
    "finite_window_attention_like": "Finite-window attention-like",
    "learned_robust_observer": "Learned robust observer",
    "predictive_coding": "Predictive-coding correction",
    "predictive_coding_open_loop": "Predictive coding, open loop",
}
COLORS = {
    "simple_recurrent": "#4c78a8",
    "finite_window_attention_like": "#f58518",
    "learned_robust_observer": "#54a24b",
    "predictive_coding": "#b279a2",
    "predictive_coding_open_loop": "#777777",
}


@dataclass(frozen=True)
class Config:
    seed: int = 23
    trials: int = 48
    steps: int = 110
    dt: float = 0.08
    window: int = 12
    correction_iterations: int = 6
    correction_step: float = 0.34
    complexity: float = 0.16
    calibration_samples: int = 24000
    process_noise: float = 0.004
    observation_noise: float = 0.008
    max_accel: float = 3.2
    max_speed: float = 1.55
    tail_steps: int = 35
    success_tail_rmse: float = 0.255
    success_final_error: float = 0.30


@dataclass(frozen=True)
class Condition:
    name: str
    value: float
    occlusion: int = 0
    delay: int = 0
    disturbance: float = 0.0
    sensor_bias: float = 0.0
    correction_iterations: int = 6


@dataclass
class Packet:
    timestamp: int
    y: np.ndarray
    mask: np.ndarray


@dataclass
class Model:
    A: np.ndarray
    B: np.ndarray
    H: np.ndarray
    k_clean: np.ndarray
    b_clean: np.ndarray
    k_robust: np.ndarray
    b_robust: np.ndarray


def rng_for(*values: int) -> np.random.Generator:
    seed = 2166136261
    for value in values:
        seed = ((seed ^ int(value)) * 16777619) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def nominal_matrices(dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.eye(STATE_DIM)
    B = np.zeros((STATE_DIM, ACTION_DIM))
    A[0:2, 2:4] = dt * np.eye(2)
    A[2:4, 2:4] = 0.985 * np.eye(2)
    B[0:2, :] = 0.5 * dt * dt * np.eye(2)
    B[2:4, :] = dt * np.eye(2)
    A[4:6, 6:8] = dt * np.eye(2)
    A[6:8, 6:8] = 0.996 * np.eye(2)
    H = np.zeros((OBS_DIM, STATE_DIM))
    H[0:4, 0:4] = np.eye(4)
    H[4:6, 4:6] = np.eye(2)
    return A, B, H


def identify_model(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify the shared recurrent dynamics from synthetic demonstrations."""
    A0, B0, H = nominal_matrices(cfg.dt)
    rng = rng_for(cfg.seed, 101)
    n = cfg.calibration_samples
    x = rng.normal(0.0, 0.65, size=(n, STATE_DIM))
    x[:, 2:4] *= 0.45
    x[:, 6:8] *= 0.32
    u = rng.uniform(-cfg.max_accel, cfg.max_accel, size=(n, ACTION_DIM))
    phase = rng.uniform(-math.pi, math.pi, size=n)
    target_acc = 0.10 * np.stack([np.sin(phase), np.cos(0.8 * phase)], axis=1)
    xn = x @ A0.T + u @ B0.T
    xn[:, 6:8] += cfg.dt * target_acc
    xn += rng.normal(0.0, cfg.process_noise, size=xn.shape)
    design = np.concatenate([x, u], axis=1)
    ridge = 2e-4
    weights = np.linalg.solve(design.T @ design + ridge * np.eye(design.shape[1]), design.T @ xn)
    A = weights[:STATE_DIM].T
    B = weights[STATE_DIM:].T
    return A, B, H


def fit_observer_gain(
    cfg: Config,
    H: np.ndarray,
    robust: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a 56-coefficient innovation-to-state correction block."""
    rng = rng_for(cfg.seed, 202 if robust else 201)
    n = cfg.calibration_samples
    true_x = rng.normal(0.0, 0.65, size=(n, STATE_DIM))
    true_x[:, 2:4] *= 0.5
    true_x[:, 6:8] *= 0.35
    if robust:
        prior_scale = np.array([0.11, 0.11, 0.16, 0.16, 0.13, 0.13, 0.16, 0.16])
        bias_scale = rng.uniform(0.0, 0.08, size=(n, 1))
    else:
        prior_scale = np.array([0.055, 0.055, 0.075, 0.075, 0.065, 0.065, 0.09, 0.09])
        bias_scale = np.zeros((n, 1))
    prior = true_x + rng.normal(size=true_x.shape) * prior_scale
    sensor_bias = rng.normal(size=(n, OBS_DIM)) * bias_scale
    sensor_bias[:, 2:4] *= 0.35
    y = true_x @ H.T + sensor_bias + rng.normal(0.0, cfg.observation_noise, size=(n, OBS_DIM))
    innovation = y - prior @ H.T
    if robust:
        # Random visual dropout teaches a conservative gain when only proprioception is present.
        drop = rng.random(n) < 0.28
        innovation[drop, 4:6] = 0.0
    target = true_x - prior
    design = np.concatenate([innovation, np.ones((n, 1))], axis=1)
    ridge = 7e-3 if robust else 2e-3
    weights = np.linalg.solve(design.T @ design + ridge * np.eye(OBS_DIM + 1), design.T @ target)
    return weights[:OBS_DIM].T, weights[OBS_DIM]


def build_model(cfg: Config) -> Model:
    A, B, H = identify_model(cfg)
    k_clean, b_clean = fit_observer_gain(cfg, H, robust=False)
    k_robust, b_robust = fit_observer_gain(cfg, H, robust=True)
    return Model(A=A, B=B, H=H, k_clean=k_clean, b_clean=b_clean, k_robust=k_robust, b_robust=b_robust)


def task_prior(case_seed: int) -> Tuple[np.ndarray, float]:
    """Episode conditioning: nominal reset and trajectory phase, not sensor feedback."""
    rng = rng_for(case_seed, 301)
    phase = rng.uniform(-math.pi, math.pi)
    x = np.array([-0.62, -0.42, 0.0, 0.0, 0.62, 0.46, -0.20, 0.13], dtype=float)
    x[4:6] += 0.08 * np.array([math.sin(phase), math.cos(phase)])
    x[6:8] += 0.035 * np.array([math.cos(phase), -math.sin(phase)])
    return x, phase


def initial_truth(case_seed: int) -> Tuple[np.ndarray, float]:
    prior, phase = task_prior(case_seed)
    rng = rng_for(case_seed, 302)
    x = prior.copy()
    x[0:2] += rng.normal(0.0, 0.055, size=2)
    x[2:4] += rng.normal(0.0, 0.025, size=2)
    x[4:6] += rng.normal(0.0, 0.06, size=2)
    x[6:8] += rng.normal(0.0, 0.028, size=2)
    return x, phase


def controller(xhat: np.ndarray, cfg: Config) -> np.ndarray:
    rel = xhat[4:6] - xhat[0:2]
    vel_rel = xhat[6:8] - xhat[2:4]
    u = 5.1 * rel + 2.1 * vel_rel
    norm = float(np.linalg.norm(u))
    if norm > cfg.max_accel:
        u *= cfg.max_accel / norm
    return u


def true_step(
    x: np.ndarray,
    u: np.ndarray,
    t: int,
    phase: float,
    disturbance: float,
    disturbance_time: int,
    impulse_dir: np.ndarray,
    process_noise: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    dt = cfg.dt
    xn = x.copy()
    target_acc = 0.105 * np.array(
        [math.sin(0.13 * t + phase) + 0.35 * math.sin(0.041 * t), math.cos(0.105 * t + 0.7 * phase)]
    )
    robot_acc = u + np.array([0.055 * math.sin(0.09 * t + phase), -0.04 * math.cos(0.075 * t)])
    xn[0:2] = x[0:2] + dt * x[2:4] + 0.5 * dt * dt * robot_acc
    xn[2:4] = 0.982 * x[2:4] + dt * robot_acc
    xn[4:6] = x[4:6] + dt * x[6:8] + 0.5 * dt * dt * target_acc
    xn[6:8] = 0.995 * x[6:8] + dt * target_acc
    if t == disturbance_time and disturbance > 0:
        xn[2:4] += disturbance * impulse_dir
        xn[6:8] += 0.32 * disturbance * np.array([-impulse_dir[1], impulse_dir[0]])
    xn += process_noise
    speed = np.linalg.norm(xn[2:4])
    if speed > cfg.max_speed:
        xn[2:4] *= cfg.max_speed / speed
    # Soft workspace reflections keep all methods in a comparable bounded task.
    for start in (0, 4):
        for j in range(2):
            idx = start + j
            vidx = start + 2 + j
            if xn[idx] > 1.15:
                xn[idx] = 1.15 - 0.55 * (xn[idx] - 1.15)
                xn[vidx] *= -0.65
            elif xn[idx] < -1.15:
                xn[idx] = -1.15 - 0.55 * (xn[idx] + 1.15)
                xn[vidx] *= -0.65
    return xn


def make_packet(
    x: np.ndarray,
    t: int,
    condition: Condition,
    bias: np.ndarray,
    noise: np.ndarray,
    cfg: Config,
) -> Packet:
    _, _, H = nominal_matrices(cfg.dt)
    y = H @ x + bias + noise
    mask = np.ones(OBS_DIM, dtype=float)
    if condition.occlusion > 0:
        start = cfg.steps // 2 - condition.occlusion // 2
        if start <= t < start + condition.occlusion:
            mask[4:6] = 0.0
    return Packet(timestamp=t, y=y, mask=mask)


def matrix_power(A: np.ndarray, n: int) -> np.ndarray:
    return np.linalg.matrix_power(A, max(0, int(n)))


def clipped_innovation(packet: Packet, prediction: np.ndarray, H: np.ndarray, limit: float = 0.34) -> np.ndarray:
    e = packet.mask * (packet.y - H @ prediction)
    return np.clip(e, -limit, limit)


def newest_packet(available: Mapping[int, Packet], t: int) -> Packet | None:
    candidates = [tau for tau in available if tau <= t]
    return available[max(candidates)] if candidates else None


def update_simple_recurrent(
    prior_current: np.ndarray,
    histories: Sequence[np.ndarray],
    available: Mapping[int, Packet],
    t: int,
    model: Model,
) -> np.ndarray:
    packet = newest_packet(available, t)
    if packet is None:
        return prior_current
    tau = packet.timestamp
    e = clipped_innovation(packet, histories[tau], model.H, limit=0.45)
    correction_tau = model.k_clean @ e + model.b_clean
    transported = matrix_power(model.A, t - tau) @ correction_tau
    return prior_current + 0.82 * transported


def update_learned_observer(
    prior_current: np.ndarray,
    histories: Sequence[np.ndarray],
    available: Mapping[int, Packet],
    t: int,
    model: Model,
) -> np.ndarray:
    packet = newest_packet(available, t)
    if packet is None:
        return prior_current
    tau = packet.timestamp
    e = clipped_innovation(packet, histories[tau], model.H, limit=0.25)
    correction_tau = model.k_robust @ e + model.b_robust
    transported = matrix_power(model.A, t - tau) @ correction_tau
    gate = 1.0 / (1.0 + 1.8 * float(np.linalg.norm(e)))
    return prior_current + (0.72 + 0.22 * gate) * transported


def update_finite_window(
    prior_current: np.ndarray,
    histories: Sequence[np.ndarray],
    available: Mapping[int, Packet],
    t: int,
    model: Model,
    window: int,
) -> np.ndarray:
    start = max(0, t - window + 1)
    packets = [available[tau] for tau in sorted(available) if start <= tau <= t]
    if not packets:
        return prior_current
    corrections: List[np.ndarray] = []
    scores: List[float] = []
    for packet in packets:
        tau = packet.timestamp
        e = clipped_innovation(packet, histories[tau], model.H, limit=0.34)
        corr = matrix_power(model.A, t - tau) @ (model.k_clean @ e + model.b_clean)
        age = t - tau
        score = math.exp(-age / max(2.0, 0.45 * window)) / (0.28 + float(np.linalg.norm(e)))
        corrections.append(corr)
        scores.append(score)
    weights = np.asarray(scores, dtype=float)
    weights /= weights.sum() + 1e-12
    aggregate = np.sum(np.stack(corrections) * weights[:, None], axis=0)
    return prior_current + 0.74 * aggregate


def rollout_from_anchor(
    anchor: np.ndarray,
    start: int,
    end: int,
    actions: Sequence[np.ndarray],
    model: Model,
) -> List[np.ndarray]:
    states = [anchor.copy()]
    x = anchor.copy()
    for k in range(start, end):
        x = model.A @ x + model.B @ actions[k]
        states.append(x.copy())
    return states


def update_predictive_coding(
    histories: List[np.ndarray],
    actions: Sequence[np.ndarray],
    available: Mapping[int, Packet],
    t: int,
    model: Model,
    cfg: Config,
    iterations: int,
) -> np.ndarray:
    """Sliding-window prediction-error inference over an anchor latent state.

    The learned 8x6 error lift and 8-vector offset have the same 56-coefficient
    observer budget as the linear baselines. Observations are never concatenated
    into the recurrence; they appear only through y-H x prediction errors here.
    """
    if iterations <= 0:
        return histories[t]
    start = max(0, t - cfg.window + 1)
    anchor_prior = histories[start].copy()
    anchor = anchor_prior.copy()
    packets = [available[tau] for tau in sorted(available) if start <= tau <= t]
    if not packets:
        return histories[t]
    for _ in range(iterations):
        candidate = rollout_from_anchor(anchor, start, t, actions, model)
        direction = np.zeros(STATE_DIM, dtype=float)
        normalizer = 0.0
        for packet in packets:
            tau = packet.timestamp
            local_state = candidate[tau - start]
            e = clipped_innovation(packet, local_state, model.H, limit=0.28)
            correction_tau = model.k_robust @ e + model.b_robust
            # Backpropagation-through-time analogue for the linear recurrence.
            back = matrix_power(model.A, tau - start).T @ correction_tau
            reliability = 0.45 + 0.55 * float(packet.mask.mean())
            direction += reliability * back
            normalizer += reliability
        direction /= max(normalizer, 1e-9)
        direction -= cfg.complexity * (anchor - anchor_prior)
        step = cfg.correction_step * direction
        step_norm = float(np.linalg.norm(step))
        if step_norm > 0.20:
            step *= 0.20 / step_norm
        anchor += step
    corrected = rollout_from_anchor(anchor, start, t, actions, model)
    for offset, state in enumerate(corrected):
        histories[start + offset] = state
    return histories[t]


def simulate(
    method: str,
    condition: Condition,
    trial: int,
    cfg: Config,
    model: Model,
    keep_trace: bool = False,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray] | None]:
    case_seed = cfg.seed * 100000 + trial * 97
    rng = rng_for(case_seed, 401)
    true_x, phase = initial_truth(case_seed)
    estimate, _ = task_prior(case_seed)
    disturbance_time = int(rng.integers(cfg.steps // 3, 2 * cfg.steps // 3))
    angle = rng.uniform(-math.pi, math.pi)
    impulse_dir = np.array([math.cos(angle), math.sin(angle)])
    process_noise = rng.normal(0.0, cfg.process_noise, size=(cfg.steps, STATE_DIM))
    process_noise[:, 4:8] *= 0.55
    obs_noise = rng.normal(0.0, cfg.observation_noise, size=(cfg.steps, OBS_DIM))
    bias_dir = rng.normal(size=OBS_DIM)
    bias_dir /= np.linalg.norm(bias_dir) + 1e-12
    bias = condition.sensor_bias * bias_dir
    bias[2:4] *= 0.35

    true_hist: List[np.ndarray] = [true_x.copy()]
    estimate_hist: List[np.ndarray] = [estimate.copy()]
    actions: List[np.ndarray] = []
    generated: Dict[int, Packet] = {}
    available: Dict[int, Packet] = {}
    tracking_errors: List[float] = []
    state_errors: List[float] = []
    prediction_errors: List[float] = []

    for t in range(cfg.steps):
        if t > 0:
            prior = model.A @ estimate_hist[t - 1] + model.B @ actions[t - 1]
            estimate_hist.append(prior)
        generated[t] = make_packet(true_hist[t], t, condition, bias, obs_noise[t], cfg)
        delivered_tau = t - condition.delay
        if delivered_tau >= 0:
            available[delivered_tau] = generated[delivered_tau]

        prior_current = estimate_hist[t].copy()
        if method == "simple_recurrent":
            estimate_hist[t] = update_simple_recurrent(prior_current, estimate_hist, available, t, model)
        elif method == "finite_window_attention_like":
            estimate_hist[t] = update_finite_window(prior_current, estimate_hist, available, t, model, cfg.window)
        elif method == "learned_robust_observer":
            estimate_hist[t] = update_learned_observer(prior_current, estimate_hist, available, t, model)
        elif method == "predictive_coding":
            estimate_hist[t] = update_predictive_coding(
                estimate_hist, actions, available, t, model, cfg, condition.correction_iterations
            )
        elif method == "predictive_coding_open_loop":
            estimate_hist[t] = update_predictive_coding(estimate_hist, actions, available, t, model, cfg, 0)
        else:
            raise ValueError(f"Unknown method: {method}")

        action = controller(estimate_hist[t], cfg)
        actions.append(action)
        tracking_errors.append(float(np.linalg.norm(true_hist[t][4:6] - true_hist[t][0:2])))
        state_errors.append(float(np.sqrt(np.mean((estimate_hist[t] - true_hist[t]) ** 2))))
        packet = generated[t]
        prediction_errors.append(float(np.sqrt(np.sum(packet.mask * (packet.y - model.H @ estimate_hist[t]) ** 2) / max(packet.mask.sum(), 1.0))))
        next_true = true_step(
            true_hist[t], action, t, phase, condition.disturbance, disturbance_time,
            impulse_dir, process_noise[t], cfg,
        )
        true_hist.append(next_true)

    errors = np.asarray(tracking_errors)
    tail_rmse = float(np.sqrt(np.mean(errors[-cfg.tail_steps :] ** 2)))
    final_error = float(errors[-1])
    success = float(tail_rmse < cfg.success_tail_rmse and final_error < cfg.success_final_error)
    metrics = {
        "success": success,
        "tracking_rmse": float(np.sqrt(np.mean(errors**2))),
        "tail_tracking_rmse": tail_rmse,
        "final_error": final_error,
        "state_rmse": float(np.mean(state_errors)),
        "sensory_prediction_rmse": float(np.mean(prediction_errors)),
        "max_action": float(np.max(np.linalg.norm(np.asarray(actions), axis=1))),
    }
    trace = None
    if keep_trace:
        trace = {
            "truth": np.asarray(true_hist[:-1]),
            "estimate": np.asarray(estimate_hist),
            "actions": np.asarray(actions),
            "tracking_error": errors,
        }
    return metrics, trace


def condition_sweeps(cfg: Config, quick: bool) -> Dict[str, List[Condition]]:
    if quick:
        occlusions = [0, 24]
        delays = [0, 5]
        disturbances = [0.0, 0.42]
        biases = [0.0, 0.08]
        iterations = [0, 2, 6]
    else:
        occlusions = [0, 10, 20, 32, 44]
        delays = [0, 1, 3, 5, 8]
        disturbances = [0.0, 0.16, 0.30, 0.44, 0.60]
        biases = [0.0, 0.02, 0.05, 0.08, 0.12]
        iterations = [0, 1, 2, 4, 6, 10, 16]
    return {
        "occlusion": [Condition("occlusion", float(v), occlusion=v, correction_iterations=cfg.correction_iterations) for v in occlusions],
        "delay": [Condition("delay", float(v), delay=v, correction_iterations=cfg.correction_iterations) for v in delays],
        "disturbance": [Condition("disturbance", float(v), disturbance=v, correction_iterations=cfg.correction_iterations) for v in disturbances],
        "sensor_bias": [Condition("sensor_bias", float(v), sensor_bias=v, correction_iterations=cfg.correction_iterations) for v in biases],
        "correction_iterations": [Condition("correction_iterations", float(v), disturbance=0.30, delay=3, occlusion=20, sensor_bias=0.025, correction_iterations=v) for v in iterations],
    }


def sanity_checks(cfg: Config, model: Model) -> Dict[str, object]:
    base = Condition("sanity", 0.0, disturbance=0.38, delay=3, occlusion=18, correction_iterations=cfg.correction_iterations)
    trial = 777
    pc_metrics, pc_trace = simulate("predictive_coding", base, trial, cfg, model, keep_trace=True)
    ol_metrics_a, ol_trace_a = simulate("predictive_coding_open_loop", base, trial, cfg, model, keep_trace=True)
    changed = replace(base, sensor_bias=0.11, occlusion=42, delay=7)
    ol_metrics_b, ol_trace_b = simulate("predictive_coding_open_loop", changed, trial, cfg, model, keep_trace=True)
    assert pc_trace is not None and ol_trace_a is not None and ol_trace_b is not None
    open_loop_action_delta = float(np.max(np.abs(ol_trace_a["actions"] - ol_trace_b["actions"])))
    closed_open_action_delta = float(np.mean(np.linalg.norm(pc_trace["actions"] - ol_trace_a["actions"], axis=1)))
    checks = {
        "identified_dynamics_finite": bool(np.isfinite(model.A).all() and np.isfinite(model.B).all()),
        "observer_gains_finite": bool(np.isfinite(model.k_clean).all() and np.isfinite(model.k_robust).all()),
        "all_actions_bounded": bool(pc_metrics["max_action"] <= cfg.max_accel + 1e-9),
        "open_loop_sensor_invariance": bool(open_loop_action_delta < 1e-12),
        "closed_loop_differs_from_open_loop": bool(closed_open_action_delta > 1e-3),
        "prediction_error_finite": bool(np.isfinite(pc_metrics["sensory_prediction_rmse"])),
        "zero_iterations_matches_open_loop": True,
        "open_loop_action_max_delta_under_sensor_changes": open_loop_action_delta,
        "closed_vs_open_mean_action_delta": closed_open_action_delta,
        "example_closed_tail_rmse": pc_metrics["tail_tracking_rmse"],
        "example_open_tail_rmse": ol_metrics_a["tail_tracking_rmse"],
    }
    zero_condition = replace(base, correction_iterations=0)
    zero_metrics, zero_trace = simulate("predictive_coding", zero_condition, trial, cfg, model, keep_trace=True)
    assert zero_trace is not None
    zero_delta = float(np.max(np.abs(zero_trace["actions"] - ol_trace_a["actions"])))
    checks["zero_iterations_action_max_delta"] = zero_delta
    checks["zero_iterations_matches_open_loop"] = bool(zero_delta < 1e-12)
    required = [v for k, v in checks.items() if isinstance(v, bool)]
    checks["all_passed"] = bool(all(required))
    if not checks["all_passed"]:
        raise AssertionError(f"Sanity checks failed: {checks}")
    return checks


def run_trials(cfg: Config, model: Model, quick: bool) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    sweeps = condition_sweeps(cfg, quick)
    n_trials = min(cfg.trials, 8) if quick else cfg.trials
    for sweep_name, conditions in sweeps.items():
        methods = METHODS if sweep_name != "correction_iterations" else ["predictive_coding"]
        for condition in conditions:
            for method in methods:
                for trial in range(n_trials):
                    metrics, _ = simulate(method, condition, trial, cfg, model)
                    rows.append({
                        "sweep": sweep_name,
                        "value": condition.value,
                        "method": method,
                        "trial": trial,
                        **metrics,
                    })
    return rows


def summarize(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, float, str], List[Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row["sweep"]), float(row["value"]), str(row["method"]))
        groups.setdefault(key, []).append(row)
    output: List[Dict[str, object]] = []
    metric_names = ["success", "tracking_rmse", "tail_tracking_rmse", "final_error", "state_rmse", "sensory_prediction_rmse"]
    for (sweep, value, method), group in sorted(groups.items()):
        item: Dict[str, object] = {"sweep": sweep, "value": value, "method": method, "n": len(group)}
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in group])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_sem"] = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
        output.append(item)
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_summary(summary: Sequence[Mapping[str, object]], sweep: str, value: float, method: str) -> Mapping[str, object]:
    matches = [r for r in summary if r["sweep"] == sweep and abs(float(r["value"]) - value) < 1e-9 and r["method"] == method]
    if len(matches) != 1:
        raise KeyError((sweep, value, method))
    return matches[0]


def plot_sweeps(summary: Sequence[Mapping[str, object]]) -> None:
    sweeps = ["occlusion", "delay", "disturbance", "sensor_bias"]
    labels = {
        "occlusion": "occlusion length (steps)",
        "delay": "observation delay (steps)",
        "disturbance": "impulse magnitude",
        "sensor_bias": "sensor-bias magnitude",
    }
    fig, axes = plt.subplots(2, 4, figsize=(17.0, 7.4), sharey="row")
    for col, sweep in enumerate(sweeps):
        for method in METHODS:
            rows = sorted([r for r in summary if r["sweep"] == sweep and r["method"] == method], key=lambda r: float(r["value"]))
            x = np.asarray([float(r["value"]) for r in rows])
            success = np.asarray([float(r["success_mean"]) for r in rows])
            success_sem = np.asarray([float(r["success_sem"]) for r in rows])
            rmse = np.asarray([float(r["tracking_rmse_mean"]) for r in rows])
            rmse_sem = np.asarray([float(r["tracking_rmse_sem"]) for r in rows])
            axes[0, col].plot(x, success, marker="o", lw=2, ms=4, color=COLORS[method], label=DISPLAY[method])
            axes[0, col].fill_between(x, np.clip(success - success_sem, 0, 1), np.clip(success + success_sem, 0, 1), color=COLORS[method], alpha=0.10)
            axes[1, col].plot(x, rmse, marker="o", lw=2, ms=4, color=COLORS[method])
            axes[1, col].fill_between(x, rmse - rmse_sem, rmse + rmse_sem, color=COLORS[method], alpha=0.10)
        axes[0, col].set_title(sweep.replace("_", " ").title())
        axes[0, col].set_ylim(-0.03, 1.03)
        axes[0, col].grid(alpha=0.25)
        axes[1, col].grid(alpha=0.25)
        axes[1, col].set_xlabel(labels[sweep])
    axes[0, 0].set_ylabel("success rate")
    axes[1, 0].set_ylabel("tracking RMSE")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Predictive-state correction robustness sweeps", y=1.01, fontsize=14)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "robustness_sweeps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_iterations(summary: Sequence[Mapping[str, object]]) -> None:
    rows = sorted([r for r in summary if r["sweep"] == "correction_iterations"], key=lambda r: float(r["value"]))
    x = np.asarray([float(r["value"]) for r in rows])
    success = np.asarray([float(r["success_mean"]) for r in rows])
    rmse = np.asarray([float(r["tracking_rmse_mean"]) for r in rows])
    state = np.asarray([float(r["state_rmse_mean"]) for r in rows])
    pred = np.asarray([float(r["sensory_prediction_rmse_mean"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    ax = axes[0]
    ax.plot(x, success, marker="o", color=COLORS["predictive_coding"], lw=2.2, label="success")
    ax.set_xlabel("correction iterations")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(x, rmse, marker="s", color="#e45756", lw=2, label="tracking RMSE")
    ax2.set_ylabel("tracking RMSE")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="center right")
    axes[1].plot(x, state, marker="o", lw=2, color="#4c78a8", label="state RMSE")
    axes[1].plot(x, pred, marker="s", lw=2, color="#f58518", label="sensory prediction RMSE")
    axes[1].set_xlabel("correction iterations")
    axes[1].set_ylabel("estimation error")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("Prediction-error correction iteration sweep")
    fig.tight_layout()
    fig.savefig(OUT / "correction_iteration_sweep.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rollout(cfg: Config, model: Model) -> None:
    condition = Condition("example", 0.0, occlusion=28, delay=4, disturbance=0.42, sensor_bias=0.035, correction_iterations=cfg.correction_iterations)
    traces: Dict[str, Dict[str, np.ndarray]] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    for method in METHODS:
        m, trace = simulate(method, condition, 9, cfg, model, keep_trace=True)
        assert trace is not None
        traces[method] = trace
        metrics[method] = m
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for method in METHODS:
        tr = traces[method]
        axes[0].plot(tr["truth"][:, 0], tr["truth"][:, 1], color=COLORS[method], lw=1.8, label=DISPLAY[method])
        axes[1].plot(tr["tracking_error"], color=COLORS[method], lw=1.8, label=DISPLAY[method])
    target = traces["predictive_coding"]["truth"]
    axes[0].plot(target[:, 4], target[:, 5], "k--", lw=2.2, label="moving target")
    axes[0].scatter(target[0, 0], target[0, 1], s=35, c="k", marker="x")
    axes[0].set_title("Paired point-robot trajectories")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.25)
    start = cfg.steps // 2 - condition.occlusion // 2
    axes[1].axvspan(start, start + condition.occlusion, color="#cccccc", alpha=0.35, label="visual occlusion")
    axes[1].set_title("Target tracking error")
    axes[1].set_xlabel("control step")
    axes[1].set_ylabel("distance")
    axes[1].grid(alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(OUT / "representative_rollout.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    with (OUT / "representative_rollout_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def parameter_budget(model: Model) -> Dict[str, object]:
    shared = STATE_DIM * STATE_DIM + STATE_DIM * ACTION_DIM
    observer = STATE_DIM * OBS_DIM + STATE_DIM
    methods = {}
    for method in METHODS[:-1]:
        methods[method] = {
            "identified_dynamics_coefficients": shared,
            "observer_or_error_lift_coefficients": observer,
            "total_fitted_coefficient_budget": shared + observer,
        }
    methods["predictive_coding_open_loop"] = {
        "identified_dynamics_coefficients": shared,
        "available_but_disabled_error_lift_coefficients": observer,
        "total_model_coefficient_budget": shared + observer,
        "active_observation_correction_coefficients": 0,
    }
    return {
        "state_dimension": STATE_DIM,
        "observation_dimension": OBS_DIM,
        "action_dimension": ACTION_DIM,
        "budget_definition": "80 identified linear recurrent-dynamics coefficients plus 56 observation-update/error-lift coefficients per closed-loop method.",
        "important_caveat": "Equal scalar coefficient slots do not make these lightweight algorithms equivalent to parameter-matched neural LSTM/Transformer/PredVLA models.",
        "methods": methods,
        "spectral_radius_A": float(max(abs(np.linalg.eigvals(model.A)))),
    }


def headline_metrics(summary: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    keys = {
        "long_occlusion": ("occlusion", 44.0),
        "delay_5": ("delay", 5.0),
        "disturbance_044": ("disturbance", 0.44),
        "bias_008": ("sensor_bias", 0.08),
    }
    result: Dict[str, object] = {}
    for name, (sweep, value) in keys.items():
        values = [r for r in summary if r["sweep"] == sweep]
        if not values:
            continue
        available_values = sorted(set(float(r["value"]) for r in values))
        chosen = min(available_values, key=lambda v: abs(v - value))
        result[name] = {
            method: {
                "success": float(select_summary(summary, sweep, chosen, method)["success_mean"]),
                "tracking_rmse": float(select_summary(summary, sweep, chosen, method)["tracking_rmse_mean"]),
            }
            for method in METHODS
        }
    iteration_rows = [r for r in summary if r["sweep"] == "correction_iterations"]
    if iteration_rows:
        best = min(iteration_rows, key=lambda r: float(r["tracking_rmse_mean"]))
        result["best_iteration_by_tracking_rmse"] = {
            "iterations": int(float(best["value"])),
            "success": float(best["success_mean"]),
            "tracking_rmse": float(best["tracking_rmse_mean"]),
        }
    return result


def write_generated_report(cfg: Config, headline: Mapping[str, object]) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    best = headline.get("best_iteration_by_tracking_rmse", {})
    delay = headline.get("delay_5", {})
    pc = delay.get("predictive_coding", {}) if isinstance(delay, dict) else {}
    ol = delay.get("predictive_coding_open_loop", {}) if isinstance(delay, dict) else {}
    text = f"""# Generated experiment note\n\nThe full deterministic run used `{cfg.trials}` paired trials per method and sweep value.\n\nAt observation delay 5, predictive-coding correction achieved {100*float(pc.get('success', 0.0)):.1f}% success with tracking RMSE {float(pc.get('tracking_rmse', float('nan'))):.3f}; its exact zero-iteration/open-loop counterpart achieved {100*float(ol.get('success', 0.0)):.1f}% success with RMSE {float(ol.get('tracking_rmse', float('nan'))):.3f}.\n\nThe lowest tracking RMSE in the mixed perturbation iteration sweep occurred at {int(best.get('iterations', 0))} iterations: success {100*float(best.get('success', 0.0)):.1f}%, tracking RMSE {float(best.get('tracking_rmse', float('nan'))):.3f}.\n\nSee `../outputs/summary.csv`, `../outputs/metrics.json`, and the LaTeX/PDF report for complete results and claim boundaries.\n"""
    (DOCS / "generated_results.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--trials", type=int, default=48)
    parser.add_argument("--quick", action="store_true", help="Run a small smoke test and overwrite outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(seed=args.seed, trials=args.trials)
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg)
    checks = sanity_checks(cfg, model)
    rows = run_trials(cfg, model, args.quick)
    summary = summarize(rows)
    write_csv(OUT / "trials.csv", rows)
    write_csv(OUT / "summary.csv", summary)
    budget = parameter_budget(model)
    headline = headline_metrics(summary)
    payload = {
        "config": asdict(cfg),
        "quick": bool(args.quick),
        "methods": DISPLAY,
        "headline": headline,
        "claim_boundary": "Synthetic low-dimensional mechanism test; not a PredVLA reproduction or evidence about LIBERO/real-robot performance.",
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "sanity_check.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    (OUT / "parameter_budget.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")
    plot_sweeps(summary)
    plot_iterations(summary)
    plot_rollout(cfg, model)
    write_generated_report(cfg, headline)
    print(json.dumps({"all_sanity_checks_passed": checks["all_passed"], "headline": headline}, indent=2))


if __name__ == "__main__":
    main()
