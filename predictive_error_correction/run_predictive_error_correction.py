#!/usr/bin/env python3
"""PredVLA-inspired predictive-error correction toy.

A one-dimensional robot tracks a moving target. The nominal recurrent latent rolls forward
robot/target state from the previous action. Hidden forces, target maneuvers, and slowly drifting
sensors make that open-loop rollout wrong. One controller performs online error regression over a
finite observation window; feed-forward baselines map fixed sensory histories directly to actions.

This is a deterministic mechanism test, not a reproduction of PredVLA or a VLA benchmark.
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
from typing import Iterable

BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = [
    "oracle",
    "open_loop",
    "pred_error_w8",
    "history_h4",
    "history_h16",
    "history_h16_chunk4",
]
LABELS = {
    "oracle": "Oracle state feedback",
    "open_loop": "Open-loop latent rollout (0 iterations)",
    "pred_error_w8": "Prediction-error correction (W=8)",
    "history_h4": "Feed-forward history (H=4)",
    "history_h16": "Feed-forward history (H=16)",
    "history_h16_chunk4": "History H=16, action chunk=4",
}
COLORS = {
    "oracle": "#4c78a8",
    "open_loop": "#f58518",
    "pred_error_w8": "#54a24b",
    "history_h4": "#e45756",
    "history_h16": "#b279a2",
    "history_h16_chunk4": "#72b7b2",
}


@dataclass(frozen=True)
class Config:
    dt: float = 0.08
    steps: int = 180
    max_action: float = 4.0
    robot_drag: float = 0.22
    target_drag: float = 0.025
    kp: float = 3.0
    kd: float = 1.9
    success_tail_steps: int = 50
    success_tail_rmse: float = 0.42
    success_final_error: float = 0.58
    error_window: int = 8
    error_iterations: int = 80
    error_complexity: float = 0.25
    # The paper reports module time constants (T, V, A_top, A_bottom)=(16, 8, 5, 2).
    # This toy reuses the same values in fast-to-slow order for four state coordinates;
    # there is no one-to-one correspondence between those coordinates and paper modules.
    latent_time_constants: tuple[float, ...] = (2.0, 5.0, 8.0, 16.0)
    history_values: tuple[int, ...] = (1, 2, 4, 8, 16)
    severities: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    history_sweep_severities: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    iteration_values: tuple[int, ...] = (0, 1, 2, 4, 8, 20, 40, 80)
    train_episodes: int = 180
    trials: int = 64
    ridge_alpha: float = 0.08


@dataclass
class RidgeModel:
    history: int
    chunk: int
    x_mean: np.ndarray
    x_std: np.ndarray
    weights: np.ndarray

    def predict(self, feature: np.ndarray) -> np.ndarray:
        z = (feature - self.x_mean) / self.x_std
        z = np.concatenate([z, np.ones(1, dtype=np.float64)])
        return z @ self.weights


@dataclass
class Schedule:
    command: float
    initial_state: np.ndarray
    target_accel: np.ndarray
    disturbance: np.ndarray
    obs_noise: np.ndarray
    proprio_bias: np.ndarray
    vision_bias: np.ndarray
    fingerprint: str


def matrices(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = cfg.dt
    a = np.array(
        [
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0 - cfg.robot_drag * dt, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0 - cfg.target_drag * dt],
        ],
        dtype=np.float64,
    )
    b = np.array([0.5 * dt**2, dt, 0.0, 0.0], dtype=np.float64)
    c = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    # A small language-conditioned target-velocity prior: move in the commanded direction.
    e = np.array([0.0, 0.0, 0.0, 0.015 * dt], dtype=np.float64)
    return a, b, c, e


def nominal_step(state: np.ndarray, action: float, command: float, cfg: Config) -> np.ndarray:
    a, b, _, e = matrices(cfg)
    return a @ state + b * float(action) + e * float(command)


def true_step(
    state: np.ndarray,
    action: float,
    target_accel: float,
    disturbance: float,
    command: float,
    cfg: Config,
) -> np.ndarray:
    r, v, g, gv = state
    robot_accel = float(action) - cfg.robot_drag * v + float(disturbance)
    target_a = 0.015 * float(command) - cfg.target_drag * gv + float(target_accel)
    v2 = v + cfg.dt * robot_accel
    r2 = r + cfg.dt * v + 0.5 * cfg.dt**2 * robot_accel
    gv2 = gv + cfg.dt * target_a
    g2 = g + cfg.dt * gv + 0.5 * cfg.dt**2 * target_a
    return np.array([r2, v2, g2, gv2], dtype=np.float64)


def observe(state: np.ndarray, schedule: Schedule, t: int) -> np.ndarray:
    return np.array(
        [
            state[0] + schedule.proprio_bias[t],
            state[1],
            state[2] + schedule.vision_bias[t],
        ],
        dtype=np.float64,
    ) + schedule.obs_noise[t]


def oracle_action(state: np.ndarray, cfg: Config) -> float:
    """Frozen visual-to-action bottleneck used by every state-based controller.

    The target/robot latent is compressed to two bounded error channels before action decoding.
    The applied action is then copied into the next nominal prediction (efference copy).
    """
    position_code = math.tanh(1.35 * (state[2] - state[0]))
    velocity_code = math.tanh(0.90 * (state[3] - state[1]))
    interaction = position_code * velocity_code
    raw = 2.75 * position_code + 1.55 * velocity_code + 0.72 * interaction
    return float(cfg.max_action * np.tanh(raw / cfg.max_action))


def make_schedule(seed: int, severity: float, cfg: Config) -> Schedule:
    rng = np.random.default_rng(seed)
    command = float(rng.choice([-1.0, 1.0]))
    target0 = rng.uniform(-0.7, 0.7)
    initial_state = np.array(
        [target0 + rng.uniform(-1.3, 1.3), rng.normal(0.0, 0.08), target0, 0.24 * command],
        dtype=np.float64,
    )
    n = cfg.steps

    target_accel = np.zeros(n, dtype=np.float64)
    disturbance = np.zeros(n, dtype=np.float64)
    colored = 0.0
    target_colored = 0.0
    for t in range(n):
        colored = 0.93 * colored + rng.normal(0.0, 0.16)
        target_colored = 0.96 * target_colored + rng.normal(0.0, 0.055)
        disturbance[t] = severity * colored
        target_accel[t] = severity * (
            0.18 * math.sin(0.065 * t + 0.4 * command) + target_colored
        )

    pulse_t = int(rng.integers(max(8, n // 4), max(9, 2 * n // 3)))
    pulse_len = int(rng.integers(max(5, n // 18), max(6, n // 10)))
    pulse_len = min(pulse_len, n - pulse_t)
    pulse_sign = float(rng.choice([-1.0, 1.0]))
    window = np.sin(np.linspace(0.0, math.pi, pulse_len))
    disturbance[pulse_t : pulse_t + pulse_len] += severity * pulse_sign * 1.25 * window

    maneuver_t = int(rng.integers(max(10, n // 3), max(11, 3 * n // 4)))
    maneuver_len = int(rng.integers(max(6, n // 14), max(7, n // 7)))
    maneuver_len = min(maneuver_len, n - maneuver_t)
    target_accel[maneuver_t : maneuver_t + maneuver_len] += (
        severity * float(rng.choice([-1.0, 1.0])) * 0.55
    )

    proprio_bias = np.zeros(n, dtype=np.float64)
    vision_bias = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        proprio_bias[t] = 0.994 * proprio_bias[t - 1] + severity * rng.normal(0.0, 0.0018)
        vision_bias[t] = 0.997 * vision_bias[t - 1] + severity * rng.normal(0.0, 0.0028)
    drift_t = int(rng.integers(max(8, n // 3), max(9, 3 * n // 4)))
    proprio_bias[drift_t:] += severity * rng.normal(0.0, 0.055)
    vision_bias[drift_t:] += severity * rng.normal(0.0, 0.095)

    obs_noise = rng.normal(0.0, np.array([0.012, 0.020, 0.016]), size=(n, 3))
    payload = np.concatenate(
        [target_accel, disturbance, proprio_bias, vision_bias, obs_noise.reshape(-1)]
    ).tobytes()
    fingerprint = hashlib.sha256(payload).hexdigest()[:16]
    return Schedule(
        command=command,
        initial_state=initial_state,
        target_accel=target_accel,
        disturbance=disturbance,
        obs_noise=obs_noise,
        proprio_bias=proprio_bias,
        vision_bias=vision_bias,
        fingerprint=fingerprint,
    )


def pad_history(observations: list[np.ndarray], actions: list[float], history: int, command: float) -> np.ndarray:
    if not observations:
        raise ValueError("at least one observation is required")
    features: list[float] = []
    start = len(observations) - history
    for idx in range(start, len(observations)):
        if idx < 0:
            obs = observations[0]
            prev_action = 0.0
        else:
            obs = observations[idx]
            prev_action = actions[idx - 1] if idx > 0 and idx - 1 < len(actions) else 0.0
        features.extend(float(x) for x in obs)
        features.append(float(prev_action))
    features.append(float(command))
    return np.asarray(features, dtype=np.float64)


def optimize_prediction_error(
    prior: np.ndarray,
    observations: list[np.ndarray],
    window: int,
    iterations: int,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Correct the current latent from current sensory error and a windowed velocity cue.

    The recurrent state first produces a current prior from only its previous posterior and the
    copied action. Four current-state correction variables are then optimized against current
    proprioception/vision plus a target-velocity cue estimated from the recent visual window.
    Unlike the sibling ``prediction_error_policy_state`` experiment, this function does not
    regenerate a full latent trajectory. Observations are loss targets, never recurrence inputs.
    Iterations=0 is a strict inference-off path and does not read them.
    """
    if iterations == 0:
        return prior.copy(), np.zeros(4), 0.0

    prior = prior.copy()
    recent = np.asarray(observations[-window:], dtype=np.float64)
    if len(recent) >= 2:
        time = cfg.dt * np.arange(len(recent), dtype=np.float64)
        centered = time - np.mean(time)
        denom = float(np.sum(centered**2))
        target_velocity = float(
            np.sum(centered * (recent[:, 2] - np.mean(recent[:, 2]))) / max(denom, 1e-9)
        )
    else:
        target_velocity = float(prior[3])
    sensory_target = np.array(
        [recent[-1, 0], recent[-1, 1], recent[-1, 2], target_velocity], dtype=np.float64
    )

    sensory_weight = np.array([4.0, 2.4, 4.0, 1.35], dtype=np.float64)
    prior_weight = cfg.error_complexity * np.array([1.0, 1.2, 1.0, 1.8], dtype=np.float64)
    hessian_diag = sensory_weight + prior_weight
    gradient0 = sensory_weight * (prior - sensory_target)
    correction = np.zeros(4, dtype=np.float64)
    step = 0.92 / float(np.max(hessian_diag))
    # Fast proprioceptive variables and slow target context use the reported 2,5,8,16 ordering
    # as a compact leaky-coordinate abstraction, not as a reproduction of PredVLA's RNN layers.
    rates = 2.0 / np.asarray(cfg.latent_time_constants, dtype=np.float64)
    correction_limit = np.array([0.9, 1.2, 1.0, 0.75], dtype=np.float64)
    for _ in range(iterations):
        correction -= step * rates * (hessian_diag * correction + gradient0)
        correction = np.clip(correction, -correction_limit, correction_limit)

    current = prior + correction
    current[1] = float(np.clip(current[1], -3.0, 3.0))
    current[3] = float(np.clip(current[3], -1.6, 1.6))
    predicted_sensory = np.array([current[0], current[1], current[2], current[3]])
    final_error = float(np.mean((predicted_sensory - sensory_target) ** 2))
    return current, correction, final_error


def collect_training_episodes(seed: int, episodes: int, cfg: Config) -> list[dict[str, np.ndarray | float]]:
    data: list[dict[str, np.ndarray | float]] = []
    for ep in range(episodes):
        rng = np.random.default_rng(seed + 19_003 * ep)
        severity = float(rng.uniform(0.0, 1.0))
        schedule = make_schedule(seed + 47_117 * ep, severity, cfg)
        state = schedule.initial_state.copy()
        observations: list[np.ndarray] = []
        actions: list[float] = []
        states: list[np.ndarray] = []
        for t in range(cfg.steps):
            obs = observe(state, schedule, t)
            action = oracle_action(state, cfg)
            states.append(state.copy())
            observations.append(obs)
            actions.append(action)
            state = true_step(
                state,
                action,
                schedule.target_accel[t],
                schedule.disturbance[t],
                schedule.command,
                cfg,
            )
        data.append(
            {
                "observations": np.asarray(observations),
                "actions": np.asarray(actions),
                "states": np.asarray(states),
                "command": schedule.command,
            }
        )
    return data


def train_ridge_models(
    episodes: list[dict[str, np.ndarray | float]], cfg: Config
) -> tuple[dict[tuple[int, int], RidgeModel], dict[str, float]]:
    models: dict[tuple[int, int], RidgeModel] = {}
    metrics: dict[str, float] = {}
    specs = [(h, 1) for h in cfg.history_values] + [(16, 4)]
    split = max(1, int(0.8 * len(episodes)))

    for history, chunk in specs:
        x_rows: list[np.ndarray] = []
        y_rows: list[np.ndarray] = []
        episode_ids: list[int] = []
        for ep_idx, ep in enumerate(episodes):
            observations = [x for x in np.asarray(ep["observations"])]
            actions = [float(x) for x in np.asarray(ep["actions"])]
            command = float(ep["command"])
            for t in range(len(actions)):
                x_rows.append(pad_history(observations[: t + 1], actions[:t], history, command))
                future = np.asarray(actions[t : t + chunk], dtype=np.float64)
                if len(future) < chunk:
                    future = np.pad(future, (0, chunk - len(future)), mode="edge")
                y_rows.append(future)
                episode_ids.append(ep_idx)
        x = np.asarray(x_rows, dtype=np.float64)
        y = np.asarray(y_rows, dtype=np.float64)
        episode_ids_arr = np.asarray(episode_ids)
        train_mask = episode_ids_arr < split
        val_mask = ~train_mask
        mean = x[train_mask].mean(axis=0)
        std = x[train_mask].std(axis=0)
        std[std < 1e-6] = 1.0
        z = (x[train_mask] - mean) / std
        z = np.concatenate([z, np.ones((len(z), 1))], axis=1)
        reg = cfg.ridge_alpha * np.eye(z.shape[1])
        reg[-1, -1] = 0.0
        weights = np.linalg.solve(z.T @ z + reg, z.T @ y[train_mask])
        model = RidgeModel(history, chunk, mean, std, weights)
        models[(history, chunk)] = model
        pred = np.vstack([model.predict(row) for row in x[val_mask]])
        rmse = float(np.sqrt(np.mean((pred - y[val_mask]) ** 2)))
        metrics[f"history_h{history}_chunk{chunk}_validation_action_rmse"] = rmse
        metrics[f"history_h{history}_chunk{chunk}_parameter_count"] = float(weights.size)
    metrics["prediction_error_online_free_variables"] = 4.0
    metrics["prediction_error_learned_parameter_count"] = 0.0
    metrics["training_episodes"] = float(len(episodes))
    metrics["training_steps"] = float(sum(len(np.asarray(ep["actions"])) for ep in episodes))
    return models, metrics


def run_episode(
    method: str,
    schedule: Schedule,
    cfg: Config,
    models: dict[tuple[int, int], RidgeModel],
    error_window: int | None = None,
    error_iterations: int | None = None,
    capture: bool = False,
) -> dict[str, float | str | np.ndarray]:
    state = schedule.initial_state.copy()
    posterior = state.copy()  # exact initial observation/state, shared by all recurrent variants
    observations: list[np.ndarray] = []
    actions: list[float] = []
    anchor_states: list[np.ndarray] = []
    states: list[np.ndarray] = []
    estimates: list[np.ndarray] = []
    oracle_actions: list[float] = []
    errors: list[float] = []
    inference_losses: list[float] = []
    queue: list[float] = []

    for t in range(cfg.steps):
        if t == 0:
            anchor = posterior.copy()
        else:
            anchor = nominal_step(posterior, actions[-1], schedule.command, cfg)
        anchor_states.append(anchor)
        obs = observe(state, schedule, t)
        observations.append(obs)
        oracle_u = oracle_action(state, cfg)

        if method == "oracle":
            estimate = state.copy()
            action = oracle_u
        elif method in {"open_loop", "pred_error_w8"}:
            iterations = 0 if method == "open_loop" else (
                cfg.error_iterations if error_iterations is None else error_iterations
            )
            window = cfg.error_window if error_window is None else error_window
            estimate, _, loss = optimize_prediction_error(
                anchor,
                observations,
                window,
                iterations,
                cfg,
            )
            action = oracle_action(estimate, cfg)
            inference_losses.append(loss)
        elif method.startswith("history_h"):
            if method == "history_h4":
                spec = (4, 1)
            elif method == "history_h16":
                spec = (16, 1)
            elif method == "history_h16_chunk4":
                spec = (16, 4)
            else:
                parts = method.split("_h", 1)[1]
                history = int(parts.split("_")[0])
                spec = (history, 1)
            estimate = np.full(4, np.nan)
            if not queue:
                feature = pad_history(observations, actions, spec[0], schedule.command)
                pred = models[spec].predict(feature)
                queue = [float(np.clip(x, -cfg.max_action, cfg.max_action)) for x in pred]
            action = queue.pop(0)
        else:
            raise ValueError(f"unknown method: {method}")

        posterior = estimate if method in {"open_loop", "pred_error_w8"} else anchor
        states.append(state.copy())
        estimates.append(estimate.copy())
        oracle_actions.append(oracle_u)
        errors.append(float(abs(state[2] - state[0])))
        actions.append(float(action))
        state = true_step(
            state,
            action,
            schedule.target_accel[t],
            schedule.disturbance[t],
            schedule.command,
            cfg,
        )

    state_arr = np.asarray(states)
    estimate_arr = np.asarray(estimates)
    action_arr = np.asarray(actions)
    oracle_arr = np.asarray(oracle_actions)
    err_arr = np.asarray(errors)
    tail = err_arr[-cfg.success_tail_steps :]
    tracking_rmse = float(np.sqrt(np.mean(err_arr**2)))
    tail_rmse = float(np.sqrt(np.mean(tail**2)))
    final_error = float(err_arr[-1])
    result: dict[str, float | str | np.ndarray] = {
        "method": method,
        "schedule_fingerprint": schedule.fingerprint,
        "success": float(
            tail_rmse < cfg.success_tail_rmse and final_error < cfg.success_final_error
        ),
        "tracking_rmse": tracking_rmse,
        "tail_rmse": tail_rmse,
        "final_error": final_error,
        "policy_rmse": float(np.sqrt(np.mean((action_arr - oracle_arr) ** 2))),
        "action_jerk": float(np.sqrt(np.mean(np.diff(action_arr) ** 2))),
        "control_effort": float(np.sqrt(np.mean(action_arr**2))),
        "latent_rmse": float(
            np.sqrt(np.nanmean((estimate_arr - state_arr) ** 2))
            if np.any(np.isfinite(estimate_arr))
            else np.nan
        ),
        "mean_inference_loss": float(np.mean(inference_losses)) if inference_losses else np.nan,
    }
    if capture:
        result.update(
            {
                "states": state_arr,
                "estimates": estimate_arr,
                "observations": np.asarray(observations),
                "actions": action_arr,
                "oracle_actions": oracle_arr,
                "errors": err_arr,
                "disturbance": schedule.disturbance.copy(),
                "target_accel": schedule.target_accel.copy(),
            }
        )
    return result


def aggregate(rows: list[dict[str, float | str]], group_keys: Iterable[str]) -> list[dict[str, float | str]]:
    group_keys = tuple(group_keys)
    groups: dict[tuple[object, ...], list[dict[str, float | str]]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)
    metric_names = [
        "success",
        "tracking_rmse",
        "tail_rmse",
        "final_error",
        "policy_rmse",
        "action_jerk",
        "control_effort",
        "latent_rmse",
    ]
    summaries: list[dict[str, float | str]] = []
    for key, items in groups.items():
        out: dict[str, float | str] = dict(zip(group_keys, key))
        out["n"] = float(len(items))
        for metric in metric_names:
            vals = np.asarray([float(x[metric]) for x in items], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                out[f"{metric}_mean"] = float(np.mean(vals))
                out[f"{metric}_sem"] = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
            else:
                out[f"{metric}_mean"] = np.nan
                out[f"{metric}_sem"] = np.nan
        summaries.append(out)
    return summaries


def write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_sanity_checks(
    cfg: Config, models: dict[tuple[int, int], RidgeModel], output_dir: pathlib.Path
) -> dict[str, object]:
    schedule = make_schedule(9091, 1.5, cfg)
    base_obs = schedule.obs_noise.copy()
    altered = replace(schedule, obs_noise=base_obs + np.array([8.0, -5.0, 6.0]))
    open_a = run_episode("open_loop", schedule, cfg, models, capture=True)
    open_b = run_episode("open_loop", altered, cfg, models, capture=True)
    pe = run_episode("pred_error_w8", schedule, cfg, models, capture=True)
    pe_alt = run_episode("pred_error_w8", altered, cfg, models, capture=True)
    pe_zero = run_episode(
        "pred_error_w8", schedule, cfg, models, error_iterations=0, capture=True
    )
    hist_a = run_episode("history_h4", schedule, cfg, models, capture=True)
    hist_b = run_episode("history_h4", schedule, cfg, models, capture=True)
    zero = make_schedule(31337, 0.0, cfg)
    zero_open = run_episode("open_loop", zero, cfg, models)
    zero_oracle = run_episode("oracle", zero, cfg, models)

    checks = {
        "open_loop_observation_invariance": bool(np.array_equal(open_a["actions"], open_b["actions"])),
        "prediction_error_uses_observations": bool(
            np.max(np.abs(np.asarray(pe["actions"]) - np.asarray(pe_alt["actions"]))) > 1e-3
        ),
        "zero_iterations_match_open_loop": bool(
            np.array_equal(np.asarray(open_a["actions"]), np.asarray(pe_zero["actions"]))
        ),
        "correction_reduces_latent_rmse": bool(float(pe["latent_rmse"]) < float(open_a["latent_rmse"])),
        "correction_reduces_tracking_rmse": bool(float(pe["tracking_rmse"]) < float(open_a["tracking_rmse"])),
        "feedforward_is_deterministic": bool(np.array_equal(hist_a["actions"], hist_b["actions"])),
        "zero_disturbance_open_loop_near_oracle": bool(
            float(zero_open["tracking_rmse"]) <= float(zero_oracle["tracking_rmse"]) + 0.06
        ),
        "paired_schedule_fingerprint": bool(
            open_a["schedule_fingerprint"] == pe["schedule_fingerprint"] == hist_a["schedule_fingerprint"]
        ),
    }
    values = {
        "open_loop_observation_action_max_diff": float(
            np.max(np.abs(np.asarray(open_a["actions"]) - np.asarray(open_b["actions"])))
        ),
        "prediction_error_observation_action_max_diff": float(
            np.max(np.abs(np.asarray(pe["actions"]) - np.asarray(pe_alt["actions"])))
        ),
        "zero_iteration_action_max_diff": float(
            np.max(np.abs(np.asarray(open_a["actions"]) - np.asarray(pe_zero["actions"])))
        ),
        "open_loop_latent_rmse": float(open_a["latent_rmse"]),
        "prediction_error_latent_rmse": float(pe["latent_rmse"]),
        "open_loop_tracking_rmse": float(open_a["tracking_rmse"]),
        "prediction_error_tracking_rmse": float(pe["tracking_rmse"]),
        "zero_disturbance_open_loop_tracking_rmse": float(zero_open["tracking_rmse"]),
        "zero_disturbance_oracle_tracking_rmse": float(zero_oracle["tracking_rmse"]),
    }
    payload = {"all_passed": bool(all(checks.values())), "checks": checks, "values": values}
    (output_dir / "sanity_check.json").write_text(json.dumps(json_ready(payload), indent=2) + "\n")
    if not payload["all_passed"]:
        failed = [k for k, v in checks.items() if not v]
        raise RuntimeError(f"sanity checks failed: {failed}")
    return payload


def plot_disturbance(summary: list[dict[str, object]], output_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    specs = [
        ("success", "Success rate", (0.0, 1.05)),
        ("tracking_rmse", "Tracking RMSE", None),
        ("policy_rmse", "Action error vs oracle", None),
        ("action_jerk", "Action jerk RMS", None),
    ]
    for method in METHOD_ORDER:
        rows = sorted(
            [r for r in summary if r["method"] == method], key=lambda r: float(r["severity"])
        )
        xs = [float(r["severity"]) for r in rows]
        for ax, (metric, title, ylim) in zip(axes.flat, specs):
            ys = [float(r[f"{metric}_mean"]) for r in rows]
            sem = [float(r[f"{metric}_sem"]) for r in rows]
            ax.plot(xs, ys, marker="o", linewidth=2, label=LABELS[method], color=COLORS[method])
            ax.fill_between(xs, np.asarray(ys) - sem, np.asarray(ys) + sem, alpha=0.12, color=COLORS[method])
            ax.set_title(title)
            ax.set_xlabel("Hidden-disturbance / drift severity")
            ax.grid(alpha=0.25)
            if ylim:
                ax.set_ylim(*ylim)
    axes[0, 0].legend(fontsize=8, ncol=2, loc="lower left")
    fig.suptitle("PredVLA-inspired online correction: paired disturbance sweep", fontsize=14)
    fig.savefig(output_dir / "disturbance_sweep.png", dpi=180)
    plt.close(fig)


def plot_history(summary: list[dict[str, object]], output_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for severity in sorted({float(r["severity"]) for r in summary}):
        rows = sorted(
            [r for r in summary if float(r["severity"]) == severity], key=lambda r: int(r["history"])
        )
        xs = [int(r["history"]) for r in rows]
        axes[0].plot(xs, [float(r["tracking_rmse_mean"]) for r in rows], marker="o", label=f"severity {severity:g}")
        axes[1].plot(xs, [float(r["success_mean"]) for r in rows], marker="o", label=f"severity {severity:g}")
    axes[0].set_title("Feed-forward history sweep")
    axes[0].set_ylabel("Tracking RMSE")
    axes[1].set_title("Success vs history")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(-0.03, 1.03)
    for ax in axes:
        ax.set_xlabel("History length H")
        ax.set_xticks(list(Config.history_values))
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(output_dir / "history_sweep.png", dpi=180)
    plt.close(fig)


def plot_iterations(summary: list[dict[str, object]], output_dir: pathlib.Path) -> None:
    rows = sorted(summary, key=lambda r: int(r["iterations"]))
    x = [int(r["iterations"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "tracking_rmse", "Tracking RMSE"),
        (axes[1], "latent_rmse", "Latent-state RMSE"),
        (axes[2], "success", "Success rate"),
    ]:
        y = [float(r[f"{metric}_mean"]) for r in rows]
        sem = [float(r[f"{metric}_sem"]) for r in rows]
        ax.errorbar(x, y, yerr=sem, marker="o", linewidth=2, capsize=3, color=COLORS["pred_error_w8"])
        ax.axvline(0, color=COLORS["open_loop"], linestyle="--", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Error-regression iterations (0 = exact open loop)")
        ax.grid(alpha=0.25)
    axes[2].set_ylim(-0.03, 1.03)
    fig.savefig(output_dir / "iteration_ablation.png", dpi=180)
    plt.close(fig)


def plot_rollout(
    rollouts: dict[str, dict[str, float | str | np.ndarray]], output_dir: pathlib.Path, cfg: Config
) -> None:
    t = np.arange(cfg.steps) * cfg.dt
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.4), sharex=True, constrained_layout=True)
    target = np.asarray(rollouts["oracle"]["states"])[:, 2]
    axes[0].plot(t, target, color="black", linewidth=2.5, label="Target")
    for method in METHOD_ORDER:
        states = np.asarray(rollouts[method]["states"])
        axes[0].plot(t, states[:, 0], color=COLORS[method], label=LABELS[method], alpha=0.9)
        axes[1].plot(t, np.asarray(rollouts[method]["errors"]), color=COLORS[method], label=LABELS[method])
        axes[2].plot(t, np.asarray(rollouts[method]["actions"]), color=COLORS[method], label=LABELS[method], alpha=0.9)
    ax2 = axes[1].twinx()
    ax2.fill_between(t, np.asarray(rollouts["oracle"]["disturbance"]), color="gray", alpha=0.16)
    ax2.set_ylabel("Hidden force", color="gray")
    axes[0].set_ylabel("Position")
    axes[1].set_ylabel("Absolute tracking error")
    axes[2].set_ylabel("Action")
    axes[2].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(alpha=0.22)
    axes[0].legend(fontsize=7.5, ncol=3)
    fig.suptitle("Representative paired rollout at severity 1.5", fontsize=14)
    fig.savefig(output_dir / "single_rollout.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--train-episodes", type=int, default=None)
    p.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    if args.trials is not None:
        cfg = replace(cfg, trials=args.trials)
    if args.train_episodes is not None:
        cfg = replace(cfg, train_episodes=args.train_episodes)
    if args.smoke:
        cfg = replace(
            cfg,
            steps=90,
            train_episodes=18,
            trials=5,
            severities=(0.0, 1.0, 2.0),
            history_sweep_severities=(1.0, 2.0),
            iteration_values=(0, 4, 12),
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {cfg.train_episodes} deterministic training episodes...")
    training = collect_training_episodes(args.seed + 100_000, cfg.train_episodes, cfg)
    models, training_metrics = train_ridge_models(training, cfg)
    (output_dir / "training_metrics.json").write_text(
        json.dumps(json_ready(training_metrics), indent=2) + "\n"
    )
    sanity = run_sanity_checks(cfg, models, output_dir)
    print("Sanity checks passed.")

    trial_rows: list[dict[str, object]] = []
    for severity in cfg.severities:
        for trial in range(cfg.trials):
            schedule = make_schedule(args.seed + 1_000_003 * trial, severity, cfg)
            for method in METHOD_ORDER:
                result = run_episode(method, schedule, cfg, models)
                trial_rows.append(
                    {
                        "method": method,
                        "severity": severity,
                        "trial": trial,
                        **{k: v for k, v in result.items() if not isinstance(v, np.ndarray)},
                    }
                )
    disturbance_summary = aggregate(trial_rows, ("method", "severity"))
    write_csv(output_dir / "disturbance_sweep_trials.csv", trial_rows)
    write_csv(output_dir / "disturbance_sweep_summary.csv", disturbance_summary)

    history_rows: list[dict[str, object]] = []
    for severity in cfg.history_sweep_severities:
        for trial in range(cfg.trials):
            schedule = make_schedule(args.seed + 7_000_021 * trial, severity, cfg)
            for history in cfg.history_values:
                method = f"history_h{history}"
                result = run_episode(method, schedule, cfg, models)
                history_rows.append(
                    {
                        "method": method,
                        "history": history,
                        "severity": severity,
                        "trial": trial,
                        **{k: v for k, v in result.items() if not isinstance(v, np.ndarray)},
                    }
                )
    history_summary = aggregate(history_rows, ("method", "history", "severity"))
    write_csv(output_dir / "history_sweep_summary.csv", history_summary)

    iteration_rows: list[dict[str, object]] = []
    for trial in range(cfg.trials):
        schedule = make_schedule(args.seed + 11_000_009 * trial, 1.5, cfg)
        for iterations in cfg.iteration_values:
            result = run_episode(
                "pred_error_w8", schedule, cfg, models, error_iterations=iterations
            )
            iteration_rows.append(
                {
                    "iterations": iterations,
                    "severity": 1.5,
                    "trial": trial,
                    **{k: v for k, v in result.items() if not isinstance(v, np.ndarray)},
                }
            )
    iteration_summary = aggregate(iteration_rows, ("iterations", "severity"))
    write_csv(output_dir / "iteration_ablation_summary.csv", iteration_summary)

    representative = make_schedule(args.seed + 424_242, 1.5, cfg)
    rollouts = {
        method: run_episode(method, representative, cfg, models, capture=True)
        for method in METHOD_ORDER
    }
    plot_disturbance(disturbance_summary, output_dir)
    plot_history(history_summary, output_dir)
    plot_iterations(iteration_summary, output_dir)
    plot_rollout(rollouts, output_dir, cfg)

    metrics = {
        "seed": args.seed,
        "config": asdict(cfg),
        "method_labels": LABELS,
        "training_metrics": training_metrics,
        "sanity_check": sanity,
        "disturbance_sweep": disturbance_summary,
        "history_sweep": history_summary,
        "iteration_ablation": iteration_summary,
    }
    (output_dir / "metrics.json").write_text(json.dumps(json_ready(metrics), indent=2) + "\n")

    severe = {
        str(r["method"]): r
        for r in disturbance_summary
        if float(r["severity"]) == max(cfg.severities)
    }
    print("\nMaximum-severity summary:")
    for method in METHOD_ORDER:
        row = severe[method]
        print(
            f"  {method:22s} success={float(row['success_mean']):.3f} "
            f"tracking_rmse={float(row['tracking_rmse_mean']):.3f} "
            f"policy_rmse={float(row['policy_rmse_mean']):.3f}"
        )
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
