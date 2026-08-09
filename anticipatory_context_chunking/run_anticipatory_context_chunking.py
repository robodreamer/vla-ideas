#!/usr/bin/env python3
"""Future-context prediction toy for asynchronous action-chunk handoffs.

This is a deliberately small synthetic experiment, not a FutureRTC reproduction.  A chunked
controller tracks a maneuvering target while inference is delayed.  At each handoff it may plan
from stale context, roll only robot state forward, correct that state with a learned residual, or
also predict the execution-time environment latent using transport and learned innovation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
from dataclasses import asdict, dataclass, replace
from typing import Callable

BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


METHOD_ORDER = [
    "oracle_fresh",
    "naive_stale",
    "state_rollout",
    "state_correction",
    "obs_transport",
    "obs_innovation",
    "policy_consistency",
]
LABELS = {
    "oracle_fresh": "Oracle fresh context",
    "naive_stale": "Naive async (stale)",
    "state_rollout": "State rollout only",
    "state_correction": "State correction",
    "obs_transport": "Obs. transport only",
    "obs_innovation": "Transport + innovation",
    "policy_consistency": "+ policy consistency",
}
COLORS = {
    "oracle_fresh": "#4c78a8",
    "naive_stale": "#f58518",
    "state_rollout": "#eeca3b",
    "state_correction": "#54a24b",
    "obs_transport": "#72b7b2",
    "obs_innovation": "#b279a2",
    "policy_consistency": "#e45756",
}


@dataclass
class Config:
    dt: float = 0.06
    chunk_horizon: int = 12
    max_delay: int = 10
    episode_steps: int = 216
    max_accel: float = 4.5
    robot_drag_true: float = 0.24
    robot_drag_nominal: float = 0.12
    actuator_gain: float = 0.82
    wind_force: float = 0.58
    target_accel: float = 1.05
    cue_turn_rate: float = 1.15
    wind_decay: float = 0.94
    target_noise: float = 0.11
    cue_noise: float = 0.055
    wind_noise: float = 0.055
    robot_noise: float = 0.035
    kp: float = 2.45
    kd: float = 1.55
    target_lookahead: float = 0.28
    cue_feedforward: float = 0.22
    wind_feedforward: float = 0.42
    success_tail_steps: int = 72
    success_tail_rmse: float = 0.72
    success_final_error: float = 0.90


@dataclass
class NormalizedRegressor:
    model: nn.Module
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        x2 = np.atleast_2d(x).astype(np.float32)
        xn = (x2 - self.x_mean) / self.x_std
        with torch.no_grad():
            yn = self.model(torch.from_numpy(xn)).cpu().numpy()
        y = yn * self.y_std + self.y_mean
        return y[0] if np.asarray(x).ndim == 1 else y


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def unit_vectors(theta: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)


def rotate(v: np.ndarray, angle: float | np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    x = c * v[..., 0] - s * v[..., 1]
    y = s * v[..., 0] + c * v[..., 1]
    return np.stack([x, y], axis=-1)


def bounded_action(raw: np.ndarray, cfg: Config) -> np.ndarray:
    return cfg.max_accel * np.tanh(raw / cfg.max_accel)


def policy_action(state: np.ndarray, latent: np.ndarray, cfg: Config) -> np.ndarray:
    """Action from synthetic proprioception + environment latent.

    state = [robot position(2), velocity(2)]
    latent = [target position(2), velocity(2), maneuver cue(2), wind(2)]
    """
    robot_p, robot_v = state[..., :2], state[..., 2:4]
    target_p, target_v = latent[..., :2], latent[..., 2:4]
    cue, wind = latent[..., 4:6], latent[..., 6:8]
    desired_p = target_p + cfg.target_lookahead * target_v
    raw = (
        cfg.kp * (desired_p - robot_p)
        + cfg.kd * (target_v - robot_v)
        + cfg.cue_feedforward * cue
        - cfg.wind_feedforward * wind
    )
    return bounded_action(raw, cfg)


def policy_action_torch(state: torch.Tensor, latent: torch.Tensor, cfg: Config) -> torch.Tensor:
    robot_p, robot_v = state[..., :2], state[..., 2:4]
    target_p, target_v = latent[..., :2], latent[..., 2:4]
    cue, wind = latent[..., 4:6], latent[..., 6:8]
    desired_p = target_p + cfg.target_lookahead * target_v
    raw = (
        cfg.kp * (desired_p - robot_p)
        + cfg.kd * (target_v - robot_v)
        + cfg.cue_feedforward * cue
        - cfg.wind_feedforward * wind
    )
    return cfg.max_accel * torch.tanh(raw / cfg.max_accel)


def nominal_robot_step(state: np.ndarray, action: np.ndarray, cfg: Config) -> np.ndarray:
    p, v = state[..., :2], state[..., 2:4]
    accel = action - cfg.robot_drag_nominal * v
    v_next = v + cfg.dt * accel
    p_next = p + cfg.dt * v + 0.5 * cfg.dt**2 * accel
    return np.concatenate([p_next, v_next], axis=-1)


def true_step(
    state: np.ndarray,
    latent: np.ndarray,
    action: np.ndarray,
    cfg: Config,
    target_noise: np.ndarray,
    cue_noise: np.ndarray,
    wind_noise: np.ndarray,
    robot_noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Coupled true dynamics. Environment evolution is action-independent for paired trials."""
    p, v = state[..., :2], state[..., 2:4]
    target_p, target_v = latent[..., :2], latent[..., 2:4]
    cue, wind = latent[..., 4:6], latent[..., 6:8]

    target_a = cfg.target_accel * cue + target_noise
    target_v_next = target_v + cfg.dt * target_a
    target_p_next = target_p + cfg.dt * target_v + 0.5 * cfg.dt**2 * target_a

    cue_next = rotate(cue, cfg.cue_turn_rate * cfg.dt) + cue_noise
    cue_next = cue_next / np.maximum(np.linalg.norm(cue_next, axis=-1, keepdims=True), 1e-6)
    wind_next = cfg.wind_decay * wind + wind_noise

    gain = cfg.actuator_gain + 0.10 * np.tanh(np.sum(cue * action, axis=-1, keepdims=True) / 5.0)
    robot_a = gain * action - cfg.robot_drag_true * v + cfg.wind_force * wind + robot_noise
    v_next = v + cfg.dt * robot_a
    p_next = p + cfg.dt * v + 0.5 * cfg.dt**2 * robot_a
    next_state = np.concatenate([p_next, v_next], axis=-1)
    next_latent = np.concatenate([target_p_next, target_v_next, cue_next, wind_next], axis=-1)
    return next_state, next_latent


def transport_latent(latent: np.ndarray, delay: int, cfg: Config) -> np.ndarray:
    """Analytic transport-only ablation: constant target velocity, frozen cue, decayed wind."""
    out = np.array(latent, dtype=np.float64, copy=True)
    tau = delay * cfg.dt
    out[..., :2] = latent[..., :2] + tau * latent[..., 2:4]
    out[..., 6:8] = (cfg.wind_decay**delay) * latent[..., 6:8]
    return out


def rollout_state(state: np.ndarray, actions: np.ndarray, cfg: Config) -> np.ndarray:
    out = np.array(state, dtype=np.float64, copy=True)
    for action in actions:
        out = nominal_robot_step(out, action, cfg)
    return out


def rollout_policy_chunk(state: np.ndarray, latent: np.ndarray, cfg: Config) -> np.ndarray:
    """Base chunk policy; all methods share it and differ only in handoff context."""
    s = np.array(state, dtype=np.float64, copy=True)
    z = np.array(latent, dtype=np.float64, copy=True)
    actions = []
    for _ in range(cfg.chunk_horizon):
        action = policy_action(s, z, cfg)
        actions.append(action)
        s = nominal_robot_step(s, action, cfg)
        # The base policy has a simple motion prior, but no stochastic future information.
        target_a = 0.72 * cfg.target_accel * z[4:6]
        z[:2] = z[:2] + cfg.dt * z[2:4] + 0.5 * cfg.dt**2 * target_a
        z[2:4] = z[2:4] + cfg.dt * target_a
        z[4:6] = rotate(z[4:6], cfg.cue_turn_rate * cfg.dt)
        z[6:8] = cfg.wind_decay * z[6:8]
    return np.asarray(actions, dtype=np.float64)


def pad_actions(actions: np.ndarray, cfg: Config) -> np.ndarray:
    padded = np.zeros((cfg.max_delay, 2), dtype=np.float64)
    n = min(len(actions), cfg.max_delay)
    if n:
        padded[:n] = actions[:n]
    return padded.reshape(-1)


def state_features(state: np.ndarray, latent: np.ndarray, actions: np.ndarray, delay: int, cfg: Config) -> np.ndarray:
    summaries = np.zeros(6, dtype=np.float64)
    if len(actions):
        summaries[:2] = np.mean(actions, axis=0)
        summaries[2:4] = np.sum(actions, axis=0) * cfg.dt
        summaries[4] = float(np.mean(np.linalg.norm(actions, axis=1)))
        summaries[5] = float(len(actions)) / cfg.max_delay
    return np.concatenate(
        [state, latent, pad_actions(actions, cfg), summaries, [delay / cfg.max_delay]], axis=0
    )


def obs_features(
    stale_state: np.ndarray,
    stale_latent: np.ndarray,
    estimated_state: np.ndarray,
    transported_latent: np.ndarray,
    actions: np.ndarray,
    delay: int,
    cfg: Config,
) -> np.ndarray:
    return np.concatenate(
        [
            stale_state,
            stale_latent,
            estimated_state,
            transported_latent,
            pad_actions(actions, cfg),
            [delay / cfg.max_delay],
        ],
        axis=0,
    )


def generate_transition_dataset(n: int, cfg: Config, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    robot_p = rng.uniform(-4.5, 4.5, size=(n, 2))
    robot_v = rng.uniform(-2.0, 2.0, size=(n, 2))
    target_p = robot_p + rng.uniform(-3.5, 3.5, size=(n, 2))
    target_v = rng.uniform(-1.4, 1.4, size=(n, 2))
    cue = unit_vectors(rng.uniform(-np.pi, np.pi, size=n))
    wind = rng.normal(0.0, 0.55, size=(n, 2))
    state0 = np.concatenate([robot_p, robot_v], axis=1)
    latent0 = np.concatenate([target_p, target_v, cue, wind], axis=1)
    delay = rng.integers(1, cfg.max_delay + 1, size=n)

    # On-policy-ish committed prefixes with exploration, rather than arbitrary white-noise actions.
    actions = np.zeros((n, cfg.max_delay, 2), dtype=np.float64)
    plan_s, plan_z = state0.copy(), latent0.copy()
    for k in range(cfg.max_delay):
        base = policy_action(plan_s, plan_z, cfg)
        actions[:, k] = np.clip(base + rng.normal(0.0, 0.70, size=(n, 2)), -cfg.max_accel, cfg.max_accel)
        plan_s = nominal_robot_step(plan_s, actions[:, k], cfg)
        plan_z[:, :2] += cfg.dt * plan_z[:, 2:4]
        plan_z[:, 4:6] = rotate(plan_z[:, 4:6], cfg.cue_turn_rate * cfg.dt)

    state, latent = state0.copy(), latent0.copy()
    for k in range(cfg.max_delay):
        mask = (k < delay)[:, None]
        next_state, next_latent = true_step(
            state,
            latent,
            actions[:, k],
            cfg,
            rng.normal(0.0, cfg.target_noise, size=(n, 2)),
            rng.normal(0.0, cfg.cue_noise, size=(n, 2)),
            rng.normal(0.0, cfg.wind_noise, size=(n, 2)),
            rng.normal(0.0, cfg.robot_noise, size=(n, 2)),
        )
        state = np.where(mask, next_state, state)
        latent = np.where(mask, next_latent, latent)

    nominal_state = np.empty_like(state)
    transported = np.empty_like(latent)
    state_x, obs_x = [], []
    for i in range(n):
        prefix = actions[i, : delay[i]]
        nominal_state[i] = rollout_state(state0[i], prefix, cfg)
        transported[i] = transport_latent(latent0[i], int(delay[i]), cfg)
        state_x.append(state_features(state0[i], latent0[i], prefix, int(delay[i]), cfg))
        # Train observation prediction with the true future state to isolate visual prediction.
        obs_x.append(
            obs_features(
                state0[i], latent0[i], state[i], transported[i], prefix, int(delay[i]), cfg
            )
        )
    return {
        "state_x": np.asarray(state_x, dtype=np.float32),
        "state_residual": (state - nominal_state).astype(np.float32),
        "obs_x": np.asarray(obs_x, dtype=np.float32),
        "obs_residual": (latent - transported).astype(np.float32),
        "future_state": state.astype(np.float32),
        "future_latent": latent.astype(np.float32),
        "transported_latent": transported.astype(np.float32),
        "nominal_state": nominal_state.astype(np.float32),
        "delay": delay.astype(np.int64),
    }


def train_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    extra_loss: Callable[[torch.Tensor, np.ndarray], torch.Tensor] | None = None,
    extra_train: dict[str, np.ndarray] | None = None,
) -> tuple[NormalizedRegressor, dict[str, float]]:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-5
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True) + 1e-5
    xn = ((x_train - x_mean) / x_std).astype(np.float32)
    yn = ((y_train - y_mean) / y_std).astype(np.float32)
    xvn = ((x_val - x_mean) / x_std).astype(np.float32)
    yvn = ((y_val - y_mean) / y_std).astype(np.float32)

    model = MLP(x_train.shape[1], y_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-5)
    generator = torch.Generator().manual_seed(seed + 101)
    x_tensor, y_tensor = torch.from_numpy(xn), torch.from_numpy(yn)
    n = len(x_train)
    model.train()
    last_loss = 0.0
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),), generator=generator)
        pred = model(x_tensor[idx])
        loss = torch.mean((pred - y_tensor[idx]) ** 2)
        if extra_loss is not None and extra_train is not None:
            loss = loss + extra_loss(pred, idx.numpy())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        last_loss = float(loss.detach())

    model.eval()
    reg = NormalizedRegressor(model, x_mean, x_std, y_mean, y_std)
    val_pred = reg.predict(x_val)
    metrics = {
        "last_train_loss": last_loss,
        "val_rmse": float(np.sqrt(np.mean((val_pred - y_val) ** 2))),
        "val_mae": float(np.mean(np.abs(val_pred - y_val))),
    }
    return reg, metrics


def train_models(
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    cfg: Config,
    seed: int,
    train_steps: int,
) -> tuple[dict[str, NormalizedRegressor], dict[str, object]]:
    state_model, state_metrics = train_regressor(
        train["state_x"], train["state_residual"], val["state_x"], val["state_residual"],
        seed=seed + 1, steps=train_steps, batch_size=512, learning_rate=2e-3,
    )
    innovation_model, innovation_metrics = train_regressor(
        train["obs_x"], train["obs_residual"], val["obs_x"], val["obs_residual"],
        seed=seed + 2, steps=train_steps, batch_size=512, learning_rate=2e-3,
    )

    # The policy-consistency loss is differentiable through the same fixed base policy used at test.
    x_train = train["obs_x"]
    y_train = train["obs_residual"]
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-5
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True) + 1e-5
    torch.manual_seed(seed + 3)
    model = MLP(x_train.shape[1], y_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-5)
    gen = torch.Generator().manual_seed(seed + 303)
    xt = torch.from_numpy(((x_train - x_mean) / x_std).astype(np.float32))
    yt = torch.from_numpy(((y_train - y_mean) / y_std).astype(np.float32))
    y_mean_t = torch.from_numpy(y_mean.astype(np.float32))
    y_std_t = torch.from_numpy(y_std.astype(np.float32))
    transport_t = torch.from_numpy(train["transported_latent"])
    future_state_t = torch.from_numpy(train["future_state"])
    future_latent_t = torch.from_numpy(train["future_latent"])
    pc_weight = 3.0
    model.train()
    last_total = last_recon = last_policy = 0.0
    for _ in range(train_steps):
        idx = torch.randint(0, len(x_train), (min(512, len(x_train)),), generator=gen)
        pred_norm = model(xt[idx])
        recon = torch.mean((pred_norm - yt[idx]) ** 2)
        residual = pred_norm * y_std_t + y_mean_t
        z_pred = transport_t[idx] + residual
        action_pred = policy_action_torch(future_state_t[idx], z_pred, cfg)
        action_true = policy_action_torch(future_state_t[idx], future_latent_t[idx], cfg)
        policy_loss = torch.mean(((action_pred - action_true) / cfg.max_accel) ** 2)
        total = recon + pc_weight * policy_loss
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        last_total, last_recon, last_policy = map(float, (total.detach(), recon.detach(), policy_loss.detach()))
    model.eval()
    pc_model = NormalizedRegressor(model, x_mean, x_std, y_mean, y_std)
    pc_val_pred = pc_model.predict(val["obs_x"])
    pc_metrics = {
        "last_train_loss": last_total,
        "last_reconstruction_loss": last_recon,
        "last_policy_loss": last_policy,
        "val_rmse": float(np.sqrt(np.mean((pc_val_pred - val["obs_residual"]) ** 2))),
        "val_mae": float(np.mean(np.abs(pc_val_pred - val["obs_residual"]))),
        "policy_weight": pc_weight,
    }

    models = {"state": state_model, "innovation": innovation_model, "policy": pc_model}
    predictor_eval = evaluate_predictors(val, models, cfg)
    metrics = {
        "state_corrector": state_metrics,
        "innovation_predictor": innovation_metrics,
        "policy_consistency_predictor": pc_metrics,
        "held_out_ablation": predictor_eval,
    }
    return models, metrics


def evaluate_predictors(
    val: dict[str, np.ndarray], models: dict[str, NormalizedRegressor], cfg: Config
) -> dict[str, dict[str, float]]:
    methods = {
        "obs_transport": val["transported_latent"],
        "obs_innovation": val["transported_latent"] + models["innovation"].predict(val["obs_x"]),
        "policy_consistency": val["transported_latent"] + models["policy"].predict(val["obs_x"]),
    }
    out: dict[str, dict[str, float]] = {}
    for name, z_pred in methods.items():
        action_pred = policy_action(val["future_state"], z_pred, cfg)
        action_true = policy_action(val["future_state"], val["future_latent"], cfg)
        out[name] = {
            "latent_rmse": float(np.sqrt(np.mean((z_pred - val["future_latent"]) ** 2))),
            "target_position_rmse": float(np.sqrt(np.mean((z_pred[:, :2] - val["future_latent"][:, :2]) ** 2))),
            "policy_action_rmse": float(np.sqrt(np.mean((action_pred - action_true) ** 2))),
        }
    nominal_err = val["nominal_state"] - val["future_state"]
    corrected = val["nominal_state"] + models["state"].predict(val["state_x"])
    out["state_rollout"] = {"proprio_rmse": float(np.sqrt(np.mean(nominal_err**2)))}
    out["state_correction"] = {
        "proprio_rmse": float(np.sqrt(np.mean((corrected - val["future_state"]) ** 2)))
    }
    return out


def estimate_context(
    method: str,
    stale_state: np.ndarray,
    stale_latent: np.ndarray,
    true_state: np.ndarray,
    true_latent: np.ndarray,
    actions: np.ndarray,
    delay: int,
    models: dict[str, NormalizedRegressor],
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    if delay == 0 or method == "oracle_fresh":
        return true_state.copy(), true_latent.copy()
    if method == "naive_stale":
        return stale_state.copy(), stale_latent.copy()

    nominal = rollout_state(stale_state, actions, cfg)
    if method == "state_rollout":
        return nominal, stale_latent.copy()

    sx = state_features(stale_state, stale_latent, actions, delay, cfg)
    corrected = nominal + models["state"].predict(sx)
    if method == "state_correction":
        return corrected, stale_latent.copy()

    transported = transport_latent(stale_latent, delay, cfg)
    if method == "obs_transport":
        return corrected, transported

    ox = obs_features(stale_state, stale_latent, corrected, transported, actions, delay, cfg)
    if method == "obs_innovation":
        return corrected, transported + models["innovation"].predict(ox)
    if method == "policy_consistency":
        return corrected, transported + models["policy"].predict(ox)
    raise KeyError(method)


def initial_context(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    target_p = rng.uniform(-0.7, 0.7, size=2)
    target_v = rng.uniform(-0.7, 0.7, size=2)
    cue = unit_vectors(np.array([rng.uniform(-np.pi, np.pi)]))[0]
    wind = rng.normal(0.0, 0.38, size=2)
    robot_p = target_p + np.array([-3.1, -1.4]) + rng.normal(0.0, 0.25, size=2)
    robot_v = rng.normal(0.0, 0.15, size=2)
    return np.concatenate([robot_p, robot_v]), np.concatenate([target_p, target_v, cue, wind])


def simulate_episode(
    method: str,
    delay: int,
    seed: int,
    models: dict[str, NormalizedRegressor],
    cfg: Config,
    keep_trace: bool = False,
) -> tuple[dict[str, float], dict[str, np.ndarray] | None]:
    rng = np.random.default_rng(seed)
    state, latent = initial_context(rng)
    n = cfg.episode_steps
    target_noise = rng.normal(0.0, cfg.target_noise, size=(n, 2))
    cue_noise = rng.normal(0.0, cfg.cue_noise, size=(n, 2))
    wind_noise = rng.normal(0.0, cfg.wind_noise, size=(n, 2))
    robot_noise = rng.normal(0.0, cfg.robot_noise, size=(n, 2))

    state_hist = [state.copy()]
    latent_hist = [latent.copy()]
    executed_actions: list[np.ndarray] = []
    plan = rollout_policy_chunk(state, latent, cfg)
    action_index = 0
    jumps, proprio_errors, latent_errors, policy_errors = [], [], [], []
    robot_path, target_path, errors = [], [], []

    for t in range(n):
        if t > 0 and t % cfg.chunk_horizon == 0:
            stale_t = max(0, t - delay)
            prefix = np.asarray(executed_actions[stale_t:t], dtype=np.float64)
            stale_state = state_hist[stale_t]
            stale_latent = latent_hist[stale_t]
            est_state, est_latent = estimate_context(
                method, stale_state, stale_latent, state, latent, prefix, delay, models, cfg
            )
            new_plan = rollout_policy_chunk(est_state, est_latent, cfg)
            jumps.append(float(np.linalg.norm(new_plan[0] - executed_actions[-1])))
            proprio_errors.append(float(np.linalg.norm(est_state - state)))
            latent_errors.append(float(np.sqrt(np.mean((est_latent - latent) ** 2))))
            oracle_first = rollout_policy_chunk(state, latent, cfg)[0]
            policy_errors.append(float(np.linalg.norm(new_plan[0] - oracle_first)))
            plan = new_plan
            action_index = 0

        action = plan[min(action_index, len(plan) - 1)]
        action_index += 1
        executed_actions.append(action.copy())
        robot_path.append(state[:2].copy())
        target_path.append(latent[:2].copy())
        errors.append(float(np.linalg.norm(state[:2] - latent[:2])))
        state, latent = true_step(
            state, latent, action, cfg,
            target_noise[t], cue_noise[t], wind_noise[t], robot_noise[t],
        )
        state_hist.append(state.copy())
        latent_hist.append(latent.copy())

    errors_a = np.asarray(errors)
    actions_a = np.asarray(executed_actions)
    tail = errors_a[-min(cfg.success_tail_steps, len(errors_a)) :]
    final_error = float(errors_a[-1])
    tail_rmse = float(np.sqrt(np.mean(tail**2)))
    metrics = {
        "tracking_rmse": float(np.sqrt(np.mean(errors_a**2))),
        "tail_tracking_rmse": tail_rmse,
        "final_error": final_error,
        "success": float(tail_rmse < cfg.success_tail_rmse and final_error < cfg.success_final_error),
        "handoff_action_jump": float(np.mean(jumps)) if jumps else 0.0,
        "proprio_context_error": float(np.mean(proprio_errors)) if proprio_errors else 0.0,
        "latent_context_rmse": float(np.mean(latent_errors)) if latent_errors else 0.0,
        "handoff_policy_error": float(np.mean(policy_errors)) if policy_errors else 0.0,
        "control_effort": float(np.mean(np.sum(actions_a**2, axis=1))),
    }
    trace = None
    if keep_trace:
        trace = {
            "robot": np.asarray(robot_path),
            "target": np.asarray(target_path),
            "error": errors_a,
            "actions": actions_a,
        }
    return metrics, trace


def mean_sem(values: list[float]) -> tuple[float, float]:
    a = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(a))
    sem = float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return mean, sem


def run_sweep(
    delays: list[int],
    trials: int,
    models: dict[str, NormalizedRegressor],
    cfg: Config,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    trial_rows: list[dict[str, object]] = []
    for delay in delays:
        for trial in range(trials):
            episode_seed = seed + 10_000 * delay + trial
            for method in METHOD_ORDER:
                metrics, _ = simulate_episode(method, delay, episode_seed, models, cfg)
                trial_rows.append({"method": method, "delay": delay, "trial": trial, **metrics})

    summary: list[dict[str, object]] = []
    metric_names = [
        "tracking_rmse", "tail_tracking_rmse", "final_error", "success",
        "handoff_action_jump", "proprio_context_error", "latent_context_rmse",
        "handoff_policy_error", "control_effort",
    ]
    for delay in delays:
        for method in METHOD_ORDER:
            rows = [r for r in trial_rows if r["delay"] == delay and r["method"] == method]
            item: dict[str, object] = {"method": method, "delay": delay, "n_trials": len(rows)}
            for metric in metric_names:
                mean, sem = mean_sem([float(r[metric]) for r in rows])
                item[f"{metric}_mean"] = mean
                item[f"{metric}_sem"] = sem
            summary.append(item)
    return trial_rows, summary


def deterministic_sanity_check(cfg: Config) -> dict[str, object]:
    """One handoff with exact arithmetic: each ablation has a known expected failure/success."""
    local = replace(
        cfg, target_noise=0.0, cue_noise=0.0, wind_noise=0.0, robot_noise=0.0,
        actuator_gain=0.78, wind_force=0.5,
    )
    stale_state = np.array([-1.2, 0.4, 0.7, -0.2], dtype=np.float64)
    stale_latent = np.array([1.0, -0.5, 0.4, 0.25, 0.6, 0.8, -0.3, 0.2], dtype=np.float64)
    delay = 4
    actions = np.array([[1.8, -0.4], [1.4, -0.1], [0.9, 0.3], [0.4, 0.6]], dtype=np.float64)
    true_state, true_latent = stale_state.copy(), stale_latent.copy()
    for action in actions:
        true_state, true_latent = true_step(
            true_state, true_latent, action, local,
            np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2),
        )
    nominal = rollout_state(stale_state, actions, local)
    corrected_exact = nominal + (true_state - nominal)
    transported = transport_latent(stale_latent, delay, local)
    innovation_exact = transported + (true_latent - transported)
    oracle_action = policy_action(true_state, true_latent, local)
    stale_action = policy_action(stale_state, stale_latent, local)
    state_only_action = policy_action(corrected_exact, stale_latent, local)
    full_action = policy_action(corrected_exact, innovation_exact, local)
    checks = {
        "state_rollout_beats_stale_state": bool(
            np.linalg.norm(nominal - true_state) < np.linalg.norm(stale_state - true_state)
        ),
        "exact_state_correction_zero_error": bool(np.allclose(corrected_exact, true_state, atol=1e-12)),
        "transport_has_nonzero_env_error": bool(np.linalg.norm(transported - true_latent) > 1e-4),
        "innovation_zero_env_error": bool(np.allclose(innovation_exact, true_latent, atol=1e-12)),
        "full_context_matches_oracle_action": bool(np.allclose(full_action, oracle_action, atol=1e-12)),
        "state_only_still_differs_from_oracle_action": bool(
            np.linalg.norm(state_only_action - oracle_action) > 1e-4
        ),
        "stale_action_differs_from_oracle_action": bool(np.linalg.norm(stale_action - oracle_action) > 1e-4),
    }
    if not all(checks.values()):
        raise AssertionError(f"sanity check failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "errors": {
            "stale_state_l2": float(np.linalg.norm(stale_state - true_state)),
            "nominal_rollout_l2": float(np.linalg.norm(nominal - true_state)),
            "transport_latent_l2": float(np.linalg.norm(transported - true_latent)),
            "state_only_action_l2": float(np.linalg.norm(state_only_action - oracle_action)),
            "full_action_l2": float(np.linalg.norm(full_action - oracle_action)),
        },
    }


def write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_delay_sweep(summary: list[dict[str, object]], delays: list[int], out: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    specs = [
        ("success_mean", "success_sem", "Success rate", True),
        ("tracking_rmse_mean", "tracking_rmse_sem", "Tracking RMSE", False),
        ("handoff_policy_error_mean", "handoff_policy_error_sem", "Handoff policy error", False),
        ("latent_context_rmse_mean", "latent_context_rmse_sem", "Environment-latent RMSE", False),
    ]
    for ax, (mean_key, sem_key, title, percent) in zip(axes.flat, specs):
        for method in METHOD_ORDER:
            rows = [r for r in summary if r["method"] == method]
            xs = np.array([int(r["delay"]) for r in rows])
            ys = np.array([float(r[mean_key]) for r in rows])
            es = np.array([float(r[sem_key]) for r in rows])
            if percent:
                ys, es = 100 * ys, 100 * es
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, lw=1.7, capsize=2,
                        label=LABELS[method], color=COLORS[method])
        ax.set_title(title)
        ax.set_xlabel("Inference delay (control steps)")
        ax.grid(alpha=0.25)
        if percent:
            ax.set_ylim(-3, 103)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Asynchronous handoff delay sweep (mean ± SEM)", fontsize=14)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_single_rollout(
    delay: int, seed: int, models: dict[str, NormalizedRegressor], cfg: Config, out: pathlib.Path
) -> None:
    selected = ["oracle_fresh", "naive_stale", "state_correction", "obs_transport", "obs_innovation", "policy_consistency"]
    traces = {m: simulate_episode(m, delay, seed, models, cfg, keep_trace=True)[1] for m in selected}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    target = traces["oracle_fresh"]["target"]
    axes[0].plot(target[:, 0], target[:, 1], "k--", lw=2.4, label="Target")
    for method in selected:
        robot = traces[method]["robot"]
        axes[0].plot(robot[:, 0], robot[:, 1], lw=1.8, color=COLORS[method], label=LABELS[method])
    axes[0].scatter(target[0, 0], target[0, 1], c="k", marker="o", s=25)
    axes[0].set_title(f"Paired rollout paths, delay={delay}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.25)
    for method in selected:
        axes[1].plot(np.arange(cfg.episode_steps) * cfg.dt, traces[method]["error"],
                     color=COLORS[method], lw=1.7, label=LABELS[method])
    for t in range(cfg.chunk_horizon, cfg.episode_steps, cfg.chunk_horizon):
        axes[1].axvline(t * cfg.dt, color="0.8", lw=0.5)
    axes[1].axhline(cfg.success_tail_rmse, color="k", ls=":", lw=1.2, label="Success RMSE threshold")
    axes[1].set_title("Robot-target distance")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Distance")
    axes[1].grid(alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_predictor_ablation(metrics: dict[str, object], out: pathlib.Path) -> None:
    held = metrics["held_out_ablation"]
    methods = ["obs_transport", "obs_innovation", "policy_consistency"]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)
    axes[0].bar(x, [held[m]["latent_rmse"] for m in methods], color=[COLORS[m] for m in methods])
    axes[0].set_title("Held-out latent prediction")
    axes[0].set_ylabel("Latent RMSE")
    axes[1].bar(x, [held[m]["policy_action_rmse"] for m in methods], color=[COLORS[m] for m in methods])
    axes[1].set_title("Held-out policy consistency")
    axes[1].set_ylabel("First-action RMSE")
    for ax in axes:
        ax.set_xticks(x, ["Transport", "+ innovation", "+ policy loss"], rotation=12)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_delays(text: str, cfg: Config) -> list[int]:
    delays = sorted({int(x) for x in text.split(",") if x.strip()})
    if not delays or delays[0] < 0 or delays[-1] > cfg.max_delay:
        raise ValueError(f"delays must be within [0, {cfg.max_delay}]")
    return delays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--delays", default="0,2,4,6,8,10")
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--val-samples", type=int, default=2500)
    parser.add_argument("--train-steps", type=int, default=650)
    parser.add_argument("--episode-steps", type=int, default=216)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    cfg = Config(episode_steps=args.episode_steps)
    if args.smoke:
        args.train_samples = min(args.train_samples, 1200)
        args.val_samples = min(args.val_samples, 300)
        args.train_steps = min(args.train_steps, 55)
        args.trials = min(args.trials, 4)
        args.delays = "0,6"
        cfg = replace(cfg, episode_steps=min(cfg.episode_steps, 72), success_tail_steps=24)
    delays = parse_delays(args.delays, cfg)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    sanity = deterministic_sanity_check(cfg)
    train = generate_transition_dataset(args.train_samples, cfg, args.seed + 100)
    val = generate_transition_dataset(args.val_samples, cfg, args.seed + 200)
    models, training_metrics = train_models(train, val, cfg, args.seed, args.train_steps)
    trial_rows, summary = run_sweep(delays, args.trials, models, cfg, args.seed + 1000)

    write_csv(out / "delay_sweep_trials.csv", trial_rows)
    write_csv(out / "delay_sweep_summary.csv", summary)
    (out / "sanity_check.json").write_text(json.dumps(sanity, indent=2) + "\n")
    (out / "training_metrics.json").write_text(json.dumps(training_metrics, indent=2) + "\n")
    run_metadata = {
        "note": "Synthetic toy; not a reproduction of FutureRTC.",
        "seed": args.seed,
        "config": asdict(cfg),
        "delays": delays,
        "trials_per_method_delay": args.trials,
        "train_samples": args.train_samples,
        "validation_samples": args.val_samples,
        "train_steps_per_model": args.train_steps,
        "methods": METHOD_ORDER,
        "summary": summary,
    }
    (out / "metrics.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    plot_delay_sweep(summary, delays, out / "delay_sweep.png")
    plot_single_rollout(delays[-1], args.seed + 4242, models, cfg, out / "single_rollout.png")
    plot_predictor_ablation(training_metrics, out / "predictor_ablation.png")

    max_delay = delays[-1]
    print(f"Sanity check: passed ({len(sanity['checks'])} checks)")
    print(f"Wrote outputs to {out}")
    print(f"Delay={max_delay} summary:")
    for method in METHOD_ORDER:
        row = next(r for r in summary if r["delay"] == max_delay and r["method"] == method)
        print(
            f"  {method:20s} success={100*float(row['success_mean']):5.1f}% "
            f"tracking={float(row['tracking_rmse_mean']):.3f} "
            f"policy_err={float(row['handoff_policy_error_mean']):.3f}"
        )


if __name__ == "__main__":
    main()
