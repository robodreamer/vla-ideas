#!/usr/bin/env python3
"""Deterministic contact-consequence prediction toy inspired by Facet-0.

This is a low-dimensional mechanism probe, not a Facet-0 reproduction, robot
benchmark, or claim about real force safety. A visually aliased insertion toy
compares action-only behavior cloning, force-conditioned behavior cloning,
future-wrench prediction, action--wrench critic reranking, and bounded local
adaptation under jams and shifted part dynamics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = (
    "action_only_bc",
    "force_conditioned_bc",
    "future_wrench_aux",
    "critic_reranking",
    "bounded_adaptation",
)
LABELS = {
    "action_only_bc": "Action-only BC",
    "force_conditioned_bc": "Force-conditioned BC",
    "future_wrench_aux": "Future-wrench auxiliary",
    "critic_reranking": "Action–wrench critic reranking",
    "bounded_adaptation": "Bounded local adaptation",
}
COLORS = {
    "action_only_bc": "#9c755f",
    "force_conditioned_bc": "#4c78a8",
    "future_wrench_aux": "#59a14f",
    "critic_reranking": "#f28e2b",
    "bounded_adaptation": "#e15759",
}


@dataclass(frozen=True)
class Config:
    seed: int = 41
    train_episodes: int = 720
    critic_states: int = 5200
    eval_episodes: int = 320
    sweep_episodes: int = 120
    max_steps: int = 58
    candidate_count: int = 21
    contact_depth: float = 0.0
    success_depth: float = 1.0
    clearance: float = 0.045
    dx_limit: float = 0.075
    dz_push_limit: float = 0.072
    dz_retract_limit: float = 0.032
    soft_force_limit: float = 2.20
    hard_force_limit: float = 4.30
    adaptation_bound: float = 0.030
    robustness_levels: tuple[float, ...] = (0.70, 1.00, 1.35, 1.70, 2.05)


@dataclass(frozen=True)
class Part:
    stiffness: float
    jam_gain: float
    friction: float
    transmission: float
    clearance_scale: float


@dataclass
class State:
    depth: float
    lateral_error: float
    last_progress: float = 0.0
    last_dx: float = 0.0
    last_dz: float = 0.0


@dataclass
class Transition:
    visual: np.ndarray
    wrench: np.ndarray
    action: np.ndarray
    next_wrench: np.ndarray
    progress_gain: float
    next_abs_error: float
    jammed: bool
    violated: bool
    contact: bool


@dataclass
class EpisodeResult:
    trial: int
    split: str
    severity: float
    method: str
    success: bool
    damaged: bool
    intervention: bool
    jammed: bool
    recovered: bool
    steps: int
    final_depth: float
    final_abs_error: float
    peak_force: float
    peak_lateral_force: float
    force_excess: float
    contact_steps: int
    action_variation: float
    mean_wrench_prediction_error: float
    final_stiffness_estimate: float
    trace: dict[str, list[float]] | None = None


class Ridge:
    def __init__(self, alpha: float = 1.0e-3) -> None:
        self.alpha = alpha
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Ridge":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        self.mean = np.mean(x, axis=0)
        self.scale = np.std(x, axis=0)
        self.scale[self.scale < 1.0e-8] = 1.0
        z = (x - self.mean) / self.scale
        design = np.concatenate([np.ones((len(z), 1)), z], axis=1)
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None or self.weights is None:
            raise RuntimeError("model has not been fitted")
        x = np.asarray(x, dtype=float)
        one = x.ndim == 1
        if one:
            x = x[None, :]
        z = (x - self.mean) / self.scale
        design = np.concatenate([np.ones((len(z), 1)), z], axis=1)
        result = design @ self.weights
        return result[0] if one else result


@dataclass
class Models:
    action_only: Ridge
    force_conditioned: Ridge
    wrench_model: Ridge
    critic: Ridge
    nominal_stiffness: float


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def sample_part(seed: int, severity: float, split: str) -> Part:
    rng = np.random.default_rng(seed)
    if split == "train":
        stiffness = rng.uniform(0.82, 1.18)
        jam_gain = rng.uniform(0.90, 1.12)
        friction = rng.uniform(0.88, 1.12)
        transmission = rng.uniform(0.94, 1.06)
        clearance_scale = rng.uniform(0.96, 1.04)
    else:
        # Severity expands part-specific contact dynamics while preserving the task.
        direction = rng.choice([-1.0, 1.0])
        stiffness = np.clip(1.0 + direction * rng.uniform(0.18, 0.42) * severity, 0.38, 2.35)
        jam_gain = np.clip(1.0 + rng.uniform(0.10, 0.36) * severity, 0.70, 1.85)
        friction = np.clip(1.0 + rng.uniform(-0.20, 0.32) * severity, 0.55, 1.85)
        transmission = np.clip(1.0 + direction * rng.uniform(0.06, 0.18) * severity, 0.62, 1.42)
        clearance_scale = np.clip(1.0 - rng.uniform(0.02, 0.11) * severity, 0.72, 1.05)
    return Part(float(stiffness), float(jam_gain), float(friction), float(transmission), float(clearance_scale))


def initial_state(seed: int, severity: float = 1.0) -> State:
    rng = np.random.default_rng(seed)
    sign = float(rng.choice([-1.0, 1.0]))
    magnitude = rng.uniform(0.115, 0.235) * (1.0 + 0.08 * max(0.0, severity - 1.0))
    return State(depth=-0.18, lateral_error=sign * float(magnitude))


def physical_wrench(state: State, part: Part, cfg: Config) -> np.ndarray:
    if state.depth < cfg.contact_depth:
        return np.zeros(2, dtype=float)
    clearance = cfg.clearance * part.clearance_scale
    misalignment = max(abs(state.lateral_error) - clearance, 0.0)
    engagement = 0.32 + 0.68 * np.clip(state.depth / 0.12, 0.0, 1.0)
    lateral = engagement * part.stiffness * state.lateral_error / 0.16
    push_term = max(state.last_dz, 0.0) / cfg.dz_push_limit
    axial = 0.16 + part.friction * 0.18 * push_term
    axial += engagement * part.jam_gain * 1.48 * misalignment / 0.16
    if state.last_dz < 0.0:
        axial *= 0.38
    return part.transmission * np.array([axial, lateral], dtype=float)


def visual_observation(state: State, cfg: Config) -> np.ndarray:
    # Deliberately hide lateral sign and quantize magnitude near contact.
    abs_error = abs(state.lateral_error)
    quantized = round(abs_error / 0.055) * 0.055
    contact = float(state.depth >= cfg.contact_depth)
    stalled = float(contact > 0.0 and state.last_dz > 0.01 and state.last_progress < 0.008)
    return np.array(
        [
            np.clip(state.depth, -0.2, 1.0),
            quantized,
            contact,
            stalled,
            np.clip(state.last_progress / 0.07, 0.0, 1.0),
            np.clip(1.0 - state.depth, 0.0, 1.2),
        ],
        dtype=float,
    )


def action_features(visual: np.ndarray, wrench: np.ndarray | None = None) -> np.ndarray:
    z, abs_error, contact, stalled, progress, remaining = visual
    base = [
        1.0,
        z,
        z * z,
        abs_error,
        abs_error * abs_error,
        contact,
        stalled,
        progress,
        remaining,
        contact * abs_error,
        stalled * abs_error,
    ]
    if wrench is not None:
        axial, lateral = wrench
        base.extend(
            [
                axial,
                lateral,
                abs(lateral),
                axial * axial,
                lateral * abs(lateral),
                contact * axial,
                contact * lateral,
                stalled * axial,
                stalled * lateral,
            ]
        )
    return np.asarray(base, dtype=float)


def wrench_features(visual: np.ndarray, wrench: np.ndarray, action: np.ndarray) -> np.ndarray:
    dx, dz = action
    base = action_features(visual, wrench).tolist()
    axial, lateral = wrench
    base.extend(
        [
            dx,
            dz,
            abs(dx),
            max(dz, 0.0),
            min(dz, 0.0),
            dx * lateral,
            dz * axial,
            dx * dx,
            dz * dz,
            abs(dx) * abs(lateral),
        ]
    )
    return np.asarray(base, dtype=float)


def wrench_target(wrench: np.ndarray) -> np.ndarray:
    """Vector consequence plus sign-invariant contact intensity targets."""
    return np.array(
        [wrench[0], wrench[1], abs(wrench[1]), np.linalg.norm(wrench)], dtype=float
    )


def critic_features(
    visual: np.ndarray, wrench: np.ndarray, action: np.ndarray, predicted_next_wrench: np.ndarray
) -> np.ndarray:
    next_axial, next_lateral, next_abs_lateral, next_norm = predicted_next_wrench
    dx, dz = action
    base = wrench_features(visual, wrench, action).tolist()
    base.extend(
        [
            next_axial,
            next_lateral,
            next_abs_lateral,
            next_norm,
            max(next_norm - 1.25, 0.0),
            next_axial * max(dz, 0.0),
            next_lateral * dx,
        ]
    )
    return np.asarray(base, dtype=float)


def clip_action(action: np.ndarray, cfg: Config) -> np.ndarray:
    return np.array(
        [
            np.clip(action[0], -cfg.dx_limit, cfg.dx_limit),
            np.clip(action[1], -cfg.dz_retract_limit, cfg.dz_push_limit),
        ],
        dtype=float,
    )


def step(state: State, action: np.ndarray, part: Part, cfg: Config) -> tuple[State, Transition]:
    action = clip_action(action, cfg)
    visual = visual_observation(state, cfg)
    wrench = physical_wrench(state, part, cfg)
    dx, dz = action
    next_error = state.lateral_error + part.transmission * dx
    clearance = cfg.clearance * part.clearance_scale
    misalignment = max(abs(next_error) - clearance, 0.0)

    if state.depth < cfg.contact_depth:
        gain = max(dz, 0.0)
    elif dz <= 0.0:
        gain = dz
    elif misalignment <= 0.0:
        gain = dz * (0.98 / max(part.friction, 0.55))
    else:
        jam_ratio = np.clip(misalignment / 0.18, 0.0, 1.0)
        gain = dz * (1.0 - jam_ratio) ** 3 / max(part.friction, 0.55)

    next_depth = float(np.clip(state.depth + gain, -0.20, 1.08))
    next_state = State(
        depth=next_depth,
        lateral_error=float(next_error),
        last_progress=float(max(gain, 0.0)),
        last_dx=float(dx),
        last_dz=float(dz),
    )
    next_wrench = physical_wrench(next_state, part, cfg)
    force_norm = float(np.linalg.norm(next_wrench))
    jammed = bool(state.depth >= 0.015 and dz > 0.01 and gain < 0.008)
    violated = bool(force_norm > cfg.hard_force_limit)
    transition = Transition(
        visual=visual,
        wrench=wrench,
        action=action,
        next_wrench=next_wrench,
        progress_gain=float(gain),
        next_abs_error=float(abs(next_error)),
        jammed=jammed,
        violated=violated,
        contact=bool(next_depth >= cfg.contact_depth),
    )
    return next_state, transition


def expert_action(state: State, part: Part, cfg: Config) -> np.ndarray:
    wrench = physical_wrench(state, part, cfg)
    clearance = cfg.clearance * part.clearance_scale
    if state.depth < cfg.contact_depth:
        return np.array([0.0, 0.062], dtype=float)
    if abs(state.lateral_error) > clearance * 0.82:
        correction = np.clip(-0.86 * state.lateral_error / max(part.transmission, 0.5), -0.071, 0.071)
        retract = -0.020 if (wrench[0] > 0.88 or state.last_progress < 0.006) else 0.006
        return np.array([correction, retract], dtype=float)
    target_push = 0.058 / max(part.friction, 0.65)
    if wrench[0] > 1.20:
        target_push *= 0.42
    return np.array([np.clip(-0.25 * state.lateral_error, -0.018, 0.018), target_push], dtype=float)


def collect_demonstrations(cfg: Config) -> list[Transition]:
    rows: list[Transition] = []
    for episode in range(cfg.train_episodes):
        part = sample_part(stable_seed(cfg.seed, "train-part", episode), 1.0, "train")
        state = initial_state(stable_seed(cfg.seed, "train-state", episode), 0.6)
        for _ in range(cfg.max_steps):
            action = expert_action(state, part, cfg)
            rng = np.random.default_rng(stable_seed(cfg.seed, "demo-noise", episode, len(rows)))
            action = action + rng.normal(0.0, [0.0035, 0.0025])
            state, transition = step(state, action, part, cfg)
            rows.append(transition)
            rows.append(
                Transition(
                    visual=transition.visual.copy(),
                    wrench=transition.wrench * np.array([1.0, -1.0]),
                    action=transition.action * np.array([-1.0, 1.0]),
                    next_wrench=transition.next_wrench * np.array([1.0, -1.0]),
                    progress_gain=transition.progress_gain,
                    next_abs_error=transition.next_abs_error,
                    jammed=transition.jammed,
                    violated=transition.violated,
                    contact=transition.contact,
                )
            )
            if state.depth >= cfg.success_depth or transition.violated:
                break
    return rows


def one_step_utility(transition: Transition, cfg: Config) -> float:
    force = float(np.linalg.norm(transition.next_wrench))
    progress = max(transition.progress_gain, 0.0)
    alignment_bonus = max(0.0, cfg.clearance * 1.4 - transition.next_abs_error)
    score = 15.0 * progress + 2.5 * alignment_bonus
    score -= 0.72 * max(force - 0.75, 0.0) ** 2
    score -= 1.25 * float(transition.jammed)
    score -= 4.0 * float(transition.violated)
    score -= 0.12 * abs(float(transition.action[0]))
    return float(score)


def collect_critic_data(cfg: Config) -> list[Transition]:
    rows: list[Transition] = []
    for index in range(cfg.critic_states):
        severity = float(np.random.default_rng(stable_seed(cfg.seed, "critic-sev", index)).uniform(0.45, 1.45))
        part = sample_part(stable_seed(cfg.seed, "critic-part", index), severity, "eval")
        state = initial_state(stable_seed(cfg.seed, "critic-state", index), severity)
        rng = np.random.default_rng(stable_seed(cfg.seed, "critic-roll", index))
        warmup = int(rng.integers(2, 13))
        for _ in range(warmup):
            nominal = expert_action(state, part, cfg)
            noisy = nominal + rng.normal(0.0, [0.034, 0.028])
            state, _ = step(state, noisy, part, cfg)
        candidate = np.array([rng.uniform(-cfg.dx_limit, cfg.dx_limit), rng.uniform(-0.025, cfg.dz_push_limit)])
        _, transition = step(state, candidate, part, cfg)
        rows.append(transition)
        rows.append(
            Transition(
                visual=transition.visual.copy(),
                wrench=transition.wrench * np.array([1.0, -1.0]),
                action=transition.action * np.array([-1.0, 1.0]),
                next_wrench=transition.next_wrench * np.array([1.0, -1.0]),
                progress_gain=transition.progress_gain,
                next_abs_error=transition.next_abs_error,
                jammed=transition.jammed,
                violated=transition.violated,
                contact=transition.contact,
            )
        )
    return rows


def fit_models(cfg: Config) -> tuple[Models, dict[str, float]]:
    demonstrations = collect_demonstrations(cfg)
    x_visual = np.stack([action_features(row.visual) for row in demonstrations])
    x_force = np.stack([action_features(row.visual, row.wrench) for row in demonstrations])
    actions = np.stack([row.action for row in demonstrations])
    wrench_x = np.stack([wrench_features(row.visual, row.wrench, row.action) for row in demonstrations])
    next_wrench = np.stack([wrench_target(row.next_wrench) for row in demonstrations])

    action_only = Ridge(2.0e-2).fit(x_visual, actions)
    force_conditioned = Ridge(1.2e-2).fit(x_force, actions)
    wrench_model = Ridge(2.0e-2).fit(wrench_x, next_wrench)

    critic_rows = collect_critic_data(cfg)
    critic_x = []
    critic_y = []
    for row in critic_rows:
        predicted = sanitize_wrench_prediction(
            wrench_model.predict(wrench_features(row.visual, row.wrench, row.action))
        )
        critic_x.append(critic_features(row.visual, row.wrench, row.action, predicted))
        critic_y.append(one_step_utility(row, cfg))
    critic = Ridge(5.0e-2).fit(np.stack(critic_x), np.asarray(critic_y))

    nominal_stiffness = float(np.median([sample_part(stable_seed(cfg.seed, "nominal", i), 1.0, "train").stiffness for i in range(200)]))
    holdout = demonstrations[:: max(1, len(demonstrations) // 1200)]
    wrench_mae = float(
        np.mean(
            [
                np.mean(
                    np.abs(
                        sanitize_wrench_prediction(
                            wrench_model.predict(wrench_features(row.visual, row.wrench, row.action))
                        )
                        - wrench_target(row.next_wrench)
                    )
                )
                for row in holdout
            ]
        )
    )
    diagnostics = {
        "demonstration_transitions": float(len(demonstrations)),
        "critic_transitions": float(len(critic_rows)),
        "demonstration_wrench_mae": wrench_mae,
        "nominal_stiffness": nominal_stiffness,
    }
    return Models(action_only, force_conditioned, wrench_model, critic, nominal_stiffness), diagnostics


def sanitize_wrench_prediction(prediction: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=float).copy()
    prediction[0] = max(0.0, prediction[0])
    prediction[2] = max(abs(prediction[1]), prediction[2], 0.0)
    prediction[3] = max(float(np.linalg.norm(prediction[:2])), prediction[2], prediction[3], 0.0)
    return prediction


def predict_wrench(models: Models, visual: np.ndarray, wrench: np.ndarray, action: np.ndarray) -> np.ndarray:
    return sanitize_wrench_prediction(
        models.wrench_model.predict(wrench_features(visual, wrench, action))
    )


def aux_governed_action(models: Models, state: State, part: Part, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    visual = visual_observation(state, cfg)
    wrench = physical_wrench(state, part, cfg)
    action = clip_action(models.force_conditioned.predict(action_features(visual, wrench)), cfg)
    predicted = predict_wrench(models, visual, wrench, action)
    predicted_norm = float(predicted[3])
    contact_risk = max(predicted_norm / 1.05, predicted[2] / 0.58)
    if action[1] > 0.0 and contact_risk > 0.62:
        risk = np.clip((contact_risk - 0.62) / 0.58, 0.0, 1.0)
        action[1] = (1.0 - risk) * action[1] - risk * 0.014
        if state.depth < cfg.contact_depth:
            # Make a shallow contact probe so the next measured wrench can
            # disambiguate lateral sign before a deep push.
            action[1] = min(action[1], max(0.012, -state.depth + 0.008))
    if predicted[2] > 1.0:
        action[0] *= 1.10
    action = clip_action(action, cfg)
    return action, predict_wrench(models, visual, wrench, action)


def candidate_actions(base: np.ndarray, cfg: Config, *, allow_lateral: bool = True) -> list[np.ndarray]:
    lateral_offsets = (
        np.array([-0.030, -0.020, -0.010, 0.0, 0.010, 0.020, 0.030])
        if allow_lateral
        else np.array([0.0])
    )
    axial_offsets = np.array([-0.018, 0.0, 0.016])
    candidates = [clip_action(base + np.array([dx, dz]), cfg) for dx in lateral_offsets for dz in axial_offsets]
    return candidates[: cfg.candidate_count]


def critic_action(models: Models, state: State, part: Part, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    base, _ = aux_governed_action(models, state, part, cfg)
    visual = visual_observation(state, cfg)
    wrench = physical_wrench(state, part, cfg)
    if state.depth < cfg.contact_depth:
        predicted = predict_wrench(models, visual, wrench, base)
        return base, predicted
    best_action = base
    best_predicted = predict_wrench(models, visual, wrench, base)
    best_score = float(
        models.critic.predict(critic_features(visual, wrench, base, best_predicted))[0]
    )
    # Keep the reranker in a conservative axial trust region; local lateral
    # remapping is reserved for the explicit adaptation stage below.
    for action in candidate_actions(base, cfg, allow_lateral=False):
        predicted = predict_wrench(models, visual, wrench, action)
        score = float(models.critic.predict(critic_features(visual, wrench, action, predicted))[0])
        # A tiny model-independent safety tie-break prevents extrapolative force chasing.
        score -= 0.08 * max(float(predicted[3]) - cfg.soft_force_limit, 0.0) ** 2
        score -= 0.70 * abs(float(action[0] - base[0]))
        score -= 0.35 * abs(float(action[1] - base[1]))
        if score > best_score + 0.035:
            best_score = score
            best_action = action
            best_predicted = predicted
    return best_action.copy(), best_predicted.copy()


class AdaptationState:
    def __init__(self, nominal_stiffness: float) -> None:
        self.stiffness = nominal_stiffness
        self.previous_lateral_force: float | None = None
        self.previous_dx: float = 0.0
        self.risk_bias: float = 0.0
        self.previous_prediction: np.ndarray | None = None

    def observe(self, wrench: np.ndarray) -> None:
        if self.previous_lateral_force is not None and abs(self.previous_dx) > 0.008:
            estimate = abs((float(wrench[1]) - self.previous_lateral_force) / self.previous_dx) * 0.16
            if 0.25 < estimate < 3.4:
                self.stiffness = 0.72 * self.stiffness + 0.28 * estimate
        if self.previous_prediction is not None:
            residual = float(np.linalg.norm(wrench) - self.previous_prediction[3])
            self.risk_bias = 0.78 * self.risk_bias + 0.22 * residual

    def record(self, wrench: np.ndarray, action: np.ndarray, predicted: np.ndarray) -> None:
        self.previous_lateral_force = float(wrench[1])
        self.previous_dx = float(action[0])
        self.previous_prediction = predicted.copy()

def policy_action(
    method: str,
    models: Models,
    state: State,
    part: Part,
    cfg: Config,
    adaptation: AdaptationState,
) -> tuple[np.ndarray, np.ndarray]:
    visual = visual_observation(state, cfg)
    wrench = physical_wrench(state, part, cfg)
    adaptation.observe(wrench)

    if method == "action_only_bc":
        action = clip_action(models.action_only.predict(action_features(visual)), cfg)
        predicted = predict_wrench(models, visual, wrench, action)
    elif method == "force_conditioned_bc":
        action = clip_action(models.force_conditioned.predict(action_features(visual, wrench)), cfg)
        predicted = predict_wrench(models, visual, wrench, action)
    elif method == "future_wrench_aux":
        action, predicted = aux_governed_action(models, state, part, cfg)
    else:
        action, predicted = critic_action(models, state, part, cfg)
        if method == "bounded_adaptation" and state.depth >= cfg.contact_depth:
            # Estimate error from measured lateral force using the locally updated
            # stiffness, then make a bounded correction around the frozen proposal.
            estimated_error = float(wrench[1]) * 0.16 / max(adaptation.stiffness * part.transmission, 0.25)
            desired_dx = np.clip(-0.84 * estimated_error / max(part.transmission, 0.55), -cfg.dx_limit, cfg.dx_limit)
            residual = np.clip(desired_dx - action[0], -cfg.adaptation_bound, cfg.adaptation_bound)
            action[0] += residual
            if adaptation.risk_bias > 0.10 and action[1] > 0.0:
                action[1] -= min(0.026, 0.018 * adaptation.risk_bias)
            action = clip_action(action, cfg)
            predicted = predict_wrench(models, visual, wrench, action)
    adaptation.record(wrench, action, predicted)
    return action, predicted


def run_episode(
    method: str,
    models: Models,
    cfg: Config,
    trial: int,
    split: str,
    severity: float,
    capture: bool = False,
) -> EpisodeResult:
    part = sample_part(stable_seed(cfg.seed, split, severity, "part", trial), severity, "eval")
    state = initial_state(stable_seed(cfg.seed, split, severity, "state", trial), severity)
    adaptation = AdaptationState(models.nominal_stiffness)
    peak_force = 0.0
    peak_lateral = 0.0
    force_excess = 0.0
    contact_steps = 0
    any_jam = False
    damaged = False
    intervention = False
    actions: list[np.ndarray] = []
    wrench_errors: list[float] = []
    trace = {key: [] for key in ("depth", "error", "axial_force", "lateral_force", "dx", "dz", "predicted_force")} if capture else None

    for step_index in range(1, cfg.max_steps + 1):
        current_wrench = physical_wrench(state, part, cfg)
        action, predicted = policy_action(method, models, state, part, cfg, adaptation)
        next_state, transition = step(state, action, part, cfg)
        force_norm = float(np.linalg.norm(transition.next_wrench))
        peak_force = max(peak_force, force_norm)
        peak_lateral = max(peak_lateral, abs(float(transition.next_wrench[1])))
        force_excess += max(force_norm - cfg.soft_force_limit, 0.0)
        contact_steps += int(transition.contact)
        any_jam = any_jam or transition.jammed
        intervention = intervention or force_norm > cfg.soft_force_limit
        damaged = damaged or transition.violated
        actions.append(action.copy())
        wrench_errors.append(float(np.mean(np.abs(predicted - wrench_target(transition.next_wrench)))))
        if trace is not None:
            trace["depth"].append(float(next_state.depth))
            trace["error"].append(float(next_state.lateral_error))
            trace["axial_force"].append(float(transition.next_wrench[0]))
            trace["lateral_force"].append(float(transition.next_wrench[1]))
            trace["dx"].append(float(action[0]))
            trace["dz"].append(float(action[1]))
            trace["predicted_force"].append(float(predicted[3]))
        state = next_state
        if damaged or state.depth >= cfg.success_depth:
            break

    success = bool(state.depth >= cfg.success_depth and not damaged)
    if len(actions) > 1:
        action_variation = float(np.mean(np.linalg.norm(np.diff(np.stack(actions), axis=0), axis=1)))
    else:
        action_variation = 0.0
    return EpisodeResult(
        trial=trial,
        split=split,
        severity=severity,
        method=method,
        success=success,
        damaged=damaged,
        intervention=intervention,
        jammed=any_jam,
        recovered=bool(any_jam and success),
        steps=step_index,
        final_depth=float(state.depth),
        final_abs_error=float(abs(state.lateral_error)),
        peak_force=peak_force,
        peak_lateral_force=peak_lateral,
        force_excess=force_excess,
        contact_steps=contact_steps,
        action_variation=action_variation,
        mean_wrench_prediction_error=float(np.mean(wrench_errors)),
        final_stiffness_estimate=float(adaptation.stiffness),
        trace=trace,
    )


def evaluate(
    models: Models,
    cfg: Config,
    episodes: int,
    split: str,
    severity: float,
    capture_trial: int | None = None,
) -> list[EpisodeResult]:
    rows: list[EpisodeResult] = []
    for trial in range(episodes):
        for method in METHODS:
            rows.append(
                run_episode(
                    method,
                    models,
                    cfg,
                    trial,
                    split,
                    severity,
                    capture=(capture_trial is not None and trial == capture_trial),
                )
            )
    return rows


def summarize(results: Sequence[EpisodeResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = sorted({(result.split, result.severity, result.method) for result in results})
    for split, severity, method in groups:
        selected = [row for row in results if row.split == split and row.severity == severity and row.method == method]
        jammed = [row for row in selected if row.jammed]
        rows.append(
            {
                "split": split,
                "severity": severity,
                "method": method,
                "episodes": len(selected),
                "success_rate": float(np.mean([row.success for row in selected])),
                "damage_rate": float(np.mean([row.damaged for row in selected])),
                "intervention_rate": float(np.mean([row.intervention for row in selected])),
                "jam_rate": float(np.mean([row.jammed for row in selected])),
                "recovery_rate_given_jam": float(np.mean([row.recovered for row in jammed])) if jammed else 0.0,
                "mean_steps": float(np.mean([row.steps for row in selected])),
                "mean_peak_force": float(np.mean([row.peak_force for row in selected])),
                "p95_peak_force": float(np.quantile([row.peak_force for row in selected], 0.95)),
                "mean_final_abs_error": float(np.mean([row.final_abs_error for row in selected])),
                "mean_action_variation": float(np.mean([row.action_variation for row in selected])),
                "mean_wrench_prediction_error": float(np.mean([row.mean_wrench_prediction_error for row in selected])),
            }
        )
    return rows


def run_contact_role_ablation(models: Models, cfg: Config, episodes: int) -> list[dict[str, Any]]:
    variants = ("full_action_wrench_critic", "action_only_score", "no_candidate_reranking")
    rows: list[dict[str, Any]] = []
    severity = 1.70
    for name in variants:
        outcomes = []
        for trial in range(episodes):
            part = sample_part(stable_seed(cfg.seed, "ablation", severity, "part", trial), severity, "eval")
            state = initial_state(stable_seed(cfg.seed, "ablation", severity, "state", trial), severity)
            peak = 0.0
            damaged = False
            jammed = False
            for _ in range(cfg.max_steps):
                base, _ = aux_governed_action(models, state, part, cfg)
                visual = visual_observation(state, cfg)
                wrench = physical_wrench(state, part, cfg)
                if name == "full_action_wrench_critic":
                    best_action, _ = critic_action(models, state, part, cfg)
                elif name == "no_candidate_reranking":
                    best_action = base
                else:
                    # Same axial candidate set, but score only commanded progress
                    # and action magnitude; anticipated wrench is withheld.
                    candidates = candidate_actions(base, cfg, allow_lateral=False)
                    best_action = max(
                        candidates,
                        key=lambda action: 1.8 * float(action[1]) - 0.08 * abs(float(action[0])),
                    )
                state, transition = step(state, best_action, part, cfg)
                peak = max(peak, float(np.linalg.norm(transition.next_wrench)))
                damaged = damaged or transition.violated
                jammed = jammed or transition.jammed
                if damaged or state.depth >= cfg.success_depth:
                    break
            outcomes.append((state.depth >= cfg.success_depth and not damaged, damaged, jammed, peak))
        rows.append(
            {
                "setting": name,
                "episodes": episodes,
                "success_rate": float(np.mean([row[0] for row in outcomes])),
                "damage_rate": float(np.mean([row[1] for row in outcomes])),
                "jam_rate": float(np.mean([row[2] for row in outcomes])),
                "mean_peak_force": float(np.mean([row[3] for row in outcomes])),
            }
        )
    return rows


def sanity_checks(models: Models, cfg: Config) -> dict[str, Any]:
    part = Part(1.0, 1.0, 1.0, 1.0, 1.0)
    plus = State(depth=0.06, lateral_error=0.17, last_progress=0.0, last_dz=0.05)
    minus = State(depth=0.06, lateral_error=-0.17, last_progress=0.0, last_dz=0.05)
    v_plus = visual_observation(plus, cfg)
    v_minus = visual_observation(minus, cfg)
    w_plus = physical_wrench(plus, part, cfg)
    w_minus = physical_wrench(minus, part, cfg)
    action_only_plus = clip_action(models.action_only.predict(action_features(v_plus)), cfg)
    action_only_minus = clip_action(models.action_only.predict(action_features(v_minus)), cfg)
    force_plus = clip_action(models.force_conditioned.predict(action_features(v_plus, w_plus)), cfg)
    force_minus = clip_action(models.force_conditioned.predict(action_features(v_minus, w_minus)), cfg)
    base = run_episode("critic_reranking", models, cfg, 7, "sanity", 1.55)
    adapted = run_episode("bounded_adaptation", models, cfg, 7, "sanity", 1.55)
    repeated = run_episode("bounded_adaptation", models, cfg, 7, "sanity", 1.55)
    checks = {
        "visual_alias_is_exact": bool(np.array_equal(v_plus, v_minus)),
        "force_reveals_opposite_lateral_sign": bool(w_plus[1] > 0.0 and w_minus[1] < 0.0),
        "action_only_cannot_change_with_hidden_sign": bool(np.allclose(action_only_plus, action_only_minus, atol=1.0e-12)),
        "force_conditioned_correction_changes_sign": bool(force_plus[0] < -0.01 and force_minus[0] > 0.01),
        "bounded_residual_limit_is_positive_and_below_action_limit": bool(0.0 < cfg.adaptation_bound < cfg.dx_limit),
        "deterministic_episode_repeat": bool(asdict(adapted) == asdict(repeated)),
        "adaptation_not_more_damaging_in_probe": bool((not adapted.damaged) or base.damaged),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "probe": {
            "action_only_plus": action_only_plus.tolist(),
            "action_only_minus": action_only_minus.tolist(),
            "force_plus": force_plus.tolist(),
            "force_minus": force_minus.tolist(),
            "critic_success": base.success,
            "adapted_success": adapted.success,
        },
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def episode_rows(results: Sequence[EpisodeResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        row = asdict(result)
        row.pop("trace")
        rows.append(row)
    return rows


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def plot_headline(summary: Sequence[dict[str, Any]], output: Path) -> None:
    rows = [row for row in summary if row["split"] == "main"]
    rows.sort(key=lambda row: METHODS.index(str(row["method"])))
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))
    colors = [COLORS[str(row["method"])] for row in rows]
    axes[0].bar(x, [row["success_rate"] for row in rows], color=colors)
    axes[0].set_title("Completion under aliased contact")
    axes[0].set_ylabel("Rate")
    axes[1].bar(x, [row["recovery_rate_given_jam"] for row in rows], color=colors)
    axes[1].set_title("Recovery after a jam")
    axes[2].bar(x, [row["mean_peak_force"] for row in rows], color=colors)
    axes[2].axhline(2.20, color="black", linestyle="--", linewidth=1, label="soft limit")
    axes[2].set_title("Mean peak wrench norm")
    for axis in axes[:2]:
        axis.set_ylim(0.0, 1.02)
    for axis in axes:
        axis.set_xticks(x, [LABELS[str(row["method"])] for row in rows], rotation=24, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.24)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def plot_robustness(rows: Sequence[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for method in METHODS:
        selected = sorted((row for row in rows if row["method"] == method), key=lambda row: row["severity"])
        severity = [row["severity"] for row in selected]
        axes[0].plot(severity, [row["success_rate"] for row in selected], marker="o", color=COLORS[method], label=LABELS[method])
        axes[1].plot(severity, [row["mean_peak_force"] for row in selected], marker="o", color=COLORS[method], label=LABELS[method])
    axes[0].set_title("Success under part-dynamics shift")
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(0.0, 1.02)
    axes[1].set_title("Contact load under part-dynamics shift")
    axes[1].set_ylabel("Mean peak wrench norm")
    axes[1].axhline(2.20, color="black", linestyle="--", linewidth=1)
    for axis in axes:
        axis.set_xlabel("Shift severity")
        axis.grid(alpha=0.24)
    axes[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def plot_trajectories(results: Sequence[EpisodeResult], output: Path) -> None:
    selected = {row.method: row for row in results if row.trace is not None}
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 7.5), sharex=True)
    for method in METHODS:
        row = selected[method]
        steps = np.arange(1, len(row.trace["depth"]) + 1)
        axes[0].plot(steps, row.trace["depth"], color=COLORS[method], label=LABELS[method])
        axes[1].plot(steps, row.trace["lateral_force"], color=COLORS[method], label=LABELS[method])
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Insertion depth")
    axes[0].set_title("Paired visually aliased hard-shift episode")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Lateral wrench")
    axes[1].set_xlabel("Control step")
    for axis in axes:
        axis.grid(alpha=0.24)
    axes[0].legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def plot_prediction_calibration(models: Models, cfg: Config, output: Path) -> None:
    rng = np.random.default_rng(stable_seed(cfg.seed, "calibration-plot"))
    actual = []
    predicted = []
    for index in range(700):
        part = sample_part(stable_seed(cfg.seed, "calibration-part", index), 1.0, "train")
        state = State(
            depth=float(rng.uniform(-0.05, 0.45)),
            lateral_error=float(rng.uniform(-0.24, 0.24)),
            last_progress=float(rng.uniform(0.0, 0.06)),
            last_dz=float(rng.uniform(-0.02, 0.07)),
        )
        visual = visual_observation(state, cfg)
        wrench = physical_wrench(state, part, cfg)
        action = np.array([rng.uniform(-0.07, 0.07), rng.uniform(-0.02, 0.07)])
        _, transition = step(state, action, part, cfg)
        prediction = predict_wrench(models, visual, wrench, action)
        actual.append(float(np.linalg.norm(transition.next_wrench)))
        predicted.append(float(prediction[3]))
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.scatter(actual, predicted, s=12, alpha=0.35, color="#4c78a8")
    limit = max(max(actual), max(predicted)) * 1.03
    ax.plot([0.0, limit], [0.0, limit], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0.0, limit)
    ax.set_ylim(0.0, limit)
    ax.set_xlabel("Realized next wrench norm")
    ax.set_ylabel("Predicted next wrench norm")
    ax.set_title("Auxiliary consequence prediction (nominal dynamics)")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sanity", "quick", "full"), default="full")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-episodes", type=int, default=None)
    return parser.parse_args()


def config_for_mode(args: argparse.Namespace) -> Config:
    cfg = Config(seed=args.seed)
    if args.mode == "sanity":
        cfg = replace(cfg, train_episodes=90, critic_states=500, eval_episodes=8, sweep_episodes=8, max_steps=38, robustness_levels=(1.0, 1.7))
    elif args.mode == "quick":
        cfg = replace(cfg, train_episodes=260, critic_states=1800, eval_episodes=80, sweep_episodes=40, max_steps=48, robustness_levels=(0.7, 1.35, 2.05))
    if args.eval_episodes is not None:
        cfg = replace(cfg, eval_episodes=args.eval_episodes)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = config_for_mode(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    models, training_diagnostics = fit_models(cfg)
    sanity = sanity_checks(models, cfg)
    if not sanity["passed"]:
        raise RuntimeError(f"sanity checks failed: {sanity}")

    if args.mode == "sanity":
        results = evaluate(models, cfg, cfg.eval_episodes, "sanity", 1.35, capture_trial=0)
        robustness_results = results
        ablations: list[dict[str, Any]] = []
    else:
        results = evaluate(models, cfg, cfg.eval_episodes, "main", 1.35, capture_trial=0)
        robustness_results = []
        for severity in cfg.robustness_levels:
            robustness_results.extend(evaluate(models, cfg, cfg.sweep_episodes, "robustness", severity))
        ablations = run_contact_role_ablation(models, cfg, max(32, cfg.sweep_episodes))

    summary = summarize(results)
    robustness_summary = summarize(robustness_results)
    runtime_seconds = time.perf_counter() - start
    lookup = {row["method"]: row for row in summary}
    claims = {
        "force_conditioning_gain_over_action_only": lookup["force_conditioned_bc"]["success_rate"] - lookup["action_only_bc"]["success_rate"],
        "future_wrench_gain_over_force_conditioning": lookup["future_wrench_aux"]["success_rate"] - lookup["force_conditioned_bc"]["success_rate"],
        "critic_gain_over_auxiliary": lookup["critic_reranking"]["success_rate"] - lookup["future_wrench_aux"]["success_rate"],
        "bounded_adaptation_gain_over_critic": lookup["bounded_adaptation"]["success_rate"] - lookup["critic_reranking"]["success_rate"],
        "bounded_adaptation_peak_force_change_vs_critic": lookup["bounded_adaptation"]["mean_peak_force"] - lookup["critic_reranking"]["mean_peak_force"],
        "scope": "Synthetic 2-D insertion mechanism probe; not a Facet-0 reproduction, robot result, or certified force-safety result.",
    }

    write_csv(output_dir / "trial_metrics.csv", episode_rows(results))
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "robustness.csv", robustness_summary)
    if ablations:
        write_csv(output_dir / "contact_role_ablation.csv", ablations)
    (output_dir / "sanity_checks.json").write_text(json.dumps(json_safe(sanity), indent=2, sort_keys=True) + "\n")
    metrics = {
        "config": asdict(cfg) | {"mode": args.mode, "output_dir": str(output_dir)},
        "method_definitions": {
            "action_only_bc": "Ridge BC from sign-aliased visual state only.",
            "force_conditioned_bc": "Same BC family with current axial/lateral wrench inputs.",
            "future_wrench_aux": "Force-conditioned BC plus a learned next-wrench predictor used by an anticipatory axial governor.",
            "critic_reranking": "Deployment-trained scalar critic reranks bounded action candidates using their predicted next wrench.",
            "bounded_adaptation": "Critic policy plus a causally updated local stiffness estimate and bounded residual correction.",
        },
        "training_diagnostics": training_diagnostics,
        "sanity": sanity,
        "summary": summary,
        "robustness": robustness_summary,
        "contact_role_ablation": ablations,
        "claims_supported_by_this_run": claims,
        "runtime_seconds": runtime_seconds,
    }
    (output_dir / "metrics.json").write_text(json.dumps(json_safe(metrics), indent=2, sort_keys=True) + "\n")

    plot_headline(summary, output_dir / "headline_metrics.png")
    if robustness_summary:
        plot_robustness(robustness_summary, output_dir / "robustness.png")
    plot_trajectories(results, output_dir / "paired_trajectories.png")
    plot_prediction_calibration(models, cfg, output_dir / "wrench_prediction.png")

    print(json.dumps(json_safe({"mode": args.mode, "output_dir": str(output_dir), "sanity": sanity, "claims": claims, "summary": summary, "runtime_seconds": runtime_seconds}), indent=2))


if __name__ == "__main__":
    main()
