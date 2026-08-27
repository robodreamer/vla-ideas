#!/usr/bin/env python3
"""PR-MPPI-inspired constraint-manifold action-filter toy.

This is an explanatory chunked-policy experiment, not a PR-MPPI reproduction.
A behavior-cloned policy proposes finite action chunks for a rigid tray moving
through circular obstacles. The execution layer compares soft penalties,
one-step correction, rollout-time equality/inequality projection, and
projection followed by equality-manifold retraction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vla_ideas_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

METHODS = [
    "penalty_only",
    "one_step_correction",
    "rollout_projection",
    "projection_retraction",
]
LABELS = {
    "penalty_only": "Penalty-only proposals",
    "one_step_correction": "One-step correction",
    "rollout_projection": "Rollout projection",
    "projection_retraction": "Projection + retraction",
}
COLORS = {
    "penalty_only": "#e45756",
    "one_step_correction": "#f2cf5b",
    "rollout_projection": "#4c78a8",
    "projection_retraction": "#54a24b",
}


@dataclass(frozen=True)
class Config:
    seed: int = 23
    trials: int = 120
    train_samples: int = 7000
    test_samples: int = 1400
    dt: float = 0.08
    chunk_horizon: int = 7
    max_steps: int = 126
    goal_x: float = 6.0
    goal_tolerance: float = 0.30
    tray_half_length: float = 0.38
    tray_radius: float = 0.09
    tray_constraint_samples: int = 9
    tray_eval_samples: int = 31
    safety_buffer: float = 0.055
    workspace_xmin: float = -0.25
    workspace_xmax: float = 6.35
    workspace_ymin: float = -1.45
    workspace_ymax: float = 1.45
    gamma: float = 5.0
    max_vxy: float = 1.05
    max_omega: float = 1.8
    penalty_eq_gain: float = 3.0
    penalty_ineq_gain: float = 5.0
    penalty_band: float = 0.20
    penalty_correction_cap: float = 0.62
    retract_tol: float = 1e-10
    retract_iters: int = 12
    deadlock_window: int = 24
    deadlock_progress: float = 0.055


@dataclass(frozen=True)
class Scenario:
    trial: int
    regime: str
    centers: Tuple[Tuple[float, float], ...]
    radii: Tuple[float, ...]
    action_gain: float
    shortcut_gain: float
    radial_bias: float
    proposal_noise: float
    seed: int


@dataclass
class ProjectionInfo:
    feasible: bool
    active_count: int
    max_halfspace_residual: float


@dataclass
class TrialResult:
    trial: int
    regime: str
    method: str
    success: bool
    reached_goal: bool
    collision: bool
    deadlock: bool
    fallback: bool
    fallback_steps: int
    steps: int
    final_progress: float
    mean_progress_speed: float
    max_equality_residual: float
    mean_equality_residual: float
    max_inequality_violation: float
    min_margin: float
    mean_intervention: float
    max_intervention: float
    action_smoothness: float
    retraction_failures: int
    trajectory: Optional[np.ndarray] = None
    equality_trace: Optional[np.ndarray] = None
    margin_trace: Optional[np.ndarray] = None
    intervention_trace: Optional[np.ndarray] = None


def angle_wrap(x: float) -> float:
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


def equality(q: np.ndarray) -> np.ndarray:
    """Rigid orientation-vector equality: c(q)=a^2+b^2-1=0."""
    return np.array([q[2] * q[2] + q[3] * q[3] - 1.0])


def equality_jacobian(q: np.ndarray) -> np.ndarray:
    return np.array([[0.0, 0.0, 2.0 * q[2], 2.0 * q[3]]])


def tangent_basis(q: np.ndarray) -> Optional[np.ndarray]:
    """Orthonormal basis for the three-dimensional equality tangent space."""
    orient = q[2:4]
    n = float(np.linalg.norm(orient))
    if n < 1e-9:
        return None
    a, b = orient / n
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -b],
            [0.0, 0.0, a],
        ]
    )


def tray_points(q: np.ndarray, cfg: Config, n: int) -> np.ndarray:
    s = np.linspace(-cfg.tray_half_length, cfg.tray_half_length, n)
    return q[None, :2] + s[:, None] * q[None, 2:4]


def inequality_values_gradients(
    q: np.ndarray, sc: Scenario, cfg: Config, *, shifted: bool
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return differentiable h(q)>=0 values and gradients.

    Obstacle constraints use smooth Euclidean clearance at sampled points on
    the tray. Workspace constraints act on the tray center. The shifted form
    subtracts the safety buffer used by the CBF half-space filter.
    """
    values: List[float] = []
    grads: List[np.ndarray] = []
    names: List[str] = []
    s_values = np.linspace(-cfg.tray_half_length, cfg.tray_half_length, cfg.tray_constraint_samples)
    eps = 1e-12
    for obs_idx, (center, radius) in enumerate(zip(sc.centers, sc.radii)):
        o = np.asarray(center, dtype=float)
        required = radius + cfg.tray_radius
        for sample_idx, s in enumerate(s_values):
            p = q[:2] + s * q[2:4]
            d = p - o
            dist = float(np.sqrt(np.dot(d, d) + eps))
            h = dist - required
            grad_p = d / dist
            grad = np.array([grad_p[0], grad_p[1], s * grad_p[0], s * grad_p[1]])
            values.append(h)
            grads.append(grad)
            names.append(f"obs{obs_idx}_sample{sample_idx}")

    workspace = [
        (q[0] - cfg.workspace_xmin, np.array([1.0, 0.0, 0.0, 0.0]), "xmin"),
        (cfg.workspace_xmax - q[0], np.array([-1.0, 0.0, 0.0, 0.0]), "xmax"),
        (q[1] - cfg.workspace_ymin, np.array([0.0, 1.0, 0.0, 0.0]), "ymin"),
        (cfg.workspace_ymax - q[1], np.array([0.0, -1.0, 0.0, 0.0]), "ymax"),
    ]
    for h, grad, name in workspace:
        values.append(float(h))
        grads.append(grad)
        names.append(name)

    h_arr = np.asarray(values, dtype=float)
    if shifted:
        h_arr = h_arr - cfg.safety_buffer
    return h_arr, np.vstack(grads), names


def true_min_margin(q: np.ndarray, sc: Scenario, cfg: Config) -> float:
    """Dense evaluation margin; independent of the filter's sparse samples."""
    points = tray_points(q, cfg, cfg.tray_eval_samples)
    margins: List[float] = []
    for center, radius in zip(sc.centers, sc.radii):
        distances = np.linalg.norm(points - np.asarray(center)[None, :], axis=1)
        margins.append(float(np.min(distances - (radius + cfg.tray_radius))))
    margins.extend(
        [
            q[0] - cfg.workspace_xmin,
            cfg.workspace_xmax - q[0],
            q[1] - cfg.workspace_ymin,
            cfg.workspace_ymax - q[1],
        ]
    )
    return float(min(margins))


def reference_geometry(x: float, sc: Scenario) -> Tuple[float, float, float]:
    """Synthetic expert corridor path and first/second x derivatives."""
    y = 0.0
    dy = 0.0
    ddy = 0.0
    sigma = 0.62
    for idx, ((ox, oy), radius) in enumerate(zip(sc.centers, sc.radii)):
        direction = -1.0 if oy >= 0.0 else 1.0
        # Near-center obstacles use an alternating convention learned only weakly
        # from the training distribution; these are deliberately OOD cases.
        if abs(oy) < 0.06:
            direction = -1.0 if idx % 2 == 0 else 1.0
        amp = direction * (0.63 + 0.72 * radius)
        z = (x - ox) / sigma
        g = math.exp(-0.5 * z * z)
        y += amp * g
        dy += amp * g * (-z / sigma)
        ddy += amp * g * ((z * z - 1.0) / (sigma * sigma))
    # Taper to the start and goal centerline.
    start_taper = 1.0 - math.exp(-max(x, 0.0) / 0.34)
    goal_taper = 1.0 - math.exp(-max(6.0 - x, 0.0) / 0.34)
    taper = start_taper * goal_taper
    return y * taper, dy * taper, ddy * taper


def expert_action(q: np.ndarray, sc: Scenario, cfg: Config) -> np.ndarray:
    y_ref, slope, curvature_term = reference_geometry(float(q[0]), sc)
    vx = 0.72 + 0.35 * np.clip(cfg.goal_x - q[0], 0.0, 1.0)
    vy = vx * slope + 1.55 * (y_ref - q[1])
    theta = math.atan2(q[3], q[2])
    theta_ref = 0.26 * math.atan(slope)
    omega = 2.8 * angle_wrap(theta_ref - theta) + 0.18 * curvature_term * vx
    a, b = q[2], q[3]
    return np.array([vx, vy, -omega * b, omega * a])


def policy_features(q: np.ndarray, sc: Scenario, cfg: Config) -> np.ndarray:
    feats: List[float] = [q[0], q[1], q[2], q[3], cfg.goal_x - q[0], -q[1]]
    for (ox, oy), radius in zip(sc.centers, sc.radii):
        dx, dy = ox - q[0], oy - q[1]
        feats.extend([dx, dy, radius, math.exp(-0.5 * (dx / 0.8) ** 2), dx * dy])
    return np.asarray(feats, dtype=float)


def make_scenario(rng: np.random.Generator, trial: int, regime: str, seed: int) -> Scenario:
    base_x = np.array([1.70, 3.25, 4.72])
    base_y = np.array([0.29, -0.27, 0.24])
    if regime == "train":
        xs = base_x + rng.normal(0.0, 0.09, size=3)
        ys = base_y + rng.normal(0.0, 0.055, size=3)
        radii = rng.uniform(0.19, 0.255, size=3)
        gain, shortcut, radial, noise = 1.0, 0.0, 0.0, 0.0
    elif regime == "aggressive":
        xs = base_x + rng.normal(0.0, 0.15, size=3)
        ys = base_y + rng.normal(0.0, 0.10, size=3)
        radii = rng.uniform(0.22, 0.30, size=3)
        gain = float(rng.uniform(1.20, 1.48))
        shortcut = float(rng.uniform(0.16, 0.36))
        radial = float(rng.uniform(0.025, 0.075))
        noise = float(rng.uniform(0.025, 0.055))
    elif regime == "aggressive_ood":
        xs = base_x + rng.normal(0.0, 0.27, size=3)
        ys = base_y + rng.normal(0.0, 0.16, size=3)
        # Pull at least one obstacle toward the centerline and enlarge all of them.
        ys[trial % 3] *= 0.12
        radii = rng.uniform(0.27, 0.36, size=3)
        gain = float(rng.uniform(1.48, 1.90))
        shortcut = float(rng.uniform(0.38, 0.72))
        radial = float(rng.uniform(0.07, 0.16))
        noise = float(rng.uniform(0.055, 0.105))
    else:
        raise KeyError(regime)
    order = np.argsort(xs)
    return Scenario(
        trial=trial,
        regime=regime,
        centers=tuple((float(xs[i]), float(ys[i])) for i in order),
        radii=tuple(float(radii[i]) for i in order),
        action_gain=gain,
        shortcut_gain=shortcut,
        radial_bias=radial,
        proposal_noise=noise,
        seed=seed,
    )


def train_behavior_clone(cfg: Config) -> Tuple[ExtraTreesRegressor, Dict[str, float]]:
    rng = np.random.default_rng(cfg.seed)
    x_train: List[np.ndarray] = []
    y_train: List[np.ndarray] = []
    for i in range(cfg.train_samples):
        sc = make_scenario(rng, i, "train", cfg.seed + i)
        x = float(rng.uniform(0.0, cfg.goal_x))
        y_ref, slope, _ = reference_geometry(x, sc)
        y = float(y_ref + rng.normal(0.0, 0.13))
        theta_ref = 0.26 * math.atan(slope)
        theta = float(theta_ref + rng.normal(0.0, 0.16))
        q = np.array([x, y, math.cos(theta), math.sin(theta)])
        x_train.append(policy_features(q, sc, cfg))
        y_train.append(expert_action(q, sc, cfg))
    model = ExtraTreesRegressor(
        n_estimators=88,
        max_depth=19,
        min_samples_leaf=2,
        random_state=cfg.seed,
        n_jobs=1,
    )
    model.fit(np.vstack(x_train), np.vstack(y_train))

    x_test: List[np.ndarray] = []
    y_test: List[np.ndarray] = []
    for i in range(cfg.test_samples):
        sc = make_scenario(rng, i, "train", cfg.seed + 100_000 + i)
        x = float(rng.uniform(0.0, cfg.goal_x))
        y_ref, slope, _ = reference_geometry(x, sc)
        theta_ref = 0.26 * math.atan(slope)
        q = np.array(
            [
                x,
                float(y_ref + rng.normal(0.0, 0.13)),
                math.cos(theta_ref + rng.normal(0.0, 0.16)),
                math.sin(theta_ref + rng.normal(0.0, 0.16)),
            ]
        )
        x_test.append(policy_features(q, sc, cfg))
        y_test.append(expert_action(q, sc, cfg))
    pred = model.predict(np.vstack(x_test))
    truth = np.vstack(y_test)
    return model, {
        "heldout_action_rmse": float(math.sqrt(mean_squared_error(truth, pred))),
        "heldout_center_velocity_rmse": float(math.sqrt(mean_squared_error(truth[:, :2], pred[:, :2]))),
        "heldout_orientation_rate_rmse": float(math.sqrt(mean_squared_error(truth[:, 2:], pred[:, 2:]))),
    }


def cap_raw_action(u: np.ndarray, cfg: Config) -> np.ndarray:
    out = np.asarray(u, dtype=float).copy()
    out[:2] = np.clip(out[:2], -1.7 * cfg.max_vxy, 1.7 * cfg.max_vxy)
    orient_norm = float(np.linalg.norm(out[2:]))
    if orient_norm > 2.4:
        out[2:] *= 2.4 / orient_norm
    return out


def penalty_action(q: np.ndarray, u: np.ndarray, sc: Scenario, cfg: Config) -> np.ndarray:
    correction = -cfg.penalty_eq_gain * equality(q)[0] * equality_jacobian(q)[0]
    h, grads, _ = inequality_values_gradients(q, sc, cfg, shifted=False)
    near = h < cfg.penalty_band
    if np.any(near):
        weights = cfg.penalty_ineq_gain * np.maximum(cfg.penalty_band - h[near], 0.0)
        correction += np.sum(weights[:, None] * grads[near], axis=0)
    norm = float(np.linalg.norm(correction))
    if norm > cfg.penalty_correction_cap:
        correction *= cfg.penalty_correction_cap / norm
    return cap_raw_action(u + correction, cfg)


def proposal_chunk(
    q0: np.ndarray,
    sc: Scenario,
    cfg: Config,
    model: ExtraTreesRegressor,
    chunk_index: int,
) -> np.ndarray:
    rng = np.random.default_rng(sc.seed + 104729 * chunk_index)
    q = q0.copy()
    actions: List[np.ndarray] = []
    correlated = np.zeros(4)
    for k in range(cfg.chunk_horizon):
        pred = model.predict(policy_features(q, sc, cfg)[None, :])[0]
        correlated = 0.68 * correlated + rng.normal(0.0, sc.proposal_noise, size=4)
        pred = sc.action_gain * pred + correlated
        # Aggressive chunks cut toward the centerline and contain radial orientation
        # error that a plain BC model has not been trained to suppress.
        pred[1] += sc.shortcut_gain * (-q[1])
        pred[2:] += sc.radial_bias * q[2:4]
        # A repeatable burst near each obstacle creates hard chunk transitions.
        for ox, _ in sc.centers:
            proximity = math.exp(-0.5 * ((q[0] - ox) / 0.34) ** 2)
            pred[1] += proximity * sc.shortcut_gain * (-q[1])
            pred[2:] += proximity * correlated[2:] * 1.8
        pred = penalty_action(q, cap_raw_action(pred, cfg), sc, cfg)
        actions.append(pred)
        q = q + cfg.dt * pred
    return np.vstack(actions)


def _project_to_halfspaces(z0: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> Optional[np.ndarray]:
    """Project onto linear half-spaces with deterministic Dykstra iterations.

    The reduced action is only three-dimensional, but enumerating every active
    set becomes needlessly expensive once tray-point obstacle constraints are
    included. Dykstra's method retains the Euclidean-projection interpretation
    while scaling linearly in the number of half-spaces per sweep.
    """
    if len(b) == 0 or np.all(a @ z0 >= b - tol):
        return z0.copy()
    z = z0.copy()
    corrections = np.zeros_like(a)
    norm2 = np.sum(a * a, axis=1)
    valid = norm2 > 1e-12
    for _ in range(80):
        max_step = 0.0
        for i in range(len(b)):
            if not valid[i]:
                continue
            y = z + corrections[i]
            deficit = float(b[i] - np.dot(a[i], y))
            if deficit > 0.0:
                z_new = y + (deficit / norm2[i]) * a[i]
            else:
                z_new = y
            corrections[i] = y - z_new
            max_step = max(max_step, float(np.linalg.norm(z_new - z)))
            z = z_new
        if max_step < tol and np.all(a @ z >= b - 5e-7):
            return z
    return z if np.all(a @ z >= b - 2e-5) else None


def projection_matrices(q: np.ndarray, sc: Scenario, cfg: Config) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    z_basis = tangent_basis(q)
    if z_basis is None:
        return None, np.empty((0, 3)), np.empty(0)
    hbar, grads, _ = inequality_values_gradients(q, sc, cfg, shifted=True)
    a = grads @ z_basis
    b = -cfg.gamma * hbar
    # Reduced-coordinate action bounds are also half-spaces.
    limits = np.array([cfg.max_vxy, cfg.max_vxy, cfg.max_omega])
    eye = np.eye(3)
    a = np.vstack([a, eye, -eye])
    b = np.concatenate([b, -limits, -limits])
    return z_basis, a, b


def project_action(q: np.ndarray, u: np.ndarray, sc: Scenario, cfg: Config) -> Tuple[np.ndarray, ProjectionInfo]:
    z_basis, a, b = projection_matrices(q, sc, cfg)
    if z_basis is None:
        return np.zeros_like(u), ProjectionInfo(False, 0, float("inf"))
    z0 = z_basis.T @ u
    z = _project_to_halfspaces(z0, a, b)
    if z is None:
        # Equality-only stop fallback. Zero is safe only when already inside every
        # shifted margin; otherwise this explicitly records local infeasibility.
        fallback = np.zeros(3)
        residual = float(np.max(b - a @ fallback))
        return z_basis @ fallback, ProjectionInfo(False, 0, residual)
    residuals = b - a @ z
    return z_basis @ z, ProjectionInfo(True, int(np.sum(np.abs(residuals) < 2e-6)), float(np.max(residuals)))


def one_step_correction(q: np.ndarray, u: np.ndarray, sc: Scenario, cfg: Config) -> np.ndarray:
    """Naive execution-only tangent projection plus ambient sequential clipping."""
    j = equality_jacobian(q)
    denom = float((j @ j.T).item())
    out = u.copy() if denom < 1e-12 else u - j.T[:, 0] * float((j @ u).item()) / denom
    hbar, grads, _ = inequality_values_gradients(q, sc, cfg, shifted=True)
    targets = -cfg.gamma * hbar
    # A single sweep is order-dependent; ambient corrections can break Jc u=0.
    for grad, target in zip(grads, targets):
        deficit = float(target - np.dot(grad, out))
        if deficit > 0.0:
            out += deficit * grad / (float(np.dot(grad, grad)) + 1e-12)
    return cap_raw_action(out, cfg)


def retract(q: np.ndarray, cfg: Config) -> Tuple[np.ndarray, bool, int]:
    out = q.copy()
    for it in range(cfg.retract_iters + 1):
        c = equality(out)
        if float(np.linalg.norm(c)) < cfg.retract_tol:
            return out, True, it
        j = equality_jacobian(out)
        denom = float((j @ j.T).item())
        if denom < 1e-14:
            return out, False, it
        out = out - j.T[:, 0] * c[0] / denom
    return out, False, cfg.retract_iters


def execute_chunk(
    q: np.ndarray,
    proposed: np.ndarray,
    method: str,
    sc: Scenario,
    cfg: Config,
) -> Tuple[np.ndarray, List[Dict[str, float]], bool]:
    records: List[Dict[str, float]] = []
    any_fallback = False
    for k, raw in enumerate(proposed):
        if method == "penalty_only":
            applied = raw.copy()
            feasible = True
        elif method == "one_step_correction":
            applied = one_step_correction(q, raw, sc, cfg) if k == 0 else raw.copy()
            feasible = True
        elif method in ("rollout_projection", "projection_retraction"):
            applied, info = project_action(q, raw, sc, cfg)
            feasible = info.feasible
            any_fallback = any_fallback or not feasible
        else:
            raise KeyError(method)

        q_next = q + cfg.dt * applied
        retract_ok = True
        retract_iters = 0
        if method == "projection_retraction":
            q_next, retract_ok, retract_iters = retract(q_next, cfg)
            any_fallback = any_fallback or not retract_ok
        records.append(
            {
                "intervention": float(np.linalg.norm(applied - raw)),
                "margin": true_min_margin(q_next, sc, cfg),
                "equality": float(abs(equality(q_next)[0])),
                "feasible": float(feasible),
                "retract_ok": float(retract_ok),
                "retract_iters": float(retract_iters),
                "vx": float(applied[0]),
                "vy": float(applied[1]),
                "da": float(applied[2]),
                "db": float(applied[3]),
            }
        )
        q = q_next
    return q, records, any_fallback


def run_trial(
    method: str,
    sc: Scenario,
    cfg: Config,
    model: ExtraTreesRegressor,
    keep_trace: bool = False,
) -> TrialResult:
    q = np.array([0.0, 0.0, 1.0, 0.0])
    qs = [q.copy()]
    records: List[Dict[str, float]] = []
    chunk_index = 0
    while len(records) < cfg.max_steps and q[0] < cfg.goal_x - cfg.goal_tolerance:
        proposed = proposal_chunk(q, sc, cfg, model, chunk_index)
        remaining = cfg.max_steps - len(records)
        proposed = proposed[:remaining]
        q, chunk_records, _ = execute_chunk(q, proposed, method, sc, cfg)
        records.extend(chunk_records)
        for _ in chunk_records:
            # Reconstruct states for plotting from the stored action records.
            prev = qs[-1]
            action = np.array([records[len(qs) - 1][key] for key in ("vx", "vy", "da", "db")])
            nxt = prev + cfg.dt * action
            if method == "projection_retraction":
                nxt, _, _ = retract(nxt, cfg)
            qs.append(nxt)
        chunk_index += 1

    traj = np.vstack(qs)
    margins = np.asarray([r["margin"] for r in records], dtype=float)
    eqs = np.asarray([r["equality"] for r in records], dtype=float)
    interventions = np.asarray([r["intervention"] for r in records], dtype=float)
    actions = np.asarray([[r[k] for k in ("vx", "vy", "da", "db")] for r in records], dtype=float)
    fallback_steps = int(np.sum([1.0 - r["feasible"] for r in records]))
    retraction_failures = int(np.sum([1.0 - r["retract_ok"] for r in records]))
    reached = bool(q[0] >= cfg.goal_x - cfg.goal_tolerance and abs(q[1]) <= 0.48)
    collision = bool(np.min(margins) < -1e-5)
    if len(traj) > cfg.deadlock_window:
        deadlock = bool(
            not reached
            and np.max(traj[-cfg.deadlock_window :, 0]) - np.min(traj[-cfg.deadlock_window :, 0])
            < cfg.deadlock_progress
        )
    else:
        deadlock = False
    eq_ok = bool(np.max(eqs) < 2e-3)
    success = bool(reached and not collision and eq_ok and fallback_steps == 0 and retraction_failures == 0)
    if len(actions) >= 2:
        smoothness = float(np.sqrt(np.mean(np.sum((np.diff(actions, axis=0) / cfg.dt) ** 2, axis=1))))
    else:
        smoothness = 0.0
    progress = float(np.clip(q[0] / cfg.goal_x, 0.0, 1.2))
    return TrialResult(
        trial=sc.trial,
        regime=sc.regime,
        method=method,
        success=success,
        reached_goal=reached,
        collision=collision,
        deadlock=deadlock,
        fallback=bool(fallback_steps > 0 or retraction_failures > 0),
        fallback_steps=fallback_steps,
        steps=len(records),
        final_progress=progress,
        mean_progress_speed=float((q[0] - traj[0, 0]) / max(len(records) * cfg.dt, cfg.dt)),
        max_equality_residual=float(np.max(eqs)),
        mean_equality_residual=float(np.mean(eqs)),
        max_inequality_violation=float(max(0.0, -np.min(margins))),
        min_margin=float(np.min(margins)),
        mean_intervention=float(np.mean(interventions)),
        max_intervention=float(np.max(interventions)),
        action_smoothness=smoothness,
        retraction_failures=retraction_failures,
        trajectory=traj if keep_trace else None,
        equality_trace=eqs if keep_trace else None,
        margin_trace=margins if keep_trace else None,
        intervention_trace=interventions if keep_trace else None,
    )


def deterministic_sanity_checks(cfg: Config) -> Dict[str, Dict[str, object]]:
    sc = Scenario(
        trial=-1,
        regime="sanity",
        centers=((2.0, 0.15), (3.4, -0.2), (4.8, 0.25)),
        radii=(0.26, 0.25, 0.24),
        action_gain=1.0,
        shortcut_gain=0.0,
        radial_bias=0.0,
        proposal_noise=0.0,
        seed=cfg.seed,
    )
    q = np.array([1.63, -0.22, math.cos(0.22), math.sin(0.22)])
    u = np.array([0.95, 0.56, 0.42, -0.31])
    projected, info = project_action(q, u, sc, cfg)
    j = equality_jacobian(q)
    z_basis, a, b = projection_matrices(q, sc, cfg)
    tangent_residual = abs(float((j @ projected).item()))
    halfspace_residual = float(np.max(b - a @ (z_basis.T @ projected))) if z_basis is not None else float("inf")

    one = one_step_correction(q, u, sc, cfg)
    one_tangent_residual = abs(float((j @ one).item()))

    q_big = q + cfg.dt * projected
    q_half = q + 0.5 * cfg.dt * projected
    drift_big = abs(float(equality(q_big)[0]))
    drift_half = abs(float(equality(q_half)[0]))
    drift_ratio = drift_big / max(drift_half, 1e-16)

    q_ret, ret_ok, ret_iters = retract(q_big, cfg)
    retraction_residual = abs(float(equality(q_ret)[0]))

    infeasible_a = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    infeasible_b = np.array([0.8, 0.8])
    infeasible_detected = _project_to_halfspaces(np.zeros(3), infeasible_a, infeasible_b) is None

    singular_q = np.array([0.0, 0.0, 0.0, 0.0])
    _, singular_ok, _ = retract(singular_q, cfg)

    checks = {
        "tangent_projection": {
            "passed": bool(info.feasible and tangent_residual < 1e-9),
            "Jc_u_abs": tangent_residual,
        },
        "joint_halfspace_projection": {
            "passed": bool(info.feasible and halfspace_residual <= 1e-7),
            "max_residual": halfspace_residual,
        },
        "naive_ambient_correction_can_break_equality": {
            "passed": bool(one_tangent_residual > 1e-5),
            "Jc_u_abs": one_tangent_residual,
        },
        "finite_step_drift_is_second_order": {
            "passed": bool(3.8 <= drift_ratio <= 4.2),
            "dt_drift": drift_big,
            "half_dt_drift": drift_half,
            "ratio": drift_ratio,
        },
        "gauss_newton_retraction": {
            "passed": bool(ret_ok and retraction_residual < cfg.retract_tol),
            "residual": retraction_residual,
            "iterations": ret_iters,
        },
        "local_infeasibility_detected": {
            "passed": bool(infeasible_detected),
            "example": "z_x >= 0.8 and z_x <= -0.8",
        },
        "rank_degenerate_retraction_fails_cleanly": {
            "passed": bool(not singular_ok),
            "example": "orientation vector [a,b]=[0,0] gives rank-zero Jc",
        },
    }
    checks["all_passed"] = {"passed": bool(all(v["passed"] for k, v in checks.items() if k != "all_passed"))}
    return checks


def summarize(results: Sequence[TrialResult]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    regimes = ["all", "aggressive", "aggressive_ood"]
    for regime in regimes:
        for method in METHODS:
            rr = [r for r in results if r.method == method and (regime == "all" or r.regime == regime)]
            rows.append(
                {
                    "regime": regime,
                    "method": method,
                    "n": len(rr),
                    "success_rate": float(np.mean([r.success for r in rr])),
                    "reached_goal_rate": float(np.mean([r.reached_goal for r in rr])),
                    "collision_rate": float(np.mean([r.collision for r in rr])),
                    "deadlock_rate": float(np.mean([r.deadlock for r in rr])),
                    "fallback_rate": float(np.mean([r.fallback for r in rr])),
                    "median_max_equality_residual": float(np.median([r.max_equality_residual for r in rr])),
                    "p95_max_equality_residual": float(np.percentile([r.max_equality_residual for r in rr], 95)),
                    "mean_max_inequality_violation": float(np.mean([r.max_inequality_violation for r in rr])),
                    "mean_min_margin": float(np.mean([r.min_margin for r in rr])),
                    "mean_intervention": float(np.mean([r.mean_intervention for r in rr])),
                    "mean_action_smoothness": float(np.mean([r.action_smoothness for r in rr])),
                    "mean_final_progress": float(np.mean([r.final_progress for r in rr])),
                    "mean_progress_speed": float(np.mean([r.mean_progress_speed for r in rr])),
                }
            )
    return rows


def write_outputs(
    cfg: Config,
    training: Dict[str, float],
    checks: Dict[str, Dict[str, object]],
    scenarios: Sequence[Scenario],
    results: Sequence[TrialResult],
    summary: Sequence[Dict[str, object]],
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fields = [
        "trial", "regime", "method", "success", "reached_goal", "collision", "deadlock", "fallback",
        "fallback_steps", "steps", "final_progress", "mean_progress_speed", "max_equality_residual",
        "mean_equality_residual", "max_inequality_violation", "min_margin", "mean_intervention",
        "max_intervention", "action_smoothness", "retraction_failures",
    ]
    with (OUT / "trial_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            row = asdict(result)
            writer.writerow({k: row[k] for k in fields})
    with (OUT / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    with (OUT / "sanity_checks.json").open("w") as f:
        json.dump(checks, f, indent=2)
    scenario_payload = [asdict(sc) for sc in scenarios]
    with (OUT / "metrics.json").open("w") as f:
        json.dump(
            {
                "experiment": "constraint_manifold_action_filter",
                "scope": "explanatory toy, not a PR-MPPI reproduction",
                "config": asdict(cfg),
                "bc_training": training,
                "sanity_checks_all_passed": checks["all_passed"]["passed"],
                "scenarios": scenario_payload,
                "summary": list(summary),
            },
            f,
            indent=2,
        )


def plot_representative(results: Sequence[TrialResult], scenario: Scenario, cfg: Config) -> None:
    kept = {r.method: r for r in results if r.trial == scenario.trial and r.trajectory is not None}
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.8))
    ax = axes[0, 0]
    for center, radius in zip(scenario.centers, scenario.radii):
        ax.add_patch(plt.Circle(center, radius + cfg.tray_radius, color="#777777", alpha=0.22))
        ax.add_patch(plt.Circle(center, radius, color="#555555", alpha=0.45))
    for method in METHODS:
        r = kept[method]
        ax.plot(r.trajectory[:, 0], r.trajectory[:, 1], lw=2.1, color=COLORS[method], label=LABELS[method])
    ax.scatter([0.0, cfg.goal_x], [0.0, 0.0], c=["black", "gold"], s=70, marker="*")
    ax.set(xlabel="tray center x", ylabel="tray center y", title=f"Paired OOD trial {scenario.trial}: center paths")
    ax.set_xlim(-0.2, 6.25)
    ax.set_ylim(cfg.workspace_ymin - 0.05, cfg.workspace_ymax + 0.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[0, 1]
    for method in METHODS:
        r = kept[method]
        ax.semilogy(np.arange(len(r.equality_trace)) * cfg.dt, np.maximum(r.equality_trace, 1e-12), color=COLORS[method], label=LABELS[method])
    ax.axhline(2e-3, color="black", ls="--", lw=1, label="success tolerance")
    ax.set(xlabel="time [s]", ylabel=r"$|a^2+b^2-1|$", title="Equality-manifold residual")
    ax.grid(alpha=0.25, which="both")

    ax = axes[1, 0]
    for method in METHODS:
        r = kept[method]
        ax.plot(np.arange(len(r.margin_trace)) * cfg.dt, r.margin_trace, color=COLORS[method], label=LABELS[method])
    ax.axhline(0.0, color="black", lw=1)
    ax.axhline(cfg.safety_buffer, color="black", ls=":", lw=1, label="filter buffer")
    ax.set(xlabel="time [s]", ylabel="dense minimum margin [m]", title="Inequality margin")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    for method in METHODS:
        r = kept[method]
        ax.plot(np.arange(len(r.intervention_trace)) * cfg.dt, r.intervention_trace, color=COLORS[method], label=LABELS[method])
    ax.set(xlabel="time [s]", ylabel=r"$\|u_{exec}-u_{proposal}\|_2$", title="Execution intervention")
    ax.grid(alpha=0.25)
    fig.suptitle("Constraint-manifold action filtering: representative aggressive/OOD chunk sequence")
    fig.tight_layout()
    fig.savefig(OUT / "representative_rollout.png", dpi=180)
    plt.close(fig)


def plot_summary(summary: Sequence[Dict[str, object]]) -> None:
    rows = {r["method"]: r for r in summary if r["regime"] == "all"}
    metrics = [
        ("success_rate", "Success rate", 100.0),
        ("p95_max_equality_residual", "p95 max equality residual", 1.0),
        ("mean_max_inequality_violation", "Mean max inequality violation [m]", 1.0),
        ("mean_intervention", "Mean intervention", 1.0),
        ("mean_final_progress", "Mean task progress", 100.0),
        ("deadlock_rate", "Deadlock rate", 100.0),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2))
    x = np.arange(len(METHODS))
    for ax, (key, title, scale) in zip(axes.ravel(), metrics):
        vals = [float(rows[m][key]) * scale for m in METHODS]
        ax.bar(x, vals, color=[COLORS[m] for m in METHODS])
        ax.set_xticks(x, ["Penalty", "1-step", "Rollout", "Proj.+ret."], rotation=18)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if key in ("success_rate", "mean_final_progress", "deadlock_rate"):
            ax.set_ylim(0.0, 105.0)
            ax.set_ylabel("percent")
        if key == "p95_max_equality_residual":
            ax.set_yscale("log")
    fig.suptitle("Paired Monte Carlo: aggressive and OOD behavior-cloned chunks")
    fig.tight_layout()
    fig.savefig(OUT / "monte_carlo_summary.png", dpi=180)
    plt.close(fig)


def print_summary(summary: Sequence[Dict[str, object]], training: Dict[str, float]) -> None:
    print(f"BC held-out action RMSE: {training['heldout_action_rmse']:.4f}")
    print("all-regime summary:")
    for row in summary:
        if row["regime"] != "all":
            continue
        print(
            f"  {row['method']:24s} success={100*float(row['success_rate']):5.1f}% "
            f"eq_p95={float(row['p95_max_equality_residual']):.3e} "
            f"viol={float(row['mean_max_inequality_violation']):.4f} "
            f"margin={float(row['mean_min_margin']):.4f} "
            f"intervention={float(row['mean_intervention']):.3f} "
            f"progress={100*float(row['mean_final_progress']):5.1f}% "
            f"deadlock={100*float(row['deadlock_rate']):4.1f}%"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--train-samples", type=int, default=7000)
    parser.add_argument("--test-samples", type=int, default=1400)
    parser.add_argument("--smoke", action="store_true", help="small fast verification run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.trials = min(args.trials, 12)
        args.train_samples = min(args.train_samples, 1200)
        args.test_samples = min(args.test_samples, 300)
    cfg = Config(seed=args.seed, trials=args.trials, train_samples=args.train_samples, test_samples=args.test_samples)
    OUT.mkdir(parents=True, exist_ok=True)
    checks = deterministic_sanity_checks(cfg)
    if not checks["all_passed"]["passed"]:
        raise RuntimeError(f"deterministic geometry sanity checks failed: {checks}")
    model, training = train_behavior_clone(cfg)

    rng = np.random.default_rng(cfg.seed + 9001)
    scenarios: List[Scenario] = []
    for trial in range(cfg.trials):
        regime = "aggressive" if trial < cfg.trials // 2 else "aggressive_ood"
        scenarios.append(make_scenario(rng, trial, regime, cfg.seed + 1_000_003 * trial))

    representative_trial = cfg.trials // 2 + min(3, max(0, cfg.trials - cfg.trials // 2 - 1))
    results: List[TrialResult] = []
    for sc in scenarios:
        for method in METHODS:
            results.append(run_trial(method, sc, cfg, model, keep_trace=sc.trial == representative_trial))
    summary = summarize(results)
    write_outputs(cfg, training, checks, scenarios, results, summary)
    plot_representative(results, scenarios[representative_trial], cfg)
    plot_summary(summary)
    print_summary(summary, training)
    print(f"Wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
