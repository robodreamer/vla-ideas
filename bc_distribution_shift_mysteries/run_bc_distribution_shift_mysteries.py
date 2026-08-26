#!/usr/bin/env python3
"""Behavioral-cloning distribution-shift mysteries in a deterministic toy.

The package has two levels:
1. An exact 1D sanity check where every policy query causes a small handoff
   shock, so an H-step chunk compounds fewer query-boundary errors than H=1.
2. A learned 2D path-following benchmark built from narrow, smooth,
   temporally correlated demonstrations. Ridge BC policies sweep action-chunk
   horizon, history conditioning, feature scaling, and basis capacity.

The benchmark is deliberately diagnostic rather than realistic. It exposes an
expert-only clock feature that is correlated with position on demonstrations,
and history exposes previous actions. Both can improve held-out supervised fit
while becoming brittle when policy rollouts leave the demonstration tube.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


@dataclass(frozen=True)
class Config:
    seed: int = 17
    train_episodes: int = 90
    val_episodes: int = 30
    eval_episodes: int = 96
    steps: int = 72
    dt: float = 0.045
    max_horizon: int = 16
    train_start_noise: float = 0.025
    eval_start_noise: float = 0.075
    demo_action_noise: float = 0.035
    demo_noise_memory: float = 0.93
    query_shock_std: float = 0.055
    action_limit: float = 1.25
    obstacle_radius: float = 0.255
    collision_radius: float = 0.235
    success_radius: float = 0.16
    ridge_alpha_linear: float = 0.16
    ridge_alpha_quadratic: float = 0.75
    history_len: int = 4
    perturb_steps: tuple[int, ...] = (25, 45)
    perturb_scale: float = 0.15
    support_radius: float = 0.18


HORIZONS = (1, 4, 8, 16)
HISTORIES = (0, 4)
SCALINGS = ("state_focus", "balanced", "clock_focus")
CAPACITIES = ("linear", "quadratic")


def stable_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**32)


def smoothstep(u: np.ndarray | float) -> np.ndarray | float:
    return 3.0 * np.asarray(u) ** 2 - 2.0 * np.asarray(u) ** 3


def smoothstep_derivative(u: np.ndarray | float) -> np.ndarray | float:
    return 6.0 * np.asarray(u) - 6.0 * np.asarray(u) ** 2


def reference_state(
    u: float | np.ndarray, route: float, shape: float = 0.0, cfg: Config | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Nominal obstacle-avoiding path and velocity at normalized time u."""
    cfg = cfg or Config()
    u_arr = np.asarray(u, dtype=float)
    x = -1.05 + 2.10 * smoothstep(u_arr)
    arch = np.sin(np.pi * u_arr)
    skew = np.sin(2.0 * np.pi * u_arr)
    y = route * (0.47 + shape) * arch + 0.045 * shape * skew
    dx_du = 2.10 * smoothstep_derivative(u_arr)
    dy_du = route * (0.47 + shape) * np.pi * np.cos(np.pi * u_arr)
    dy_du += 0.09 * np.pi * shape * np.cos(2.0 * np.pi * u_arr)
    duration = cfg.steps * cfg.dt
    pos = np.stack([x, y], axis=-1)
    vel = np.stack([dx_du / duration, dy_du / duration], axis=-1)
    return pos, vel


def oracle_action(
    pos: np.ndarray,
    u: float,
    route: float,
    shape: float,
    cfg: Config,
) -> np.ndarray:
    ref_pos, ref_vel = reference_state(u, route, shape, cfg)
    action = ref_vel + 2.15 * (ref_pos - pos)
    norm = float(np.linalg.norm(action))
    if norm > cfg.action_limit:
        action = action * (cfg.action_limit / norm)
    return np.asarray(action, dtype=float)


def generate_episode(seed: int, cfg: Config, split: str) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    route = float(rng.choice([-1.0, 1.0]))
    shape = float(rng.normal(0.0, 0.025))
    start_noise = cfg.train_start_noise if split != "eval" else cfg.eval_start_noise
    ref0, _ = reference_state(0.0, route, shape, cfg)
    pos = np.asarray(ref0, dtype=float) + rng.normal(0.0, start_noise, 2)
    positions = np.zeros((cfg.steps, 2), dtype=float)
    actions = np.zeros((cfg.steps, 2), dtype=float)
    noise = rng.normal(0.0, cfg.demo_action_noise, 2)
    innovation = cfg.demo_action_noise * math.sqrt(1.0 - cfg.demo_noise_memory**2)
    for t in range(cfg.steps):
        u = t / (cfg.steps - 1)
        base = oracle_action(pos, u, route, shape, cfg)
        noise = cfg.demo_noise_memory * noise + rng.normal(0.0, innovation, 2)
        action = base + noise
        norm = float(np.linalg.norm(action))
        if norm > cfg.action_limit:
            action *= cfg.action_limit / norm
        positions[t] = pos
        actions[t] = action
        pos = pos + cfg.dt * action
    return {
        "positions": positions,
        "actions": actions,
        "route": route,
        "shape": shape,
    }


def base_features(pos: np.ndarray, u: float, route: float) -> np.ndarray:
    """State and nuisance features.

    Position/goal-relative coordinates support corrective feedback. The clock
    coordinates are almost interchangeable with position on narrow expert
    rollouts but do not react to perturbations. This lets ridge feature scaling
    choose between equally predictive in-distribution explanations.
    """
    goal = np.array([1.05, 0.0])
    return np.array(
        [
            pos[0],
            pos[1],
            goal[0] - pos[0],
            goal[1] - pos[1],
            route,
            u,
            u * u,
            math.sin(math.pi * u),
            math.cos(math.pi * u),
            1.0,
        ],
        dtype=float,
    )


def raw_features(
    pos: np.ndarray,
    u: float,
    route: float,
    action_history: np.ndarray,
    history: int,
) -> np.ndarray:
    values = [base_features(pos, u, route)]
    if history:
        values.append(action_history[-history:].reshape(-1))
    return np.concatenate(values)


def make_supervised(
    episodes: list[dict[str, np.ndarray | float]],
    horizon: int,
    history: int,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for episode in episodes:
        positions = np.asarray(episode["positions"])
        actions = np.asarray(episode["actions"])
        route = float(episode["route"])
        padded_history = np.zeros((cfg.steps + cfg.history_len, 2), dtype=float)
        padded_history[cfg.history_len :] = actions
        last_t = cfg.steps - horizon
        for t in range(last_t):
            hist = padded_history[t : t + cfg.history_len]
            xs.append(raw_features(positions[t], t / (cfg.steps - 1), route, hist, history))
            ys.append(actions[t : t + horizon].reshape(-1))
    return np.asarray(xs), np.asarray(ys)


def feature_group_weights(history: int, scaling: str) -> np.ndarray:
    dim = 10 + 2 * history
    weights = np.ones(dim, dtype=float)
    state_idx = np.array([0, 1, 2, 3])
    clock_idx = np.array([5, 6, 7, 8])
    if scaling == "state_focus":
        weights[state_idx] = 3.2
        weights[clock_idx] = 0.32
    elif scaling == "clock_focus":
        weights[state_idx] = 0.32
        weights[clock_idx] = 3.2
    elif scaling != "balanced":
        raise ValueError(scaling)
    if history:
        # Previous expert actions are highly predictive on demonstration data.
        # Making them cheap under ridge exposes the causal-confusion failure.
        weights[10:] = 3.0
    return weights


class BCModel:
    def __init__(
        self,
        horizon: int,
        history: int,
        scaling: str,
        capacity: str,
        cfg: Config,
    ) -> None:
        self.horizon = horizon
        self.history = history
        self.scaling = scaling
        self.capacity = capacity
        self.cfg = cfg
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.weights = feature_group_weights(history, scaling)
        degree = 1 if capacity == "linear" else 2
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        alpha = cfg.ridge_alpha_linear if capacity == "linear" else cfg.ridge_alpha_quadratic
        self.ridge = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", tol=1e-7)

    @property
    def name(self) -> str:
        return f"H{self.horizon}_hist{self.history}_{self.scaling}_{self.capacity}"

    def transform(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.mean = x.mean(axis=0)
            self.std = x.std(axis=0)
            self.std[self.std < 1e-6] = 1.0
        assert self.mean is not None and self.std is not None
        z = ((x - self.mean) / self.std) * self.weights
        return self.poly.fit_transform(z) if fit else self.poly.transform(z)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.ridge.fit(self.transform(x, fit=True), y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.ridge.predict(self.transform(x, fit=False)))

    @property
    def feature_count(self) -> int:
        return int(self.ridge.coef_.shape[1])


def validation_metrics(model: BCModel, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = model.predict(x)
    all_mse = float(np.mean((pred - y) ** 2))
    first_mse = float(np.mean((pred[:, :2] - y[:, :2]) ** 2))
    return {"val_action_mse": all_mse, "val_first_action_mse": first_mse}


def collision(pos: np.ndarray, cfg: Config) -> bool:
    return bool(np.linalg.norm(pos) < cfg.collision_radius)


def rollout_model(
    model: BCModel,
    episode_seed: int,
    cfg: Config,
    record: bool = False,
) -> dict:
    rng = np.random.default_rng(episode_seed)
    route = float(rng.choice([-1.0, 1.0]))
    shape = float(rng.normal(0.0, 0.025))
    ref0, _ = reference_state(0.0, route, shape, cfg)
    pos = np.asarray(ref0, dtype=float) + rng.normal(0.0, cfg.eval_start_noise, 2)
    perturbations = {
        step: rng.normal(0.0, cfg.perturb_scale, 2) for step in cfg.perturb_steps
    }
    action_history = np.zeros((cfg.history_len, 2), dtype=float)
    actions: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    tracking: list[float] = []
    oracle_errors: list[float] = []
    collided = False
    query_count = 0
    t = 0
    while t < cfg.steps:
        u = t / (cfg.steps - 1)
        feat = raw_features(pos, u, route, action_history, model.history)[None, :]
        chunk = model.predict(feat)[0].reshape(model.horizon, 2)
        # A small first-action query-boundary shock models stochastic policy
        # sampling / receding-horizon handoff inconsistency. Its marginal scale
        # is independent of H, but H=1 experiences it at every control step.
        shock_rng = np.random.default_rng(stable_seed(model.name, episode_seed, query_count))
        chunk[0] += shock_rng.normal(0.0, cfg.query_shock_std, 2)
        query_count += 1
        for k in range(model.horizon):
            if t >= cfg.steps:
                break
            if t in perturbations:
                pos = pos + perturbations[t]
            action = chunk[k]
            norm = float(np.linalg.norm(action))
            if norm > cfg.action_limit:
                action = action * (cfg.action_limit / norm)
            u_now = t / (cfg.steps - 1)
            oracle = oracle_action(pos, u_now, route, shape, cfg)
            ref_pos, _ = reference_state(u_now, route, shape, cfg)
            positions.append(pos.copy())
            actions.append(action.copy())
            tracking.append(float(np.linalg.norm(pos - ref_pos)))
            oracle_errors.append(float(np.mean((action - oracle) ** 2)))
            collided = collided or collision(pos, cfg)
            pos = pos + cfg.dt * action
            action_history = np.vstack([action_history[1:], action])
            t += 1
    pos_arr = np.asarray(positions)
    action_arr = np.asarray(actions)
    tracking_arr = np.asarray(tracking)
    goal_distance = float(np.linalg.norm(pos - np.array([1.05, 0.0])))
    smoothness = float(np.mean(np.sum(np.diff(action_arr, axis=0) ** 2, axis=1)))
    metrics = {
        "success": float(goal_distance < cfg.success_radius and not collided),
        "collision": float(collided),
        "goal_distance": goal_distance,
        "tracking_rmse": float(np.sqrt(np.mean(tracking_arr**2))),
        "max_state_divergence": float(np.max(tracking_arr)),
        "ood_fraction": float(np.mean(tracking_arr > cfg.support_radius)),
        "smoothness": smoothness,
        "on_policy_oracle_mse": float(np.mean(oracle_errors)),
        "query_count": float(query_count),
    }
    if record:
        metrics.update(
            positions=pos_arr.tolist(),
            actions=action_arr.tolist(),
            route=route,
            perturbations={str(k): v.tolist() for k, v in perturbations.items()},
        )
    return metrics


def exact_sanity_check() -> dict:
    steps = 20
    target_action = 0.05
    handoff_shock = 0.01
    rows = []
    for horizon in (1, 5):
        x = 0.0
        actions = []
        for t in range(steps):
            query_boundary = t % horizon == 0
            action = target_action + (handoff_shock if query_boundary else 0.0)
            actions.append(action)
            x += action
        rows.append(
            {
                "horizon": horizon,
                "queries": int(math.ceil(steps / horizon)),
                "final_error": abs(x - steps * target_action),
                "smoothness": float(np.mean(np.diff(actions) ** 2)),
            }
        )
    return {
        "description": "Exact 1D query-boundary compounding check",
        "rows": rows,
        "checks": {
            "chunk_has_fewer_queries": rows[1]["queries"] < rows[0]["queries"],
            "chunk_has_lower_final_error": rows[1]["final_error"] < rows[0]["final_error"],
        },
    }


def aggregate_trials(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["model"], []).append(row)
    summary = []
    metric_names = [
        "success",
        "collision",
        "goal_distance",
        "tracking_rmse",
        "max_state_divergence",
        "ood_fraction",
        "smoothness",
        "on_policy_oracle_mse",
        "query_count",
    ]
    for name, trials in groups.items():
        first = trials[0]
        item = {
            key: first[key]
            for key in ["model", "horizon", "history", "scaling", "capacity"]
        }
        for metric in metric_names:
            values = np.asarray([r[metric] for r in trials], dtype=float)
            item[metric] = float(values.mean())
            item[f"{metric}_sem"] = float(values.std(ddof=1) / math.sqrt(len(values)))
        summary.append(item)
    return summary


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def model_lookup(summary: list[dict], **kwargs: object) -> dict:
    matches = [r for r in summary if all(r[k] == v for k, v in kwargs.items())]
    if len(matches) != 1:
        raise KeyError(kwargs)
    return matches[0]


def plot_summary(summary: list[dict], records: dict[str, dict], cfg: Config) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selected_specs = [
        (1, 0, "state_focus", "quadratic", "one-step"),
        (1, 4, "state_focus", "quadratic", "history"),
        (8, 0, "state_focus", "quadratic", "open-loop H8"),
        (8, 0, "clock_focus", "quadratic", "H8 clock-focus"),
    ]
    selected = [
        (model_lookup(summary, horizon=h, history=hist, scaling=s, capacity=c), label)
        for h, hist, s, c, label in selected_specs
    ]
    labels = [label for _, label in selected]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    metrics = [
        ("val_action_mse", "validation action MSE", False),
        ("success", "rollout success", True),
        ("tracking_rmse", "tracking RMSE", False),
        ("ood_fraction", "OOD-state fraction", True),
        ("smoothness", "action-difference energy", False),
        ("on_policy_oracle_mse", "on-policy oracle action MSE", False),
    ]
    for ax, (metric, title, percentage) in zip(axes.flat, metrics):
        vals = [row[metric] * (100 if percentage else 1) for row, _ in selected]
        ax.bar(np.arange(len(vals)), vals, color=colors)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(vals)), labels, rotation=22, ha="right")
        if percentage:
            ax.set_ylabel("%")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("BC offline fit and closed-loop behavior can disagree", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "mystery_summary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.3), sharex=True)
    for col, scaling in enumerate(SCALINGS):
        for row_idx, history in enumerate(HISTORIES):
            ax = axes[row_idx, col]
            subset = [
                r
                for r in summary
                if r["scaling"] == scaling
                and r["history"] == history
                and r["capacity"] == "quadratic"
            ]
            subset.sort(key=lambda r: r["horizon"])
            x = [r["horizon"] for r in subset]
            ax2 = ax.twinx()
            ax.plot(x, [100 * r["success"] for r in subset], "o-", color="#54a24b", label="success")
            ax2.plot(x, [r["val_action_mse"] for r in subset], "s--", color="#4c78a8", label="val MSE")
            ax.set_ylim(-4, 104)
            ax.set_title(f"{scaling}; history={history}")
            ax.set_ylabel("success (%)", color="#54a24b")
            ax2.set_ylabel("validation MSE", color="#4c78a8")
            ax.grid(alpha=0.22)
            ax.set_xticks(HORIZONS)
    for ax in axes[-1]:
        ax.set_xlabel("chunk horizon H")
    fig.suptitle("Horizon, history, and feature-scaling sweep (quadratic basis)")
    fig.tight_layout()
    fig.savefig(OUT / "sweep_horizon_scaling.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.3, 6.4))
    markers = {"state_focus": "o", "balanced": "s", "clock_focus": "^"}
    colors_cap = {"linear": "#9ecae1", "quadratic": "#08519c"}
    for scaling in SCALINGS:
        for capacity in CAPACITIES:
            subset = [r for r in summary if r["scaling"] == scaling and r["capacity"] == capacity]
            ax.scatter(
                [r["val_action_mse"] for r in subset],
                [100 * r["success"] for r in subset],
                marker=markers[scaling],
                c=colors_cap[capacity],
                alpha=0.78,
                s=60,
                label=f"{scaling}, {capacity}",
            )
    ax.set_xscale("log")
    ax.set_xlabel("held-out expert-state action MSE")
    ax.set_ylabel("closed-loop success (%)")
    ax.set_title("Offline validation loss is an incomplete rollout metric")
    ax.grid(alpha=0.25)
    handles, labels_unique = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_unique, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "validation_vs_rollout.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    route_colors = {"state_focus": "#54a24b", "balanced": "#4c78a8", "clock_focus": "#e45756"}
    for ax, scaling in zip(axes, SCALINGS):
        key = f"H8_hist0_{scaling}_quadratic"
        rec = records[key]
        pos = np.asarray(rec["positions"])
        route = float(rec["route"])
        us = np.linspace(0.0, 1.0, cfg.steps)
        ref, _ = reference_state(us, route, 0.0, cfg)
        circle = plt.Circle((0, 0), cfg.collision_radius, color="black", alpha=0.16)
        ax.add_patch(circle)
        ax.plot(ref[:, 0], ref[:, 1], "k--", lw=1.2, label="nominal expert path")
        ax.plot(pos[:, 0], pos[:, 1], color=route_colors[scaling], lw=2.2, label="BC rollout")
        for step, delta in rec["perturbations"].items():
            idx = int(step)
            ax.scatter(pos[idx, 0], pos[idx, 1], marker="x", s=45, color="black")
        ax.scatter([-1.05, 1.05], [0, 0], c=["black", "gold"], marker="*", s=80)
        ax.set_title(scaling.replace("_", " "))
        ax.set_aspect("equal")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("y")
    for ax in axes:
        ax.set_xlabel("x")
    axes[0].legend(fontsize=8, loc="lower center")
    fig.suptitle("Same-information feature scalings extrapolate differently after perturbations")
    fig.tight_layout()
    fig.savefig(OUT / "feature_scaling_rollouts.png", dpi=180)
    plt.close(fig)


def run(cfg: Config, smoke: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    sanity = exact_sanity_check()
    (OUT / "sanity_check.json").write_text(json.dumps(sanity, indent=2) + "\n")

    train_n = 18 if smoke else cfg.train_episodes
    val_n = 8 if smoke else cfg.val_episodes
    eval_n = 12 if smoke else cfg.eval_episodes
    train_eps = [generate_episode(stable_seed(cfg.seed, "train", i), cfg, "train") for i in range(train_n)]
    val_eps = [generate_episode(stable_seed(cfg.seed, "val", i), cfg, "val") for i in range(val_n)]

    val_rows: list[dict] = []
    trial_rows: list[dict] = []
    representative: dict[str, dict] = {}
    capacities = ("linear",) if smoke else CAPACITIES
    scalings = ("state_focus", "clock_focus") if smoke else SCALINGS
    horizons = (1, 8) if smoke else HORIZONS
    histories = HISTORIES

    dataset_cache: dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray]] = {}
    for horizon in horizons:
        for history in histories:
            x_train, y_train = make_supervised(train_eps, horizon, history, cfg)
            x_val, y_val = make_supervised(val_eps, horizon, history, cfg)
            dataset_cache[(horizon, history, "train")] = (x_train, y_train)
            dataset_cache[(horizon, history, "val")] = (x_val, y_val)
            for scaling in scalings:
                for capacity in capacities:
                    model = BCModel(horizon, history, scaling, capacity, cfg)
                    model.fit(x_train, y_train)
                    vm = validation_metrics(model, x_val, y_val)
                    model_row = {
                        "model": model.name,
                        "horizon": horizon,
                        "history": history,
                        "scaling": scaling,
                        "capacity": capacity,
                        "feature_count": model.feature_count,
                        "train_samples": len(x_train),
                        **vm,
                    }
                    val_rows.append(model_row)
                    for episode in range(eval_n):
                        seed = stable_seed(cfg.seed, "eval", episode)
                        result = rollout_model(model, seed, cfg, record=False)
                        trial_rows.append(
                            {
                                "model": model.name,
                                "horizon": horizon,
                                "history": history,
                                "scaling": scaling,
                                "capacity": capacity,
                                "episode": episode,
                                **result,
                            }
                        )
                    if horizon == 8 and history == 0 and capacity == "quadratic" and not smoke:
                        representative[model.name] = rollout_model(
                            model, stable_seed(cfg.seed, "eval", 14), cfg, record=True
                        )

    summary = aggregate_trials(trial_rows)
    val_by_name = {r["model"]: r for r in val_rows}
    for row in summary:
        row.update(
            feature_count=val_by_name[row["model"]]["feature_count"],
            train_samples=val_by_name[row["model"]]["train_samples"],
            val_action_mse=val_by_name[row["model"]]["val_action_mse"],
            val_first_action_mse=val_by_name[row["model"]]["val_first_action_mse"],
        )
    summary.sort(key=lambda r: (r["capacity"], r["scaling"], r["history"], r["horizon"]))
    write_csv(OUT / "model_metrics.csv", summary)
    write_csv(OUT / "rollout_trials.csv", trial_rows)

    claims = {}
    if not smoke:
        one = model_lookup(summary, horizon=1, history=0, scaling="state_focus", capacity="quadratic")
        hist = model_lookup(summary, horizon=1, history=4, scaling="state_focus", capacity="quadratic")
        chunk = model_lookup(summary, horizon=8, history=0, scaling="state_focus", capacity="quadratic")
        clock = model_lookup(summary, horizon=8, history=0, scaling="clock_focus", capacity="quadratic")
        balanced = model_lookup(summary, horizon=8, history=0, scaling="balanced", capacity="quadratic")
        claims = {
            "open_loop_vs_one_step": {
                "one_step": one,
                "history_one_step": hist,
                "open_loop_h8": chunk,
                "success_gain_vs_one_step_pp": 100 * (chunk["success"] - one["success"]),
                "success_gain_vs_history_pp": 100 * (chunk["success"] - hist["success"]),
                "val_mse_ratio_vs_one_step": chunk["val_action_mse"] / one["val_action_mse"],
            },
            "feature_scaling": {
                "state_focus": chunk,
                "balanced": balanced,
                "clock_focus": clock,
                "success_gain_state_vs_clock_pp": 100 * (chunk["success"] - clock["success"]),
                "val_mse_ratio_state_vs_clock": chunk["val_action_mse"] / clock["val_action_mse"],
            },
        }
        plot_summary(summary, representative, cfg)

    payload = {
        "config": asdict(cfg),
        "smoke": smoke,
        "sanity_check": sanity,
        "claims": claims,
        "models": summary,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"claims": claims, "model_count": len(summary)}, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval-episodes", type=int, default=96)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = Config(seed=args.seed, eval_episodes=args.eval_episodes)
    run(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
