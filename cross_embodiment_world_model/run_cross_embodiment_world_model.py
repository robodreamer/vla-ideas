#!/usr/bin/env python3
"""Deterministic cross-embodiment world-model toy inspired by CLAP.

Two source embodiments perturb the same 2-D object through different raw control
spaces. A target embodiment supplies few demonstrations. We compare padded raw
controls, canonical end-effector controls, learned transition latents, and a
latent-to-end-effector curriculum for one-step prediction and candidate action
reranking.

This is a small vector-state mechanism test. It is not a reproduction of CLAP's
video diffusion models, datasets, latent-action VAE, or real-robot results.
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
from typing import Any, Callable, Iterable

BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("raw_joint", "canonical_ee", "learned_latent", "latent_to_ee_curriculum")
LABELS = {
    "raw_joint": "Raw / padded joints",
    "canonical_ee": "Canonical EE",
    "learned_latent": "Learned latent",
    "latent_to_ee_curriculum": "Latent→EE curriculum",
}
COLORS = {
    "raw_joint": "#7f7f7f",
    "canonical_ee": "#4c78a8",
    "learned_latent": "#f58518",
    "latent_to_ee_curriculum": "#e45756",
}


@dataclass(frozen=True)
class Config:
    seed: int = 23
    seeds: int = 8
    source_episodes: int = 150
    source_horizon: int = 12
    target_pool_episodes: int = 48
    target_horizon: int = 12
    test_transitions: int = 900
    planning_queries: int = 320
    planning_candidates: int = 10
    shots: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)
    source_ee_label_fraction: float = 0.010
    target_weight: float = 14.0
    ridge: float = 2.0e-3
    grounding_ridge: float = 8.0e-3
    correction_ridge: float = 5.0e-2
    latent_dim: int = 2
    planning_regret_threshold: float = 0.003


@dataclass(frozen=True)
class Embodiment:
    name: str
    raw_dim: int
    matrix: np.ndarray
    quadratic: np.ndarray
    bias: np.ndarray


@dataclass
class Dataset:
    state: np.ndarray
    raw: np.ndarray
    ee: np.ndarray
    next_state: np.ndarray
    embodiment: np.ndarray
    episode: np.ndarray
    step: np.ndarray

    def __len__(self) -> int:
        return int(len(self.state))

    @property
    def delta(self) -> np.ndarray:
        return self.next_state - self.state

    def subset(self, indices: np.ndarray | list[int]) -> "Dataset":
        idx = np.asarray(indices, dtype=int)
        return Dataset(
            self.state[idx], self.raw[idx], self.ee[idx], self.next_state[idx],
            self.embodiment[idx], self.episode[idx], self.step[idx]
        )


@dataclass
class LinearModel:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=float)
        xn = (xx - self.mean) / self.scale
        design = np.concatenate([np.ones((len(xn), 1)), xn], axis=1)
        return design @ self.coef


@dataclass
class PCAState:
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.mean) @ self.components.T

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float) @ self.components + self.mean


@dataclass
class MethodBundle:
    name: str
    predict_delta: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def embodiment_specs() -> list[Embodiment]:
    return [
        Embodiment(
            "source_slide2", 2,
            np.array([[1.15, 0.25], [-0.18, 0.95]], dtype=float),
            np.array([[0.16, -0.10], [0.08, 0.13]], dtype=float),
            np.array([0.04, -0.03]),
        ),
        Embodiment(
            "source_redundant3", 3,
            np.array([[-0.55, 0.95, 0.35], [0.82, 0.18, -0.72]], dtype=float),
            np.array([[0.08, -0.12, 0.10], [-0.10, 0.07, 0.12]], dtype=float),
            np.array([-0.05, 0.02]),
        ),
        Embodiment(
            "target_novel4", 4,
            np.array([[0.38, -0.72, 0.88, 0.25], [-0.78, -0.15, 0.32, 0.90]], dtype=float),
            np.array([[0.11, 0.07, -0.09, 0.06], [0.06, -0.11, 0.08, 0.10]], dtype=float),
            np.array([0.01, 0.05]),
        ),
    ]


def raw_to_ee(raw: np.ndarray, embodiment: Embodiment) -> np.ndarray:
    r = np.asarray(raw, dtype=float)[..., : embodiment.raw_dim]
    linear = r @ embodiment.matrix.T
    quad = (r * np.abs(r)) @ embodiment.quadratic.T
    if embodiment.raw_dim >= 2:
        cross = np.stack([r[..., 0] * r[..., 1], r[..., 0] * r[..., -1]], axis=-1)
        linear = linear + np.stack([0.10 * cross[..., 0], -0.08 * cross[..., 1]], axis=-1)
    return np.tanh(linear + quad + embodiment.bias)


def object_step(state: np.ndarray, ee: np.ndarray) -> np.ndarray:
    """Shared object dynamics; embodiment identity never enters this function."""
    s = np.asarray(state, dtype=float)
    a = np.asarray(ee, dtype=float)
    p = s[..., :2]
    v = s[..., 2:]
    contact = np.stack(
        [
            (1.0 + 0.22 * np.sin(1.7 * p[..., 1])) * a[..., 0] + 0.13 * a[..., 0] * a[..., 1],
            (0.92 + 0.18 * np.cos(1.4 * p[..., 0])) * a[..., 1] - 0.10 * a[..., 0] ** 2,
        ],
        axis=-1,
    )
    coupling = np.stack(
        [0.055 * np.sin(1.8 * p[..., 1]) - 0.035 * v[..., 1],
         -0.050 * np.sin(1.5 * p[..., 0]) + 0.030 * v[..., 0]],
        axis=-1,
    )
    v_next = 0.76 * v + 0.24 * contact + coupling
    p_next = p + 0.18 * v_next + 0.018 * np.stack([a[..., 1], -a[..., 0]], axis=-1)
    return np.concatenate([p_next, v_next], axis=-1)


def generate_dataset(
    embodiment_id: int,
    episodes: int,
    horizon: int,
    seed: int,
    episode_offset: int = 0,
) -> Dataset:
    embodiment = embodiment_specs()[embodiment_id]
    states: list[np.ndarray] = []
    raws: list[np.ndarray] = []
    ees: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    embodiment_ids: list[int] = []
    episode_ids: list[int] = []
    steps: list[int] = []
    for episode in range(episodes):
        rng = np.random.default_rng(stable_seed(seed, embodiment.name, episode + episode_offset))
        state = np.concatenate([rng.uniform(-0.9, 0.9, 2), rng.uniform(-0.30, 0.30, 2)])
        raw = rng.uniform(-0.7, 0.7, embodiment.raw_dim)
        target = rng.uniform(-0.95, 0.95, 2)
        for step in range(horizon):
            if step % 4 == 0:
                target = rng.uniform(-0.95, 0.95, 2)
            # Correlated exploratory commands provide trajectory structure without
            # requiring inverse kinematics.
            drift = np.zeros(embodiment.raw_dim)
            drift[:2] = 0.32 * np.tanh(target - state[:2])
            raw = np.clip(0.62 * raw + drift + rng.normal(0.0, 0.43, embodiment.raw_dim), -1.0, 1.0)
            raw_pad = np.zeros(4, dtype=float)
            raw_pad[: embodiment.raw_dim] = raw
            ee = raw_to_ee(raw_pad[None], embodiment)[0]
            nxt = object_step(state[None], ee[None])[0]
            states.append(state.copy())
            raws.append(raw_pad)
            ees.append(ee)
            next_states.append(nxt)
            embodiment_ids.append(embodiment_id)
            episode_ids.append(episode + episode_offset)
            steps.append(step)
            state = nxt
    return Dataset(
        np.asarray(states), np.asarray(raws), np.asarray(ees), np.asarray(next_states),
        np.asarray(embodiment_ids, dtype=int), np.asarray(episode_ids, dtype=int), np.asarray(steps, dtype=int)
    )


def concatenate(parts: Iterable[Dataset]) -> Dataset:
    items = list(parts)
    return Dataset(
        *[np.concatenate([getattr(item, field) for item in items], axis=0)
          for field in ("state", "raw", "ee", "next_state", "embodiment", "episode", "step")]
    )


def state_features(state: np.ndarray) -> np.ndarray:
    s = np.asarray(state, dtype=float)
    p0, p1, v0, v1 = (s[:, i] for i in range(4))
    return np.column_stack([
        s,
        s * s,
        np.sin(1.5 * s),
        p0 * p1,
        p0 * v0,
        p1 * v1,
        v0 * v1,
    ])


def conditioned_features(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    s = np.asarray(state, dtype=float)
    a = np.asarray(action, dtype=float)
    base = state_features(s)
    cross = np.einsum("ni,nj->nij", s, a).reshape(len(s), -1)
    return np.concatenate([base, a, a * a, cross], axis=1)


def raw_features(state: np.ndarray, raw: np.ndarray, embodiment: np.ndarray) -> np.ndarray:
    s = np.asarray(state, dtype=float)
    r = np.asarray(raw, dtype=float)
    emb = np.asarray(embodiment, dtype=int)
    onehot = np.eye(3)[emb]
    emb_raw = np.einsum("ni,nj->nij", onehot, r).reshape(len(r), -1)
    emb_state = np.einsum("ni,nj->nij", onehot, s[:, :2]).reshape(len(r), -1)
    return np.concatenate([state_features(s), r, r * r, onehot, emb_raw, emb_state], axis=1)


def fit_linear(x: np.ndarray, y: np.ndarray, ridge: float, weights: np.ndarray | None = None) -> LinearModel:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if yy.ndim == 1:
        yy = yy[:, None]
    mean = xx.mean(axis=0)
    scale = xx.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    xn = (xx - mean) / scale
    design = np.concatenate([np.ones((len(xn), 1)), xn], axis=1)
    if weights is not None:
        root_w = np.sqrt(np.asarray(weights, dtype=float))[:, None]
        design_w = design * root_w
        target_w = yy * root_w
    else:
        design_w = design
        target_w = yy
    gram = design_w.T @ design_w
    penalty = ridge * np.eye(gram.shape[0])
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(gram + penalty, design_w.T @ target_w)
    return LinearModel(mean, scale, coef)


def fit_pca(x: np.ndarray, dim: int) -> PCAState:
    xx = np.asarray(x, dtype=float)
    mean = xx.mean(axis=0)
    _, singular, vt = np.linalg.svd(xx - mean, full_matrices=False)
    variance = singular * singular
    ratio = variance / max(variance.sum(), 1.0e-12)
    return PCAState(mean, vt[:dim], ratio[:dim])


def target_demo_subset(pool: Dataset, shots: int) -> Dataset:
    if shots == 0:
        return pool.subset(np.array([], dtype=int))
    episodes = np.unique(pool.episode)[:shots]
    return pool.subset(np.flatnonzero(np.isin(pool.episode, episodes)))


def source_label_subset(source: Dataset, fraction: float, seed: int) -> Dataset:
    count = max(24, int(round(len(source) * fraction)))
    rng = np.random.default_rng(stable_seed(seed, "source-ee-labels"))
    # Stratify by embodiment so both source control spaces contribute.
    picked: list[int] = []
    for embodiment_id in (0, 1):
        idx = np.flatnonzero(source.embodiment == embodiment_id)
        local_n = count // 2 if embodiment_id == 0 else count - len(picked)
        picked.extend(rng.choice(idx, size=local_n, replace=False).tolist())
    return source.subset(np.asarray(sorted(picked), dtype=int))


def infer_latents(
    data: Dataset,
    passive: LinearModel,
    pca: PCAState,
) -> np.ndarray:
    residual = data.delta - passive.predict(state_features(data.state))
    return pca.transform(residual)


def build_models(source: Dataset, source_labeled: Dataset, target_demo: Dataset, cfg: Config) -> tuple[dict[str, MethodBundle], dict[str, Any]]:
    has_target = len(target_demo) > 0
    combined = concatenate([source, target_demo]) if has_target else source
    weights = np.ones(len(combined))
    if has_target:
        weights[-len(target_demo):] = cfg.target_weight

    raw_model = fit_linear(
        raw_features(combined.state, combined.raw, combined.embodiment),
        combined.delta,
        cfg.ridge,
        weights,
    )

    labeled = concatenate([source_labeled, target_demo]) if has_target else source_labeled
    labeled_weights = np.ones(len(labeled))
    if has_target:
        labeled_weights[-len(target_demo):] = cfg.target_weight
    ee_model = fit_linear(conditioned_features(labeled.state, labeled.ee), labeled.delta, cfg.ridge, labeled_weights)

    # Proxy-action stage: remove passive state change, then learn a two-dimensional
    # transition latent from all source transitions, including those without EE labels.
    passive = fit_linear(state_features(source.state), source.delta, cfg.ridge)
    source_residual = source.delta - passive.predict(state_features(source.state))
    pca = fit_pca(source_residual, cfg.latent_dim)
    source_z = pca.transform(source_residual)
    latent_dynamics = fit_linear(conditioned_features(source.state, source_z), source.delta, cfg.ridge)

    # Learned-latent deployment requires embodiment-specific alignment from raw
    # commands to inferred transition latents.
    if has_target:
        target_z = infer_latents(target_demo, passive, pca)
        mapper_state = np.concatenate([source.state, target_demo.state], axis=0)
        mapper_raw = np.concatenate([source.raw, target_demo.raw], axis=0)
        mapper_emb = np.concatenate([source.embodiment, target_demo.embodiment], axis=0)
        mapper_z = np.concatenate([source_z, target_z], axis=0)
        mapper_weights = np.concatenate([np.ones(len(source)), np.full(len(target_demo), cfg.target_weight)])
    else:
        mapper_state, mapper_raw, mapper_emb, mapper_z = source.state, source.raw, source.embodiment, source_z
        mapper_weights = np.ones(len(source))
    latent_mapper = fit_linear(raw_features(mapper_state, mapper_raw, mapper_emb), mapper_z, cfg.grounding_ridge, mapper_weights)

    # Curriculum grounding: retain the latent dynamics backbone, learn an EE->z
    # bridge from the small labeled subset, then fit only a conservative residual.
    source_label_z = infer_latents(source_labeled, passive, pca)
    if has_target:
        target_z = infer_latents(target_demo, passive, pca)
        ground_data = concatenate([source_labeled, target_demo])
        ground_z = np.concatenate([source_label_z, target_z], axis=0)
        ground_weights = np.concatenate([np.ones(len(source_labeled)), np.full(len(target_demo), cfg.target_weight)])
    else:
        ground_data, ground_z = source_labeled, source_label_z
        ground_weights = np.ones(len(source_labeled))
    ee_to_z = fit_linear(conditioned_features(ground_data.state, ground_data.ee), ground_z, cfg.grounding_ridge, ground_weights)
    ground_pred_z = ee_to_z.predict(conditioned_features(ground_data.state, ground_data.ee))
    prior_delta = latent_dynamics.predict(conditioned_features(ground_data.state, ground_pred_z))
    correction = fit_linear(
        conditioned_features(ground_data.state, ground_data.ee),
        ground_data.delta - prior_delta,
        cfg.correction_ridge,
        ground_weights,
    )

    def predict_raw(state: np.ndarray, raw: np.ndarray, ee: np.ndarray) -> np.ndarray:
        emb = np.full(len(state), 2, dtype=int)
        return raw_model.predict(raw_features(state, raw, emb))

    def predict_ee(state: np.ndarray, raw: np.ndarray, ee: np.ndarray) -> np.ndarray:
        return ee_model.predict(conditioned_features(state, ee))

    def predict_latent(state: np.ndarray, raw: np.ndarray, ee: np.ndarray) -> np.ndarray:
        emb = np.full(len(state), 2, dtype=int)
        z = latent_mapper.predict(raw_features(state, raw, emb))
        return latent_dynamics.predict(conditioned_features(state, z))

    def predict_curriculum(state: np.ndarray, raw: np.ndarray, ee: np.ndarray) -> np.ndarray:
        features = conditioned_features(state, ee)
        z = ee_to_z.predict(features)
        prior = latent_dynamics.predict(conditioned_features(state, z))
        return prior + correction.predict(features)

    bundles = {
        "raw_joint": MethodBundle("raw_joint", predict_raw),
        "canonical_ee": MethodBundle("canonical_ee", predict_ee),
        "learned_latent": MethodBundle("learned_latent", predict_latent),
        "latent_to_ee_curriculum": MethodBundle("latent_to_ee_curriculum", predict_curriculum),
    }
    diagnostics = {
        "latent_explained_variance": float(pca.explained_variance_ratio.sum()),
        "latent_components": pca.components.tolist(),
        "source_labeled_transitions": len(source_labeled),
        "source_total_transitions": len(source),
    }
    return bundles, diagnostics


def evaluate_prediction(bundle: MethodBundle, test: Dataset) -> dict[str, float]:
    pred = test.state + bundle.predict_delta(test.state, test.raw, test.ee)
    error = pred - test.next_state
    pos_error = np.linalg.norm(error[:, :2], axis=1)
    full_error = np.linalg.norm(error, axis=1)
    return {
        "prediction_rmse": float(np.sqrt(np.mean(error * error))),
        "position_rmse": float(np.sqrt(np.mean(error[:, :2] ** 2))),
        "mean_state_l2": float(full_error.mean()),
        "p90_position_l2": float(np.quantile(pos_error, 0.90)),
    }


def planning_bank(cfg: Config, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(stable_seed(seed, "planning-bank"))
    target = embodiment_specs()[2]
    states = np.concatenate([
        rng.uniform(-0.95, 0.95, (cfg.planning_queries, 2)),
        rng.uniform(-0.35, 0.35, (cfg.planning_queries, 2)),
    ], axis=1)
    raws = rng.uniform(-1.0, 1.0, (cfg.planning_queries, cfg.planning_candidates, 4))
    ees = raw_to_ee(raws.reshape(-1, 4), target).reshape(cfg.planning_queries, cfg.planning_candidates, 2)
    tiled_states = np.repeat(states[:, None, :], cfg.planning_candidates, axis=1)
    true_next = object_step(tiled_states.reshape(-1, 4), ees.reshape(-1, 2)).reshape(
        cfg.planning_queries, cfg.planning_candidates, 4
    )
    desired_ee = rng.uniform(-0.95, 0.95, (cfg.planning_queries, 2))
    goals = object_step(states, desired_ee)[:, :2]
    return {"state": states, "raw": raws, "ee": ees, "true_next": true_next, "goal": goals}


def evaluate_planning(bundle: MethodBundle, bank: dict[str, np.ndarray], cfg: Config) -> tuple[dict[str, float], np.ndarray]:
    q, k = bank["raw"].shape[:2]
    state = np.repeat(bank["state"][:, None, :], k, axis=1).reshape(-1, 4)
    raw = bank["raw"].reshape(-1, 4)
    ee = bank["ee"].reshape(-1, 2)
    pred_next = (state + bundle.predict_delta(state, raw, ee)).reshape(q, k, 4)
    true_cost = np.linalg.norm(bank["true_next"][:, :, :2] - bank["goal"][:, None, :], axis=2)
    pred_cost = np.linalg.norm(pred_next[:, :, :2] - bank["goal"][:, None, :], axis=2)
    oracle = np.argmin(true_cost, axis=1)
    chosen = np.argmin(pred_cost, axis=1)
    rows = np.arange(q)
    regret = true_cost[rows, chosen] - true_cost[rows, oracle]
    return {
        "candidate_top1": float(np.mean(chosen == oracle)),
        "candidate_regret": float(np.mean(regret)),
        "candidate_regret_p90": float(np.quantile(regret, 0.90)),
        "planning_success": float(np.mean(regret <= cfg.planning_regret_threshold)),
    }, chosen


def sem(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "prediction_rmse", "position_rmse", "mean_state_l2", "p90_position_l2",
        "candidate_top1", "candidate_regret", "candidate_regret_p90", "planning_success",
    ]
    output: list[dict[str, Any]] = []
    for shots in sorted({int(row["shots"]) for row in rows}):
        for method in METHODS:
            group = [row for row in rows if row["method"] == method and int(row["shots"]) == shots]
            record: dict[str, Any] = {"method": method, "label": LABELS[method], "shots": shots, "seeds": len(group)}
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in group])
                record[f"{metric}_mean"] = float(values.mean())
                record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                record[f"{metric}_sem"] = sem(values)
            output.append(record)
    return output


def run_sanity_checks(cfg: Config, seed: int) -> dict[str, Any]:
    specs = embodiment_specs()
    shared_state = np.array([[0.22, -0.31, 0.08, -0.04]])
    shared_ee = np.array([[0.55, -0.35]])
    next_a = object_step(shared_state, shared_ee)
    next_b = object_step(shared_state, shared_ee)
    same_raw = np.array([[0.45, -0.30, 0.0, 0.0]])
    ee_a = raw_to_ee(same_raw, specs[0])
    ee_b = raw_to_ee(same_raw, specs[1])
    mini1 = generate_dataset(0, 2, 4, seed)
    mini2 = generate_dataset(0, 2, 4, seed)
    checks = {
        "shared_physics_exact": bool(np.array_equal(next_a, next_b)),
        "same_raw_different_effect": bool(np.linalg.norm(ee_a - ee_b) > 0.25),
        "dataset_deterministic": bool(np.array_equal(mini1.next_state, mini2.next_state)),
        "finite_dynamics": bool(np.isfinite(mini1.next_state).all()),
        "target_raw_dimension_distinct": specs[2].raw_dim == 4 and specs[0].raw_dim != specs[2].raw_dim,
        "shot_schedule_monotonic": list(cfg.shots) == sorted(set(cfg.shots)),
        "planning_candidates_nontrivial": cfg.planning_candidates >= 4,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "same_raw_effect_distance": float(np.linalg.norm(ee_a - ee_b)),
        "shared_physics_max_difference": float(np.max(np.abs(next_a - next_b))),
    }


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_sweep(summary: list[dict[str, Any]], cfg: Config, output_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    specifications = [
        ("prediction_rmse", "One-step state RMSE ↓"),
        ("candidate_top1", "Candidate top-1 agreement ↑"),
        ("candidate_regret", "Candidate regret ↓"),
        ("planning_success", f"Planning success (regret ≤ {cfg.planning_regret_threshold:g}) ↑"),
    ]
    for ax, (metric, title) in zip(axes.flat, specifications):
        for method in METHODS:
            rows = sorted([r for r in summary if r["method"] == method], key=lambda r: r["shots"])
            x = np.asarray([r["shots"] for r in rows], dtype=float)
            y = np.asarray([r[f"{metric}_mean"] for r in rows], dtype=float)
            err = np.asarray([r[f"{metric}_sem"] for r in rows], dtype=float)
            ax.errorbar(x, y, yerr=err, marker="o", linewidth=2, capsize=3, color=COLORS[method], label=LABELS[method])
        ax.set_title(title)
        ax.set_xlabel("Target demonstration episodes")
        ax.set_xticks(cfg.shots)
        ax.grid(alpha=0.25)
    axes[0, 1].legend(frameon=False, fontsize=9, ncol=2)
    fig.suptitle("Cross-embodiment target adaptation and candidate reranking", fontsize=14)
    fig.savefig(output_dir / "few_shot_sweep.png", dpi=180)
    plt.close(fig)


def plot_diagnostics(source: Dataset, target: Dataset, diagnostics: dict[str, Any], output_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
    for embodiment_id, label, color in [(0, "source A", "#4c78a8"), (1, "source B", "#54a24b"), (2, "target", "#e45756")]:
        data = source if embodiment_id < 2 else target
        idx = np.flatnonzero(data.embodiment == embodiment_id)[:260]
        axes[0].scatter(data.raw[idx, 0], data.ee[idx, 0], s=10, alpha=0.45, label=label, color=color)
    axes[0].set_xlabel("Raw control coordinate 1")
    axes[0].set_ylabel("Canonical EE x")
    axes[0].set_title("Different raw-control semantics")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2)

    residual = source.delta - fit_linear(state_features(source.state), source.delta, 2.0e-3).predict(state_features(source.state))
    pca = fit_pca(residual, 2)
    z = pca.transform(residual)
    for embodiment_id, label, color in [(0, "source A", "#4c78a8"), (1, "source B", "#54a24b")]:
        idx = np.flatnonzero(source.embodiment == embodiment_id)[::8]
        axes[1].scatter(z[idx, 0], z[idx, 1], c=source.ee[idx, 0], cmap="coolwarm", vmin=-1, vmax=1, s=9, alpha=0.55, label=label)
    axes[1].set_xlabel("Latent action z₁")
    axes[1].set_ylabel("Latent action z₂")
    axes[1].set_title(f"Shared transition latent ({diagnostics['latent_explained_variance']:.1%} residual variance)")
    axes[1].grid(alpha=0.2)

    samples = np.arange(min(80, len(target)))
    axes[2].quiver(
        target.state[samples, 0], target.state[samples, 1],
        target.next_state[samples, 0] - target.state[samples, 0],
        target.next_state[samples, 1] - target.state[samples, 1],
        target.ee[samples, 0], cmap="coolwarm", angles="xy", scale_units="xy", scale=1.0, width=0.005,
    )
    axes[2].set_xlabel("Object x")
    axes[2].set_ylabel("Object y")
    axes[2].set_title("Target embodiment, shared object motion")
    axes[2].grid(alpha=0.2)
    fig.savefig(output_dir / "representation_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_planning_example(
    bank: dict[str, np.ndarray], chosen_by_method: dict[str, np.ndarray], output_dir: pathlib.Path, query: int = 3
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.2), constrained_layout=True)
    true_pos = bank["true_next"][query, :, :2]
    goal = bank["goal"][query]
    true_cost = np.linalg.norm(true_pos - goal[None], axis=1)
    oracle = int(np.argmin(true_cost))
    for ax, method in zip(axes, METHODS):
        chosen = int(chosen_by_method[method][query])
        ax.scatter(true_pos[:, 0], true_pos[:, 1], c=true_cost, cmap="viridis_r", s=40, alpha=0.75)
        ax.scatter(goal[0], goal[1], marker="*", s=150, color="black", label="goal")
        ax.scatter(true_pos[oracle, 0], true_pos[oracle, 1], marker="o", facecolors="none", edgecolors="#54a24b", s=150, linewidths=2, label="oracle")
        ax.scatter(true_pos[chosen, 0], true_pos[chosen, 1], marker="x", color=COLORS[method], s=100, linewidths=2, label="chosen")
        ax.set_title(f"{LABELS[method]}\nregret={true_cost[chosen]-true_cost[oracle]:.4f}")
        ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="datalim")
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("One held-out candidate-reranking query (true target outcomes shown)")
    fig.savefig(output_dir / "candidate_reranking_example.png", dpi=180)
    plt.close(fig)


def run_experiment(cfg: Config, output_dir: pathlib.Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sanity = run_sanity_checks(cfg, cfg.seed)
    if not sanity["passed"]:
        raise RuntimeError(f"Sanity checks failed: {sanity}")

    per_seed: list[dict[str, Any]] = []
    diagnostics_by_seed: list[dict[str, Any]] = []
    representative: dict[str, Any] = {}
    for seed_index in range(cfg.seeds):
        seed = cfg.seed + seed_index
        source = concatenate([
            generate_dataset(0, cfg.source_episodes, cfg.source_horizon, seed),
            generate_dataset(1, cfg.source_episodes, cfg.source_horizon, seed),
        ])
        target_pool = generate_dataset(2, cfg.target_pool_episodes, cfg.target_horizon, seed, episode_offset=10_000)
        target_test = generate_dataset(
            2,
            math.ceil(cfg.test_transitions / cfg.target_horizon),
            cfg.target_horizon,
            seed + 50_000,
            episode_offset=20_000,
        ).subset(np.arange(cfg.test_transitions))
        source_labeled = source_label_subset(source, cfg.source_ee_label_fraction, seed)
        bank = planning_bank(cfg, seed)

        for shots in cfg.shots:
            demo = target_demo_subset(target_pool, shots)
            bundles, diagnostics = build_models(source, source_labeled, demo, cfg)
            diagnostics_by_seed.append({"seed": seed, "shots": shots, **diagnostics})
            chosen_by_method: dict[str, np.ndarray] = {}
            for method in METHODS:
                prediction = evaluate_prediction(bundles[method], target_test)
                planning, chosen = evaluate_planning(bundles[method], bank, cfg)
                chosen_by_method[method] = chosen
                per_seed.append({
                    "seed": seed,
                    "shots": shots,
                    "target_demo_transitions": len(demo),
                    "method": method,
                    "label": LABELS[method],
                    **prediction,
                    **planning,
                })
            if seed_index == 0 and shots == 4:
                representative = {
                    "source": source,
                    "target": target_test,
                    "diagnostics": diagnostics,
                    "bank": bank,
                    "chosen": chosen_by_method,
                }

    summary = aggregate_rows(per_seed)
    write_csv(output_dir / "per_seed_metrics.csv", per_seed)
    write_csv(output_dir / "summary_metrics.csv", summary)
    write_csv(output_dir / "latent_diagnostics.csv", diagnostics_by_seed)
    (output_dir / "sanity_check.json").write_text(json.dumps(sanity, indent=2) + "\n")
    plot_sweep(summary, cfg, output_dir)
    plot_diagnostics(representative["source"], representative["target"], representative["diagnostics"], output_dir)
    plot_planning_example(representative["bank"], representative["chosen"], output_dir)

    headline_shots = 4
    headline = [row for row in summary if int(row["shots"]) == headline_shots]
    metrics = {
        "experiment": "cross_embodiment_world_model",
        "scope": "deterministic vector-state mechanism test; not a CLAP reproduction",
        "config": asdict(cfg),
        "method_definitions": {
            "raw_joint": "Padded raw controls with embodiment-specific interaction features; target branch learned from few demonstrations.",
            "canonical_ee": "Direct world model conditioned on exact two-dimensional canonical end-effector actions, trained only on EE-labeled source data plus target demonstrations.",
            "learned_latent": "PCA proxy actions inferred from transitions; target raw-to-latent alignment learned from few demonstrations.",
            "latent_to_ee_curriculum": "All-source latent dynamics pretraining, then EE-to-latent grounding and a conservative labeled residual correction.",
        },
        "headline_shots": headline_shots,
        "headline": headline,
        "sanity": sanity,
        "latent_explained_variance_mean": float(np.mean([row["latent_explained_variance"] for row in diagnostics_by_seed])),
        "all_summary_rows": summary,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--seeds", type=int, default=Config.seeds)
    parser.add_argument("--source-episodes", type=int, default=Config.source_episodes)
    parser.add_argument("--target-pool-episodes", type=int, default=Config.target_pool_episodes)
    parser.add_argument("--test-transitions", type=int, default=Config.test_transitions)
    parser.add_argument("--planning-queries", type=int, default=Config.planning_queries)
    parser.add_argument("--shots", type=int, nargs="+", default=list(Config.shots))
    parser.add_argument("--source-ee-label-fraction", type=float, default=Config.source_ee_label_fraction)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = replace(
        Config(),
        seed=args.seed,
        seeds=args.seeds,
        source_episodes=args.source_episodes,
        target_pool_episodes=args.target_pool_episodes,
        test_transitions=args.test_transitions,
        planning_queries=args.planning_queries,
        shots=tuple(args.shots),
        source_ee_label_fraction=args.source_ee_label_fraction,
    )
    if args.smoke:
        cfg = replace(
            cfg,
            seeds=min(cfg.seeds, 2),
            source_episodes=min(cfg.source_episodes, 24),
            target_pool_episodes=min(cfg.target_pool_episodes, 8),
            test_transitions=min(cfg.test_transitions, 120),
            planning_queries=min(cfg.planning_queries, 50),
            shots=tuple(x for x in cfg.shots if x <= 4) or (0, 1, 4),
        )
    metrics = run_experiment(cfg, args.output_dir)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "sanity_passed": metrics["sanity"]["passed"],
        "headline_shots": metrics["headline_shots"],
        "headline": [
            {
                "method": row["method"],
                "prediction_rmse": round(row["prediction_rmse_mean"], 6),
                "candidate_top1": round(row["candidate_top1_mean"], 4),
                "candidate_regret": round(row["candidate_regret_mean"], 6),
            }
            for row in metrics["headline"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
