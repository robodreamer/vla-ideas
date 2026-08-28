#!/usr/bin/env python3
"""Conflict-aware experience replay in a deterministic sequential-control toy.

This is a mechanism test inspired by Memory Anchors, not a reproduction of the
paper's diffusion policies, LIBERO experiments, or real-robot systems. Four
weakly distinguished tasks share physical observation support but require
conflicting steering actions around the same decision region. We compare no
replay, random replay, MIR-style loss-hard mining, latent diversity, and a
three-step latent-overlap/action-disagreement anchor selector.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import pathlib
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable

BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from torch import nn


METHODS = ("no_replay", "random", "loss_hard", "diversity", "anchors")
REPLAY_METHODS = METHODS[1:]
LABELS = {
    "no_replay": "No replay",
    "random": "Random replay",
    "loss_hard": "Loss-hard (probe)",
    "diversity": "Latent diversity",
    "anchors": "Conflict anchors",
}
COLORS = {
    "no_replay": "#7f7f7f",
    "random": "#4c78a8",
    "loss_hard": "#f58518",
    "diversity": "#54a24b",
    "anchors": "#e45756",
}


@dataclass(frozen=True)
class Config:
    seed: int = 17
    seeds: int = 6
    tasks: int = 4
    train_episodes: int = 12
    val_episodes: int = 5
    horizon: int = 49
    dt: float = 0.055
    demo_start_std: float = 0.045
    eval_start_std: float = 0.115
    demo_action_std: float = 0.025
    action_limit: float = 2.4
    feedback_gain: float = 3.2
    hidden_dim: int = 48
    initial_steps: int = 650
    train_steps: int = 480
    probe_steps: int = 24
    learning_rate: float = 0.003
    weight_decay: float = 1.0e-5
    batch_size: int = 64
    replay_fraction: float = 0.25
    eval_rollouts: int = 32
    buffer_sizes: tuple[int, ...] = (12, 32, 96, 288)
    anchor_fraction: float = 0.20
    anchor_knn: int = 5
    success_tracking_rmse: float = 0.23
    success_branch_error: float = 0.25
    success_final_error: float = 0.17


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    name: str
    cue: float
    amplitude: float


@dataclass
class Dataset:
    obs: np.ndarray
    action: np.ndarray
    task: np.ndarray
    phase: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return int(len(self.obs))

    def subset(self, indices: np.ndarray | list[int]) -> "Dataset":
        idx = np.asarray(indices, dtype=int)
        return Dataset(self.obs[idx], self.action[idx], self.task[idx], self.phase[idx], self.y[idx])


class Policy(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(obs))


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**32)


def task_specs(cfg: Config) -> list[TaskSpec]:
    base = [
        ("upper-wide", -0.60, 0.82),
        ("lower-wide", -0.20, -0.82),
        ("upper-tight", 0.20, 0.60),
        ("lower-tight", 0.60, -0.60),
    ]
    if cfg.tasks > len(base):
        raise ValueError(f"tasks must be <= {len(base)}")
    return [TaskSpec(i, *base[i]) for i in range(cfg.tasks)]


def route_profile(phase: np.ndarray | float, amplitude: float) -> tuple[np.ndarray, np.ndarray]:
    """Smooth route and dy/dphase; tasks differ mostly around the decision arc."""
    u = np.asarray(phase, dtype=float)
    envelope = np.sin(np.pi * u) ** 2
    y = amplitude * envelope
    dy_du = amplitude * np.pi * np.sin(2.0 * np.pi * u)
    return y, dy_du


def observation(phase: float | np.ndarray, y: float | np.ndarray, cue: float) -> np.ndarray:
    u = np.asarray(phase, dtype=float)
    yy = np.asarray(y, dtype=float)
    # The cue is low-dimensional and the physical state dominates most of the
    # trajectory. Before seeing a new cue, the encoder can place it near old
    # tasks with the same physical state.
    return np.stack(
        [2.0 * u - 1.0, yy, np.sin(np.pi * u), np.cos(np.pi * u), np.full_like(u, cue)],
        axis=-1,
    )


def expert_action(phase: float, y: float, task: TaskSpec, cfg: Config) -> float:
    target_y, dy_du = route_profile(phase, task.amplitude)
    phase_rate = 1.0 / ((cfg.horizon - 1) * cfg.dt)
    raw = float(dy_du) * phase_rate + cfg.feedback_gain * (float(target_y) - y)
    return float(np.clip(raw, -cfg.action_limit, cfg.action_limit))


def generate_dataset(task: TaskSpec, cfg: Config, seed: int, episodes: int, noisy: bool) -> Dataset:
    obs_rows: list[np.ndarray] = []
    actions: list[float] = []
    task_ids: list[int] = []
    phases: list[float] = []
    ys: list[float] = []
    for episode in range(episodes):
        rng = np.random.default_rng(stable_seed(seed, task.task_id, "episode", episode, noisy))
        y = float(rng.normal(0.0, cfg.demo_start_std))
        noise_state = float(rng.normal(0.0, cfg.demo_action_std))
        for step in range(cfg.horizon):
            u = step / (cfg.horizon - 1)
            action = expert_action(u, y, task, cfg)
            if noisy:
                noise_state = 0.75 * noise_state + float(rng.normal(0.0, cfg.demo_action_std * 0.66))
                action = float(np.clip(action + noise_state, -cfg.action_limit, cfg.action_limit))
            obs_rows.append(observation(u, y, task.cue))
            actions.append(action)
            task_ids.append(task.task_id)
            phases.append(u)
            ys.append(y)
            if step + 1 < cfg.horizon:
                y += cfg.dt * action
    return Dataset(
        obs=np.asarray(obs_rows, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32)[:, None],
        task=np.asarray(task_ids, dtype=np.int64),
        phase=np.asarray(phases, dtype=np.float32),
        y=np.asarray(ys, dtype=np.float32),
    )


def concat_datasets(parts: Iterable[Dataset]) -> Dataset:
    items = list(parts)
    return Dataset(
        obs=np.concatenate([x.obs for x in items], axis=0),
        action=np.concatenate([x.action for x in items], axis=0),
        task=np.concatenate([x.task for x in items], axis=0),
        phase=np.concatenate([x.phase for x in items], axis=0),
        y=np.concatenate([x.y for x in items], axis=0),
    )


def predict(model: Policy, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(obs.astype(np.float32))).cpu().numpy()


def encode(model: Policy, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model.encode(torch.from_numpy(obs.astype(np.float32))).cpu().numpy()


def train_stage(
    model: Policy,
    current: Dataset,
    replay: Dataset | None,
    cfg: Config,
    steps: int,
    seed: int,
) -> list[float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(seed)
    replay_n = 0 if replay is None else max(1, round(cfg.batch_size * cfg.replay_fraction))
    current_n = cfg.batch_size - replay_n
    losses: list[float] = []
    x_cur = torch.from_numpy(current.obs)
    y_cur = torch.from_numpy(current.action)
    x_rep = None if replay is None else torch.from_numpy(replay.obs)
    y_rep = None if replay is None else torch.from_numpy(replay.action)
    for step in range(steps):
        cur_idx = rng.integers(0, len(current), size=current_n)
        xb = x_cur[cur_idx]
        yb = y_cur[cur_idx]
        if replay is not None and x_rep is not None and y_rep is not None:
            rep_idx = rng.integers(0, len(replay), size=replay_n)
            xb = torch.cat([xb, x_rep[rep_idx]], dim=0)
            yb = torch.cat([yb, y_rep[rep_idx]], dim=0)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(xb) - yb) ** 2)
        loss.backward()
        optimizer.step()
        if step % max(1, steps // 25) == 0 or step == steps - 1:
            losses.append(float(loss.detach()))
    model.eval()
    return losses


def farthest_first(latents: np.ndarray) -> np.ndarray:
    """Deterministic full ranking by greedy farthest-first traversal."""
    n = len(latents)
    if n == 0:
        return np.empty(0, dtype=int)
    center = np.mean(latents, axis=0, keepdims=True)
    first = int(np.argmax(np.sum((latents - center) ** 2, axis=1)))
    order = np.empty(n, dtype=int)
    order[0] = first
    chosen = np.zeros(n, dtype=bool)
    chosen[first] = True
    min_dist = np.sum((latents - latents[first]) ** 2, axis=1)
    for i in range(1, n):
        min_dist[chosen] = -1.0
        nxt = int(np.argmax(min_dist))
        order[i] = nxt
        chosen[nxt] = True
        dist = np.sum((latents - latents[nxt]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
    return order


def overlap_features(new_lat: np.ndarray, old_lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = pairwise_distances(new_lat, old_lat, metric="euclidean", n_jobs=1)
    sorted_d = np.sort(distances, axis=1)
    n_old = sorted_d.shape[1]
    indices = [0, max(0, math.ceil(0.01 * n_old) - 1), max(0, math.ceil(0.05 * n_old) - 1), max(0, math.ceil(0.10 * n_old) - 1)]
    features = sorted_d[:, indices]
    return features, distances


def anchor_context(
    model: Policy,
    past_train: Dataset,
    past_val: Dataset,
    current: Dataset,
    cfg: Config,
    seed: int,
) -> dict[str, Any]:
    old_lat = encode(model, past_train.obs)
    new_lat = encode(model, current.obs)
    features, _ = overlap_features(new_lat, old_lat)
    scaled = (features - features.mean(axis=0, keepdims=True)) / (features.std(axis=0, keepdims=True) + 1e-6)
    km = KMeans(n_clusters=2, random_state=seed, n_init=10).fit(scaled)
    means = [float(np.mean(features[km.labels_ == cluster])) for cluster in range(2)]
    overlap_cluster = int(np.argmin(means))
    overlap_mask = km.labels_ == overlap_cluster

    old_val_disagreement = np.abs(predict(model, past_val.obs)[:, 0] - past_val.action[:, 0])
    threshold = float(np.mean(old_val_disagreement) + 2.0 * np.std(old_val_disagreement))
    new_disagreement = np.abs(predict(model, current.obs)[:, 0] - current.action[:, 0])
    conflict_mask = overlap_mask & (new_disagreement > threshold)
    minimum = max(4, int(round(0.04 * len(current))))
    if int(np.sum(conflict_mask)) < minimum:
        overlap_idx = np.flatnonzero(overlap_mask)
        rank = overlap_idx[np.argsort(-new_disagreement[overlap_idx])]
        conflict_mask[rank[:minimum]] = True
    conflict_lat = new_lat[conflict_mask]

    distances = pairwise_distances(old_lat, conflict_lat, metric="euclidean", n_jobs=1)
    k = min(cfg.anchor_knn, distances.shape[1])
    nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
    anchor_score = np.median(nearest, axis=1)
    anchor_rank = np.argsort(anchor_score, kind="stable")
    diversity_rank = farthest_first(old_lat)
    return {
        "old_lat": old_lat,
        "new_lat": new_lat,
        "overlap_features": features,
        "overlap_mask": overlap_mask,
        "new_disagreement": new_disagreement,
        "threshold": threshold,
        "conflict_mask": conflict_mask,
        "anchor_score": anchor_score,
        "anchor_rank": anchor_rank,
        "diversity_rank": diversity_rank,
        "anchor_fraction": cfg.anchor_fraction,
    }


def loss_hard_rank(
    model: Policy,
    past: Dataset,
    current: Dataset,
    cfg: Config,
    seed: int,
) -> np.ndarray:
    """MIR-style hard mining: old samples whose loss spikes after a new-task probe."""
    probe = copy.deepcopy(model)
    before = (predict(probe, past.obs)[:, 0] - past.action[:, 0]) ** 2
    probe_cfg = replace(cfg, replay_fraction=0.0, learning_rate=cfg.learning_rate * 0.8)
    train_stage(probe, current, None, probe_cfg, cfg.probe_steps, stable_seed(seed, "probe"))
    after = (predict(probe, past.obs)[:, 0] - past.action[:, 0]) ** 2
    return np.argsort(-(after - before), kind="stable")


def select_buffer(
    method: str,
    size: int,
    context: dict[str, Any],
    loss_rank: np.ndarray,
    seed: int,
) -> np.ndarray:
    n = len(context["anchor_rank"])
    size = min(size, n)
    if method == "random":
        return np.random.default_rng(seed).permutation(n)[:size]
    if method == "loss_hard":
        return loss_rank[:size]
    if method == "diversity":
        return context["diversity_rank"][:size]
    if method == "anchors":
        quota = min(size, max(1, int(round(size * float(context["anchor_fraction"])))))
        selected = list(context["anchor_rank"][:quota])
        used = set(selected)
        random_remainder = np.random.default_rng(seed).permutation(n)
        for idx in random_remainder:
            if int(idx) not in used:
                selected.append(int(idx))
                used.add(int(idx))
                if len(selected) == size:
                    break
        return np.asarray(selected, dtype=int)
    raise ValueError(method)


def selection_metrics(
    selected: np.ndarray,
    context: dict[str, Any],
    past: Dataset,
    method: str,
    size: int,
    seed: int,
    stage: int,
) -> dict[str, Any]:
    top_count = max(1, int(math.ceil(0.10 * len(past))))
    top_anchor = set(int(x) for x in context["anchor_rank"][:top_count])
    concentration = float(np.mean([int(int(i) in top_anchor) for i in selected])) if len(selected) else 0.0
    return {
        "seed": seed,
        "method": method,
        "buffer_size": size,
        "train_stage": stage,
        "selected_count": int(len(selected)),
        "anchor_concentration": concentration,
        "mean_anchor_score": float(np.mean(context["anchor_score"][selected])),
        "selected_early_fraction": float(np.mean(past.phase[selected] < 0.38)),
        "selected_near_center_fraction": float(np.mean(np.abs(past.y[selected]) < 0.32)),
        "overlap_fraction": float(np.mean(context["overlap_mask"])),
        "conflict_fraction": float(np.mean(context["conflict_mask"])),
        "disagreement_threshold": float(context["threshold"]),
    }


def rollout(
    model: Policy,
    task: TaskSpec,
    cfg: Config,
    start_y: float,
) -> dict[str, Any]:
    y = float(start_y)
    ys: list[float] = []
    targets: list[float] = []
    actions: list[float] = []
    oracle_actions: list[float] = []
    for step in range(cfg.horizon):
        phase = step / (cfg.horizon - 1)
        target, _ = route_profile(phase, task.amplitude)
        obs = observation(phase, y, task.cue).astype(np.float32)[None]
        action = float(np.clip(predict(model, obs)[0, 0], -cfg.action_limit, cfg.action_limit))
        ys.append(y)
        targets.append(float(target))
        actions.append(action)
        oracle_actions.append(expert_action(phase, y, task, cfg))
        if step + 1 < cfg.horizon:
            y += cfg.dt * action
    ys_arr = np.asarray(ys)
    target_arr = np.asarray(targets)
    branch_step = int(round(0.50 * (cfg.horizon - 1)))
    tracking = float(np.sqrt(np.mean((ys_arr - target_arr) ** 2)))
    branch_error = float(abs(ys_arr[branch_step] - target_arr[branch_step]))
    final_error = float(abs(ys_arr[-1] - target_arr[-1]))
    action_mse = float(np.mean((np.asarray(actions) - np.asarray(oracle_actions)) ** 2))
    success = (
        tracking < cfg.success_tracking_rmse
        and branch_error < cfg.success_branch_error
        and final_error < cfg.success_final_error
    )
    return {
        "success": float(success),
        "tracking_rmse": tracking,
        "branch_error": branch_error,
        "final_error": final_error,
        "on_policy_action_mse": action_mse,
        "ys": ys_arr,
        "targets": target_arr,
        "actions": np.asarray(actions),
    }


def evaluate_task(model: Policy, task: TaskSpec, cfg: Config, seed: int) -> dict[str, float]:
    rows = []
    rng = np.random.default_rng(seed)
    starts = rng.normal(0.0, cfg.eval_start_std, size=cfg.eval_rollouts)
    for start in starts:
        rows.append(rollout(model, task, cfg, float(start)))
    keys = ("success", "tracking_rmse", "branch_error", "final_error", "on_policy_action_mse")
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def continual_metrics(stage_rows: list[dict[str, Any]], tasks: int) -> dict[str, float]:
    success = {(int(row["eval_task"]), int(row["train_stage"])): float(row["success"]) for row in stage_rows}
    final_rows = [row for row in stage_rows if int(row["train_stage"]) == tasks - 1]
    rmse_final = [float(row["tracking_rmse"]) for row in final_rows]
    action_mse_final = [float(row["on_policy_action_mse"]) for row in final_rows]
    final_success = [success[(task, tasks - 1)] for task in range(tasks)]
    nbt_per_task = []
    final_forgetting = []
    aucs = []
    plasticity = []
    for task in range(tasks):
        learned = success[(task, task)]
        plasticity.append(learned)
        trajectory = [success[(task, stage)] for stage in range(task, tasks)]
        aucs.append(float(np.mean(trajectory)))
        if task < tasks - 1:
            nbt_per_task.append(float(np.mean([learned - success[(task, stage)] for stage in range(task + 1, tasks)])))
            final_forgetting.append(learned - success[(task, tasks - 1)])
    interaction_drops = [
        success[(task, stage - 1)] - success[(task, stage)]
        for stage in range(1, tasks)
        for task in range(stage)
    ]
    return {
        "final_avg_success": float(np.mean(final_success)),
        "final_old_success": float(np.mean(final_success[:-1])) if tasks > 1 else final_success[0],
        "final_current_success": final_success[-1],
        "final_tracking_rmse": float(np.mean(rmse_final)),
        "final_on_policy_action_mse": float(np.mean(action_mse_final)),
        "nbt": float(np.mean(nbt_per_task)) if nbt_per_task else 0.0,
        "mean_interaction_drop": float(np.mean(interaction_drops)) if interaction_drops else 0.0,
        "worst_interaction_drop": float(np.max(interaction_drops)) if interaction_drops else 0.0,
        "final_forgetting": float(np.mean(final_forgetting)) if final_forgetting else 0.0,
        "auc": float(np.mean(aucs)),
        "plasticity": float(np.mean(plasticity)),
    }


def run_condition(
    method: str,
    buffer_size: int,
    run_seed: int,
    cfg: Config,
    specs: list[TaskSpec],
    train_sets: list[Dataset],
    val_sets: list[Dataset],
    capture: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    torch.manual_seed(stable_seed(cfg.seed, "model-init", run_seed))
    model = Policy(train_sets[0].obs.shape[1], cfg.hidden_dim)
    stage_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    capture_data: dict[str, Any] = {}
    for stage in range(cfg.tasks):
        replay = None
        selected = np.empty(0, dtype=int)
        context = None
        if stage > 0 and method != "no_replay":
            past_train = concat_datasets(train_sets[:stage])
            past_val = concat_datasets(val_sets[:stage])
            context = anchor_context(
                model,
                past_train,
                past_val,
                train_sets[stage],
                cfg,
                stable_seed(run_seed, method, buffer_size, stage, "anchors"),
            )
            loss_rank = loss_hard_rank(
                model,
                past_train,
                train_sets[stage],
                cfg,
                stable_seed(run_seed, method, buffer_size, stage, "loss-hard"),
            )
            selected = select_buffer(
                method,
                buffer_size,
                context,
                loss_rank,
                stable_seed(run_seed, method, buffer_size, stage, "select"),
            )
            replay = past_train.subset(selected)
            selection_rows.append(selection_metrics(selected, context, past_train, method, buffer_size, run_seed, stage))
            if capture and stage == cfg.tasks - 1:
                candidate_indices = {
                    name: select_buffer(
                        name,
                        buffer_size,
                        context,
                        loss_rank,
                        stable_seed(run_seed, "diagnostic", name, buffer_size, stage),
                    )
                    for name in REPLAY_METHODS
                }
                capture_data["selector"] = {
                    "past": past_train,
                    "current": train_sets[stage],
                    "context": context,
                    "indices": candidate_indices,
                }
        steps = cfg.initial_steps if stage == 0 else cfg.train_steps
        train_stage(
            model,
            train_sets[stage],
            replay,
            cfg,
            steps,
            stable_seed(run_seed, method, buffer_size, stage, "train"),
        )
        for eval_task in range(stage + 1):
            values = evaluate_task(
                model,
                specs[eval_task],
                cfg,
                stable_seed(cfg.seed, run_seed, eval_task, "eval"),
            )
            stage_rows.append(
                {
                    "seed": run_seed,
                    "method": method,
                    "buffer_size": buffer_size,
                    "train_stage": stage,
                    "eval_task": eval_task,
                    **values,
                }
            )
    condition = continual_metrics(stage_rows, cfg.tasks)
    if selection_rows:
        condition["anchor_concentration"] = float(np.mean([x["anchor_concentration"] for x in selection_rows]))
    else:
        condition["anchor_concentration"] = float("nan")
    if capture:
        capture_data["final_model"] = copy.deepcopy(model)
    return stage_rows, selection_rows, condition, capture_data


def aggregate_rows(condition_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in condition_rows:
        groups.setdefault((row["method"], int(row["buffer_size"])), []).append(row)
    metrics = (
        "final_avg_success",
        "final_old_success",
        "final_current_success",
        "final_tracking_rmse",
        "final_on_policy_action_mse",
        "nbt",
        "mean_interaction_drop",
        "worst_interaction_drop",
        "final_forgetting",
        "auc",
        "plasticity",
        "anchor_concentration",
        "runtime_seconds",
    )
    output: list[dict[str, Any]] = []
    for (method, size), rows in groups.items():
        item: dict[str, Any] = {"method": method, "buffer_size": size, "seeds": len(rows)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows], dtype=float)
            finite = values[np.isfinite(values)]
            item[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
            item[f"{metric}_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
            item[f"{metric}_sem"] = float(np.std(finite, ddof=1) / math.sqrt(len(finite))) if len(finite) > 1 else 0.0
        output.append(item)
    order = {method: i for i, method in enumerate(METHODS)}
    return sorted(output, key=lambda row: (order[row["method"]], int(row["buffer_size"])))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_buffer_sweep(summary: list[dict[str, Any]], cfg: Config, out: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    panels = [
        ("final_avg_success", "Final average success", (0.0, 1.02)),
        ("nbt", "Negative backward transfer (lower is better)", None),
        ("final_tracking_rmse", "Final tracking RMSE", None),
        ("auc", "Continual-learning AUC", (0.0, 1.02)),
    ]
    no_replay = next(row for row in summary if row["method"] == "no_replay")
    for ax, (metric, title, ylim) in zip(axes.flat, panels):
        base = float(no_replay[f"{metric}_mean"])
        ax.axhline(base, color=COLORS["no_replay"], linestyle="--", linewidth=1.6, label=LABELS["no_replay"])
        for method in REPLAY_METHODS:
            rows = [row for row in summary if row["method"] == method]
            x = np.asarray([row["buffer_size"] for row in rows], dtype=float)
            y = np.asarray([row[f"{metric}_mean"] for row in rows], dtype=float)
            e = np.asarray([row[f"{metric}_sem"] for row in rows], dtype=float)
            ax.errorbar(x, y, yerr=e, marker="o", linewidth=2, capsize=3, color=COLORS[method], label=LABELS[method])
        ax.set_xscale("log", base=2)
        ax.set_xticks(cfg.buffer_sizes, [str(x) for x in cfg.buffer_sizes])
        ax.set_xlabel("Total replay buffer transitions")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        if ylim is not None:
            ax.set_ylim(*ylim)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Conflict-aware replay buffer sweep (mean ± SEM)", fontsize=14, y=0.985)
    fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.947), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(stage_rows: list[dict[str, Any]], cfg: Config, out: pathlib.Path) -> None:
    chosen_size = cfg.buffer_sizes[1] if len(cfg.buffer_sizes) > 1 else cfg.buffer_sizes[0]
    methods = ("no_replay", "random", "anchors")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), sharey=True, constrained_layout=True)
    for ax, method in zip(axes, methods):
        rows = [row for row in stage_rows if row["method"] == method and (method == "no_replay" or row["buffer_size"] == chosen_size)]
        for task in range(cfg.tasks):
            points = []
            for stage in range(task, cfg.tasks):
                vals = [float(row["success"]) for row in rows if int(row["eval_task"]) == task and int(row["train_stage"]) == stage]
                if vals:
                    points.append((stage, float(np.mean(vals)), float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0))
            if points:
                ax.errorbar([p[0] + 1 for p in points], [p[1] for p in points], yerr=[p[2] for p in points], marker="o", capsize=2, label=f"Task {task + 1}")
        ax.set_title(LABELS[method] + ("" if method == "no_replay" else f" (B={chosen_size})"))
        ax.set_xlabel("Tasks learned")
        ax.set_xticks(range(1, cfg.tasks + 1))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Closed-loop success")
    axes[0].set_ylim(-0.03, 1.03)
    axes[-1].legend(frameon=False, fontsize=9)
    fig.suptitle("Retention trajectories after each sequential training stage")
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_anchor_diagnostics(capture: dict[str, Any], cfg: Config, out: pathlib.Path) -> None:
    selector = capture["selector"]
    past: Dataset = selector["past"]
    current: Dataset = selector["current"]
    context = selector["context"]
    indices = selector["indices"]
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.1), constrained_layout=True)
    ax = axes[0, 0]
    ax.scatter(context["overlap_features"][:, 0], context["new_disagreement"], s=9, alpha=0.35, color="#999999", label="new data")
    mask = context["overlap_mask"]
    ax.scatter(context["overlap_features"][mask, 0], context["new_disagreement"][mask], s=11, alpha=0.45, color="#4c78a8", label="latent overlap")
    conflict = context["conflict_mask"]
    ax.scatter(context["overlap_features"][conflict, 0], context["new_disagreement"][conflict], s=18, color="#e45756", label="action conflict")
    ax.axhline(context["threshold"], linestyle="--", color="black", linewidth=1, label="old-data μ+2σ")
    ax.set_xlabel("Nearest old-latent distance")
    ax.set_ylabel("Old-policy action disagreement")
    ax.set_title("Three-step conflict filter")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    ax.scatter(current.phase, current.y, s=7, alpha=0.25, color="#999999")
    ax.scatter(current.phase[mask], current.y[mask], s=9, alpha=0.35, color="#4c78a8")
    ax.scatter(current.phase[conflict], current.y[conflict], s=17, color="#e45756")
    ax.set_xlabel("Trajectory phase")
    ax.set_ylabel("Physical y")
    ax.set_title("New-task overlap and disagreement")
    ax.grid(alpha=0.2)

    ax = axes[0, 2]
    ax.hist(context["anchor_score"], bins=30, color="#bab0ac", alpha=0.8)
    cutoff_count = max(1, int(math.ceil(0.10 * len(past))))
    cutoff = context["anchor_score"][context["anchor_rank"][cutoff_count - 1]]
    ax.axvline(cutoff, color="#e45756", linestyle="--", label="top 10% anchor cutoff")
    ax.set_xlabel("Median kNN distance to new conflict set")
    ax.set_ylabel("Old transitions")
    ax.set_title("Old-data anchor score")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)

    for ax, method in zip(axes[1], ("random", "diversity", "anchors")):
        ax.scatter(past.phase, past.y, s=5, alpha=0.12, color="#777777")
        idx = indices[method]
        ax.scatter(past.phase[idx], past.y[idx], s=28, alpha=0.9, color=COLORS[method], edgecolors="white", linewidths=0.3)
        ax.set_xlabel("Trajectory phase")
        ax.set_ylabel("Physical y")
        ax.set_title(f"{LABELS[method]} selection")
        ax.grid(alpha=0.2)
    fig.suptitle(f"Representative selector diagnostics at the final task (B={len(indices['anchors'])})", fontsize=14)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_closed_loop(captures: dict[str, dict[str, Any]], specs: list[TaskSpec], cfg: Config, out: pathlib.Path) -> None:
    methods = ("no_replay", "random", "anchors")
    tasks = (0, cfg.tasks - 1)
    fig, axes = plt.subplots(len(tasks), len(methods), figsize=(12.5, 6.8), sharex=True, sharey="row", constrained_layout=True)
    phase = np.linspace(0.0, 1.0, cfg.horizon)
    for row_idx, task_id in enumerate(tasks):
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx, col_idx]
            model: Policy = captures[method]["final_model"]
            result = rollout(model, specs[task_id], cfg, start_y=0.08)
            ax.plot(phase, result["targets"], color="black", linewidth=2.1, label="expert route")
            ax.plot(phase, result["ys"], color=COLORS[method], linewidth=2.0, label="policy")
            ax.axvspan(0.12, 0.42, color="#e45756", alpha=0.08)
            ax.set_title(f"{LABELS[method]} — Task {task_id + 1}")
            ax.set_xlabel("Trajectory phase")
            if col_idx == 0:
                ax.set_ylabel("y position")
            ax.grid(alpha=0.22)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Representative final closed-loop routes: oldest vs newest task")
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


def sanity_checks(cfg: Config, specs: list[TaskSpec], seed: int) -> dict[str, Any]:
    # Exact physical overlap at reset, weak cue separation, and opposite actions.
    obs_a = observation(0.18, 0.0, specs[0].cue)
    obs_b = observation(0.18, 0.0, specs[1].cue)
    act_a = expert_action(0.18, 0.0, specs[0], cfg)
    act_b = expert_action(0.18, 0.0, specs[1], cfg)
    physical_equal = bool(np.allclose(obs_a[:4], obs_b[:4]))
    action_conflict = bool(act_a * act_b < 0 and abs(act_a - act_b) > 1.0)

    trains = [generate_dataset(task, cfg, seed, max(3, min(6, cfg.train_episodes)), True) for task in specs[:2]]
    vals = [generate_dataset(task, cfg, seed + 1, 3, False) for task in specs[:2]]
    torch.manual_seed(stable_seed(seed, "sanity-model"))
    model = Policy(trains[0].obs.shape[1], cfg.hidden_dim)
    train_stage(model, trains[0], None, cfg, min(300, cfg.initial_steps), stable_seed(seed, "sanity-train"))
    context = anchor_context(model, trains[0], vals[0], trains[1], cfg, stable_seed(seed, "sanity-anchor"))
    loss_rank = loss_hard_rank(model, trains[0], trains[1], cfg, stable_seed(seed, "sanity-hard"))
    size = min(12, len(trains[0]))
    selected_a = select_buffer("anchors", size, context, loss_rank, stable_seed(seed, "same"))
    selected_b = select_buffer("anchors", size, context, loss_rank, stable_seed(seed, "same"))
    top_count = max(1, int(math.ceil(0.10 * len(trains[0]))))
    top = set(int(x) for x in context["anchor_rank"][:top_count])
    anchor_concentration = float(np.mean([int(int(i) in top) for i in selected_a]))
    random_values = []
    for i in range(32):
        idx = select_buffer("random", size, context, loss_rank, stable_seed(seed, "sanity-random", i))
        random_values.append(float(np.mean([int(int(j) in top) for j in idx])))
    checks = {
        "same_physical_observation_support": physical_equal,
        "opposite_expert_actions_at_shared_state": action_conflict,
        "overlap_cluster_nonempty": bool(0 < np.sum(context["overlap_mask"]) < len(trains[1])),
        "action_conflict_set_nonempty": bool(np.sum(context["conflict_mask"]) >= 4),
        "anchor_selection_deterministic": bool(np.array_equal(selected_a, selected_b)),
        "buffer_unique_and_exact_size": bool(len(selected_a) == size and len(np.unique(selected_a)) == size),
        "anchor_enrichment_exceeds_random": bool(anchor_concentration > float(np.mean(random_values))),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "diagnostics": {
            "shared_state_observation_l2": float(np.linalg.norm(obs_a - obs_b)),
            "expert_action_task_1": act_a,
            "expert_action_task_2": act_b,
            "overlap_fraction": float(np.mean(context["overlap_mask"])),
            "conflict_fraction": float(np.mean(context["conflict_mask"])),
            "anchor_concentration": anchor_concentration,
            "random_concentration_mean": float(np.mean(random_values)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--train-episodes", type=int, default=12)
    parser.add_argument("--val-episodes", type=int, default=5)
    parser.add_argument("--eval-rollouts", type=int, default=32)
    parser.add_argument("--initial-steps", type=int, default=650)
    parser.add_argument("--train-steps", type=int, default=480)
    parser.add_argument("--buffer-sizes", type=int, nargs="+", default=[12, 32, 96, 288])
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        seed=args.seed,
        seeds=args.seeds,
        train_episodes=args.train_episodes,
        val_episodes=args.val_episodes,
        eval_rollouts=args.eval_rollouts,
        initial_steps=args.initial_steps,
        train_steps=args.train_steps,
        buffer_sizes=tuple(args.buffer_sizes),
    )
    if args.smoke:
        cfg = replace(
            cfg,
            seeds=min(cfg.seeds, 2),
            train_episodes=min(cfg.train_episodes, 6),
            val_episodes=min(cfg.val_episodes, 3),
            eval_rollouts=min(cfg.eval_rollouts, 8),
            initial_steps=min(cfg.initial_steps, 180),
            train_steps=min(cfg.train_steps, 120),
            probe_steps=10,
            buffer_sizes=tuple(x for x in cfg.buffer_sizes if x <= 12) or (4,),
        )
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    np.set_printoptions(precision=5, suppress=True)

    specs = task_specs(cfg)
    sanity = sanity_checks(cfg, specs, cfg.seed)
    if not sanity["passed"]:
        raise RuntimeError(f"Sanity checks failed: {sanity}")

    stage_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    captures: dict[str, dict[str, Any]] = {}
    capture_size = cfg.buffer_sizes[1] if len(cfg.buffer_sizes) > 1 else cfg.buffer_sizes[0]
    for run_index in range(cfg.seeds):
        run_seed = cfg.seed + run_index
        train_sets = [generate_dataset(task, cfg, run_seed, cfg.train_episodes, True) for task in specs]
        val_sets = [generate_dataset(task, cfg, run_seed + 1000, cfg.val_episodes, False) for task in specs]
        conditions = [("no_replay", 0)] + [(method, size) for method in REPLAY_METHODS for size in cfg.buffer_sizes]
        for method, size in conditions:
            capture = run_index == 0 and (method == "no_replay" or size == capture_size) and method in ("no_replay", "random", "anchors")
            started = time.perf_counter()
            rows, select_rows, metrics, capture_data = run_condition(
                method,
                size,
                run_seed,
                cfg,
                specs,
                train_sets,
                val_sets,
                capture,
            )
            elapsed = time.perf_counter() - started
            for row in rows:
                row["runtime_seconds"] = elapsed
            stage_rows.extend(rows)
            selection_rows.extend(select_rows)
            condition_rows.append(
                {
                    "seed": run_seed,
                    "method": method,
                    "buffer_size": size,
                    **metrics,
                    "runtime_seconds": elapsed,
                }
            )
            if capture:
                captures[method] = capture_data
            print(
                f"seed={run_seed} method={method:10s} B={size:3d} "
                f"success={metrics['final_avg_success']:.3f} nbt={metrics['nbt']:.3f} "
                f"rmse={metrics['final_tracking_rmse']:.3f} time={elapsed:.2f}s"
            )

    summary = aggregate_rows(condition_rows)
    write_csv(out / "stage_metrics.csv", stage_rows)
    write_csv(out / "selection_metrics.csv", selection_rows)
    write_csv(out / "condition_metrics.csv", condition_rows)
    write_csv(out / "summary_metrics.csv", summary)
    (out / "sanity_check.json").write_text(json.dumps(sanity, indent=2) + "\n")

    plot_buffer_sweep(summary, cfg, out / "buffer_sweep.png")
    plot_learning_curves(stage_rows, cfg, out / "learning_curves.png")
    if "anchors" in captures and "selector" in captures["anchors"]:
        plot_anchor_diagnostics(captures["anchors"], cfg, out / "anchor_diagnostics.png")
    if all(method in captures for method in ("no_replay", "random", "anchors")):
        plot_closed_loop(captures, specs, cfg, out / "closed_loop_routes.png")

    lookup = {(row["method"], int(row["buffer_size"])): row for row in summary}
    smallest = cfg.buffer_sizes[0]
    largest = cfg.buffer_sizes[-1]
    headline = {
        "no_replay": lookup[("no_replay", 0)],
        "small_buffer_random": lookup[("random", smallest)],
        "small_buffer_anchors": lookup[("anchors", smallest)],
        "largest_buffer_random": lookup[("random", largest)],
        "largest_buffer_anchors": lookup[("anchors", largest)],
    }
    nbt_deltas = {
        str(size): float(lookup[("random", size)]["nbt_mean"] - lookup[("anchors", size)]["nbt_mean"])
        for size in cfg.buffer_sizes
    }
    best_by_size = {
        str(size): min(REPLAY_METHODS, key=lambda method: float(lookup[(method, size)]["nbt_mean"]))
        for size in cfg.buffer_sizes
    }
    claims = {
        "anchor_minus_random_nbt_improvement_by_buffer": nbt_deltas,
        "anchor_improves_nbt_at_n_of_m_buffers": [
            int(sum(delta > 0.0 for delta in nbt_deltas.values())),
            len(cfg.buffer_sizes),
        ],
        "best_mean_nbt_method_by_buffer": best_by_size,
        "scope": "Synthetic deterministic route-following mechanism test; not a Memory Anchors reproduction or VLA result.",
    }
    metrics_json = {
        "config": asdict(cfg),
        "task_specs": [asdict(task) for task in specs],
        "methods": {
            "no_replay": "Sequential current-task training only.",
            "random": "Uniform sample without replacement from all old transitions.",
            "loss_hard": "MIR-style ranking by old-sample loss increase after a short current-task-only probe update.",
            "diversity": "Greedy farthest-first coreset in the current policy latent space.",
            "anchors": "20% three-step conflict anchors plus 80% uniform old-data coverage.",
        },
        "sanity": sanity,
        "headline": headline,
        "claims": claims,
        "summary": summary,
    }
    (out / "metrics.json").write_text(json.dumps(json_safe(metrics_json), indent=2, allow_nan=False) + "\n")
    print(json.dumps(json_safe({"output_dir": str(out), "headline": headline, "claims": claims}), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
