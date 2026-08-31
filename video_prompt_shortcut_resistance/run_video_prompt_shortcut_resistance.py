#!/usr/bin/env python3
"""Zero-WAM-inspired in-context future-prediction mechanism toy.

The benchmark deliberately makes teacher-forced one-step actions easy to predict
from history and a seen-task text alias. Novel suffix compositions reuse that
alias, so a video-like prompt is the only reliable deployment specification.
An auxiliary future-chunk objective is tested as pressure against the shortcut.

This is a deterministic mechanism toy, not a reproduction of Zero-WAM.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

METHODS = (
    "direct_bc",
    "prompt_one_step_bc",
    "ifp_future_chunk",
    "ifp_shuffled_prompt",
)
METHOD_LABELS = {
    "direct_bc": "No-prompt direct BC",
    "prompt_one_step_bc": "Prompt one-step BC",
    "ifp_future_chunk": "Future-chunk auxiliary (IFP-inspired)",
    "ifp_shuffled_prompt": "IFP + shuffled prompt",
}
METHOD_COLORS = {
    "direct_bc": "#666666",
    "prompt_one_step_bc": "#e68613",
    "ifp_future_chunk": "#1677b8",
    "ifp_shuffled_prompt": "#b04a8b",
}

N_PRIMITIVES = 6
CHUNK_LEN = 4
N_CHUNKS = 3
HORIZON = CHUNK_LEN * N_CHUNKS
PROMPT_FRAME_DIM = 8
TEXT_DIM = N_PRIMITIVES
HISTORY_DIM = 2 + 2 + HORIZON  # previous action, current position, phase one-hot
FUTURE_STEPS = 6

SEEN_TASKS: Dict[int, Tuple[int, int, int]] = {
    i: (i, (i + 1) % N_PRIMITIVES, (i + 3) % N_PRIMITIVES)
    for i in range(N_PRIMITIVES)
}
UNSEEN_TASKS: Dict[int, Tuple[int, int, int]] = {
    i: (i, (i + 3) % N_PRIMITIVES, (i + 1) % N_PRIMITIVES)
    for i in range(N_PRIMITIVES)
}


@dataclass(frozen=True)
class RunConfig:
    mode: str
    seed: int
    train_episodes: int
    epochs: int
    eval_trials: int
    batch_size: int
    hidden_dim: int
    latent_dim: int
    future_weight: float
    language_dropout: float
    lr: float
    weight_decay: float


@dataclass(frozen=True)
class Episode:
    alias: int
    composition: Tuple[int, int, int]
    actions: np.ndarray
    prompt: np.ndarray
    prompt_clean: np.ndarray
    positions: np.ndarray


class ShortcutPolicy(nn.Module):
    """Small shared-representation policy with an optional prompt encoder."""

    def __init__(self, use_prompt: bool, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.use_prompt = use_prompt
        prompt_dim = FUTURE_STEPS * PROMPT_FRAME_DIM if use_prompt else 0
        if use_prompt:
            self.prompt_encoder = nn.Sequential(
                nn.Linear(prompt_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
                nn.Tanh(),
            )
            self.prompt_gate_logit = nn.Parameter(torch.tensor(-2.5))
        else:
            self.prompt_encoder = None
            self.prompt_gate_logit = None
        self.trunk = nn.Sequential(
            nn.Linear(HISTORY_DIM + TEXT_DIM + (latent_dim if use_prompt else 0), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(latent_dim, 2)
        self.future_head = nn.Linear(latent_dim, 2 * FUTURE_STEPS)

    def forward(
        self, history: torch.Tensor, text: torch.Tensor, prompt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pieces = [history, text]
        prompt_latent = None
        if self.use_prompt:
            assert self.prompt_encoder is not None and self.prompt_gate_logit is not None
            gate = torch.sigmoid(self.prompt_gate_logit)
            prompt_latent = gate * self.prompt_encoder(prompt)
            pieces.append(prompt_latent)
        latent = self.trunk(torch.cat(pieces, dim=-1))
        # The future loss supervises the same fused representation used by the
        # deployed action head.  The auxiliary head is never executed at test
        # time, matching the key causal structure of Zero-WAM's training-only
        # IFP modules more closely than a prompt-only rollout head would.
        return self.action_head(latent), self.future_head(latent).view(-1, FUTURE_STEPS, 2)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

def primitive_template(primitive: int) -> np.ndarray:
    theta = 2.0 * math.pi * primitive / N_PRIMITIVES
    direction = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
    perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32)
    speed = np.array([0.64, 0.94, 1.06, 0.76], dtype=np.float32)
    bend = np.array([-0.13, 0.09, 0.10, -0.06], dtype=np.float32)
    return speed[:, None] * direction[None, :] + bend[:, None] * perpendicular[None, :]


def composition_actions(
    composition: Sequence[int], rng: np.random.Generator
) -> np.ndarray:
    scale = float(rng.uniform(0.88, 1.13))
    anisotropy = np.diag(rng.uniform(0.94, 1.06, size=2)).astype(np.float32)
    chunks = []
    for chunk_index, primitive in enumerate(composition):
        chunk = primitive_template(int(primitive)).copy()
        chunk *= scale * float(rng.uniform(0.96, 1.04))
        chunk = chunk @ anisotropy
        chunk += rng.normal(0.0, 0.018, size=chunk.shape).astype(np.float32)
        # Small transition-dependent modulation makes composition more than labels.
        if chunk_index:
            previous = primitive_template(int(composition[chunk_index - 1]))[-1]
            chunk[0] += 0.10 * previous
        chunks.append(chunk)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def make_prompt(
    actions: np.ndarray,
    rng: np.random.Generator,
    distractor_level: float = 0.0,
    corruption: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a compact video-like trace: motion, position, and nuisance channels."""
    prompt_motion = actions + rng.normal(0.0, 0.045, size=actions.shape).astype(np.float32)
    prompt_position = np.cumsum(prompt_motion, axis=0)
    denom = max(float(np.max(np.linalg.norm(prompt_position, axis=1))), 1.0)
    prompt_position = prompt_position / denom
    phase = np.linspace(0.0, 2.0 * math.pi, HORIZON, endpoint=False, dtype=np.float32)
    nuisance = np.stack(
        [
            np.sin(phase + rng.uniform(-math.pi, math.pi)),
            np.cos(1.7 * phase + rng.uniform(-math.pi, math.pi)),
            rng.normal(0.0, 0.35, size=HORIZON),
            rng.normal(0.0, 0.35, size=HORIZON),
        ],
        axis=1,
    ).astype(np.float32)
    clean = np.concatenate([prompt_motion, prompt_position, nuisance], axis=1).astype(np.float32)
    prompt = clean.copy()
    if distractor_level > 0.0:
        prompt[:, 4:] += rng.normal(
            0.0, 0.55 * distractor_level, size=(HORIZON, 4)
        ).astype(np.float32)
        # A moving visual distractor leaks slightly into the position channels.
        drift = rng.normal(0.0, 0.10 * distractor_level, size=(HORIZON, 2)).astype(np.float32)
        prompt[:, 2:4] += np.cumsum(drift, axis=0) / math.sqrt(HORIZON)
    if corruption > 0.0:
        n_mask = max(1, int(round(corruption * HORIZON)))
        mask_idx = rng.choice(HORIZON, size=n_mask, replace=False)
        prompt[mask_idx, :4] = 0.0
        prompt[:, :2] += rng.normal(0.0, 0.18 * corruption, size=(HORIZON, 2)).astype(np.float32)
        if corruption >= 0.35:
            lo = int(rng.integers(1, HORIZON - 3))
            hi = min(HORIZON, lo + 3)
            prompt[lo:hi, :4] = prompt[lo:hi, :4][::-1]
    return prompt.astype(np.float32), clean


def make_episode(
    alias: int,
    composition: Sequence[int],
    seed: int,
    distractor_level: float = 0.0,
    corruption: float = 0.0,
) -> Episode:
    rng = np.random.default_rng(seed)
    actions = composition_actions(composition, rng)
    prompt, prompt_clean = make_prompt(actions, rng, distractor_level, corruption)
    positions = np.concatenate(
        [np.zeros((1, 2), dtype=np.float32), np.cumsum(actions, axis=0)], axis=0
    )
    return Episode(
        alias=int(alias),
        composition=tuple(int(v) for v in composition),
        actions=actions,
        prompt=prompt,
        prompt_clean=prompt_clean,
        positions=positions.astype(np.float32),
    )


def history_feature(previous_action: np.ndarray, position: np.ndarray, time_index: int) -> np.ndarray:
    phase = np.zeros(HORIZON, dtype=np.float32)
    phase[time_index] = 1.0
    return np.concatenate([previous_action, position, phase]).astype(np.float32)


def text_feature(alias: int) -> np.ndarray:
    text = np.zeros(TEXT_DIM, dtype=np.float32)
    text[int(alias)] = 1.0
    return text


def prompt_window(prompt: np.ndarray, time_index: int) -> np.ndarray:
    idx = np.minimum(np.arange(time_index, time_index + FUTURE_STEPS), HORIZON - 1)
    return prompt[idx].reshape(-1).astype(np.float32)


def future_target(actions: np.ndarray, time_index: int) -> np.ndarray:
    # The contiguous action window is a deliberately simpler proxy for the
    # source method's strided future-video targets.
    idx = np.minimum(np.arange(time_index, time_index + FUTURE_STEPS), HORIZON - 1)
    return actions[idx].astype(np.float32)


def build_training_arrays(config: RunConfig) -> Dict[str, np.ndarray]:
    histories: List[np.ndarray] = []
    texts: List[np.ndarray] = []
    prompts: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    futures: List[np.ndarray] = []
    episode_ids: List[int] = []
    rng = np.random.default_rng(config.seed + 101)
    for episode_index in range(config.train_episodes):
        alias = episode_index % N_PRIMITIVES
        episode_seed = int(rng.integers(0, 2**31 - 1))
        variant_draw = episode_index % 20
        if variant_draw < 18:
            composition = SEEN_TASKS[alias]
        elif variant_draw == 18:
            composition = (alias, (alias + 2) % N_PRIMITIVES, (alias + 3) % N_PRIMITIVES)
        else:
            composition = (alias, (alias + 1) % N_PRIMITIVES, (alias + 4) % N_PRIMITIVES)
        episode = make_episode(alias, composition, episode_seed, distractor_level=0.18)
        for t in range(HORIZON):
            histories.append(history_feature(episode.actions[t - 1] if t else np.zeros(2), episode.positions[t], t))
            texts.append(text_feature(alias))
            prompts.append(prompt_window(episode.prompt, t))
            actions.append(episode.actions[t])
            futures.append(future_target(episode.actions, t))
            episode_ids.append(episode_index)
    order = rng.permutation(len(histories))
    return {
        "history": np.asarray(histories, dtype=np.float32)[order],
        "text": np.asarray(texts, dtype=np.float32)[order],
        "prompt": np.asarray(prompts, dtype=np.float32)[order],
        "action": np.asarray(actions, dtype=np.float32)[order],
        "future": np.asarray(futures, dtype=np.float32)[order],
        "episode_id": np.asarray(episode_ids, dtype=np.int64)[order],
    }


def train_method(method: str, arrays: Mapping[str, np.ndarray], config: RunConfig) -> ShortcutPolicy:
    # Prompt-conditioned ablations share initialization and minibatch order;
    # only the future loss or prompt-shuffle intervention differs.  Using a
    # different seed for each row would confound a single-seed mechanism test.
    method_seed = config.seed + 1000 if method != "direct_bc" else config.seed + 2000
    set_determinism(method_seed)
    use_prompt = method != "direct_bc"
    model = ShortcutPolicy(use_prompt, config.hidden_dim, config.latent_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    history = torch.from_numpy(arrays["history"])
    text = torch.from_numpy(arrays["text"])
    prompt = torch.from_numpy(arrays["prompt"])
    action = torch.from_numpy(arrays["action"])
    future = torch.from_numpy(arrays["future"])
    n = len(history)
    generator = torch.Generator().manual_seed(method_seed + 1)
    shuffle_generator = torch.Generator().manual_seed(method_seed + 2)
    future_weight = config.future_weight if method in {"ifp_future_chunk", "ifp_shuffled_prompt"} else 0.0

    model.train()
    for _epoch in range(config.epochs):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, config.batch_size):
            idx = permutation[start : start + config.batch_size]
            batch_text = text[idx].clone()
            if use_prompt:
                keep = torch.rand((len(idx), 1), generator=generator) >= config.language_dropout
                batch_text *= keep
            batch_prompt = prompt[idx]
            if method == "ifp_shuffled_prompt":
                shuffle_idx = idx[torch.randperm(len(idx), generator=shuffle_generator)]
                batch_prompt = prompt[shuffle_idx]
            pred_action, pred_future = model(history[idx], batch_text, batch_prompt)
            action_loss = torch.mean((pred_action - action[idx]) ** 2)
            future_loss = torch.mean((pred_future - future[idx]) ** 2)
            loss = action_loss + future_weight * future_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


def rollout(
    model: ShortcutPolicy,
    episode: Episode,
    prompt_override: np.ndarray | None = None,
    text_override: np.ndarray | None = None,
) -> np.ndarray:
    predicted = []
    previous = np.zeros(2, dtype=np.float32)
    position = np.zeros(2, dtype=np.float32)
    prompt_array = episode.prompt if prompt_override is None else prompt_override
    text = text_feature(episode.alias) if text_override is None else text_override.astype(np.float32)
    with torch.no_grad():
        for t in range(HORIZON):
            history = torch.from_numpy(history_feature(previous, position, t)[None, :])
            text_tensor = torch.from_numpy(text[None, :])
            prompt_tensor = torch.from_numpy(prompt_window(prompt_array, t)[None, :])
            action, _predicted_chunk = model(history, text_tensor, prompt_tensor)
            current = action.numpy()[0].astype(np.float32)
            predicted.append(current)
            previous = current
            position = position + current
    return np.asarray(predicted, dtype=np.float32)


def primitive_ids(actions: np.ndarray) -> Tuple[int, int, int]:
    ids: List[int] = []
    base_directions = np.stack([primitive_template(i).sum(axis=0) for i in range(N_PRIMITIVES)])
    base_directions /= np.linalg.norm(base_directions, axis=1, keepdims=True)
    for chunk_index in range(N_CHUNKS):
        chunk = actions[chunk_index * CHUNK_LEN : (chunk_index + 1) * CHUNK_LEN]
        direction = chunk.sum(axis=0)
        direction /= np.linalg.norm(direction) + 1e-8
        ids.append(int(np.argmax(base_directions @ direction)))
    return tuple(ids)  # type: ignore[return-value]


def rollout_metrics(predicted: np.ndarray, episode: Episode) -> Dict[str, float]:
    error = predicted - episode.actions
    action_rmse = float(np.sqrt(np.mean(error**2)))
    pred_chunks = predicted.reshape(N_CHUNKS, CHUNK_LEN, 2).sum(axis=1)
    true_chunks = episode.actions.reshape(N_CHUNKS, CHUNK_LEN, 2).sum(axis=1)
    chunk_rmse = float(np.sqrt(np.mean((pred_chunks - true_chunks) ** 2)))
    final_error = float(np.linalg.norm(predicted.sum(axis=0) - episode.actions.sum(axis=0)))
    sequence_correct = primitive_ids(predicted) == episode.composition
    success = float(sequence_correct and action_rmse < 0.38 and final_error < 1.10)
    return {
        "success": success,
        "action_rmse": action_rmse,
        "chunk_rmse": chunk_rmse,
        "final_error": final_error,
        "sequence_correct": float(sequence_correct),
    }


def swap_prompt_for_episode(episode: Episode, seed: int) -> np.ndarray:
    # Same text alias, but the seen/default suffix: this is the shortcut's preferred prompt.
    swapped = make_episode(episode.alias, SEEN_TASKS[episode.alias], seed)
    return swapped.prompt


def evaluate_models(
    models: Mapping[str, ShortcutPolicy], config: RunConfig
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    eval_conditions = [
        ("clean", 0.0, 0.0),
        ("distractor_mild", 0.65, 0.0),
        ("distractor_hard", 1.25, 0.0),
        ("prompt_corrupt_mild", 0.0, 0.15),
        ("prompt_corrupt_hard", 0.0, 0.40),
    ]
    for split, tasks in (("seen", SEEN_TASKS), ("unseen", UNSEEN_TASKS)):
        for alias, composition in tasks.items():
            for trial in range(config.eval_trials):
                base_seed = config.seed * 100000 + (0 if split == "seen" else 50000) + alias * 1000 + trial * 17
                for condition_index, (condition, distractor, corruption) in enumerate(eval_conditions):
                    episode = make_episode(
                        alias,
                        composition,
                        base_seed + condition_index,
                        distractor_level=distractor,
                        corruption=corruption,
                    )
                    swap_prompt = swap_prompt_for_episode(episode, base_seed + 8000 + condition_index)
                    for method, model in models.items():
                        deployment_text = text_feature(alias)
                        predicted = rollout(model, episode, text_override=deployment_text)
                        metrics = rollout_metrics(predicted, episode)
                        swapped_prediction = rollout(
                            model,
                            episode,
                            prompt_override=swap_prompt,
                            text_override=deployment_text,
                        )
                        swapped_metrics = rollout_metrics(swapped_prediction, episode)
                        output_shift = float(np.sqrt(np.mean((predicted - swapped_prediction) ** 2)))
                        row: Dict[str, object] = {
                            "method": method,
                            "split": split,
                            "condition": condition,
                            "alias": alias,
                            "composition": "-".join(map(str, composition)),
                            "trial": trial,
                            **metrics,
                            "swapped_success": swapped_metrics["success"],
                            "swapped_action_rmse": swapped_metrics["action_rmse"],
                            "prompt_swap_output_shift": output_shift,
                            "prompt_swap_error_increase": float(
                                swapped_metrics["action_rmse"] - metrics["action_rmse"]
                            ),
                            "prompt_swap_success_drop": float(
                                metrics["success"] - swapped_metrics["success"]
                            ),
                        }
                        rows.append(row)

    clean_unseen = [r for r in rows if r["split"] == "unseen" and r["condition"] == "clean"]
    reliance = {}
    for method in METHODS:
        subset = [r for r in clean_unseen if r["method"] == method]
        reliance[method] = {
            "output_shift": mean_field(subset, "prompt_swap_output_shift"),
            "error_increase": mean_field(subset, "prompt_swap_error_increase"),
            "success_drop": mean_field(subset, "prompt_swap_success_drop"),
        }
    return rows, {"clean_unseen_prompt_reliance": reliance}


def mean_field(rows: Iterable[Mapping[str, object]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    return float(np.mean(values)) if values else float("nan")


def summarize(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    for method in METHODS:
        method_rows = [r for r in rows if r["method"] == method]
        clean_seen = [r for r in method_rows if r["split"] == "seen" and r["condition"] == "clean"]
        clean_unseen = [r for r in method_rows if r["split"] == "unseen" and r["condition"] == "clean"]
        hard_distractor = [
            r for r in method_rows if r["split"] == "unseen" and r["condition"] == "distractor_hard"
        ]
        hard_corrupt = [
            r for r in method_rows if r["split"] == "unseen" and r["condition"] == "prompt_corrupt_hard"
        ]
        summary.append(
            {
                "method": method,
                "seen_success": mean_field(clean_seen, "success"),
                "unseen_success": mean_field(clean_unseen, "success"),
                "seen_action_rmse": mean_field(clean_seen, "action_rmse"),
                "unseen_action_rmse": mean_field(clean_unseen, "action_rmse"),
                "unseen_chunk_rmse": mean_field(clean_unseen, "chunk_rmse"),
                "unseen_sequence_accuracy": mean_field(clean_unseen, "sequence_correct"),
                "prompt_swap_output_shift": mean_field(clean_unseen, "prompt_swap_output_shift"),
                "prompt_swap_error_increase": mean_field(clean_unseen, "prompt_swap_error_increase"),
                "prompt_swap_success_drop": mean_field(clean_unseen, "prompt_swap_success_drop"),
                "hard_distractor_success": mean_field(hard_distractor, "success"),
                "hard_corruption_success": mean_field(hard_corrupt, "success"),
            }
        )
    return summary


def sanity_check(models: Mapping[str, ShortcutPolicy], config: RunConfig) -> Dict[str, object]:
    alias = 2
    episode = make_episode(alias, UNSEEN_TASKS[alias], config.seed + 424242)
    result: Dict[str, object] = {
        "alias": alias,
        "seen_shortcut_composition": list(SEEN_TASKS[alias]),
        "unseen_prompted_composition": list(UNSEEN_TASKS[alias]),
        "methods": {},
    }
    for method, model in models.items():
        prediction = rollout(model, episode)
        metrics = rollout_metrics(prediction, episode)
        result["methods"][method] = {
            **metrics,
            "predicted_primitive_sequence": list(primitive_ids(prediction)),
        }
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plot_summary(summary: Sequence[Mapping[str, object]]) -> None:
    labels = [METHOD_LABELS[str(row["method"])] for row in summary]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))
    width = 0.36
    axes[0].bar(x - width / 2, [float(r["seen_success"]) for r in summary], width, label="seen")
    axes[0].bar(x + width / 2, [float(r["unseen_success"]) for r in summary], width, label="unseen")
    axes[0].set_ylabel("Task success")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False)
    axes[0].set_title("Shortcut gap")
    axes[1].bar(x - width / 2, [float(r["unseen_action_rmse"]) for r in summary], width, label="action")
    axes[1].bar(x + width / 2, [float(r["unseen_chunk_rmse"]) for r in summary], width, label="chunk")
    axes[1].set_ylabel("RMSE (lower is better)")
    axes[1].set_title("Unseen composition error")
    axes[1].legend(frameon=False)
    axes[2].bar(x - width / 2, [float(r["prompt_swap_output_shift"]) for r in summary], width, label="output shift")
    axes[2].bar(x + width / 2, [float(r["prompt_swap_error_increase"]) for r in summary], width, label="error increase")
    axes[2].set_ylabel("Prompt-swap response")
    axes[2].set_title("Prompt reliance")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=26, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "method_summary.png", dpi=190)
    plt.close(fig)


def plot_robustness(rows: Sequence[Mapping[str, object]]) -> None:
    conditions = ["clean", "distractor_mild", "distractor_hard", "prompt_corrupt_mild", "prompt_corrupt_hard"]
    condition_labels = ["clean", "distractor\nmild", "distractor\nhard", "corruption\nmild", "corruption\nhard"]
    fig, axis = plt.subplots(figsize=(10.6, 5.0))
    for method in METHODS:
        values = []
        for condition in conditions:
            subset = [
                row
                for row in rows
                if row["method"] == method and row["split"] == "unseen" and row["condition"] == condition
            ]
            values.append(mean_field(subset, "success"))
        axis.plot(condition_labels, values, marker="o", linewidth=2.0, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    axis.set_ylim(-0.02, 1.05)
    axis.set_ylabel("Unseen-composition task success")
    axis.set_title("Robustness to prompt distractors and corruption")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "robustness_sweeps.png", dpi=190)
    plt.close(fig)


def plot_representative(models: Mapping[str, ShortcutPolicy], config: RunConfig) -> None:
    alias = 1
    episode = make_episode(alias, UNSEEN_TASKS[alias], config.seed + 909090, distractor_level=0.65)
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.3, 7.2), sharex=True, sharey=True)
    axes_flat = list(axes.flat)
    true_positions = np.concatenate([np.zeros((1, 2)), np.cumsum(episode.actions, axis=0)], axis=0)
    for axis, method in zip(axes_flat, METHODS):
        prediction = rollout(models[method], episode)
        positions = np.concatenate([np.zeros((1, 2)), np.cumsum(prediction, axis=0)], axis=0)
        axis.plot(true_positions[:, 0], true_positions[:, 1], color="black", linewidth=2.5, label="target")
        axis.plot(positions[:, 0], positions[:, 1], color=METHOD_COLORS[method], linewidth=2.0, marker=".", label="rollout")
        for boundary in (0, CHUNK_LEN, 2 * CHUNK_LEN, 3 * CHUNK_LEN):
            axis.scatter(true_positions[boundary, 0], true_positions[boundary, 1], color="black", s=24)
        metrics = rollout_metrics(prediction, episode)
        axis.set_title(f"{METHOD_LABELS[method]}\nsuccess={metrics['success']:.0f}, RMSE={metrics['action_rmse']:.2f}", fontsize=9)
        axis.grid(alpha=0.22)
        axis.set_aspect("equal", adjustable="box")
    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.93, 0.12), frameon=False)
    fig.suptitle(f"Representative unseen composition: text alias {alias}, target {UNSEEN_TASKS[alias]}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "representative_rollout.png", dpi=190)
    plt.close(fig)


def config_for_mode(args: argparse.Namespace) -> RunConfig:
    if args.mode == "smoke":
        return RunConfig(
            mode="smoke",
            seed=args.seed,
            train_episodes=72,
            epochs=100,
            eval_trials=2,
            batch_size=144,
            hidden_dim=64,
            latent_dim=32,
            future_weight=1.0,
            language_dropout=0.15,
            lr=2.5e-3,
            weight_decay=2e-4,
        )
    return RunConfig(
        mode="full",
        seed=args.seed,
        train_episodes=args.train_episodes,
        epochs=args.epochs,
        eval_trials=args.eval_trials,
        batch_size=192,
        hidden_dim=96,
        latent_dim=48,
        future_weight=1.0,
        language_dropout=0.15,
        lr=2.0e-3,
        weight_decay=2e-4,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--train-episodes", type=int, default=720)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--eval-trials", type=int, default=24)
    args = parser.parse_args()
    warnings.filterwarnings("error")
    # PyTorch creates an internal TemporaryDirectory that is cleaned at process
    # exit; silence only that third-party ResourceWarning while keeping all
    # experiment warnings fatal.
    warnings.filterwarnings("ignore", category=ResourceWarning, module=r"tempfile")
    config = config_for_mode(args)
    set_determinism(config.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    arrays = build_training_arrays(config)
    models = {method: train_method(method, arrays, config) for method in METHODS}
    rows, diagnostics = evaluate_models(models, config)
    summary = summarize(rows)
    sanity = sanity_check(models, config)

    prefix = "smoke_" if config.mode == "smoke" else ""
    write_csv(OUT / f"{prefix}trials.csv", rows)
    write_csv(OUT / f"{prefix}summary.csv", summary)
    write_json(OUT / f"{prefix}sanity_check.json", sanity)
    metrics_payload = {
        "statement": "Deterministic mechanism toy; not a Zero-WAM reproduction.",
        "config": asdict(config),
        "task_design": {
            "seen": {str(k): list(v) for k, v in SEEN_TASKS.items()},
            "unseen": {str(k): list(v) for k, v in UNSEEN_TASKS.items()},
            "text_alias": "first primitive only; each unseen suffix reuses the corresponding seen alias",
            "horizon": HORIZON,
            "chunk_len": CHUNK_LEN,
            "future_steps": FUTURE_STEPS,
        },
        "summary": summary,
        "diagnostics": diagnostics,
        "sanity_check": sanity,
    }
    write_json(OUT / f"{prefix}metrics.json", metrics_payload)
    if config.mode == "full":
        plot_summary(summary)
        plot_robustness(rows)
        plot_representative(models, config)

    print(json.dumps({"mode": config.mode, "config": asdict(config), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
