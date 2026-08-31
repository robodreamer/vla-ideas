#!/usr/bin/env python3
"""Deterministic toy for grounded, efficient online visual adaptation.

A source behavior-cloning policy aligns to a broad object center while ignoring a
small colored mark.  At adaptation time the mark becomes the true target.  The
toy compares no adaptation, a full-policy update proxy, a pooled lightweight
adapter, reward-only visual attention, supervised visual anchors, and the same
anchor learner with cached frozen-prefix features.

This is a mechanism test inspired by GRAFT, not a VLA, robot, Q-learning, or
real-time systems reproduction.
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
from torch import nn
from torch.nn import functional as F


METHODS = (
    "frozen",
    "full_policy",
    "adapter",
    "reward_attention",
    "supervised_anchors",
    "anchors_cached",
)
ADAPTIVE_METHODS = METHODS[1:]
LABELS = {
    "frozen": "Frozen BC",
    "full_policy": "Full-policy update",
    "adapter": "Lightweight adapter",
    "reward_attention": "Reward-only attention",
    "supervised_anchors": "Supervised anchors",
    "anchors_cached": "Anchors + cached prefix",
}
COLORS = {
    "frozen": "#7f7f7f",
    "full_policy": "#e45756",
    "adapter": "#4c78a8",
    "reward_attention": "#f58518",
    "supervised_anchors": "#54a24b",
    "anchors_cached": "#7a5195",
}


@dataclass(frozen=True)
class Config:
    seed: int = 29
    seeds: int = 5
    patches: int = 15
    raw_dim: int = 11
    token_dim: int = 48
    prefix_hidden: int = 128
    prefix_depth: int = 3
    head_hidden: int = 64
    anchors: int = 2
    adapter_rank: int = 4
    source_train: int = 2400
    source_eval: int = 800
    source_steps: int = 380
    source_batch: int = 128
    interactions: int = 720
    interaction_batch: int = 24
    learner_updates: int = 14
    replay_batch: int = 96
    eval_episodes: int = 900
    eval_every: int = 48
    marker_contrast: float = 0.28
    marker_offsets: tuple[int, ...] = (-4, -3, 3, 4)
    distractors: int = 2
    reward_temperature: float = 2.5
    reward_probe_delta: float = 1.0
    success_radius: float = 0.5
    base_logit_scale: float = 1.0
    learning_rate: float = 0.006
    full_learning_rate: float = 0.002
    source_bc_weight: float = 0.40
    grounding_weight: float = 0.2
    attention_duplicate_weight: float = 0.08
    grounding_every: int = 1
    success_target: float = 0.70
    supervision_weights: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)
    contrast_values: tuple[float, ...] = (0.12, 0.20, 0.28, 0.40)


@dataclass
class VisualBatch:
    raw: torch.Tensor
    target: torch.Tensor
    coarse: torch.Tensor
    proposals: torch.Tensor

    def __len__(self) -> int:
        return int(self.raw.shape[0])

    def subset(self, indices: np.ndarray | torch.Tensor | list[int]) -> "VisualBatch":
        idx = torch.as_tensor(indices, dtype=torch.long)
        return VisualBatch(self.raw[idx], self.target[idx], self.coarse[idx], self.proposals[idx])


@dataclass
class Replay:
    raw: list[torch.Tensor]
    tokens: list[torch.Tensor]
    pooled: list[torch.Tensor]
    action: list[int]
    reward: list[float]
    proposal: list[torch.Tensor]

    @classmethod
    def empty(cls) -> "Replay":
        return cls([], [], [], [], [], [])

    def __len__(self) -> int:
        return len(self.action)


class Prefix(nn.Module):
    def __init__(self, raw_dim: int, token_dim: int, hidden: int, depth: int):
        super().__init__()
        if token_dim <= raw_dim:
            raise ValueError("token_dim must exceed raw_dim for the explicit visual-feature skip")
        layers: list[nn.Module] = [nn.Linear(raw_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
        layers.append(nn.Linear(hidden, token_dim - raw_dim))
        self.net = nn.Sequential(*layers)
        self.calls = 0
        self.token_evaluations = 0

    def reset_counters(self) -> None:
        self.calls = 0
        self.token_evaluations = 0

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.token_evaluations += int(raw.shape[0] * raw.shape[1])
        return torch.cat([raw, self.net(raw)], dim=-1)


class BaseBC(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.prefix = Prefix(cfg.raw_dim, cfg.token_dim, cfg.prefix_hidden, cfg.prefix_depth)
        self.head = nn.Sequential(
            nn.Linear(cfg.token_dim, cfg.head_hidden),
            nn.GELU(),
            nn.Linear(cfg.head_hidden, 1),
        )

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        tokens = self.prefix(raw)
        return self.head(tokens.mean(dim=1)).squeeze(-1)


class OnlinePolicy(nn.Module):
    def __init__(self, base: BaseBC, cfg: Config, method: str):
        super().__init__()
        self.cfg = cfg
        self.method = method
        self.prefix = copy.deepcopy(base.prefix)
        self.base_head = copy.deepcopy(base.head)
        self.adapter_down: nn.Linear | None = None
        self.adapter_up: nn.Linear | None = None
        self.queries: nn.Parameter | None = None
        self.anchor_head: nn.Module | None = None
        self.residual_scale: nn.Parameter | None = None
        self.anchor_position_gain: nn.Parameter | None = None

        if method != "full_policy":
            for parameter in self.prefix.parameters():
                parameter.requires_grad_(False)
            for parameter in self.base_head.parameters():
                parameter.requires_grad_(False)

        if method == "adapter":
            self.adapter_down = nn.Linear(2, cfg.adapter_rank, bias=False)
            self.adapter_up = nn.Linear(cfg.adapter_rank, 1, bias=False)
            nn.init.normal_(self.adapter_down.weight, std=0.08)
            nn.init.zeros_(self.adapter_up.weight)
            self.residual_scale = nn.Parameter(torch.tensor(1.0))
        elif method in ("reward_attention", "supervised_anchors", "anchors_cached"):
            self.queries = nn.Parameter(torch.randn(cfg.anchors, cfg.token_dim) * 0.08)
            self.anchor_head = nn.Sequential(
                nn.Linear(cfg.anchors * cfg.token_dim, cfg.head_hidden),
                nn.GELU(),
                nn.Linear(cfg.head_hidden, 1),
            )
            nn.init.zeros_(self.anchor_head[-1].weight)
            nn.init.zeros_(self.anchor_head[-1].bias)
            self.residual_scale = nn.Parameter(torch.tensor(1.0))
            self.anchor_position_gain = nn.Parameter(torch.tensor(0.0))

    def encode(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.prefix(raw)
        return tokens, tokens.mean(dim=1)

    def logits_from_features(
        self, tokens: torch.Tensor, pooled: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_logits = self.cfg.base_logit_scale * self.base_head(pooled).squeeze(-1)
        if self.method in ("frozen", "full_policy"):
            return base_logits, None
        if self.method == "adapter":
            assert self.adapter_down is not None and self.adapter_up is not None and self.residual_scale is not None
            # A tiny global-moment adapter: mark mass and mark-position moment.
            # It does not select a spatial patch, unlike the attention methods.
            adapter_input = 48.0 * torch.stack([pooled[:, 3], pooled[:, 6]], dim=-1)
            residual = self.adapter_up(torch.tanh(self.adapter_down(adapter_input))).squeeze(-1)
            mark_presence = (pooled[:, 3] * self.cfg.patches / self.cfg.marker_contrast).clamp(0.0, 1.0)
            return base_logits + mark_presence * self.residual_scale * residual, None
        assert (
            self.queries is not None
            and self.anchor_head is not None
            and self.residual_scale is not None
            and self.anchor_position_gain is not None
        )
        scores = torch.einsum("ad,bpd->bap", self.queries, tokens) / 0.35
        attention = torch.softmax(scores, dim=-1)
        anchors = torch.einsum("bap,bpd->bad", attention, tokens).flatten(1)
        residual = self.anchor_head(anchors).squeeze(-1)
        patch_position = (tokens[:, :, 0] + 1.0) * 0.5 * (self.cfg.patches - 1)
        anchor_position = torch.einsum("bap,bp->ba", attention, patch_position)
        red_evidence = torch.einsum("bap,bp->ba", attention, tokens[:, :, 3])
        anchor_relevance = torch.softmax(12.0 * red_evidence, dim=-1)
        grounded_position = torch.sum(anchor_relevance * anchor_position, dim=-1)
        position_residual = self.anchor_position_gain * (grounded_position - base_logits)
        mark_presence = (tokens[:, :, 3].sum(dim=1) / self.cfg.marker_contrast).clamp(0.0, 1.0)
        return base_logits + mark_presence * (position_residual + self.residual_scale * residual), attention

    def forward(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        tokens, pooled = self.encode(raw)
        return self.logits_from_features(tokens, pooled)

    def trainable_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class FrozenPolicy(nn.Module):
    def __init__(self, base: BaseBC):
        super().__init__()
        self.base = copy.deepcopy(base)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, raw: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.base(raw), None

    def trainable_count(self) -> int:
        return 0


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**31)


def configure_determinism() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    np.set_printoptions(precision=5, suppress=True)


def positions(cfg: Config) -> np.ndarray:
    return np.linspace(-1.0, 1.0, cfg.patches, dtype=np.float32)


def make_visual_batch(
    cfg: Config,
    n: int,
    seed: int,
    mode: str,
    marker_contrast: float | None = None,
) -> VisualBatch:
    """Create source or fine-alignment strip observations.

    Source examples target the broad object's center and contain no red mark.
    Fine examples target the red mark, offset from the same coarse object center.
    """
    rng = np.random.default_rng(seed)
    pos = positions(cfg)
    raw_rows = np.zeros((n, cfg.patches, cfg.raw_dim), dtype=np.float32)
    target = np.zeros(n, dtype=np.int64)
    coarse = np.zeros(n, dtype=np.int64)
    proposals = np.zeros((n, 2), dtype=np.int64)
    contrast = cfg.marker_contrast if marker_contrast is None else marker_contrast
    margin = max(abs(x) for x in cfg.marker_offsets) + 1
    for i in range(n):
        center = int(rng.integers(margin, cfg.patches - margin))
        if mode == "source":
            mark = int(rng.integers(1, cfg.patches - 1))
            while abs(mark - center) <= 1:
                mark = int(rng.integers(1, cfg.patches - 1))
            target[i] = center
        elif mode == "fine":
            mark = center + int(rng.choice(cfg.marker_offsets))
            target[i] = mark
        else:
            raise ValueError(mode)
        distractor_indices: list[int] = []
        while len(distractor_indices) < cfg.distractors:
            candidate = int(rng.integers(0, cfg.patches))
            if candidate not in distractor_indices and candidate not in (mark, center):
                distractor_indices.append(candidate)
        blob = np.exp(-0.5 * ((pos - pos[center]) / 0.25) ** 2).astype(np.float32)
        red = np.zeros(cfg.patches, dtype=np.float32)
        if mode == "fine":
            red[mark] = contrast
        blue = np.zeros(cfg.patches, dtype=np.float32)
        for candidate in distractor_indices:
            blue[candidate] = contrast * float(rng.uniform(0.75, 1.15))
        cursor = float(rng.uniform(-0.9, 0.9))
        features = np.stack(
            [
                pos,
                pos**2,
                blob,
                red,
                blue,
                blob * pos,
                red * pos,
                blue * pos,
                np.full_like(pos, cursor),
                np.full_like(pos, math.sin(math.pi * cursor)),
                np.ones_like(pos),
            ],
            axis=1,
        )
        raw_rows[i] = features
        coarse[i] = center
        proposals[i] = [mark, center]
    return VisualBatch(
        raw=torch.from_numpy(raw_rows),
        target=torch.from_numpy(target),
        coarse=torch.from_numpy(coarse),
        proposals=torch.from_numpy(proposals),
    )


def train_source_bc(cfg: Config, seed: int) -> tuple[BaseBC, VisualBatch, VisualBatch]:
    torch.manual_seed(stable_seed(seed, "source-model"))
    train = make_visual_batch(cfg, cfg.source_train, stable_seed(seed, "source-train"), "source")
    evaluation = make_visual_batch(cfg, cfg.source_eval, stable_seed(seed, "source-eval"), "source")
    model = BaseBC(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1.0e-5)
    generator = torch.Generator().manual_seed(stable_seed(seed, "source-batches"))
    model.train()
    for _ in range(cfg.source_steps):
        idx = torch.randint(0, len(train), (cfg.source_batch,), generator=generator)
        loss = F.mse_loss(model(train.raw[idx]), train.target[idx].to(torch.float32))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return model, train, evaluation


def reward_for_action(action: torch.Tensor, target: torch.Tensor, cfg: Config) -> torch.Tensor:
    distance = torch.abs(action.to(torch.float32) - target.to(torch.float32))
    return torch.exp(-distance / cfg.reward_temperature)


def target_from_two_rewards(
    minus_action: torch.Tensor,
    plus_action: torch.Tensor,
    minus_reward: torch.Tensor,
    plus_reward: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    """Invert the toy's known scalar reward shape using two black-box probes.

    Each probe supplies only a scalar task reward.  For exp(-|a-target|/T),
    each reward defines two possible targets.  We choose the closest-consistent
    pair across the two probes.  This deliberately favorable 1-D verifier is a
    toy stand-in for fitting a critic from task-level rewards.
    """
    minus_distance = -cfg.reward_temperature * torch.log(minus_reward.clamp_min(1.0e-8))
    plus_distance = -cfg.reward_temperature * torch.log(plus_reward.clamp_min(1.0e-8))
    minus_candidates = torch.stack([minus_action - minus_distance, minus_action + minus_distance], dim=1)
    plus_candidates = torch.stack([plus_action - plus_distance, plus_action + plus_distance], dim=1)
    pair_error = torch.abs(minus_candidates[:, :, None] - plus_candidates[:, None, :]).reshape(-1, 4)
    best = torch.argmin(pair_error, dim=1)
    minus_choice = best // 2
    plus_choice = best % 2
    row = torch.arange(len(best))
    estimate = 0.5 * (minus_candidates[row, minus_choice] + plus_candidates[row, plus_choice])
    return estimate.clamp(0.0, cfg.patches - 1.0)


def success_for_action(action: torch.Tensor, target: torch.Tensor, cfg: Config) -> torch.Tensor:
    return (torch.abs(action.to(torch.float32) - target.to(torch.float32)) <= cfg.success_radius).to(torch.float32)


def policy_metrics(
    policy: nn.Module,
    batch: VisualBatch,
    cfg: Config,
    cached: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, float]:
    policy.eval()
    with torch.no_grad():
        if cached is not None and isinstance(policy, OnlinePolicy):
            logits, attention = policy.logits_from_features(*cached)
        else:
            logits, attention = policy(batch.raw)
        action = logits.clamp(0.0, cfg.patches - 1.0)
        success = success_for_action(action, batch.target, cfg)
        distance = torch.abs(action - batch.target.to(torch.float32))
        reward = reward_for_action(action, batch.target, cfg)
        out = {
            "success": float(success.mean()),
            "mean_abs_error": float(distance.mean()),
            "mean_reward": float(reward.mean()),
        }
        if attention is not None:
            mark_mass = attention.gather(2, batch.proposals[:, 0, None, None].expand(-1, cfg.anchors, 1)).squeeze(-1)
            proposal_mass = []
            for k in range(batch.proposals.shape[1]):
                proposal_mass.append(
                    attention.gather(2, batch.proposals[:, k, None, None].expand(-1, cfg.anchors, 1)).squeeze(-1)
                )
            stacked = torch.stack(proposal_mass, dim=-1)
            out["marker_attention_max"] = float(mark_mass.max(dim=1).values.mean())
            out["proposal_coverage"] = float(stacked.max(dim=1).values.mean())
            out["attention_overlap"] = float((attention[:, 0] * attention[:, 1]).sum(dim=-1).mean())
        else:
            out["marker_attention_max"] = float("nan")
            out["proposal_coverage"] = float("nan")
            out["attention_overlap"] = float("nan")
    return out


def identity_free_grounding_loss(attention: torch.Tensor, proposals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-anchor/two-proposal identity-free assignment plus overlap penalty."""
    eps = 1.0e-7
    gathered = []
    for k in range(proposals.shape[1]):
        gathered.append(attention.gather(2, proposals[:, k, None, None].expand(-1, attention.shape[1], 1)).squeeze(-1))
    probability = torch.stack(gathered, dim=-1).clamp_min(eps)  # [B, anchors, proposals]
    direct = -torch.log(probability[:, 0, 0]) - torch.log(probability[:, 1, 1])
    swapped = -torch.log(probability[:, 0, 1]) - torch.log(probability[:, 1, 0])
    assignment = torch.minimum(direct, swapped).mean() * 0.5
    duplicate = (attention[:, 0] * attention[:, 1]).sum(dim=-1).mean()
    return assignment, duplicate


def append_replay(
    replay: Replay,
    batch: VisualBatch,
    action: torch.Tensor,
    reward: torch.Tensor,
    features: tuple[torch.Tensor, torch.Tensor] | None,
) -> None:
    for i in range(len(batch)):
        replay.raw.append(batch.raw[i].detach().clone())
        if features is None:
            replay.tokens.append(torch.empty(0))
            replay.pooled.append(torch.empty(0))
        else:
            replay.tokens.append(features[0][i].detach().clone())
            replay.pooled.append(features[1][i].detach().clone())
        replay.action.append(int(action[i]))
        replay.reward.append(float(reward[i]))
        replay.proposal.append(batch.proposals[i].detach().clone())


def replay_batch(
    replay: Replay,
    size: int,
    generator: torch.Generator,
    cached: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    idx = torch.randint(0, len(replay), (size,), generator=generator)
    raw = torch.stack([replay.raw[int(i)] for i in idx])
    action = torch.tensor([replay.action[int(i)] for i in idx], dtype=torch.long)
    reward = torch.tensor([replay.reward[int(i)] for i in idx], dtype=torch.float32)
    proposal = torch.stack([replay.proposal[int(i)] for i in idx])
    if cached:
        tokens = torch.stack([replay.tokens[int(i)] for i in idx])
        pooled = torch.stack([replay.pooled[int(i)] for i in idx])
        return raw, action, reward, proposal, tokens, pooled
    return raw, action, reward, proposal, None, None


def source_regularizer(
    policy: OnlinePolicy,
    source_batch: VisualBatch,
    base_logits: torch.Tensor,
    cached_features: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    if cached_features is None:
        logits, _ = policy(source_batch.raw)
    else:
        logits, _ = policy.logits_from_features(*cached_features)
    return F.mse_loss(logits, base_logits)


def run_adaptation(
    method: str,
    cfg: Config,
    seed: int,
    base: BaseBC,
    source_train: VisualBatch,
    source_eval: VisualBatch,
    supervision_weight: float | None = None,
    capture: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    randomization_key = "supervised_anchor_pair" if method in ("supervised_anchors", "anchors_cached") else method
    torch.manual_seed(stable_seed(seed, randomization_key, "policy"))
    if method == "frozen":
        policy: nn.Module = FrozenPolicy(base)
    else:
        policy = OnlinePolicy(base, cfg, method)
    cached_mode = method == "anchors_cached"
    supervision = cfg.grounding_weight if supervision_weight is None else supervision_weight
    if method not in ("supervised_anchors", "anchors_cached"):
        supervision = 0.0

    adaptive_parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = None
    if adaptive_parameters:
        lr = cfg.full_learning_rate if method == "full_policy" else cfg.learning_rate
        optimizer = torch.optim.AdamW(adaptive_parameters, lr=lr, weight_decay=1.0e-5)

    fine_eval = make_visual_batch(cfg, cfg.eval_episodes, stable_seed(seed, "fine-eval"), "fine")
    source_regularization = source_train.subset(np.arange(min(512, len(source_train))))
    with torch.no_grad():
        frozen_source_logits = (cfg.base_logit_scale * base(source_regularization.raw)).detach()
        cached_source_features = None
        if cached_mode and isinstance(policy, OnlinePolicy):
            cached_source_features = policy.encode(source_regularization.raw)
            cached_source_features = tuple(value.detach() for value in cached_source_features)

    update_gen = torch.Generator().manual_seed(stable_seed(seed, randomization_key, "updates"))
    replay = Replay.empty()
    curve: list[dict[str, Any]] = []
    runtime_update = 0.0
    runtime_act = 0.0
    update_steps = 0
    grounding_updates = 0
    learner_prefix_calls = 0
    learner_prefix_token_evaluations = 0
    captures: dict[str, Any] = {}

    if isinstance(policy, OnlinePolicy):
        policy.prefix.reset_counters()
        prefix = policy.prefix
    elif isinstance(policy, FrozenPolicy):
        policy.base.prefix.reset_counters()
        prefix = policy.base.prefix

    initial_fine = policy_metrics(policy, fine_eval, cfg)
    initial_source = policy_metrics(policy, source_eval, cfg)
    curve.append(
        {
            "seed": seed,
            "method": method,
            "interactions": 0,
            "fine_success": initial_fine["success"],
            "fine_reward": initial_fine["mean_reward"],
            "fine_abs_error": initial_fine["mean_abs_error"],
            "source_success": initial_source["success"],
            "marker_attention": initial_fine["marker_attention_max"],
        }
    )

    batches = math.ceil(cfg.interactions / cfg.interaction_batch)
    for round_index in range(batches):
        interaction_count = min(cfg.interaction_batch, cfg.interactions - round_index * cfg.interaction_batch)
        if interaction_count <= 0:
            break
        context_count = max(1, interaction_count // 2)
        batch = make_visual_batch(
            cfg,
            context_count,
            stable_seed(seed, "online-context", round_index),
            "fine",
        )
        started = time.perf_counter()
        with torch.no_grad():
            if isinstance(policy, OnlinePolicy):
                tokens, pooled = policy.encode(batch.raw)
                logits, _ = policy.logits_from_features(tokens, pooled)
                cached_features = (tokens, pooled) if cached_mode else None
            else:
                logits, _ = policy(batch.raw)
                cached_features = None
            mean_action = logits.clamp(0.0, cfg.patches - 1.0)
            minus_action = (mean_action - cfg.reward_probe_delta).clamp(0.0, cfg.patches - 1.0)
            plus_action = (mean_action + cfg.reward_probe_delta).clamp(0.0, cfg.patches - 1.0)
            minus_reward = reward_for_action(minus_action, batch.target, cfg)
            plus_reward = reward_for_action(plus_action, batch.target, cfg)
            reward_target = target_from_two_rewards(
                minus_action, plus_action, minus_reward, plus_reward, cfg
            )
            action = torch.round(mean_action).to(torch.long)
        runtime_act += time.perf_counter() - started
        append_replay(replay, batch, action, reward_target, cached_features)

        if optimizer is not None:
            policy.train()
            for local_update in range(cfg.learner_updates):
                started = time.perf_counter()
                raw, _old_action, old_reward, proposal, tokens, pooled = replay_batch(
                    replay, cfg.replay_batch, update_gen, cached_mode
                )
                prefix_calls_before = prefix.calls
                prefix_tokens_before = prefix.token_evaluations
                if cached_mode:
                    assert isinstance(policy, OnlinePolicy) and tokens is not None and pooled is not None
                    logits, attention = policy.logits_from_features(tokens, pooled)
                else:
                    logits, attention = policy(raw)
                mean_action = logits
                policy_loss = F.mse_loss(mean_action, old_reward.detach())
                source_idx = torch.randint(0, len(source_regularization), (64,), generator=update_gen)
                src = source_regularization.subset(source_idx)
                source_features = None
                if cached_source_features is not None:
                    source_features = (cached_source_features[0][source_idx], cached_source_features[1][source_idx])
                source_loss = source_regularizer(policy, src, frozen_source_logits[source_idx], source_features)
                loss = policy_loss + cfg.source_bc_weight * source_loss
                if supervision > 0.0 and attention is not None and local_update % cfg.grounding_every == 0:
                    grounding, duplicate = identity_free_grounding_loss(attention, proposal)
                    loss = loss + supervision * grounding + cfg.attention_duplicate_weight * duplicate
                    grounding_updates += 1
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adaptive_parameters, 5.0)
                optimizer.step()
                update_steps += 1
                runtime_update += time.perf_counter() - started
                learner_prefix_calls += prefix.calls - prefix_calls_before
                learner_prefix_token_evaluations += prefix.token_evaluations - prefix_tokens_before
            policy.eval()

        interactions_so_far = min(cfg.interactions, (round_index + 1) * cfg.interaction_batch)
        if interactions_so_far % cfg.eval_every == 0 or interactions_so_far == cfg.interactions:
            fine_values = policy_metrics(policy, fine_eval, cfg)
            source_values = policy_metrics(policy, source_eval, cfg)
            curve.append(
                {
                    "seed": seed,
                    "method": method,
                    "interactions": interactions_so_far,
                    "fine_success": fine_values["success"],
                    "fine_reward": fine_values["mean_reward"],
                    "fine_abs_error": fine_values["mean_abs_error"],
                    "source_success": source_values["success"],
                    "marker_attention": fine_values["marker_attention_max"],
                }
            )

    final_fine = policy_metrics(policy, fine_eval, cfg)
    final_source = policy_metrics(policy, source_eval, cfg)
    reached = [row["interactions"] for row in curve if row["fine_success"] >= cfg.success_target]
    interactions_to_target = min(reached) if reached else None
    x = np.asarray([row["interactions"] for row in curve], dtype=float)
    y = np.asarray([row["fine_success"] for row in curve], dtype=float)
    success_auc = float(np.trapezoid(y, x) / max(1.0, float(cfg.interactions)))
    condition = {
        "seed": seed,
        "method": method,
        "trainable_parameters": policy.trainable_count(),
        "initial_fine_success": initial_fine["success"],
        "final_fine_success": final_fine["success"],
        "final_fine_reward": final_fine["mean_reward"],
        "final_fine_abs_error": final_fine["mean_abs_error"],
        "success_auc": success_auc,
        "interactions_to_70_success": interactions_to_target,
        "initial_source_success": initial_source["success"],
        "final_source_success": final_source["success"],
        "forgetting": initial_source["success"] - final_source["success"],
        "marker_attention": final_fine["marker_attention_max"],
        "proposal_coverage": final_fine["proposal_coverage"],
        "attention_overlap": final_fine["attention_overlap"],
        "update_steps": update_steps,
        "grounding_updates": grounding_updates,
        "actor_seconds": runtime_act,
        "update_seconds": runtime_update,
        "seconds_per_update": runtime_update / max(1, update_steps),
        "prefix_calls": prefix.calls,
        "prefix_token_evaluations": prefix.token_evaluations,
        "learner_prefix_calls": learner_prefix_calls,
        "learner_prefix_token_evaluations": learner_prefix_token_evaluations,
        "cached_prefix": cached_mode,
        "grounding_weight": supervision,
    }
    if capture:
        captures = {"policy": copy.deepcopy(policy)}
        if isinstance(policy, OnlinePolicy) and policy.queries is not None:
            captures["evaluation"] = fine_eval.subset(np.arange(min(8, len(fine_eval))))
    return condition, curve, captures


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = [
        "trainable_parameters",
        "initial_fine_success",
        "final_fine_success",
        "final_fine_reward",
        "final_fine_abs_error",
        "success_auc",
        "interactions_to_70_success",
        "initial_source_success",
        "final_source_success",
        "forgetting",
        "marker_attention",
        "proposal_coverage",
        "attention_overlap",
        "update_steps",
        "actor_seconds",
        "update_seconds",
        "seconds_per_update",
        "prefix_calls",
        "prefix_token_evaluations",
        "learner_prefix_calls",
        "learner_prefix_token_evaluations",
    ]
    output: list[dict[str, Any]] = []
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        if not group:
            continue
        item: dict[str, Any] = {"method": method, "seeds": len(group)}
        for metric in metric_names:
            values = np.asarray(
                [float(row[metric]) for row in group if row[metric] is not None and math.isfinite(float(row[metric]))],
                dtype=float,
            )
            item[f"{metric}_mean"] = float(np.mean(values)) if len(values) else float("nan")
            item[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            item[f"{metric}_sem"] = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
            item[f"{metric}_n"] = int(len(values))
        output.append(item)
    return output


def contrast_sweep(
    policies: dict[tuple[int, str], nn.Module], cfg: Config
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_index in range(cfg.seeds):
        seed = cfg.seed + seed_index
        for contrast in cfg.contrast_values:
            batch = make_visual_batch(
                cfg,
                cfg.eval_episodes,
                stable_seed(seed, "contrast", contrast),
                "fine",
                marker_contrast=contrast,
            )
            for method in METHODS:
                policy = policies[(seed, method)]
                values = policy_metrics(policy, batch, cfg)
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "marker_contrast": contrast,
                        "success": values["success"],
                        "mean_abs_error": values["mean_abs_error"],
                        "marker_attention": values["marker_attention_max"],
                    }
                )
    return rows


def run_supervision_sweep(cfg: Config) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sweep_cfg = replace(
        cfg,
        seeds=min(3, cfg.seeds),
        interactions=cfg.interactions,
        eval_episodes=min(600, cfg.eval_episodes),
    )
    for seed_index in range(sweep_cfg.seeds):
        seed = cfg.seed + seed_index
        base, source_train, source_eval = train_source_bc(sweep_cfg, seed)
        for weight in cfg.supervision_weights:
            method = "supervised_anchors"
            condition, _, _ = run_adaptation(
                method,
                sweep_cfg,
                seed,
                base,
                source_train,
                source_eval,
                supervision_weight=weight,
            )
            rows.append(
                {
                    "seed": seed,
                    "grounding_weight": weight,
                    "final_fine_success": condition["final_fine_success"],
                    "success_auc": condition["success_auc"],
                    "forgetting": condition["forgetting"],
                    "marker_attention": condition["marker_attention"],
                    "update_seconds": condition["update_seconds"],
                }
            )
    return rows


def sanity_checks(cfg: Config) -> dict[str, Any]:
    seed = cfg.seed
    a = make_visual_batch(cfg, 32, stable_seed(seed, "deterministic"), "fine")
    b = make_visual_batch(cfg, 32, stable_seed(seed, "deterministic"), "fine")
    base, source_train, source_eval = train_source_bc(replace(cfg, source_steps=min(cfg.source_steps, 420)), seed)
    fine_eval = make_visual_batch(cfg, 600, stable_seed(seed, "sanity-fine"), "fine")
    frozen = FrozenPolicy(base)
    source_values = policy_metrics(frozen, source_eval, cfg)
    fine_values = policy_metrics(frozen, fine_eval, cfg)

    online = OnlinePolicy(base, cfg, "supervised_anchors")
    online.eval()
    with torch.no_grad():
        tokens, pooled = online.encode(fine_eval.raw[:64])
        uncached_logits, _ = online(fine_eval.raw[:64])
        cached_logits, _ = online.logits_from_features(tokens, pooled)
    action = fine_eval.target[:32]
    optimal_reward = reward_for_action(action, fine_eval.target[:32], cfg)
    shifted_reward = reward_for_action((action + 2).clamp_max(cfg.patches - 1), fine_eval.target[:32], cfg)

    full = OnlinePolicy(base, cfg, "full_policy")
    adapter = OnlinePolicy(base, cfg, "adapter")
    anchor = OnlinePolicy(base, cfg, "anchors_cached")
    checks = {
        "data_generation_deterministic": bool(torch.equal(a.raw, b.raw) and torch.equal(a.target, b.target)),
        "marker_changes_target_not_coarse_object": bool(torch.all(a.target != a.coarse)),
        "source_bc_success_above_90pct": source_values["success"] > 0.90,
        "frozen_policy_fails_fine_alignment": fine_values["success"] < 0.25,
        "cached_and_uncached_logits_match": bool(torch.allclose(uncached_logits, cached_logits, atol=1e-7, rtol=0.0)),
        "task_reward_prefers_alignment_mark": bool(float(optimal_reward.mean()) > float(shifted_reward.mean()) + 0.35),
        "adapter_lighter_than_full_policy": adapter.trainable_count() < full.trainable_count(),
        "anchor_path_lighter_than_full_policy": anchor.trainable_count() < full.trainable_count(),
        "deployment_forward_needs_no_proposals": bool(online(fine_eval.raw[:2])[0].shape == (2,)),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "diagnostics": {
            "source_bc_success": source_values["success"],
            "frozen_fine_success": fine_values["success"],
            "full_trainable_parameters": full.trainable_count(),
            "adapter_trainable_parameters": adapter.trainable_count(),
            "anchor_trainable_parameters": anchor.trainable_count(),
            "optimal_reward": float(optimal_reward.mean()),
            "two_patch_error_reward": float(shifted_reward.mean()),
            "cached_logit_max_abs_delta": float(torch.max(torch.abs(uncached_logits - cached_logits))),
        },
    }


def write_csv(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def mean_sem(rows: list[dict[str, Any]], key: str) -> tuple[float, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def plot_learning(curves: list[dict[str, Any]], cfg: Config, out: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), constrained_layout=True)
    for method in METHODS:
        rows = [row for row in curves if row["method"] == method]
        xs = sorted(set(int(row["interactions"]) for row in rows))
        success_mean, success_sem, source_mean = [], [], []
        for x in xs:
            group = [row for row in rows if int(row["interactions"]) == x]
            mean, sem = mean_sem(group, "fine_success")
            success_mean.append(mean)
            success_sem.append(sem)
            source_mean.append(float(np.mean([row["source_success"] for row in group])))
        axes[0].plot(xs, success_mean, marker="o", linewidth=2, color=COLORS[method], label=LABELS[method])
        axes[0].fill_between(xs, np.asarray(success_mean) - success_sem, np.asarray(success_mean) + success_sem, color=COLORS[method], alpha=0.13)
        axes[1].plot(xs, source_mean, marker="o", linewidth=2, color=COLORS[method], label=LABELS[method])
    axes[0].axhline(cfg.success_target, color="black", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].set_title("Fine-mark adaptation (mean ± SEM)")
    axes[0].set_ylabel("Exact alignment success")
    axes[1].set_title("Source-task retention")
    axes[1].set_ylabel("Broad-object alignment success")
    for ax in axes:
        ax.set_xlabel("Online interactions")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoffs(summary: list[dict[str, Any]], out: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), constrained_layout=True)
    x = np.arange(len(summary))
    labels = [LABELS[row["method"]] for row in summary]
    metrics = [
        ("final_fine_success", "Final fine-mark success", (0.0, 1.02)),
        ("forgetting", "Source forgetting (lower is better)", None),
        ("update_seconds", "Measured learner update time (s)", None),
    ]
    for ax, (metric, title, ylim) in zip(axes, metrics):
        values = [row[f"{metric}_mean"] for row in summary]
        errors = [row[f"{metric}_sem"] for row in summary]
        ax.bar(x, values, yerr=errors, color=[COLORS[row["method"]] for row in summary], capsize=3)
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if ylim is not None:
            ax.set_ylim(*ylim)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_sweeps(contrast_rows: list[dict[str, Any]], supervision_rows: list[dict[str, Any]], cfg: Config, out: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3), constrained_layout=True)
    for method in METHODS:
        means, sems = [], []
        for contrast in cfg.contrast_values:
            group = [row for row in contrast_rows if row["method"] == method and row["marker_contrast"] == contrast]
            mean, sem = mean_sem(group, "success")
            means.append(mean)
            sems.append(sem)
        axes[0].errorbar(cfg.contrast_values, means, yerr=sems, marker="o", capsize=3, linewidth=2, color=COLORS[method], label=LABELS[method])
    axes[0].set_xlabel("Tiny-mark contrast")
    axes[0].set_ylabel("Exact alignment success")
    axes[0].set_title("Visual-cue contrast sweep")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)

    weights = sorted(set(float(row["grounding_weight"]) for row in supervision_rows))
    success, success_sem, attention = [], [], []
    for weight in weights:
        group = [row for row in supervision_rows if float(row["grounding_weight"]) == weight]
        mean, sem = mean_sem(group, "final_fine_success")
        success.append(mean)
        success_sem.append(sem)
        attention.append(float(np.mean([row["marker_attention"] for row in group])))
    axes[1].errorbar(weights, success, yerr=success_sem, marker="o", capsize=3, linewidth=2, color=COLORS["supervised_anchors"], label="Success")
    axes[1].plot(weights, attention, marker="s", linestyle="--", linewidth=2, color="#b279a2", label="Marker attention")
    axes[1].set_xlabel("Grounding-loss weight")
    axes[1].set_ylabel("Mean metric")
    axes[1].set_title("Visual-anchor supervision ablation")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_cache(summary: list[dict[str, Any]], out: pathlib.Path) -> None:
    methods = ["full_policy", "supervised_anchors", "anchors_cached"]
    rows = [next(row for row in summary if row["method"] == method) for method in methods]
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.9), constrained_layout=True)
    labels = [LABELS[row["method"]] for row in rows]
    colors = [COLORS[row["method"]] for row in rows]
    axes[0].bar(labels, [row["learner_prefix_token_evaluations_mean"] for row in rows], color=colors)
    axes[0].set_ylabel("Prefix token evaluations")
    axes[0].set_title("Frozen-prefix recomputation proxy")
    axes[0].tick_params(axis="x", rotation=22)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, [row["seconds_per_update_mean"] * 1000 for row in rows], color=colors)
    axes[1].set_ylabel("Milliseconds / learner update")
    axes[1].set_title("Measured CPU update cost")
    axes[1].tick_params(axis="x", rotation=22)
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_attention(captures: dict[str, dict[str, Any]], cfg: Config, out: pathlib.Path) -> None:
    selected = ["reward_attention", "supervised_anchors", "anchors_cached"]
    fig, axes = plt.subplots(len(selected), 2, figsize=(11.5, 7.6), constrained_layout=True)
    x = np.arange(cfg.patches)
    for row_index, method in enumerate(selected):
        capture = captures[method]
        policy: OnlinePolicy = capture["policy"]
        batch: VisualBatch = capture["evaluation"]
        example = batch.subset([0])
        with torch.no_grad():
            _, attention = policy(example.raw)
        assert attention is not None
        blob = example.raw[0, :, 2].numpy()
        red = example.raw[0, :, 3].numpy()
        blue = example.raw[0, :, 4].numpy()
        axes[row_index, 0].plot(x, blob, color="#999999", linewidth=2, label="broad object")
        axes[row_index, 0].stem(x, red, linefmt="#d62728", markerfmt="ro", basefmt=" ", label="target mark")
        axes[row_index, 0].stem(x, blue, linefmt="#1f77b4", markerfmt="bo", basefmt=" ", label="distractors")
        axes[row_index, 0].axvline(int(example.target[0]), color="black", linestyle="--", linewidth=1)
        axes[row_index, 0].set_title(f"{LABELS[method]}: visual strip")
        axes[row_index, 0].set_ylabel("Feature intensity")
        for anchor_index in range(cfg.anchors):
            axes[row_index, 1].plot(x, attention[0, anchor_index].numpy(), marker="o", label=f"anchor {anchor_index + 1}")
        axes[row_index, 1].axvline(int(example.target[0]), color="#d62728", linestyle="--", linewidth=1, label="target mark")
        axes[row_index, 1].axvline(int(example.coarse[0]), color="#777777", linestyle=":", linewidth=1, label="object center")
        axes[row_index, 1].set_title(f"{LABELS[method]}: learned attention")
        axes[row_index, 1].set_ylabel("Attention mass")
        axes[row_index, 1].set_ylim(0.0, 1.0)
        axes[row_index, 1].grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Patch index")
    axes[-1, 1].set_xlabel("Patch index")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=3)
    axes[0, 1].legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--interactions", type=int, default=720)
    parser.add_argument("--source-steps", type=int, default=380)
    parser.add_argument("--eval-episodes", type=int, default=900)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        seed=args.seed,
        seeds=args.seeds,
        interactions=args.interactions,
        source_steps=args.source_steps,
        eval_episodes=args.eval_episodes,
    )
    if args.smoke:
        cfg = replace(
            cfg,
            seeds=min(2, cfg.seeds),
            source_train=700,
            source_eval=240,
            source_steps=min(180, cfg.source_steps),
            interactions=min(96, cfg.interactions),
            interaction_batch=24,
            learner_updates=3,
            replay_batch=32,
            eval_episodes=min(240, cfg.eval_episodes),
            eval_every=48,
            supervision_weights=(0.0, 1.2),
            contrast_values=(0.20, 0.28),
        )
    configure_determinism()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    sanity = sanity_checks(cfg)
    if not sanity["passed"]:
        raise RuntimeError(f"Sanity checks failed: {sanity}")

    conditions: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    captures: dict[str, dict[str, Any]] = {}
    final_policies: dict[tuple[int, str], nn.Module] = {}
    for seed_index in range(cfg.seeds):
        seed = cfg.seed + seed_index
        base, source_train, source_eval = train_source_bc(cfg, seed)
        for method in METHODS:
            condition, method_curves, method_capture = run_adaptation(
                method,
                cfg,
                seed,
                base,
                source_train,
                source_eval,
                capture=True,
            )
            conditions.append(condition)
            curves.extend(method_curves)
            final_policies[(seed, method)] = copy.deepcopy(method_capture["policy"])
            if seed_index == 0 and method in ("reward_attention", "supervised_anchors", "anchors_cached"):
                captures[method] = method_capture
            print(
                f"seed={seed} method={method:20s} fine={condition['final_fine_success']:.3f} "
                f"forget={condition['forgetting']:.3f} auc={condition['success_auc']:.3f} "
                f"update={condition['update_seconds']:.3f}s"
            )

    parity_rows: list[dict[str, Any]] = []
    for seed_index in range(cfg.seeds):
        seed = cfg.seed + seed_index
        batch = make_visual_batch(
            cfg, min(256, cfg.eval_episodes), stable_seed(seed, "final-cache-parity"), "fine"
        )
        with torch.no_grad():
            uncached_logits, _ = final_policies[(seed, "supervised_anchors")](batch.raw)
            cached_logits, _ = final_policies[(seed, "anchors_cached")](batch.raw)
        delta = float(torch.max(torch.abs(uncached_logits - cached_logits)))
        parity_rows.append({"seed": seed, "max_abs_logit_delta": delta})
    max_parity_delta = max(row["max_abs_logit_delta"] for row in parity_rows)
    if max_parity_delta > 1.0e-7:
        raise RuntimeError(f"cached/uncached final-policy parity failed: {max_parity_delta}")

    summary = aggregate(conditions)
    contrast_rows = contrast_sweep(final_policies, cfg)
    supervision_rows = run_supervision_sweep(cfg)

    write_csv(out / "condition_metrics.csv", conditions)
    write_csv(out / "learning_curve.csv", curves)
    write_csv(out / "summary_metrics.csv", summary)
    write_csv(out / "contrast_sweep.csv", contrast_rows)
    write_csv(out / "supervision_sweep.csv", supervision_rows)
    (out / "sanity_check.json").write_text(json.dumps(json_safe(sanity), indent=2) + "\n")

    plot_learning(curves, cfg, out / "learning_curves.png")
    plot_tradeoffs(summary, out / "method_tradeoffs.png")
    plot_sweeps(contrast_rows, supervision_rows, cfg, out / "sweeps.png")
    plot_cache(summary, out / "cache_runtime.png")
    if all(method in captures for method in ("reward_attention", "supervised_anchors", "anchors_cached")):
        plot_attention(captures, cfg, out / "attention_diagnostics.png")

    lookup = {row["method"]: row for row in summary}
    cache_speedup = (
        lookup["supervised_anchors"]["seconds_per_update_mean"]
        / max(1.0e-12, lookup["anchors_cached"]["seconds_per_update_mean"])
    )
    prefix_reduction = 1.0 - (
        lookup["anchors_cached"]["learner_prefix_token_evaluations_mean"]
        / max(1.0, lookup["supervised_anchors"]["learner_prefix_token_evaluations_mean"])
    )
    claims = {
        "supervised_anchor_gain_over_reward_attention": lookup["supervised_anchors"]["final_fine_success_mean"]
        - lookup["reward_attention"]["final_fine_success_mean"],
        "cached_prefix_measured_update_speedup": cache_speedup,
        "cached_prefix_token_evaluation_reduction": prefix_reduction,
        "cached_uncached_final_max_abs_logit_delta": max_parity_delta,
        "full_policy_forgetting_minus_cached_anchor_forgetting": lookup["full_policy"]["forgetting_mean"]
        - lookup["anchors_cached"]["forgetting_mean"],
        "scope": "Synthetic 1-D visual-strip contextual-bandit mechanism test; not a GRAFT reproduction or robot/VLA result.",
    }
    metrics = {
        "config": asdict(cfg),
        "methods": {
            "frozen": "Frozen source behavior-cloning policy; no online update.",
            "full_policy": "Reward update of prefix and BC head; full-update cost/forgetting proxy.",
            "adapter": "Low-rank residual on globally pooled frozen-prefix features.",
            "reward_attention": "Two trainable visual anchors and residual head, trained only from scalar task reward plus source distillation.",
            "supervised_anchors": "Reward-attention policy plus training-only identity-free proposal supervision.",
            "anchors_cached": "Same supervised anchor pathway with frozen-prefix tokens cached for replay learner updates.",
        },
        "sanity": sanity,
        "final_cache_parity": {
            "all_passed": True,
            "tolerance": 1.0e-7,
            "max_abs_logit_delta": max_parity_delta,
            "per_seed": parity_rows,
        },
        "summary": summary,
        "sweeps": {
            "contrast": contrast_rows,
            "grounding_weight": supervision_rows,
        },
        "claims": claims,
    }
    (out / "metrics.json").write_text(json.dumps(json_safe(metrics), indent=2, allow_nan=False) + "\n")
    print(json.dumps(json_safe({"output_dir": str(out), "claims": claims}), indent=2))


if __name__ == "__main__":
    main()
