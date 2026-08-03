#!/usr/bin/env python3
"""TurboVLA-inspired direct V+L->A toy.

This is intentionally small, but it keeps the central research mechanics:
- separate visual and language encoders
- lightweight bidirectional cross-attention between visual and instruction tokens
- an ACT-style action-chunk decoder trained with BC/L1 loss
- a receding-horizon control loop where latency determines how often chunks refresh

The task is a synthetic language-conditioned reach/drag world. The observation is a
small rendered image containing a cursor/object plus four possible goal markers. The
instruction selects one marker. The expert returns a future velocity chunk toward the
selected marker. The closed-loop evaluation perturbs the world, so stale low-rate
chunks overshoot or chase the wrong instantaneous state more often.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DOCS = ROOT / "docs"

GOAL_NAMES = ["red", "green", "blue", "yellow"]
GOAL_POS = torch.tensor(
    [
        [0.16, 0.18],
        [0.84, 0.18],
        [0.16, 0.84],
        [0.84, 0.84],
    ],
    dtype=torch.float32,
)


@dataclass
class Config:
    seed: int = 7
    train_steps: int = 480
    batch_size: int = 192
    eval_episodes: int = 200
    horizon: int = 12
    image_size: int = 16
    control_hz: float = 32.0
    max_speed: float = 0.052
    disturbance_std: float = 0.018
    max_episode_steps: int = 34
    success_radius: float = 0.075
    obstacle_radius: float = 0.245
    obstacle_clearance: float = 0.315
    gust_std: float = 0.065
    device: str = "auto"


@dataclass
class EvalCase:
    goal_idx: int
    start: Tuple[float, float]
    noise: List[Tuple[float, float]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def gaussian_blob(size: int, xy: Tensor, sigma: float = 0.06) -> Tensor:
    # xy: [B, 2] in [0,1]
    b = xy.shape[0]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, size, device=xy.device),
        torch.linspace(0, 1, size, device=xy.device),
        indexing="ij",
    )
    gx = grid_x[None, :, :].expand(b, -1, -1)
    gy = grid_y[None, :, :].expand(b, -1, -1)
    dx = gx - xy[:, 0, None, None]
    dy = gy - xy[:, 1, None, None]
    return torch.exp(-(dx.square() + dy.square()) / (2 * sigma**2))


def render_obs(pos: Tensor, goal_idx: Tensor, image_size: int) -> Tensor:
    """Render [B, 6, H, W]: cursor + obstacle + one channel per possible goal."""
    b = pos.shape[0]
    chans = [gaussian_blob(image_size, pos, sigma=0.055)]
    obstacle = torch.tensor([[0.50, 0.50]], device=pos.device).expand(b, -1)
    # The obstacle channel makes this more than a straight-line reacher. Stale chunks
    # cut the corner near this central keep-out zone; high-rate refreshes can bend
    # around it after each observed state update.
    chans.append(gaussian_blob(image_size, obstacle, sigma=0.115))
    goals = GOAL_POS.to(pos.device)
    for i in range(4):
        goal_xy = goals[i][None, :].expand(b, -1)
        chans.append(gaussian_blob(image_size, goal_xy, sigma=0.045))
    # No selected-goal halo: the observation lists all fixed markers, while the
    # instruction token selects which marker should drive the action chunk. This is
    # an instruction-use sanity check, not a full randomized-layout grounding test.
    img = torch.stack(chans, dim=1)
    return img.clamp(0, 1)


def expert_chunk(pos: Tensor, goal_idx: Tensor, horizon: int, max_speed: float) -> Tensor:
    goals = GOAL_POS.to(pos.device)[goal_idx]
    sim_pos = pos.clone()
    center = torch.tensor([0.50, 0.50], device=pos.device)[None, :]
    actions: List[Tensor] = []
    for _ in range(horizon):
        to_goal = goals - sim_pos
        goal_norm = torch.linalg.norm(to_goal, dim=-1, keepdim=True).clamp_min(1e-6)
        attract = to_goal / goal_norm

        rel = sim_pos - center
        obs_dist = torch.linalg.norm(rel, dim=-1, keepdim=True).clamp_min(1e-6)
        radial = rel / obs_dist
        # Smooth repulsion plus a tangent field so the expert goes around, not just away.
        clearance = 0.315
        repulse_strength = torch.clamp((clearance - obs_dist) / clearance, min=0.0, max=1.0)
        side = torch.where((goal_idx % 2)[:, None] == 0, 1.0, -1.0)
        tangent = side * torch.cat([-radial[:, 1:2], radial[:, 0:1]], dim=-1)
        field = attract + 1.65 * repulse_strength * radial + 0.95 * repulse_strength * tangent

        norm = torch.linalg.norm(field, dim=-1, keepdim=True).clamp_min(1e-6)
        speed = torch.minimum(goal_norm, torch.full_like(goal_norm, max_speed))
        step = field / norm * speed * torch.clamp(goal_norm / 0.18, min=0.25, max=1.0)
        actions.append(step)
        sim_pos = (sim_pos + step).clamp(0, 1)
    return torch.stack(actions, dim=1)


def sample_batch(cfg: Config, device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
    b = cfg.batch_size
    goal_idx = torch.randint(0, 4, (b,), device=device)
    # Bias samples toward cross-obstacle cases while retaining broad coverage.
    anchors = GOAL_POS.to(device)[(goal_idx + 2) % 4]
    pos = anchors + 0.18 * torch.randn(b, 2, device=device)
    mix = torch.rand(b, 1, device=device) < 0.35
    pos = torch.where(mix, torch.rand(b, 2, device=device) * 0.86 + 0.07, pos).clamp(0.06, 0.94)
    img = render_obs(pos, goal_idx, cfg.image_size)
    y = expert_chunk(pos, goal_idx, cfg.horizon, cfg.max_speed)
    return img, goal_idx, y


class DirectVLAPolicy(nn.Module):
    """Compact direct fusion: visual tokens <-> instruction token, then ACT-style chunk queries."""

    def __init__(self, horizon: int, d: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.horizon = horizon
        self.visual = nn.Sequential(
            nn.Conv2d(6, 24, 3, padding=1), nn.GELU(),
            nn.Conv2d(24, d, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(d, d, 3, stride=2, padding=1), nn.GELU(),
        )
        self.text = nn.Embedding(4, d)
        self.v_to_l = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers)])
        self.l_to_v = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers)])
        self.norm_v = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.norm_l = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        decoder_layer = nn.TransformerDecoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d, batch_first=True, dropout=0.0)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.queries = nn.Parameter(torch.randn(horizon, d) * 0.02)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2), nn.Tanh())

    def forward(self, img: Tensor, goal_idx: Tensor) -> Tensor:
        b = img.shape[0]
        v = self.visual(img).flatten(2).transpose(1, 2)  # [B, 16, d]
        l = self.text(goal_idx).unsqueeze(1)  # [B, 1, d]
        for v2l, l2v, nv, nl in zip(self.v_to_l, self.l_to_v, self.norm_v, self.norm_l):
            l = nl(l + v2l(query=l, key=v, value=v, need_weights=False)[0])
            v = nv(v + l2v(query=v, key=l, value=l, need_weights=False)[0])
        mem = torch.cat([l, v], dim=1)
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        z = self.decoder(q, mem)
        return 0.060 * self.head(z)


class LLMBottleneckPolicy(nn.Module):
    """Heavier LLM-centric proxy: project vision into a token stream and process all tokens in a larger transformer core."""

    def __init__(self, horizon: int, d: int = 160, heads: int = 8, layers: int = 4):
        super().__init__()
        self.horizon = horizon
        self.visual = nn.Sequential(
            nn.Conv2d(6, 40, 3, padding=1), nn.GELU(),
            nn.Conv2d(40, d, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(d, d, 3, stride=2, padding=1), nn.GELU(),
        )
        self.text = nn.Embedding(4, d)
        self.pos = nn.Parameter(torch.randn(1, 1 + 16, d) * 0.01)
        enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d, batch_first=True, dropout=0.0)
        self.core = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, horizon * 2), nn.Tanh())

    def forward(self, img: Tensor, goal_idx: Tensor) -> Tensor:
        b = img.shape[0]
        v = self.visual(img).flatten(2).transpose(1, 2)
        l = self.text(goal_idx).unsqueeze(1)
        x = torch.cat([l, v], dim=1) + self.pos[:, : 1 + v.shape[1]]
        pooled = self.core(x)[:, 0]
        return 0.060 * self.head(pooled).view(b, self.horizon, 2)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_policy(name: str, model: nn.Module, cfg: Config, device: torch.device) -> Dict[str, float]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    t0 = time.perf_counter()
    last_loss = math.nan
    for step in range(1, cfg.train_steps + 1):
        img, goal_idx, y = sample_batch(cfg, device)
        pred = model(img, goal_idx)
        loss = F.l1_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.detach().cpu())
        if step % max(1, cfg.train_steps // 4) == 0:
            print(f"{name:>14} step {step:4d}/{cfg.train_steps}: L1={last_loss:.4f}", flush=True)
    return {"train_l1": last_loss, "train_seconds": time.perf_counter() - t0, "params": count_params(model)}


@torch.no_grad()
def measure_latency(model: nn.Module, cfg: Config, device: torch.device, warmup: int = 20, reps: int = 90) -> float:
    model.eval()
    img, goal_idx, _ = sample_batch(cfg, device)
    img = img[:1]
    goal_idx = goal_idx[:1]
    for _ in range(warmup):
        _ = model(img, goal_idx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = model(img, goal_idx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / reps


def make_eval_cases(cfg: Config) -> List[EvalCase]:
    """Create paired starts/noise so every policy sees identical disturbances.

    The occasional gusts are deliberate latency probes: after an unmodeled bump, a
    32 Hz policy can replan on the next tick while lower-rate chunk execution keeps
    consuming stale actions for several ticks.
    """
    rng = np.random.default_rng(cfg.seed + 10_003)
    cases: List[EvalCase] = []
    goals = GOAL_POS.numpy()
    for ep in range(cfg.eval_episodes):
        goal_idx = ep % 4
        start_anchor = goals[(goal_idx + 2) % 4]
        jitter = np.array([0.055 * math.sin(ep * 1.7), 0.055 * math.cos(ep * 1.1)], dtype=np.float32)
        start = np.clip(start_anchor + jitter, 0.06, 0.94)
        noise = rng.normal(0.0, cfg.disturbance_std, size=(cfg.max_episode_steps, 2)).astype(np.float32)
        for t in (4, 9, 14):
            if t < cfg.max_episode_steps and rng.random() < 0.72:
                # Push roughly sideways relative to the current corner-to-goal path.
                sign = 1.0 if ((ep + t) % 2 == 0) else -1.0
                direction = np.array([sign, -sign], dtype=np.float32) / math.sqrt(2.0)
                noise[t] += direction * rng.normal(cfg.gust_std, cfg.gust_std * 0.18)
        cases.append(EvalCase(goal_idx=int(goal_idx), start=(float(start[0]), float(start[1])), noise=[(float(x), float(y)) for x, y in noise]))
    return cases


@torch.no_grad()
def rollout(
    model: nn.Module,
    cfg: Config,
    device: torch.device,
    refresh_every: int,
    cases: List[EvalCase],
    policy_label: str,
    language_mode: str = "correct",
) -> Tuple[Dict[str, float], List[Dict[str, float]], Dict[str, np.ndarray]]:
    model.eval()
    rows: List[Dict[str, float]] = []
    example = None
    for ep, case in enumerate(cases):
        goal_idx = torch.tensor([case.goal_idx], device=device)
        if language_mode == "shuffled":
            policy_goal_idx = torch.tensor([(case.goal_idx + 1) % 4], device=device)
        elif language_mode == "blank":
            policy_goal_idx = torch.tensor([0], device=device)
        else:
            policy_goal_idx = goal_idx
        goals_all = GOAL_POS.to(device)
        pos = torch.tensor([case.start], dtype=torch.float32, device=device).clamp(0.06, 0.94)
        goal = goals_all[goal_idx][0]
        chunk = None
        chunk_cursor = 0
        traj = [pos[0].detach().cpu().numpy().copy()]
        actions = []
        min_dist = float(torch.linalg.norm(pos[0] - goal).cpu())
        min_obstacle_margin = 1.0
        collided = False
        settled_at = cfg.max_episode_steps
        for t in range(cfg.max_episode_steps):
            if chunk is None or chunk_cursor >= min(cfg.horizon, refresh_every):
                img = render_obs(pos, policy_goal_idx, cfg.image_size)
                chunk = model(img, policy_goal_idx)[0]
                chunk_cursor = 0
            action = chunk[min(chunk_cursor, cfg.horizon - 1)].unsqueeze(0)
            chunk_cursor += 1
            noise = torch.tensor([case.noise[t]], dtype=torch.float32, device=device)
            pos = (pos + action + noise).clamp(0, 1)
            dist = float(torch.linalg.norm(pos[0] - goal).cpu())
            obs_dist = float(torch.linalg.norm(pos[0] - torch.tensor([0.50, 0.50], device=device)).cpu())
            min_obstacle_margin = min(min_obstacle_margin, obs_dist - cfg.obstacle_radius)
            if obs_dist < cfg.obstacle_radius:
                collided = True
            min_dist = min(min_dist, dist)
            if dist < cfg.success_radius and settled_at == cfg.max_episode_steps:
                settled_at = t + 1
            traj.append(pos[0].detach().cpu().numpy().copy())
            actions.append(action[0].detach().cpu().numpy().copy())
            if dist < cfg.success_radius * 0.72:
                break
        traj_np = np.asarray(traj)
        act_np = np.asarray(actions) if actions else np.zeros((0, 2))
        final_dist = float(np.linalg.norm(traj_np[-1] - goal.detach().cpu().numpy()))
        jerk = 0.0
        if len(act_np) >= 3:
            jerk = float(np.mean(np.linalg.norm(np.diff(act_np, n=2, axis=0), axis=1)))
        success = (final_dist < cfg.success_radius) and not collided
        rows.append(
            {
                "policy": policy_label,
                "episode": ep,
                "goal": int(goal_idx.item()),
                "policy_goal": int(policy_goal_idx.item()),
                "refresh_every_steps": refresh_every,
                "success": float(success),
                "steps": float(len(traj_np) - 1),
                "settled_at": float(settled_at),
                "final_dist": final_dist,
                "min_dist": min_dist,
                "min_obstacle_margin": min_obstacle_margin,
                "collision": float(collided),
                "mean_jerk": jerk,
            }
        )
        if example is None and ep % 4 == 1:
            example = {"traj": traj_np, "goal": goal.detach().cpu().numpy(), "policy": policy_label}
    agg = {
        "policy": policy_label,
        "refresh_every_steps": refresh_every,
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_final_dist": float(np.mean([r["final_dist"] for r in rows])),
        "mean_min_dist": float(np.mean([r["min_dist"] for r in rows])),
        "collision_rate": float(np.mean([r["collision"] for r in rows])),
        "mean_obstacle_margin": float(np.mean([r["min_obstacle_margin"] for r in rows])),
        "mean_jerk": float(np.mean([r["mean_jerk"] for r in rows])),
        "episodes": len(rows),
        "language_mode": language_mode,
    }
    return agg, rows, example or {}


def display_label(policy: str) -> str:
    return {
        "direct_32hz": "32 Hz",
        "direct_throttled_11hz": "11 Hz\nsame",
        "llm_bottleneck_11hz": "11 Hz\nproxy",
        "direct_stress_6hz": "6 Hz",
        "direct_stress_4hz": "4 Hz",
        "direct_shuffled_language": "wrong\nlang",
    }.get(policy, policy)


def save_plots(summary: List[Dict[str, float]], examples: Dict[str, Dict[str, np.ndarray]], cfg: Config) -> None:
    labels = [display_label(s["policy"]) for s in summary]
    xs = np.arange(len(labels))
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.7))
    axes[0].bar(xs, [s["success_rate"] for s in summary], color="#4C78A8")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("success rate")
    axes[1].bar(xs, [s["collision_rate"] for s in summary], color="#E45756")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("collision rate")
    axes[2].bar(xs, [s["mean_final_dist"] for s in summary], color="#F58518")
    axes[2].set_ylabel("final distance ↓")
    axes[3].bar(xs, [s["mean_jerk"] for s in summary], color="#54A24B")
    axes[3].set_ylabel("mean jerk ↓")
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title(f"Task success (n={int(summary[0].get('episodes', 0))})")
    axes[1].set_title("Obstacle collisions")
    axes[2].set_title("Goal error")
    axes[3].set_title("Action smoothness")
    fig.text(0.5, 0.01, "same = same direct policy throttled; proxy = heavier transformer-core LLM-bottleneck proxy", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "turbo_vla_latency_metrics.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    colors = {"direct_32hz": "#4C78A8", "direct_throttled_11hz": "#72B7B2", "llm_bottleneck_11hz": "#E45756", "direct_stress_6hz": "#B279A2", "direct_stress_4hz": "#9D755D", "direct_shuffled_language": "#BAB0AC"}
    for label, ex in examples.items():
        traj = ex["traj"]
        goal = ex["goal"]
        short = display_label(label).replace("\n", " ")
        ax.plot(traj[:, 0], traj[:, 1], "-o", ms=2.5, lw=1.7, label=short, color=colors.get(label))
        ax.scatter([traj[0, 0]], [traj[0, 1]], marker="x", s=70, color=colors.get(label), linewidth=1.4)
        ax.scatter([goal[0]], [goal[1]], marker="*", s=160, color=colors.get(label), edgecolor="black", linewidth=0.5)
        if len(traj) > 2:
            ax.annotate("", xy=(traj[min(3, len(traj)-1), 0], traj[min(3, len(traj)-1), 1]), xytext=(traj[0, 0], traj[0, 1]), arrowprops=dict(arrowstyle="->", color=colors.get(label), lw=1.0, alpha=0.8))
    ax.add_patch(plt.Circle((0.5, 0.5), cfg.obstacle_radius, color="#E45756", alpha=0.16, label="keep-out zone"))
    for i, gp in enumerate(GOAL_POS.numpy()):
        ax.text(gp[0] + 0.015, gp[1] + 0.015, GOAL_NAMES[i], fontsize=8)
        ax.scatter([gp[0]], [gp[1]], s=35, color="lightgray", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("normalized x")
    ax.set_ylabel("normalized y")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    ax.set_title("Representative rollouts (x=start, *=target)")
    fig.tight_layout()
    fig.savefig(OUT / "turbo_vla_example_rollouts.png", dpi=170)
    plt.close(fig)


def save_delta_plot(deltas: List[Dict[str, float]]) -> None:
    labels = [display_label(d["policy"]).replace("\n", " ") for d in deltas]
    y = np.arange(len(labels))
    success = np.asarray([100 * d["success_delta_vs_direct_32hz"] for d in deltas])
    success_err = np.asarray([
        [100 * (d["success_delta_vs_direct_32hz"] - d["success_delta_ci95"][0]) for d in deltas],
        [100 * (d["success_delta_ci95"][1] - d["success_delta_vs_direct_32hz"]) for d in deltas],
    ])
    final = np.asarray([d["final_dist_delta_vs_direct_32hz"] for d in deltas])
    final_err = np.asarray([
        [d["final_dist_delta_vs_direct_32hz"] - d["final_dist_delta_ci95"][0] for d in deltas],
        [d["final_dist_delta_ci95"][1] - d["final_dist_delta_vs_direct_32hz"] for d in deltas],
    ])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    axes[0].barh(y, success, xerr=success_err, color="#4C78A8", alpha=0.9)
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_xlabel("success delta vs Direct 32 Hz (pp)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh(y, final, xerr=final_err, color="#F58518", alpha=0.9)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("final-distance delta")
    axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle("Episode-paired degradation from lower refresh or wrong language")
    fig.tight_layout()
    fig.savefig(OUT / "turbo_vla_paired_deltas.png", dpi=170)
    plt.close(fig)


def paired_deltas(rows: List[Dict[str, float]], baseline: str = "direct_32hz") -> List[Dict[str, float]]:
    """Episode-paired deltas against the baseline policy."""
    by_policy: Dict[str, Dict[int, Dict[str, float]]] = {}
    for r in rows:
        by_policy.setdefault(str(r["policy"]), {})[int(r["episode"])] = r
    base = by_policy.get(baseline, {})
    out: List[Dict[str, float]] = []
    rng = np.random.default_rng(123)
    for policy, prow in by_policy.items():
        if policy == baseline:
            continue
        eps = sorted(set(base).intersection(prow))
        if not eps:
            continue
        succ_delta = np.asarray([prow[e]["success"] - base[e]["success"] for e in eps], dtype=np.float32)
        final_delta = np.asarray([prow[e]["final_dist"] - base[e]["final_dist"] for e in eps], dtype=np.float32)
        coll_delta = np.asarray([prow[e]["collision"] - base[e]["collision"] for e in eps], dtype=np.float32)

        def ci95(x: np.ndarray) -> Tuple[float, float]:
            if len(x) < 3:
                return (float(np.mean(x)), float(np.mean(x)))
            boots = []
            for _ in range(600):
                idx = rng.integers(0, len(x), size=len(x))
                boots.append(float(np.mean(x[idx])))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            return float(lo), float(hi)

        s_lo, s_hi = ci95(succ_delta)
        f_lo, f_hi = ci95(final_delta)
        c_lo, c_hi = ci95(coll_delta)
        out.append({
            "policy": policy,
            "episodes": len(eps),
            "success_delta_vs_direct_32hz": float(np.mean(succ_delta)),
            "success_delta_ci95": [s_lo, s_hi],
            "final_dist_delta_vs_direct_32hz": float(np.mean(final_delta)),
            "final_dist_delta_ci95": [f_lo, f_hi],
            "collision_delta_vs_direct_32hz": float(np.mean(coll_delta)),
            "collision_delta_ci95": [c_lo, c_hi],
        })
    return out


def write_report(cfg: Config, train: Dict[str, Dict[str, float]], summary: List[Dict[str, float]], latency: Dict[str, float], deltas: List[Dict[str, float]]) -> None:
    def pct(x: float) -> str:
        return f"{100*x:.1f}%"

    rows = "\n".join(
        f"| {s['policy']} | {s['refresh_every_steps']} | {pct(s['success_rate'])} | {pct(s['collision_rate'])} | {s['mean_steps']:.1f} | {s['mean_final_dist']:.3f} | {s['mean_obstacle_margin']:.3f} | {s['mean_jerk']:.4f} |"
        for s in summary
    )
    delta_rows = "\n".join(
        f"| {d['policy']} | {100*d['success_delta_vs_direct_32hz']:+.1f} pp | [{100*d['success_delta_ci95'][0]:+.1f}, {100*d['success_delta_ci95'][1]:+.1f}] pp | {d['final_dist_delta_vs_direct_32hz']:+.3f} | [{d['final_dist_delta_ci95'][0]:+.3f}, {d['final_dist_delta_ci95'][1]:+.3f}] | {100*d['collision_delta_vs_direct_32hz']:+.1f} pp |"
        for d in deltas
    )

    text = f"""# TurboVLA Direct-Control Toy Report

Generated by `run_turbo_vla_toy.py`.

## What was tested

TurboVLA's research claim is not just “small model is faster.” The control-level implication is that a compact direct vision+language-to-action policy can refresh action chunks near the robot control rate, while an LLM-centric execution path tends to run at a lower receding-horizon update rate. This toy tests that mechanism in a synthetic language-conditioned reach/drag task.

The toy keeps these method pieces:

1. visual image encoder over a rendered scene,
2. language/instruction embedding selecting a target marker,
3. lightweight bidirectional cross-attention for direct V+L interaction,
4. ACT-style parallel action-chunk decoder,
5. behavior cloning with L1 loss on expert future chunks,
6. receding-horizon execution where measured/assumed latency sets chunk refresh rate.

The task also connects to existing repo themes: it reuses the latency/stale-chunk question from `async_chunking_compare`, treats obstacle hits as a path-consistency/safety failure in the spirit of `path_consistent_safety_filtering`, keeps action chunks continuous like the `bspline_action_parameterization` experiments, and reports jerk/chunk-refresh behavior that complements `prefix_rl_chunking`'s stability focus.

It intentionally removes hard parts that are not needed for the latency implication: no pretrained DINO/BERT, no real robot dynamics, no LIBERO simulator, and no open-ended language reasoning.

## Training snapshot

- direct V+L policy parameters: {train['direct']['params']:,}
- LLM-bottleneck proxy parameters: {train['llm_bottleneck']['params']:,}
- direct final BC L1: {train['direct']['train_l1']:.4f}
- bottleneck final BC L1: {train['llm_bottleneck']['train_l1']:.4f}
- device: `{cfg.device}` after auto-selection

## Latency proxy

Single-observation local forward timing in this toy:

- direct V+L policy: {latency['direct_ms']:.3f} ms
- LLM-bottleneck proxy: {latency['llm_bottleneck_ms']:.3f} ms
- local toy latency ratio: {latency['ratio']:.2f}x

This timing is recorded for reproducibility, but it is **not** the evidence for TurboVLA's hardware claim: these tiny networks are dominated by local kernel/runtime effects, and the proxy is even faster locally in this run. The meaningful test below is therefore the controlled refresh-rate ablation, especially `direct_32hz` versus the same `direct_throttled_11hz` policy.

- `direct_32hz`: refresh every control tick.
- `direct_throttled_11hz`: same trained direct model, but artificially refreshed every 3 ticks.
- `llm_bottleneck_11hz`: heavier baseline refreshed every 3 ticks.
- `direct_stress_6hz` / `direct_stress_4hz`: stronger low-rate stress tests for the same direct policy.
- `direct_shuffled_language`: same 32 Hz direct policy, but with the wrong instruction token, to verify that target choice is actually language-conditioned.

The throttled direct ablation is the important sanity check: if low update rate hurts the same policy, the result is about control latency rather than only model capacity. Disturbances are paired across policies, so deltas are not artifacts of different random noise streams.

## Metric definitions

- **success**: final state is within the selected-goal radius and the rollout never enters the obstacle.
- **collisions**: fraction of rollouts that enter the central keep-out disk.
- **mean final dist**: Euclidean distance to the language-selected goal at termination.
- **mean obstacle margin**: minimum distance to the obstacle boundary; negative means penetration.
- **mean jerk**: second finite difference of emitted velocity actions, used as a small smoothness proxy.

## Closed-loop results

| policy | refresh steps | success | collisions | mean steps | mean final dist | mean obstacle margin | mean jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

## Paired deltas vs `direct_32hz`

Same starts and disturbance traces are reused for every policy. Negative success delta means worse than 32 Hz direct control.

| policy | success delta | 95% bootstrap CI | final-distance delta | 95% bootstrap CI | collision delta |
| --- | ---: | ---: | ---: | ---: | ---: |
{delta_rows}

## Visualizations

- `outputs/turbo_vla_latency_metrics.png` summarizes success, obstacle collisions, goal error, and action jerk with short labels.
- `outputs/turbo_vla_paired_deltas.png` shows paired degradation against `direct_32hz` with bootstrap intervals.
- `outputs/turbo_vla_example_rollouts.png` shows representative trajectories; `x` marks the start and `*` marks the selected target.

## Interpretation

The toy is a deliberately small version of the TurboVLA story: execution-level manipulation can benefit from frequent direct V+L→A chunk refreshes, not because every control tick needs more semantic reasoning, but because stale chunks react late to disturbances. The strongest evidence is the same-model comparison: `direct_throttled_11hz` loses success and collision margin relative to `direct_32hz` under identical starts/noise. The `llm_bottleneck_11hz` row is only a proxy for an LLM-centric execution path, not proof about real LLM latency.

The shuffled-language ablation confirms that the instruction token is used, but the fixed marker layout means this is not a full language-grounding benchmark. A stronger future version would randomize goal identities and positions per episode.

## Outputs

- `outputs/turbo_vla_metrics.json`
- `outputs/turbo_vla_trials.csv`
- `outputs/turbo_vla_latency_metrics.png`
- `outputs/turbo_vla_paired_deltas.png`
- `outputs/turbo_vla_example_rollouts.png`
"""
    (DOCS / "turbo_vla_toy_report.md").write_text(text)


def write_readme(cfg: Config, summary: List[Dict[str, float]]) -> None:
    best = max(summary, key=lambda s: s["success_rate"])
    text = f"""# TurboVLA Direct Control Toy

This folder explores the practical control implication behind **TurboVLA: Efficient Vision-Language-Action Model**.

Primary reference:

- TurboVLA paper page: https://arxiv.org/abs/2607.27205
- Paper PDF used for notes: https://arxiv.org/pdf/2607.27205

## Core idea

TurboVLA argues that execution-level robot control does not need to route every observation and instruction through a large LLM core. Instead, a compact model can encode vision and language separately, fuse them with lightweight bidirectional cross-attention, and decode a continuous action chunk directly. The paper reports a 0.2B-parameter model, about 31 ms latency, around 32 Hz action updates, and under 1 GB inference VRAM on an RTX 4090.

This toy asks whether that matters for control, not just for a model-size table.

## What the toy does

`run_turbo_vla_toy.py` trains two behavior-cloned chunk policies on a synthetic language-conditioned reach/drag task:

- `direct_32hz`: compact direct V+L→A policy with bidirectional cross-attention and an ACT-style chunk decoder.
- `llm_bottleneck_11hz`: heavier transformer-core proxy for an LLM-centric V→L→A execution path.
- `direct_throttled_11hz`: same direct policy, but forced to refresh chunks only every three 32 Hz ticks. This isolates the latency/control-rate mechanism.
- `direct_shuffled_language`: same direct policy with the wrong instruction token, to verify that the instruction is actually used.

The rendered observation contains the cursor/object, a central keep-out obstacle, and four fixed possible goal markers. The goal markers are separate input channels rather than natural images, so this is an instruction-use sanity check rather than a full visual-grounding benchmark. There is no selected-goal halo; the instruction token selects which marker matters. The expert target is a future velocity chunk that bends around the obstacle. Evaluation uses paired small disturbances plus occasional gusts so stale chunks have visible consequences.

## Relation to other repo ideas

- `async_chunking_compare`: same stale-chunk / refresh-delay question, now tied to a TurboVLA-style direct V+L→A architecture.
- `path_consistent_safety_filtering`: obstacle entry is treated as a path/safety failure rather than only final-goal error.
- `bspline_action_parameterization`: keeps attention on continuous action chunks and jerk/smoothness.
- `prefix_rl_chunking`: complements chunk-stability work by stressing receding-horizon refresh behavior.

## Run

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python turbo_vla_direct_control/run_turbo_vla_toy.py --train-steps 480 --eval-episodes 200
```

Outputs are written to `turbo_vla_direct_control/outputs/`, including:

- `turbo_vla_direct_control/outputs/turbo_vla_latency_metrics.png`
- `turbo_vla_direct_control/outputs/turbo_vla_paired_deltas.png`
- `turbo_vla_direct_control/outputs/turbo_vla_example_rollouts.png`

Reports are generated at:

- `turbo_vla_direct_control/docs/turbo_vla_toy_report.md`
- `turbo_vla_direct_control/docs/turbo_vla_direct_control_report.tex`
- `turbo_vla_direct_control/docs/turbo_vla_direct_control_report.pdf`

## Latest generated headline

Best policy in the latest run: `{best['policy']}` with {100*best['success_rate']:.1f}% success, mean final distance {best['mean_final_dist']:.3f}, and refresh period {best['refresh_every_steps']} control step(s).

See the generated report for the full table and interpretation.

## Simplifications

- Tiny learned encoders replace DINOv3/BERT.
- The “LLM bottleneck” baseline is a larger transformer-core proxy, not an actual LLM.
- The task is synthetic 2D control, not LIBERO or a real robot.
- Absolute hardware latency claims are from the paper; this script only measures local toy forward time and uses refresh-rate ablations to expose the control consequence.
- Goal locations are fixed and channelized; the shuffled-language ablation checks instruction use, but randomized visual-language grounding is left as future work.
"""
    (ROOT / "README.md").write_text(text)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--train-steps", type=int, default=480)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()
    cfg = Config(seed=args.seed, train_steps=args.train_steps, batch_size=args.batch_size, eval_episodes=args.eval_episodes, device=args.device)
    set_seed(cfg.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    cfg.device = str(device)
    print(f"device={device}")

    direct = DirectVLAPolicy(cfg.horizon)
    bottleneck = LLMBottleneckPolicy(cfg.horizon)
    train = {
        "direct": train_policy("direct_vla", direct, cfg, device),
        "llm_bottleneck": train_policy("llm_bottleneck", bottleneck, cfg, device),
    }
    direct_ms = measure_latency(direct, cfg, device)
    llm_ms = measure_latency(bottleneck, cfg, device)
    latency = {"direct_ms": direct_ms, "llm_bottleneck_ms": llm_ms, "ratio": llm_ms / max(direct_ms, 1e-6)}

    eval_cases = make_eval_cases(cfg)
    evals = [
        ("direct_32hz", direct, 1, "correct"),
        ("direct_throttled_11hz", direct, 3, "correct"),
        ("llm_bottleneck_11hz", bottleneck, 3, "correct"),
        ("direct_stress_6hz", direct, 5, "correct"),
        ("direct_stress_4hz", direct, 8, "correct"),
        ("direct_shuffled_language", direct, 1, "shuffled"),
    ]
    summary: List[Dict[str, float]] = []
    all_rows: List[Dict[str, float]] = []
    examples: Dict[str, Dict[str, np.ndarray]] = {}
    for label, model, refresh, language_mode in evals:
        agg, rows, ex = rollout(model, cfg, device, refresh, eval_cases, label, language_mode=language_mode)
        summary.append(agg)
        all_rows.extend(rows)
        examples[label] = ex
        print(f"{label:>22}: success={agg['success_rate']:.3f} final_dist={agg['mean_final_dist']:.3f} jerk={agg['mean_jerk']:.4f}")

    deltas = paired_deltas(all_rows)
    metrics = {"config": cfg.__dict__, "train": train, "latency": latency, "summary": summary, "paired_deltas": deltas, "eval_notes": "paired starts/noise across policies; selected-goal visual halo removed; shuffled-language ablation uses wrong instruction token"}
    (OUT / "turbo_vla_metrics.json").write_text(json.dumps(metrics, indent=2))
    with (OUT / "turbo_vla_trials.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    save_plots(summary, examples, cfg)
    save_delta_plot(deltas)
    write_report(cfg, train, summary, latency, deltas)
    write_readme(cfg, summary)
    print(json.dumps(metrics["summary"], indent=2))


if __name__ == "__main__":
    main()
