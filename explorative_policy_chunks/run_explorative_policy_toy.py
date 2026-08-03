#!/usr/bin/env python3
"""Toy Explorative Policy experiment for multimodal VLA action chunks.

The toy is a small analogue of the Explorative Modeling / Explorative Policy
claim: if demonstrations contain several valid futures, a one-shot MSE policy
learns the blurry average, while a best-of-K explorative objective can keep
multiple committed action chunks and still run in a single forward pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE_DIR, ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

COLORS = {
    "bc_k1": "#4c78a8",
    "xm_k2": "#f58518",
    "xm_k4": "#54a24b",
    "xm_k8": "#e45756",
}


@dataclass
class Config:
    seed: int = 11
    train_size: int = 4096
    test_size: int = 1024
    horizon: int = 32
    obstacle_radius: float = 0.165
    collision_margin: float = 0.018
    steps: int = 900
    batch_size: int = 256
    lr: float = 2.5e-3
    hidden: int = 96
    train_noise: float = 0.010
    endpoint_tol: float = 0.075
    head_bias: float = 0.24
    diversity_weight: float = 0.002
    k_values: Tuple[int, ...] = (1, 2, 4, 8)


def make_dataset(n: int, cfg: Config, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ambiguous two-route demonstrations around a circular keep-out zone."""
    # Context visible to the policy: start/goal vertical offsets and obstacle y.
    start_y = rng.uniform(-0.10, 0.10, size=n)
    goal_y = rng.uniform(-0.10, 0.10, size=n)
    obs_y = rng.uniform(-0.055, 0.055, size=n)
    obs_r = np.full(n, cfg.obstacle_radius)
    ctx = np.stack([start_y, goal_y, obs_y, obs_r], axis=1).astype(np.float32)

    # Hidden expert mode: both over and under routes are valid; demos are mixed.
    mode = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)
    t = np.linspace(0.0, 1.0, cfg.horizon, dtype=np.float32)
    x = t[None, :]
    base_y = (1.0 - t)[None, :] * start_y[:, None] + t[None, :] * goal_y[:, None]
    # Push the trajectory above/below the obstacle mostly near the middle.
    bump = np.sin(np.pi * t)[None, :]
    clearance = cfg.obstacle_radius + 0.17 + rng.normal(0.0, 0.015, size=(n, 1))
    y = base_y + bump * (obs_y[:, None] + mode[:, None] * clearance)
    y += rng.normal(0.0, cfg.train_noise, size=y.shape) * np.sin(np.pi * t)[None, :]
    traj = np.stack([np.broadcast_to(x, y.shape), y], axis=-1).astype(np.float32)
    traj[:, 0, 0] = 0.0
    traj[:, 0, 1] = start_y
    traj[:, -1, 0] = 1.0
    traj[:, -1, 1] = goal_y
    return ctx, traj, mode.astype(np.float32)


class ChunkPolicy(nn.Module):
    def __init__(self, k: int, horizon: int, hidden: int, head_bias: float):
        super().__init__()
        self.k = k
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, k * horizon * 2),
        )
        # Break head symmetry so best-of-K has something to specialize from.
        with torch.no_grad():
            final = self.net[-1]
            final.bias.zero_()
            t = torch.linspace(0.0, 1.0, horizon)
            for head in range(k):
                if k == 1:
                    sign = 0.0
                else:
                    sign = -1.0 + 2.0 * head / max(1, k - 1)
                y_bias = head_bias * sign * torch.sin(torch.pi * t)
                final.bias[head * horizon * 2 + 1 : (head + 1) * horizon * 2 : 2] = y_bias

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        out = self.net(ctx)
        return out.view(ctx.shape[0], self.k, self.horizon, 2)


def train_policy(k: int, cfg: Config, train_ctx: np.ndarray, train_traj: np.ndarray) -> Tuple[ChunkPolicy, List[float]]:
    torch.manual_seed(cfg.seed + k)
    model = ChunkPolicy(k, cfg.horizon, cfg.hidden, cfg.head_bias)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    ctx_t = torch.from_numpy(train_ctx)
    traj_t = torch.from_numpy(train_traj)
    losses: List[float] = []
    n = ctx_t.shape[0]
    for step in range(cfg.steps):
        idx = torch.randint(0, n, (cfg.batch_size,))
        pred = model(ctx_t[idx])
        target = traj_t[idx][:, None, :, :]
        per_head = ((pred - target) ** 2).mean(dim=(2, 3))
        recon_loss = per_head.min(dim=1).values.mean()
        loss = recon_loss
        # A tiny anti-collapse regularizer keeps duplicate heads from becoming identical,
        # but the reconstruction term remains the dominant XM best-of-K objective.
        if k > 1 and cfg.diversity_weight > 0.0:
            head_means = pred.mean(dim=2)
            pair_dist = torch.cdist(head_means, head_means).mean()
            loss = loss - cfg.diversity_weight * pair_dist.clamp(max=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 25 == 0:
            losses.append(float(recon_loss.detach().cpu()))
    return model, losses


def collision_penalty(paths: np.ndarray, ctx: np.ndarray, cfg: Config) -> np.ndarray:
    """Return a soft collision score for B x K x T x 2 paths."""
    obs = np.stack([np.full(ctx.shape[0], 0.50), ctx[:, 2]], axis=1)[:, None, None, :]
    dist = np.linalg.norm(paths - obs, axis=-1)
    threshold = ctx[:, 3][:, None, None] + cfg.collision_margin
    penetration = np.maximum(0.0, threshold - dist)
    return penetration.max(axis=-1)  # B x K


def score_candidates(paths: np.ndarray, ctx: np.ndarray, cfg: Config) -> np.ndarray:
    collisions = collision_penalty(paths, ctx, cfg)
    goal = np.stack([np.ones(ctx.shape[0]), ctx[:, 1]], axis=1)[:, None, :]
    goal_err = np.linalg.norm(paths[:, :, -1, :] - goal, axis=-1)
    jerk = np.diff(paths, n=2, axis=2)
    smooth = np.mean(np.linalg.norm(jerk, axis=-1), axis=-1)
    # Large weight for collision, then endpoint and smoothness as tie breakers.
    return 250.0 * collisions + 3.0 * goal_err + 0.15 * smooth


def evaluate_model(
    model: ChunkPolicy,
    cfg: Config,
    test_ctx: np.ndarray,
    test_traj: np.ndarray,
    test_mode: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    with torch.no_grad():
        cand = model(torch.from_numpy(test_ctx)).cpu().numpy()
    scores = score_candidates(cand, test_ctx, cfg)
    chosen_idx = np.argmin(scores, axis=1)
    chosen = cand[np.arange(cand.shape[0]), chosen_idx]
    coll = collision_penalty(chosen[:, None, :, :], test_ctx, cfg)[:, 0] > 1e-6
    candidate_coll = collision_penalty(cand, test_ctx, cfg) > 1e-6
    goal = np.stack([np.ones(test_ctx.shape[0]), test_ctx[:, 1]], axis=1)
    final_err = np.linalg.norm(chosen[:, -1, :] - goal, axis=1)
    # The hidden route mode is intentionally not in the context. Feasibility success asks
    # whether the policy emits at least one safe committed future; oracle MSE asks whether
    # one candidate also matches the particular held-out demonstration mode.
    per_head_mse = ((cand - test_traj[:, None, :, :]) ** 2).mean(axis=(2, 3))
    nearest_mse = per_head_mse.min(axis=1)
    chosen_mse = ((chosen - test_traj) ** 2).mean(axis=(1, 2))
    y_mid = cand[:, :, cfg.horizon // 2, 1] - test_ctx[:, None, 2]
    head_sides = np.sign(y_mid)
    both_sides = (np.any(head_sides > 0, axis=1) & np.any(head_sides < 0, axis=1)).mean()
    side_separation = (np.max(y_mid, axis=1) - np.min(y_mid, axis=1)).mean()
    oracle_mode_match = np.any(head_sides == test_mode[:, None], axis=1).mean()
    any_collision_free = np.any(~candidate_coll, axis=1).mean()
    chosen_over_rate = ((chosen[:, cfg.horizon // 2, 1] - test_ctx[:, 2]) > 0).mean()
    success = (~coll & (final_err < cfg.endpoint_tol)).mean()
    metrics = {
        "k": float(model.k),
        "success_rate": float(success),
        "collision_rate": float(coll.mean()),
        "any_collision_free_candidate_rate": float(any_collision_free),
        "final_error_mean": float(final_err.mean()),
        "chosen_mse_mean": float(chosen_mse.mean()),
        "oracle_best_mse_mean": float(nearest_mse.mean()),
        "both_sides_coverage": float(both_sides),
        "oracle_mode_match_rate": float(oracle_mode_match),
        "midpoint_side_separation_mean": float(side_separation),
        "chosen_over_rate": float(chosen_over_rate),
        "inference_forwards": 1.0,
        "candidate_chunks": float(model.k),
    }
    return metrics, cand, chosen


def write_metrics(metrics: List[Dict[str, float]], cfg: Config) -> None:
    json_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "metrics": metrics}, f, indent=2)
    csv_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)


def plot_k_sweep(metrics: List[Dict[str, float]]) -> None:
    ks = [int(m["k"]) for m in metrics]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    ax[0].plot(ks, [m["success_rate"] for m in metrics], marker="o", color="#54a24b")
    ax[0].set_ylabel("success rate")
    ax[0].set_ylim(-0.03, 1.03)
    ax[1].plot(ks, [m["collision_rate"] for m in metrics], marker="o", color="#e45756")
    ax[1].set_ylabel("collision rate")
    ax[1].set_ylim(-0.03, 1.03)
    ax[2].plot(ks, [m["oracle_best_mse_mean"] for m in metrics], marker="o", color="#4c78a8")
    ax[2].set_ylabel("oracle best trajectory MSE")
    for a in ax:
        a.set_xlabel("exploration K")
        a.grid(alpha=0.25)
        a.set_xticks(ks)
    fig.suptitle("Best-of-K exploration turns ambiguous demos into committed chunks")
    fig.savefig(os.path.join(OUTPUT_DIR, "exploration_k_sweep.png"), dpi=180)
    plt.close(fig)


def plot_training_curves(losses_by_k: Dict[int, List[float]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    for k, losses in losses_by_k.items():
        xs = np.arange(1, len(losses) + 1) * 25
        ax.plot(xs, losses, label=f"K={k}", color=COLORS.get(f"xm_k{k}", None))
    ax.set_xlabel("training step")
    ax.set_ylabel("pure best-of-K reconstruction loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=180)
    plt.close(fig)


def plot_representative(
    cfg: Config,
    test_ctx: np.ndarray,
    test_traj: np.ndarray,
    candidates_by_k: Dict[int, np.ndarray],
    chosen_by_k: Dict[int, np.ndarray],
) -> None:
    idx = int(np.argmin(np.abs(test_ctx[:, 2])))
    ctx = test_ctx[idx]
    obs = plt.Circle((0.5, ctx[2]), ctx[3] + cfg.collision_margin, color="#e45756", alpha=0.22)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for a in ax:
        a.add_patch(plt.Circle((0.5, ctx[2]), ctx[3] + cfg.collision_margin, color="#e45756", alpha=0.22))
        a.plot(test_traj[idx, :, 0], test_traj[idx, :, 1], "k--", lw=1.7, label="held-out demo")
        a.scatter([0, 1], [ctx[0], ctx[1]], c=["#4c78a8", "#54a24b"], s=45, zorder=5)
        a.set_xlim(-0.04, 1.04)
        a.set_ylim(-0.62, 0.62)
        a.set_aspect("equal", adjustable="box")
        a.grid(alpha=0.2)
    ax[0].set_title("BC / K=1 averages modes")
    ax[0].plot(chosen_by_k[1][idx, :, 0], chosen_by_k[1][idx, :, 1], color=COLORS["bc_k1"], lw=2.4, label="chosen chunk")
    ax[1].set_title("XM / K=4 offers committed chunks")
    cand = candidates_by_k[4][idx]
    for j, p in enumerate(cand):
        ax[1].plot(p[:, 0], p[:, 1], lw=1.3, alpha=0.42, color="#777777")
    ax[1].plot(chosen_by_k[4][idx, :, 0], chosen_by_k[4][idx, :, 1], color=COLORS["xm_k4"], lw=2.7, label="feasibility-selected chunk")
    for a in ax:
        a.legend(frameon=False, loc="upper right")
    fig.savefig(os.path.join(OUTPUT_DIR, "representative_trajectories.png"), dpi=180)
    plt.close(fig)


def write_report(metrics: List[Dict[str, float]], cfg: Config) -> None:
    by_k = {int(m["k"]): m for m in metrics}
    def pct(v: float) -> str:
        return f"{100.0 * v:.1f}%"
    report = f"""# Explorative Policy Chunks Toy Report

## Why this belongs in `vla-ideas`

Explorative Modeling argues that multimodality can be handled by factoring *training* instead of repeatedly factoring *generation*. The VLA-relevant version is an action-chunk policy: train K candidate chunks with a best-of-K reconstruction loss, then execute with one network forward pass instead of averaging all valid futures into an unsafe mean chunk.

This toy keeps the robotics part intentionally small. A point robot must move from left to right around a circular keep-out zone. Demonstrations contain two equally good modes, above and below the obstacle, while the policy input does not reveal which mode the demonstrator picked. A standard one-shot behavior-cloning policy trained with MSE therefore predicts the mean of the two modes, which cuts through the obstacle. An Explorative Policy-style multi-head chunker trains only the closest candidate head for each demo, so different heads can specialize to different route modes. The implementation deliberately includes small head-specific route biases and a weak anti-collapse term; the claim here is not that best-of-K alone always discovers modes from a symmetric initialization, but that best-of-K credit assignment can preserve seeded candidate futures instead of averaging them away.

## Mapping to the XM idea

- **K=1 / BC**: ordinary reconstructive behavior cloning; one generated chunk is compared to the expert chunk.
- **K>1 / XM**: the policy emits K candidate chunks; the loss backpropagates only through the candidate closest to the expert chunk. This mirrors Forward XM's best-of-K credit assignment, amortized here as K action heads rather than K separate latent samples. The tiny route-bias initialization is a toy stand-in for whatever stochasticity, latent conditioning, or architectural asymmetry seeds distinct futures in a larger model.
- **Single forward inference**: all candidate chunks come from one model call; a cheap geometric critic selects a feasible committed chunk for this toy. The critic is deliberately simple and only proves that the candidates include safe committed modes, not that the hidden demonstrator choice can be recovered from unobserved information.
- **Simplification**: the real paper studies images, video, language, robot manipulation, and world models. This script isolates only the multimodal action-chunk mechanism.

## Source notes

- Project page: https://explorative-modeling.github.io/
- Code repository: https://github.com/alexiglad/XM
- The repo README describes Forward XM as exploring K candidate outputs and backpropagating only through the closest candidate; it also notes that `K = 1` is the no-exploration baseline.

## Latest metrics

| method | success | collision | any safe cand. | oracle best MSE | chosen MSE | both-side coverage | side sep. | forwards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for k in cfg.k_values:
        m = by_k[k]
        label = "BC K=1" if k == 1 else f"XM K={k}"
        report += (
            f"| {label} | {pct(m['success_rate'])} | {pct(m['collision_rate'])} | "
            f"{pct(m['any_collision_free_candidate_rate'])} | {m['oracle_best_mse_mean']:.5f} | "
            f"{m['chosen_mse_mean']:.5f} | {pct(m['both_sides_coverage'])} | "
            f"{m['midpoint_side_separation_mean']:.3f} | {m['inference_forwards']:.0f} |\n"
        )
    report += f"""

## Result sanity checks

- The held-out demo route is intentionally hidden, so the selected safe chunk is not expected to match the demonstrator's arbitrary over/under choice. That is why `chosen MSE` can be worse than the K=1 average even when task success is perfect.
- `oracle best MSE` measures whether one of the emitted candidates matches the hidden route. It drops from K=1's averaged path to near-zero for K≥2.
- `both-side coverage` checks that the multi-head model is actually representing over/under alternatives, not just outputting duplicate safe curves. `side sep.` is only a rough diagnostic because extra heads for K=4/8 can drift into non-useful outlier routes.
- `any safe cand.` separates candidate expressivity from the toy critic, but it is not the strongest evidence here; `oracle best MSE` is the main mode-preservation check.
- Limitation: K is a candidate budget, not a guarantee that every head becomes a meaningful mode. In this two-mode toy, K=4/8 usually still reduce to two useful over/under candidates plus redundant or outlier heads. Without the route-bias initialization, this small deterministic toy often collapses back toward the averaged solution.

## Generated artifacts

- `outputs/metrics.json` and `outputs/metrics.csv`
- `outputs/exploration_k_sweep.png`
- `outputs/representative_trajectories.png`
- `outputs/training_curves.png`

## Takeaway

The useful distilled idea is not “use XM as-is for everything,” nor that best-of-K removes the need to seed diversity. It is: for VLA action chunks, best-of-K credit assignment plus a source of candidate asymmetry is a compact way to keep several plausible futures alive without paying diffusion-style iterative inference at deployment time. In this toy, K=1 blurs the two route modes; K≥2 with seeded heads can represent both over/under chunks and lets a light safety or task critic pick a committed trajectory.

## Run

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python explorative_policy_chunks/run_explorative_policy_toy.py --steps {cfg.steps}
```
"""
    with open(os.path.join(DOCS_DIR, "explorative_policy_toy_report.md"), "w", encoding="utf-8") as f:
        f.write(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--train-size", type=int, default=Config.train_size)
    parser.add_argument("--test-size", type=int, default=Config.test_size)
    args = parser.parse_args()
    cfg = Config(seed=args.seed, steps=args.steps, train_size=args.train_size, test_size=args.test_size)
    rng = np.random.default_rng(cfg.seed)
    train_ctx, train_traj, _ = make_dataset(cfg.train_size, cfg, rng)
    test_ctx, test_traj, test_mode = make_dataset(cfg.test_size, cfg, rng)

    metrics: List[Dict[str, float]] = []
    losses_by_k: Dict[int, List[float]] = {}
    candidates_by_k: Dict[int, np.ndarray] = {}
    chosen_by_k: Dict[int, np.ndarray] = {}
    for k in cfg.k_values:
        model, losses = train_policy(k, cfg, train_ctx, train_traj)
        m, cand, chosen = evaluate_model(model, cfg, test_ctx, test_traj, test_mode)
        metrics.append(m)
        losses_by_k[k] = losses
        candidates_by_k[k] = cand
        chosen_by_k[k] = chosen
        print(
            f"K={k}: success={m['success_rate']:.3f} collision={m['collision_rate']:.3f} "
            f"oracle_mse={m['oracle_best_mse_mean']:.5f} coverage={m['both_sides_coverage']:.3f}"
        )

    write_metrics(metrics, cfg)
    plot_k_sweep(metrics)
    plot_training_curves(losses_by_k)
    plot_representative(cfg, test_ctx, test_traj, candidates_by_k, chosen_by_k)
    write_report(metrics, cfg)
    print(f"Wrote outputs to {OUTPUT_DIR}")
    print(f"Wrote report to {os.path.join(DOCS_DIR, 'explorative_policy_toy_report.md')}")


if __name__ == "__main__":
    main()
