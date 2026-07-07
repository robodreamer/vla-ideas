import argparse
import csv
import os
import random
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE_DIR, ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

SEED = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    horizon: int = 14
    prefix_len: int = 5
    execute_len: int = 4
    action_dim: int = 2
    max_chunks: int = 18
    batch_episodes: int = 96
    ppo_iters: int = 42
    ppo_epochs: int = 3
    minibatch: int = 512
    gamma: float = 0.965
    gae_lambda: float = 0.90
    clip_range: float = 0.18
    value_coef: float = 0.35
    entropy_coef: float = 0.003
    prefix_coef: float = 0.70
    bc_steps: int = 120
    lr: float = 2.5e-4
    eval_episodes: int = 192
    goal: float = 1.0
    dt: float = 0.08


class ChunkPolicy(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        obs_dim = 4 + cfg.prefix_len * cfg.action_dim
        out_dim = cfg.horizon * cfg.action_dim
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 96),
            nn.Tanh(),
            nn.Linear(96, 96),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(96, out_dim)
        self.value_head = nn.Linear(96, 1)
        self.log_std = nn.Parameter(torch.full((cfg.horizon, cfg.action_dim), -0.55))

    def forward(self, obs):
        h = self.net(obs)
        mean = self.action_head(h).view(-1, self.cfg.horizon, self.cfg.action_dim)
        mean = 2.2 * torch.tanh(mean / 2.2)
        value = self.value_head(h).squeeze(-1)
        return mean, value

    def dist(self, obs):
        mean, value = self(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std, value


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_obs(x, v, idle, prefix, cfg: Config):
    goal_delta = cfg.goal - x
    return torch.cat([x[:, None], v[:, None], idle[:, None], goal_delta[:, None], prefix.flatten(1)], dim=1)


def teacher_chunk(x, v, idle, prefix, cfg: Config):
    """BC target: copy committed prefix, then continue with a smooth PD chunk."""
    batch = x.shape[0]
    out = torch.zeros(batch, cfg.horizon, cfg.action_dim, device=x.device)
    out[:, : cfg.prefix_len] = prefix

    xs = x.clone()
    vs = v.clone()
    idles = idle.clone()
    prev = prefix[:, -1].clone()
    for t in range(cfg.prefix_len, cfg.horizon):
        arm = 4.1 * (cfg.goal - xs) - 1.25 * vs
        # BC demonstrator is conservative and keeps the idle joint near zero.
        idle_u = -2.1 * idles - 0.18 * prev[:, 1]
        act = torch.stack([arm, idle_u], dim=1).clamp(-1.7, 1.7)
        # Smooth continuation from the copied prefix.
        act = 0.65 * act + 0.35 * prev
        out[:, t] = act
        xs, vs, idles, _, _ = env_step(xs, vs, idles, act, cfg, deterministic=True)
        prev = act
    return out


def env_step(x, v, idle, action, cfg: Config, deterministic=False):
    arm = action[:, 0].clamp(-2.6, 2.6)
    idle_u = action[:, 1].clamp(-2.6, 2.6)
    if deterministic:
        noise_v = torch.zeros_like(v)
        noise_idle = torch.zeros_like(idle)
    else:
        noise_v = 0.006 * torch.randn_like(v)
        noise_idle = 0.003 * torch.randn_like(idle)

    # Simple actuator-limited one-dimensional reacher plus an inactive joint.
    v_next = 0.86 * v + 0.105 * arm + noise_v
    x_next = x + cfg.dt * v_next
    idle_next = 0.975 * idle + 0.045 * idle_u + 0.010 * torch.tanh(arm) + noise_idle

    # Safety monitor: high acceleration near/through the hard stop or idle-arm drift trips a stop.
    hard_stop_force = torch.relu(x_next - 1.06) * 8.0
    impact_force = 0.18 * torch.abs(arm) + 0.65 * torch.relu(torch.abs(v_next) - 1.35)
    idle_force = 1.35 * torch.relu(torch.abs(idle_next) - 0.46)
    force = hard_stop_force + impact_force + idle_force
    safety = force > 2.10
    success = (torch.abs(x_next - cfg.goal) < 0.045) & (torch.abs(v_next) < 0.45) & (torch.abs(idle_next) < 0.22)
    return x_next, v_next, idle_next, safety, success


def gaussian_log_prob(actions, mean, std):
    var = std.pow(2)
    logp = -0.5 * (((actions - mean).pow(2) / var) + 2.0 * torch.log(std) + np.log(2.0 * np.pi))
    return logp.sum(dim=(1, 2))


def pretrain_bc(policy: ChunkPolicy, cfg: Config):
    opt = torch.optim.Adam(policy.parameters(), lr=7e-4)
    for _ in range(cfg.bc_steps):
        n = 256
        x = torch.empty(n, device=DEVICE).uniform_(-0.10, 0.75)
        v = torch.empty(n, device=DEVICE).uniform_(-0.35, 0.45)
        idle = torch.empty(n, device=DEVICE).uniform_(-0.18, 0.18)
        prefix = torch.randn(n, cfg.prefix_len, cfg.action_dim, device=DEVICE) * 0.24
        obs = make_obs(x, v, idle, prefix, cfg)
        target = teacher_chunk(x, v, idle, prefix, cfg)
        mean, value = policy(obs)
        # Tiny bootstrap value target: closer/faster states are better.
        v_target = 1.0 - torch.abs(cfg.goal - x) - 0.15 * torch.abs(v) - 0.25 * torch.abs(idle)
        loss = F.mse_loss(mean, target) + 0.10 * F.mse_loss(value, v_target)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()


def rollout(policy: ChunkPolicy, cfg: Config, deterministic=False, seed_offset=0):
    n = cfg.batch_episodes if not deterministic else cfg.eval_episodes
    x = torch.zeros(n, device=DEVICE) + 0.02 * torch.randn(n, device=DEVICE)
    v = 0.06 * torch.randn(n, device=DEVICE)
    idle = 0.03 * torch.randn(n, device=DEVICE)
    prefix = torch.zeros(n, cfg.prefix_len, cfg.action_dim, device=DEVICE)
    done = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    success_seen = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    safety_seen = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    chunks_to_finish = torch.full((n,), cfg.max_chunks, dtype=torch.float32, device=DEVICE)

    obs_buf, actions_buf, logp_buf, value_buf, reward_buf, done_buf, prefix_mse_buf = [], [], [], [], [], [], []
    trace = {"x": [], "idle": [], "a": []}
    prefix_mse_values = []

    for chunk_idx in range(cfg.max_chunks):
        obs = make_obs(x, v, idle, prefix, cfg)
        with torch.no_grad():
            mean, std, value = policy.dist(obs)
            if deterministic:
                actions = mean
            else:
                # Toy ODE-to-SDE stand-in: the chunk mean is the deterministic flow endpoint;
                # Gaussian noise supplies exploration and a tractable log probability.
                actions = mean + std * torch.randn_like(mean)
            logp = gaussian_log_prob(actions, mean, std)
            prefix_mse = (mean[:, : cfg.prefix_len] - prefix).pow(2).mean(dim=(1, 2))
            prefix_mse_values.append(prefix_mse.detach())

        chunk_reward = torch.zeros(n, device=DEVICE)
        for t in range(cfg.execute_len):
            active = ~done
            step_action = actions[:, t]
            x, v, idle, safety, success = env_step(x, v, idle, step_action, cfg, deterministic=deterministic)
            # Sparse/generic reward: success, recoverable overshoot, and safety termination.
            overshoot = (x > cfg.goal + 0.10) & (~safety) & (~success)
            r = -0.012 + 1.40 * success.float() - 0.22 * overshoot.float() - 1.15 * safety.float()
            r = r * active.float()
            chunk_reward += r
            newly_done = active & (success | safety)
            chunks_to_finish = torch.where(newly_done, torch.full_like(chunks_to_finish, float(chunk_idx + 1)), chunks_to_finish)
            success_seen |= active & success
            safety_seen |= active & safety
            done |= success | safety

            if deterministic:
                trace["x"].append(x.detach().cpu().numpy().copy())
                trace["idle"].append(idle.detach().cpu().numpy().copy())
                trace["a"].append(step_action[:, 0].detach().cpu().numpy().copy())

        # The model's unexecuted continuation becomes the next async prefix.
        prefix = actions[:, cfg.execute_len : cfg.execute_len + cfg.prefix_len].detach()

        if not deterministic:
            obs_buf.append(obs)
            actions_buf.append(actions)
            logp_buf.append(logp)
            value_buf.append(value)
            reward_buf.append(chunk_reward)
            done_buf.append(done.float())
            prefix_mse_buf.append(prefix_mse)

    final_error = torch.abs(x - cfg.goal)
    metrics = {
        "success": float(success_seen.float().mean().item()),
        "safety_stop": float(safety_seen.float().mean().item()),
        "mean_chunks": float(chunks_to_finish.mean().item()),
        "final_error": float(final_error.mean().item()),
        "idle_drift": float(torch.abs(idle).mean().item()),
        "prefix_mse": float(torch.cat(prefix_mse_values).mean().item()),
    }

    if deterministic:
        return metrics, trace

    return {
        "obs": torch.cat(obs_buf),
        "actions": torch.cat(actions_buf),
        "old_logp": torch.cat(logp_buf),
        "values": torch.stack(value_buf),
        "rewards": torch.stack(reward_buf),
        "dones": torch.stack(done_buf),
        "prefix_mse": torch.cat(prefix_mse_buf),
        "metrics": metrics,
    }


def compute_gae(values, rewards, dones, cfg: Config):
    t_steps, n = rewards.shape
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros(n, device=DEVICE)
    next_value = torch.zeros(n, device=DEVICE)
    for t in reversed(range(t_steps)):
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + cfg.gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
        adv[t] = last_gae
        next_value = values[t]
    returns = adv + values
    return adv.flatten(), returns.flatten()


def train_variant(name: str, use_prefix_loss: bool, initial_state: dict, cfg: Config):
    policy = ChunkPolicy(cfg).to(DEVICE)
    policy.load_state_dict(initial_state)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    history = []

    for it in range(cfg.ppo_iters + 1):
        eval_metrics, _ = rollout(policy, cfg, deterministic=True)
        eval_metrics.update({"iteration": it, "method": name})
        history.append(eval_metrics)
        if it == cfg.ppo_iters:
            break

        batch = rollout(policy, cfg, deterministic=False)
        adv, returns = compute_gae(batch["values"], batch["rewards"], batch["dones"], cfg)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        obs = batch["obs"].detach()
        actions = batch["actions"].detach()
        old_logp = batch["old_logp"].detach()
        returns = returns.detach()
        adv = adv.detach()

        total = obs.shape[0]
        for _ in range(cfg.ppo_epochs):
            order = torch.randperm(total, device=DEVICE)
            for start in range(0, total, cfg.minibatch):
                idx = order[start : start + cfg.minibatch]
                mean, std, value = policy.dist(obs[idx])
                logp = gaussian_log_prob(actions[idx], mean, std)
                ratio = torch.exp(logp - old_logp[idx])
                unclipped = -adv[idx] * ratio
                clipped = -adv[idx] * torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
                policy_loss = torch.maximum(unclipped, clipped).mean()
                value_loss = F.mse_loss(value, returns[idx])
                entropy = (0.5 + 0.5 * np.log(2 * np.pi) + torch.log(std)).sum(dim=(1, 2)).mean()
                prefix_target = obs[idx, 4:].view(-1, cfg.prefix_len, cfg.action_dim)
                prefix_loss = F.mse_loss(mean[:, : cfg.prefix_len], prefix_target)
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                if use_prefix_loss:
                    loss = loss + cfg.prefix_coef * prefix_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.8)
                opt.step()

    final_metrics, trace = rollout(policy, cfg, deterministic=True)
    return policy, history, final_metrics, trace


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def plot_results(histories, traces, cfg: Config):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {"bc_reference": "#4c78a8", "ppo_no_prefix_loss": "#e45756", "ppo_with_prefix_loss": "#54a24b"}
    labels = {"bc_reference": "BC reference", "ppo_no_prefix_loss": "PPO only", "ppo_with_prefix_loss": "PPO + prefix loss"}

    for name, rows in histories.items():
        it = [r["iteration"] for r in rows]
        axes[0, 0].plot(it, [r["success"] for r in rows], label=labels[name], color=colors[name])
        axes[0, 1].plot(it, [r["safety_stop"] for r in rows], label=labels[name], color=colors[name])
        axes[1, 0].plot(it, [r["prefix_mse"] for r in rows], label=labels[name], color=colors[name])

    for name, trace in traces.items():
        x = np.asarray(trace["x"])
        idle = np.asarray(trace["idle"])
        if x.size == 0:
            continue
        axes[1, 1].plot(x[:, :12].mean(axis=1), label=f"{labels[name]} x", color=colors[name])
        axes[1, 1].plot(idle[:, :12].mean(axis=1), linestyle="--", color=colors[name], alpha=0.75)

    axes[0, 0].set_title("success rate")
    axes[0, 1].set_title("safety-triggered stops")
    axes[1, 0].set_title("prefix-copy error")
    axes[1, 1].set_title("mean deterministic rollout: x solid, idle dashed")
    axes[1, 1].axhline(cfg.goal, color="#999999", lw=1, ls=":")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("PPO iteration" if ax is not axes[1, 1] else "control step")
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 0].legend(loc="lower right")
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "prefix_rl_chunking_summary.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description="Toy chunked-VLA PPO + prefix-loss chunked control demo")
    parser.add_argument("--quick", action="store_true", help="short smoke-test run")
    args = parser.parse_args()

    cfg = Config()
    if args.quick:
        cfg.bc_steps = 80
        cfg.ppo_iters = 4
        cfg.batch_episodes = 32
        cfg.eval_episodes = 48
        cfg.ppo_epochs = 1

    set_seed(SEED)
    base = ChunkPolicy(cfg).to(DEVICE)
    pretrain_bc(base, cfg)
    initial_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    histories = {}
    traces = {}
    final_rows = []

    bc_metrics, bc_trace = rollout(base, cfg, deterministic=True)
    bc_metrics.update({"iteration": 0, "method": "bc_reference"})
    histories["bc_reference"] = [bc_metrics for _ in range(cfg.ppo_iters + 1)]
    traces["bc_reference"] = bc_trace
    final_rows.append({"method": "bc_reference", **bc_metrics})

    for name, use_prefix in [("ppo_no_prefix_loss", False), ("ppo_with_prefix_loss", True)]:
        set_seed(SEED + (13 if use_prefix else 11))
        _, hist, final_metrics, trace = train_variant(name, use_prefix, initial_state, cfg)
        histories[name] = hist
        traces[name] = trace
        final_rows.append({"method": name, **final_metrics})

    curve_rows = []
    for rows in histories.values():
        curve_rows.extend(rows)

    curve_path = os.path.join(OUTPUT_DIR, "prefix_rl_chunking_training_curves.csv")
    metrics_path = os.path.join(OUTPUT_DIR, "prefix_rl_chunking_metrics.csv")
    fields = ["method", "iteration", "success", "safety_stop", "mean_chunks", "final_error", "idle_drift", "prefix_mse"]
    write_csv(curve_path, curve_rows, fields)
    write_csv(metrics_path, final_rows, ["method", "success", "safety_stop", "mean_chunks", "final_error", "idle_drift", "prefix_mse"])
    plot_path = plot_results(histories, traces, cfg)

    print("Wrote:")
    print(f"  {metrics_path}")
    print(f"  {curve_path}")
    print(f"  {plot_path}")
    print("Final metrics:")
    for row in final_rows:
        print(
            f"  {row['method']}: success={row['success']:.3f}, safety={row['safety_stop']:.3f}, "
            f"chunks={row['mean_chunks']:.2f}, idle_drift={row['idle_drift']:.3f}, "
            f"prefix_mse={row['prefix_mse']:.4f}"
        )


if __name__ == "__main__":
    main()
