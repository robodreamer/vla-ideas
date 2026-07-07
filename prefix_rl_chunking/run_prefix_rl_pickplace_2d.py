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
from matplotlib.patches import Circle, Rectangle

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

SEED = 23
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    horizon: int = 24
    prefix_len: int = 7
    execute_len: int = 5
    action_dim: int = 3  # ax, ay, grip/open command
    max_chunks: int = 22
    batch_episodes: int = 80
    eval_episodes: int = 160
    ppo_iters: int = 34
    ppo_epochs: int = 3
    minibatch: int = 512
    bc_steps: int = 420
    lr: float = 2.2e-4
    gamma: float = 0.965
    gae_lambda: float = 0.90
    clip_range: float = 0.18
    value_coef: float = 0.35
    entropy_coef: float = 0.002
    prefix_coef: float = 0.85
    dt: float = 0.075
    obj_radius: float = 0.045
    grasp_radius: float = 0.105
    goal_radius: float = 0.095
    max_accel: float = 2.8


OBSTACLE = torch.tensor([0.42, 0.33, 0.62, 0.68], device=DEVICE)  # xmin, ymin, xmax, ymax
GOAL = torch.tensor([0.88, 0.82], device=DEVICE)
OBJ_START = torch.tensor([0.20, 0.24], device=DEVICE)


class PickPlaceChunkPolicy(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        obs_dim = 12 + cfg.prefix_len * cfg.action_dim
        out_dim = cfg.horizon * cfg.action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 160),
            nn.Tanh(),
            nn.Linear(160, 160),
            nn.Tanh(),
            nn.Linear(160, 96),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(96, out_dim)
        self.value_head = nn.Linear(96, 1)
        self.log_std = nn.Parameter(torch.tensor([-0.45, -0.45, -0.70]).repeat(cfg.horizon, 1))

    def forward(self, obs):
        h = self.net(obs)
        raw = self.action_head(h).view(-1, self.cfg.horizon, self.cfg.action_dim)
        xy = self.cfg.max_accel * torch.tanh(raw[:, :, :2] / self.cfg.max_accel)
        grip = 1.7 * torch.tanh(raw[:, :, 2:3] / 1.7)
        mean = torch.cat([xy, grip], dim=2)
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


def inside_obstacle(pos):
    return (
        (pos[:, 0] > OBSTACLE[0])
        & (pos[:, 0] < OBSTACLE[2])
        & (pos[:, 1] > OBSTACLE[1])
        & (pos[:, 1] < OBSTACLE[3])
    )


def make_obs(ee, vel, obj, grip, carried, prefix, cfg: Config):
    goal = GOAL.expand(ee.shape[0], 2)
    rel_obj = obj - ee
    rel_goal = goal - obj
    scalar = torch.stack([grip, carried], dim=1)
    return torch.cat([ee, vel, obj, rel_obj, rel_goal, scalar, prefix.flatten(1)], dim=1)


def reset_world(n, jitter=True):
    if jitter:
        ee = torch.tensor([0.08, 0.16], device=DEVICE).expand(n, 2).clone() + 0.025 * torch.randn(n, 2, device=DEVICE)
        obj = OBJ_START.expand(n, 2).clone() + 0.030 * torch.randn(n, 2, device=DEVICE)
    else:
        ee = torch.tensor([0.08, 0.16], device=DEVICE).expand(n, 2).clone()
        obj = OBJ_START.expand(n, 2).clone()
    vel = torch.zeros(n, 2, device=DEVICE)
    grip = torch.zeros(n, device=DEVICE)
    carried = torch.zeros(n, device=DEVICE)
    return ee, vel, obj, grip, carried


def env_step(ee, vel, obj, grip, carried, action, cfg: Config, deterministic=False):
    accel = action[:, :2].clamp(-cfg.max_accel, cfg.max_accel)
    grip_cmd = action[:, 2].clamp(-1.7, 1.7)
    noise = torch.zeros_like(vel) if deterministic else 0.004 * torch.randn_like(vel)
    vel_next = 0.84 * vel + 0.080 * accel + noise
    ee_next = ee + cfg.dt * vel_next
    grip_next = (0.92 * grip + 0.16 * grip_cmd).clamp(-1.0, 1.0)

    dist_to_obj = torch.linalg.norm(ee_next - obj, dim=1)
    newly_grasped = (carried < 0.5) & (grip_next > 0.35) & (dist_to_obj < cfg.grasp_radius)
    still_carried = (carried > 0.5) & (grip_next > -0.25)
    carried_next = (newly_grasped | still_carried).float()

    obj_follow = ee_next + torch.tensor([0.018, -0.020], device=DEVICE)
    obj_next = torch.where(carried_next[:, None] > 0.5, 0.72 * obj + 0.28 * obj_follow, obj)

    out_of_bounds = (ee_next[:, 0] < -0.10) | (ee_next[:, 0] > 1.12) | (ee_next[:, 1] < -0.10) | (ee_next[:, 1] > 1.12)
    obstacle_hit = inside_obstacle(ee_next) | inside_obstacle(obj_next)
    hard_contact = (torch.linalg.norm(accel, dim=1) > 2.55) & (dist_to_obj < cfg.obj_radius * 1.4) & (carried < 0.5)
    force_proxy = obstacle_hit | out_of_bounds | hard_contact
    success = (torch.linalg.norm(obj_next - GOAL, dim=1) < cfg.goal_radius) & (carried_next > 0.5) & (torch.linalg.norm(vel_next, dim=1) < 0.75)
    return ee_next, vel_next, obj_next, grip_next, carried_next, force_proxy, success


def teacher_action(ee, vel, obj, grip, carried, cfg: Config, speed_scale=1.0):
    n = ee.shape[0]
    pregrasp = obj + torch.tensor([-0.035, -0.015], device=DEVICE)
    lift = torch.tensor([0.26, 0.82], device=DEVICE).expand(n, 2)
    over = torch.tensor([0.74, 0.84], device=DEVICE).expand(n, 2)
    goal = GOAL.expand(n, 2)

    d_obj = torch.linalg.norm(ee - pregrasp, dim=1)
    d_lift = torch.linalg.norm(ee - lift, dim=1)
    d_over = torch.linalg.norm(ee - over, dim=1)
    obj_to_goal = torch.linalg.norm(obj - goal, dim=1)

    target = torch.where((carried < 0.5)[:, None], pregrasp, lift)
    target = torch.where(((carried > 0.5) & (d_lift < 0.13))[:, None], over, target)
    target = torch.where(((carried > 0.5) & (d_over < 0.18))[:, None], goal, target)

    close = ((carried < 0.5) & (d_obj < 0.11)) | ((carried > 0.5) & (obj_to_goal > cfg.goal_radius))
    grip_target = torch.where(close, torch.ones(n, device=DEVICE), -0.55 * torch.ones(n, device=DEVICE))

    kp = 4.3 * speed_scale
    kd = 1.25
    accel = kp * (target - ee) - kd * vel
    # Conservative demonstrator slows near obstacle to avoid contact.
    near_obstacle = (ee[:, 0] > 0.34) & (ee[:, 0] < 0.70) & (ee[:, 1] > 0.22) & (ee[:, 1] < 0.76)
    accel = torch.where(near_obstacle[:, None], 0.78 * accel, accel)
    return torch.cat([accel.clamp(-cfg.max_accel, cfg.max_accel), grip_target[:, None]], dim=1)


def teacher_chunk(ee, vel, obj, grip, carried, prefix, cfg: Config, speed_scale=1.0):
    n = ee.shape[0]
    chunk = torch.zeros(n, cfg.horizon, cfg.action_dim, device=DEVICE)
    chunk[:, : cfg.prefix_len] = prefix
    ee_i, vel_i, obj_i, grip_i, carried_i = ee.clone(), vel.clone(), obj.clone(), grip.clone(), carried.clone()
    for t in range(cfg.prefix_len, cfg.horizon):
        act = teacher_action(ee_i, vel_i, obj_i, grip_i, carried_i, cfg, speed_scale=speed_scale)
        if t > cfg.prefix_len:
            act[:, :2] = 0.75 * act[:, :2] + 0.25 * chunk[:, t - 1, :2]
        chunk[:, t] = act
        ee_i, vel_i, obj_i, grip_i, carried_i, _, _ = env_step(
            ee_i, vel_i, obj_i, grip_i, carried_i, act, cfg, deterministic=True
        )
    return chunk


def gaussian_log_prob(actions, mean, std):
    var = std.pow(2)
    logp = -0.5 * (((actions - mean).pow(2) / var) + 2.0 * torch.log(std) + np.log(2.0 * np.pi))
    return logp.sum(dim=(1, 2))


def pretrain_bc(policy, cfg: Config):
    opt = torch.optim.Adam(policy.parameters(), lr=7e-4)
    for _ in range(cfg.bc_steps):
        n = 192
        ee, vel, obj, grip, carried = reset_world(n, jitter=True)
        # Mix initial, near-grasp, carrying, and near-goal states so BC covers all phases.
        phase = torch.rand(n, device=DEVICE)
        near_grasp = phase > 0.35
        carrying = phase > 0.58
        near_goal = phase > 0.82
        ee = torch.where(near_grasp[:, None], obj + 0.06 * torch.randn(n, 2, device=DEVICE), ee)
        obj = torch.where(carrying[:, None], ee + torch.tensor([0.02, -0.02], device=DEVICE), obj)
        carried = torch.where(carrying, torch.ones_like(carried), carried)
        grip = torch.where(carrying, torch.ones_like(grip) * 0.65, grip)
        obj = torch.where(near_goal[:, None], GOAL + 0.12 * torch.randn(n, 2, device=DEVICE), obj)
        ee = torch.where(near_goal[:, None], obj + 0.06 * torch.randn(n, 2, device=DEVICE), ee)

        prefix = 0.22 * torch.randn(n, cfg.prefix_len, cfg.action_dim, device=DEVICE)
        prefix[:, :, 2] *= 0.5
        obs = make_obs(ee, vel, obj, grip, carried, prefix, cfg)
        target = teacher_chunk(ee, vel, obj, grip, carried, prefix, cfg, speed_scale=1.0)
        mean, value = policy(obs)
        value_target = 1.0 - torch.linalg.norm(obj - GOAL, dim=1) - 0.25 * torch.linalg.norm(ee - obj, dim=1)
        loss = F.mse_loss(mean, target) + 0.08 * F.mse_loss(value, value_target)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()


def rollout(policy, cfg: Config, deterministic=False):
    n = cfg.eval_episodes if deterministic else cfg.batch_episodes
    # Evaluation keeps deterministic dynamics but uses reset jitter to measure robustness
    # across small object/robot pose variations rather than a single identical rollout.
    ee, vel, obj, grip, carried = reset_world(n, jitter=True)
    prefix = torch.zeros(n, cfg.prefix_len, cfg.action_dim, device=DEVICE)
    done = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    success_seen = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    safety_seen = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    grasp_seen = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    chunks_to_finish = torch.full((n,), cfg.max_chunks, dtype=torch.float32, device=DEVICE)

    obs_buf, action_buf, logp_buf, value_buf, reward_buf, done_buf = [], [], [], [], [], []
    prefix_mse_values = []
    trace = {"ee": [], "obj": [], "grip": [], "carried": []}

    for chunk_idx in range(cfg.max_chunks):
        obs = make_obs(ee, vel, obj, grip, carried, prefix, cfg)
        with torch.no_grad():
            mean, std, value = policy.dist(obs)
            # Flow-policy proxy: deterministic chunk mean plus Gaussian SDE-style noise.
            actions = mean if deterministic else mean + std * torch.randn_like(mean)
            logp = gaussian_log_prob(actions, mean, std)
            prefix_mse = (mean[:, : cfg.prefix_len] - prefix).pow(2).mean(dim=(1, 2))
            prefix_mse_values.append(prefix_mse.detach())

        chunk_reward = torch.zeros(n, device=DEVICE)
        for t in range(cfg.execute_len):
            active = ~done
            step_action = actions[:, cfg.prefix_len + t]
            prev_dist_goal = torch.linalg.norm(obj - GOAL, dim=1)
            ee, vel, obj, grip, carried, safety, success = env_step(
                ee, vel, obj, grip, carried, step_action, cfg, deterministic=deterministic
            )
            dist_goal = torch.linalg.norm(obj - GOAL, dim=1)
            dist_obj = torch.linalg.norm(ee - obj, dim=1)
            grasp_bonus = ((carried > 0.5) & (~grasp_seen)).float() * 0.20
            progress = (prev_dist_goal - dist_goal).clamp(-0.03, 0.03)
            # Mostly sparse reward, with tiny progress/grasp hints so the toy is stable.
            r = -0.010 + 0.10 * progress + grasp_bonus + 1.65 * success.float() - 1.15 * safety.float()
            r = r * active.float()
            chunk_reward += r
            newly_done = active & (success | safety)
            chunks_to_finish = torch.where(newly_done, torch.full_like(chunks_to_finish, float(chunk_idx + 1)), chunks_to_finish)
            success_seen |= active & success
            safety_seen |= active & safety
            grasp_seen |= active & (carried > 0.5)
            done |= success | safety
            if deterministic:
                trace["ee"].append(ee.detach().cpu().numpy().copy())
                trace["obj"].append(obj.detach().cpu().numpy().copy())
                trace["grip"].append(grip.detach().cpu().numpy().copy())
                trace["carried"].append(carried.detach().cpu().numpy().copy())

        prefix_start = cfg.prefix_len + cfg.execute_len
        prefix = actions[:, prefix_start : prefix_start + cfg.prefix_len].detach()
        if not deterministic:
            obs_buf.append(obs)
            action_buf.append(actions)
            logp_buf.append(logp)
            value_buf.append(value)
            reward_buf.append(chunk_reward)
            done_buf.append(done.float())

    metrics = {
        "success": float(success_seen.float().mean().item()),
        "grasp": float(grasp_seen.float().mean().item()),
        "safety_stop": float(safety_seen.float().mean().item()),
        "mean_chunks": float(chunks_to_finish.mean().item()),
        "object_goal_error": float(torch.linalg.norm(obj - GOAL, dim=1).mean().item()),
        "prefix_mse": float(torch.cat(prefix_mse_values).mean().item()),
    }
    if deterministic:
        return metrics, trace
    return {
        "obs": torch.cat(obs_buf),
        "actions": torch.cat(action_buf),
        "old_logp": torch.cat(logp_buf),
        "values": torch.stack(value_buf),
        "rewards": torch.stack(reward_buf),
        "dones": torch.stack(done_buf),
        "metrics": metrics,
    }


def compute_gae(values, rewards, dones, cfg: Config):
    t_steps, n = rewards.shape
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros(n, device=DEVICE)
    next_value = torch.zeros(n, device=DEVICE)
    for t in reversed(range(t_steps)):
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + cfg.gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
        adv[t] = last_gae
        next_value = values[t]
    return adv.flatten(), (adv + values).flatten()


def train_variant(name, use_prefix_loss, initial_state, cfg: Config):
    policy = PickPlaceChunkPolicy(cfg).to(DEVICE)
    policy.load_state_dict(initial_state)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    history = []
    for it in range(cfg.ppo_iters + 1):
        eval_metrics, _ = rollout(policy, cfg, deterministic=True)
        eval_metrics.update({"method": name, "iteration": it})
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
                prefix_target = obs[idx, 12:].view(-1, cfg.prefix_len, cfg.action_dim)
                prefix_loss = F.mse_loss(mean[:, : cfg.prefix_len], prefix_target)
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                if use_prefix_loss:
                    loss = loss + cfg.prefix_coef * prefix_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.9)
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = {"bc_reference": "#4c78a8", "ppo_no_prefix_loss": "#e45756", "ppo_with_prefix_loss": "#54a24b"}
    labels = {"bc_reference": "BC reference", "ppo_no_prefix_loss": "PPO only", "ppo_with_prefix_loss": "PPO + prefix loss"}

    for name, rows in histories.items():
        it = [r["iteration"] for r in rows]
        axes[0, 0].plot(it, [r["success"] for r in rows], label=labels[name], color=colors[name])
        axes[0, 1].plot(it, [r["safety_stop"] for r in rows], label=labels[name], color=colors[name])
        axes[1, 0].plot(it, [r["prefix_mse"] for r in rows], label=labels[name], color=colors[name])

    ax = axes[1, 1]
    ax.add_patch(Rectangle((float(OBSTACLE[0].cpu()), float(OBSTACLE[1].cpu())), float((OBSTACLE[2] - OBSTACLE[0]).cpu()), float((OBSTACLE[3] - OBSTACLE[1]).cpu()), color="#bbbbbb", alpha=0.45, label="obstacle"))
    ax.add_patch(Circle(tuple(GOAL.cpu().numpy()), cfg.goal_radius, color="#ffd166", alpha=0.35, label="goal"))
    ax.add_patch(Circle(tuple(OBJ_START.cpu().numpy()), cfg.obj_radius, color="#118ab2", alpha=0.25, label="object start"))
    for name, trace in traces.items():
        ee = np.asarray(trace["ee"])
        obj = np.asarray(trace["obj"])
        if ee.size == 0:
            continue
        ax.plot(ee[:, :8, 0].mean(axis=1), ee[:, :8, 1].mean(axis=1), color=colors[name], lw=2, label=f"{labels[name]} ee")
        ax.plot(obj[:, :8, 0].mean(axis=1), obj[:, :8, 1].mean(axis=1), color=colors[name], lw=1.5, ls="--", label=f"{labels[name]} obj")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.04, 1.05)
    ax.set_ylim(-0.02, 1.04)
    ax.set_title("deterministic 2D pick/place rollouts")

    axes[0, 0].set_title("success rate")
    axes[0, 1].set_title("safety-triggered stops")
    axes[1, 0].set_title("prefix-copy error")
    for a in axes.ravel():
        a.grid(True, alpha=0.25)
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 0].legend(loc="lower right")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "prefix_rl_pickplace_2d_summary.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description="2D pick/place chunked PPO + prefix-loss toy")
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
    base = PickPlaceChunkPolicy(cfg).to(DEVICE)
    pretrain_bc(base, cfg)
    initial_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    histories = {}
    traces = {}
    final_rows = []
    bc_metrics, bc_trace = rollout(base, cfg, deterministic=True)
    bc_metrics.update({"method": "bc_reference", "iteration": 0})
    histories["bc_reference"] = [bc_metrics for _ in range(cfg.ppo_iters + 1)]
    traces["bc_reference"] = bc_trace
    final_rows.append({"method": "bc_reference", **bc_metrics})

    for name, use_prefix in [("ppo_no_prefix_loss", False), ("ppo_with_prefix_loss", True)]:
        set_seed(SEED + (101 if use_prefix else 97))
        _, hist, final_metrics, trace = train_variant(name, use_prefix, initial_state, cfg)
        histories[name] = hist
        traces[name] = trace
        final_rows.append({"method": name, **final_metrics})

    curve_rows = [row for rows in histories.values() for row in rows]
    fields = ["method", "iteration", "success", "grasp", "safety_stop", "mean_chunks", "object_goal_error", "prefix_mse"]
    metrics_path = os.path.join(OUTPUT_DIR, "prefix_rl_pickplace_2d_metrics.csv")
    curves_path = os.path.join(OUTPUT_DIR, "prefix_rl_pickplace_2d_training_curves.csv")
    write_csv(metrics_path, final_rows, ["method", "success", "grasp", "safety_stop", "mean_chunks", "object_goal_error", "prefix_mse"])
    write_csv(curves_path, curve_rows, fields)
    plot_path = plot_results(histories, traces, cfg)

    print("Wrote:")
    print(f"  {metrics_path}")
    print(f"  {curves_path}")
    print(f"  {plot_path}")
    print("Final metrics:")
    for row in final_rows:
        print(
            f"  {row['method']}: success={row['success']:.3f}, grasp={row['grasp']:.3f}, "
            f"safety={row['safety_stop']:.3f}, chunks={row['mean_chunks']:.2f}, "
            f"goal_err={row['object_goal_error']:.3f}, prefix_mse={row['prefix_mse']:.4f}"
        )


if __name__ == "__main__":
    main()
