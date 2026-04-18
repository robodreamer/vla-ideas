import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE_DIR, ".mplconfig"))

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle


OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class NavigationEnv:
    def __init__(self):
        self.dt = 0.08
        self.max_acc = 1.0
        self.goal = np.array([1.0, 1.0], dtype=np.float32)
        self.goal_radius = 0.09
        self.bounds = (-0.15, 1.15, -0.2, 1.2)
        self.obstacles = [
            [0.38, 0.16, 0.66, 0.78],
            [0.62, -0.02, 0.92, 0.48],
        ]
        self.state = None

    def reset(self, jitter_scale=0.0):
        pos_noise = np.random.normal(0.0, jitter_scale, size=2)
        vel_noise = np.random.normal(0.0, jitter_scale * 0.5, size=2)
        self.state = np.array([0.02, 0.02, 0.0, 0.0], dtype=np.float32)
        self.state[:2] += pos_noise.astype(np.float32)
        self.state[2:] += vel_noise.astype(np.float32)
        return self.state.copy()

    def inside_obstacle(self, pos):
        return any(o[0] <= pos[0] <= o[2] and o[1] <= pos[1] <= o[3] for o in self.obstacles)

    def out_of_bounds(self, pos):
        xmin, xmax, ymin, ymax = self.bounds
        return pos[0] < xmin or pos[0] > xmax or pos[1] < ymin or pos[1] > ymax

    def step(self, action):
        action = np.clip(action, -self.max_acc, self.max_acc)
        self.state[2:] = np.clip(self.state[2:] + action * self.dt, -1.2, 1.2)
        self.state[:2] = self.state[:2] + self.state[2:] * self.dt
        collided = self.inside_obstacle(self.state[:2]) or self.out_of_bounds(self.state[:2])
        done = np.linalg.norm(self.state[:2] - self.goal) < self.goal_radius
        return self.state.copy(), done, collided


env = NavigationEnv()


@dataclass
class Demo:
    states: np.ndarray
    actions: np.ndarray
    score: float
    success: bool
    steps: int
    name: str


def pd_action(state, target, kp=5.0, kd=2.6):
    pos_err = target - state[:2]
    vel_err = -state[2:]
    act = kp * pos_err + kd * vel_err
    return np.clip(act, -1.0, 1.0)


def run_waypoints(waypoints, noise=0.0, max_steps=220, aggressive=False, label="expert"):
    s = env.reset()
    states = []
    actions = []
    waypoint_idx = 0
    collided = False
    done = False
    for _ in range(max_steps):
        target = np.array(waypoints[min(waypoint_idx, len(waypoints) - 1)], dtype=np.float32)
        if np.linalg.norm(s[:2] - target) < 0.10 and waypoint_idx < len(waypoints) - 1:
            waypoint_idx += 1
            target = np.array(waypoints[waypoint_idx], dtype=np.float32)

        kp = 6.0 if aggressive else 4.6
        kd = 3.1 if aggressive else 2.3
        act = pd_action(s, target, kp=kp, kd=kd)
        if noise > 0.0:
            act = np.clip(act + np.random.normal(0.0, noise, size=2), -1.0, 1.0)
        states.append(s.copy())
        actions.append(act.copy())
        s, done, collided = env.step(act)
        if done or collided:
            break

    if done and not collided:
        score = 2.0 - 0.004 * len(states)
    elif collided:
        score = -1.8 - 0.002 * len(states)
    else:
        score = -0.8 - 0.6 * np.linalg.norm(s[:2] - env.goal)
    return Demo(
        states=np.array(states, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        score=float(score),
        success=bool(done and not collided),
        steps=len(states),
        name=label,
    )


def run_bad_demo(kind, max_steps=240):
    s = env.reset()
    states = []
    actions = []
    collided = False
    done = False

    for t in range(max_steps):
        if kind == "straight":
            target = env.goal
            act = pd_action(s, target, kp=3.8, kd=0.8)
            act += np.random.normal(0.0, 0.10, size=2)
        elif kind == "hesitate":
            target = np.array([0.42, 0.52], dtype=np.float32) if t < 110 else env.goal
            act = pd_action(s, target, kp=1.9, kd=0.7)
            act += 0.36 * np.array([np.sin(t * 0.22), np.cos(t * 0.19)])
        elif kind == "wrong_detour":
            target = np.array([0.84, 0.06], dtype=np.float32) if t < 130 else env.goal
            act = pd_action(s, target, kp=4.0, kd=1.0)
            act += np.array([0.06, -0.06])
        else:
            target = np.array([0.54, 0.30], dtype=np.float32) if t < 100 else np.array([0.72, 0.52], dtype=np.float32)
            act = pd_action(s, target, kp=2.5, kd=0.7)
            act += np.random.normal(0.0, 0.20, size=2)

        act = np.clip(act, -1.0, 1.0)
        states.append(s.copy())
        actions.append(act.copy())
        s, done, collided = env.step(act)
        if done or collided:
            break

    if done and not collided:
        score = 0.5 - 0.005 * len(states)
    elif collided:
        score = -2.0 - 0.001 * len(states)
    else:
        score = -1.1 - 0.5 * np.linalg.norm(s[:2] - env.goal)
    return Demo(
        states=np.array(states, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        score=float(score),
        success=bool(done and not collided),
        steps=len(states),
        name=kind,
    )


def build_dataset():
    demos = []

    top_route = [(0.08, 0.10), (0.15, 0.90), (0.76, 1.02), (1.0, 1.0)]
    wide_top_route = [(0.10, 0.12), (0.22, 0.98), (0.88, 1.04), (1.0, 1.0)]

    for _ in range(280):
        demos.append(run_waypoints(top_route, noise=0.05, aggressive=True, label="expert_top"))
    for _ in range(60):
        demos.append(run_waypoints(wide_top_route, noise=0.09, aggressive=False, label="slow_top"))
    for _ in range(440):
        demos.append(run_bad_demo("straight"))
    for _ in range(320):
        demos.append(run_bad_demo("hesitate"))
    for _ in range(320):
        demos.append(run_bad_demo("wrong_detour"))
    for _ in range(180):
        demos.append(run_bad_demo("jitter"))

    scores = np.array([d.score for d in demos], dtype=np.float32)
    threshold = np.quantile(scores, 0.82)

    states = []
    actions = []
    conds = []
    returns = []
    for demo in demos:
        cond = 1.0 if demo.score >= threshold else 0.0
        for s, a in zip(demo.states, demo.actions):
            states.append(s)
            actions.append(a)
            conds.append(cond)
            returns.append(demo.score)

    return (
        torch.tensor(np.array(states), dtype=torch.float32),
        torch.tensor(np.array(actions), dtype=torch.float32),
        torch.tensor(np.array(conds), dtype=torch.float32),
        torch.tensor(np.array(returns), dtype=torch.float32).unsqueeze(1),
        demos,
    )


class Policy(nn.Module):
    def __init__(self, conditioned):
        super().__init__()
        self.conditioned = conditioned
        in_dim = 5 if conditioned else 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, state, cond=None):
        if self.conditioned:
            if cond is None:
                raise ValueError("Conditioned policy requires cond.")
            x = torch.cat([state, cond.unsqueeze(1)], dim=1)
        else:
            x = state
        return torch.tanh(self.net(x))


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        return self.net(state)


def train_models(states, actions, conds, returns, epochs=160, batch_size=1024):
    plain = Policy(conditioned=False)
    recap = Policy(conditioned=True)
    value = ValueNet()

    popt = torch.optim.Adam(plain.parameters(), lr=1e-3)
    ropt = torch.optim.Adam(recap.parameters(), lr=1e-3)
    vopt = torch.optim.Adam(value.parameters(), lr=1e-3)

    num = states.shape[0]
    for ep in range(epochs):
        order = torch.randperm(num)
        for start in range(0, num, batch_size):
            idx = order[start : start + batch_size]
            s = states[idx]
            a = actions[idx]
            c = conds[idx]
            r = returns[idx]

            vloss = F.mse_loss(value(s), r)
            vopt.zero_grad()
            vloss.backward()
            vopt.step()

            ploss = F.mse_loss(plain(s), a)
            popt.zero_grad()
            ploss.backward()
            popt.step()

            pred = recap(s, c)
            rloss = F.mse_loss(pred, a)
            null_mask = torch.rand_like(c) < 0.20
            if null_mask.any():
                rloss = rloss + 0.25 * F.mse_loss(recap(s[null_mask], torch.zeros_like(c[null_mask])), a[null_mask])
            ropt.zero_grad()
            rloss.backward()
            ropt.step()

        if ep in {0, 39, 79, 119, epochs - 1}:
            print(
                f"epoch {ep + 1:03d}/{epochs} | "
                f"value {vloss.item():.4f} | plain {ploss.item():.4f} | recap {rloss.item():.4f}"
            )

    return plain, recap, value


def rollout(policy, conditioned, episodes=48, max_steps=220):
    trajectories = []
    steps = []
    effective_steps = []
    successes = []
    collisions = []

    for _ in range(episodes):
        s = env.reset(jitter_scale=0.018)
        traj = [s.copy()]
        collided = False
        done = False
        for _ in range(max_steps):
            st = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if conditioned:
                    act = policy(st, torch.ones(1))[0].numpy()
                else:
                    act = policy(st)[0].numpy()
            act = np.clip(act + np.random.normal(0.0, 0.015, size=2), -1.0, 1.0)
            s, done, collided = env.step(act)
            traj.append(s.copy())
            if done or collided:
                break

        trajectories.append(np.array(traj, dtype=np.float32))
        step_count = len(traj) - 1
        success = bool(done and not collided)
        steps.append(step_count)
        effective_steps.append(step_count if success else max_steps)
        successes.append(success)
        collisions.append(bool(collided))

    return {
        "trajectories": trajectories,
        "steps": np.array(steps),
        "effective_steps": np.array(effective_steps),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
    }


def draw_env(ax):
    xmin, xmax, ymin, ymax = env.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.scatter([0.02], [0.02], c="black", s=45, label="start", zorder=5)
    ax.scatter([env.goal[0]], [env.goal[1]], c="gold", edgecolors="black", s=180, label="goal", zorder=5)
    goal_circle = plt.Circle(env.goal, env.goal_radius, color="gold", alpha=0.15, zorder=1)
    ax.add_patch(goal_circle)
    for obs in env.obstacles:
        ax.add_patch(
            Rectangle(
                (obs[0], obs[1]),
                obs[2] - obs[0],
                obs[3] - obs[1],
                color="#b23a48",
                alpha=0.45,
                zorder=2,
            )
        )
    ax.grid(alpha=0.2)


def choose_examples(results, count=8):
    trajs = results["trajectories"]
    lengths = [len(t) for t in trajs]
    order = np.argsort(lengths)
    idx = np.linspace(0, len(order) - 1, count, dtype=int)
    return [trajs[order[i]] for i in idx]


def make_comparison_plot(plain_results, recap_results, value_net, plain_policy, recap_policy):
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0], wspace=0.22, hspace=0.22)

    ax_plain = fig.add_subplot(gs[0, 0])
    ax_recap = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_plain_field = fig.add_subplot(gs[1, 0])
    ax_recap_field = fig.add_subplot(gs[1, 1])
    ax_value = fig.add_subplot(gs[1, 2])

    for ax, title in [(ax_plain, "Plain Imitation Rollouts"), (ax_recap, "RECAP Rollouts")]:
        draw_env(ax)
        ax.set_title(title)

    for traj in plain_results["trajectories"][:20]:
        ax_plain.plot(traj[:, 0], traj[:, 1], color="#4c78a8", alpha=0.35, lw=1.7)
    for traj in recap_results["trajectories"][:20]:
        ax_recap.plot(traj[:, 0], traj[:, 1], color="#54a24b", alpha=0.45, lw=1.8)

    labels = ["Plain", "RECAP"]
    avg_steps = [plain_results["effective_steps"].mean(), recap_results["effective_steps"].mean()]
    success = [100 * plain_results["success_rate"], 100 * recap_results["success_rate"]]
    bars = ax_bar.bar(labels, avg_steps, color=["#4c78a8", "#54a24b"], alpha=0.85)
    ax_bar.set_ylabel("Average Effective Steps")
    ax_bar.set_title("Episode Cost (Failures Count As Timeout)")
    for bar, succ in zip(bars, success):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{succ:.0f}% success", ha="center")

    xs = np.linspace(-0.02, 1.05, 20)
    ys = np.linspace(-0.05, 1.05, 20)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel(), np.zeros_like(xx).ravel(), np.zeros_like(yy).ravel()], axis=1)
    grid_t = torch.tensor(grid, dtype=torch.float32)
    with torch.no_grad():
        plain_vec = plain_policy(grid_t).numpy()
        recap_vec = recap_policy(grid_t, torch.ones(grid_t.shape[0])).numpy()
        values = value_net(grid_t).numpy().reshape(xx.shape)

    for ax, vec, title, color in [
        (ax_plain_field, plain_vec, "Plain Policy Field", "#4c78a8"),
        (ax_recap_field, recap_vec, "RECAP Policy Field", "#54a24b"),
    ]:
        draw_env(ax)
        ax.quiver(xx, yy, vec[:, 0].reshape(xx.shape), vec[:, 1].reshape(xx.shape), color=color, alpha=0.75)
        ax.set_title(title)

    draw_env(ax_value)
    im = ax_value.imshow(
        values,
        extent=(xs.min(), xs.max(), ys.min(), ys.max()),
        origin="lower",
        cmap="viridis",
        alpha=0.9,
        aspect="auto",
    )
    ax_value.set_title("Learned Value Heatmap")
    cbar = fig.colorbar(im, ax=ax_value, fraction=0.046, pad=0.04)
    cbar.set_label("Predicted return")

    fig.suptitle("Double-Integrator Navigation: Plain BC vs RECAP Advantage Prompt", fontsize=18)
    fig.savefig(os.path.join(OUTPUT_DIR, "complex_comparison.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_rollout_gif(results, path, title, color):
    examples = choose_examples(results, count=8)
    max_len = max(len(t) for t in examples)

    fig, ax = plt.subplots(figsize=(7.4, 7.1))
    draw_env(ax)
    ax.set_title(title)
    lines = [ax.plot([], [], color=color, alpha=0.35 + 0.07 * (i / len(examples)), lw=2.0)[0] for i in range(len(examples))]
    dots = [ax.plot([], [], "o", color=color, markersize=5)[0] for _ in examples]
    counter = ax.text(0.02, 1.02, "", transform=ax.transAxes, fontsize=12)

    def init():
        counter.set_text("")
        for line, dot in zip(lines, dots):
            line.set_data([], [])
            dot.set_data([], [])
        return lines + dots + [counter]

    def update(frame):
        counter.set_text(f"t = {frame:03d}")
        for line, dot, traj in zip(lines, dots, examples):
            idx = min(frame, len(traj) - 1)
            line.set_data(traj[: idx + 1, 0], traj[: idx + 1, 1])
            dot.set_data([traj[idx, 0]], [traj[idx, 1]])
        return lines + dots + [counter]

    anim = FuncAnimation(fig, update, frames=max_len, init_func=init, interval=80, blit=True)
    anim.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)


def main():
    print("Generating mixed expert/noisy dataset...")
    states, actions, conds, returns, demos = build_dataset()
    success_rate = np.mean([d.success for d in demos])
    print(
        f"dataset demos: {len(demos)} | transitions: {len(states)} | "
        f"successful demos: {success_rate * 100:.1f}%"
    )

    print("\nTraining plain imitation and RECAP-conditioned policies...")
    plain_policy, recap_policy, value_net = train_models(states, actions, conds, returns)

    print("\nEvaluating policies...")
    plain_results = rollout(plain_policy, conditioned=False)
    recap_results = rollout(recap_policy, conditioned=True)

    plain_steps = plain_results["effective_steps"].mean()
    recap_steps = recap_results["effective_steps"].mean()
    plain_raw_steps = plain_results["steps"].mean()
    recap_raw_steps = recap_results["steps"].mean()
    plain_success = plain_results["success_rate"]
    recap_success = recap_results["success_rate"]

    print("\n=== RESULTS ===")
    print(
        f"Plain Imitation  -> Avg steps: {plain_steps:.0f} | "
        f"Success: {plain_success * 100:.0f}% | Collisions: {plain_results['collision_rate'] * 100:.0f}% | "
        f"Raw term steps: {plain_raw_steps:.0f}"
    )
    print(
        f"RECAP Advantage  -> Avg steps: {recap_steps:.0f} | "
        f"Success: {recap_success * 100:.0f}% | Collisions: {recap_results['collision_rate'] * 100:.0f}% | "
        f"Raw term steps: {recap_raw_steps:.0f}"
    )
    print(f"Duration improvement: {plain_steps / recap_steps:.2f}x faster")
    print(f"Success improvement: {recap_success / max(plain_success, 1e-6):.2f}x higher")

    print("\nSaving visuals...")
    make_comparison_plot(plain_results, recap_results, value_net, plain_policy, recap_policy)
    make_rollout_gif(
        plain_results,
        os.path.join(OUTPUT_DIR, "plain_rollouts.gif"),
        "Plain Imitation Rollouts",
        "#4c78a8",
    )
    make_rollout_gif(
        recap_results,
        os.path.join(OUTPUT_DIR, "recap_rollouts.gif"),
        "RECAP Advantage Rollouts",
        "#54a24b",
    )
    print(f"Saved {os.path.join(OUTPUT_DIR, 'complex_comparison.png')}")
    print(f"Saved {os.path.join(OUTPUT_DIR, 'plain_rollouts.gif')}")
    print(f"Saved {os.path.join(OUTPUT_DIR, 'recap_rollouts.gif')}")


if __name__ == "__main__":
    main()
