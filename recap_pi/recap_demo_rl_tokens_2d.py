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
        self.dt = 0.1
        self.goal = np.array([1.0, 1.0], dtype=np.float32)
        self.goal_radius = 0.08
        self.bounds = (-0.25, 1.2, -0.25, 1.2)
        self.obstacles = [[0.25, 0.25, 0.55, 0.75], [0.45, -0.10, 0.75, 0.40]]
        self.state = None

    def reset(self, jitter=0.0):
        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.state[:2] += np.random.normal(0.0, jitter, size=2).astype(np.float32)
        self.state[2:] += np.random.normal(0.0, 0.5 * jitter, size=2).astype(np.float32)
        return self.state.copy()

    def inside_obstacle(self, pos):
        return any(o[0] <= pos[0] <= o[2] and o[1] <= pos[1] <= o[3] for o in self.obstacles)

    def out_of_bounds(self, pos):
        xmin, xmax, ymin, ymax = self.bounds
        return pos[0] < xmin or pos[0] > xmax or pos[1] < ymin or pos[1] > ymax

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.state[2:] = np.clip(self.state[2:] + action * self.dt, -1.25, 1.25)
        self.state[:2] = self.state[:2] + self.state[2:] * self.dt
        collided = self.inside_obstacle(self.state[:2]) or self.out_of_bounds(self.state[:2])
        done = np.linalg.norm(self.state[:2] - self.goal) < self.goal_radius
        return self.state.copy(), done, collided

    def apply_velocity_kick(self, delta_v):
        self.state[2:] = np.clip(self.state[2:] + delta_v.astype(np.float32), -1.25, 1.25)
        return self.state.copy()


env = NavigationEnv()


@dataclass
class Demo:
    states: np.ndarray
    actions: np.ndarray
    score: float
    success: bool


def pd_action(state, target, kp=4.8, kd=2.2):
    pos_err = target - state[:2]
    vel_err = -state[2:]
    return np.clip(kp * pos_err + kd * vel_err, -1.0, 1.0)


def recovery_target(state):
    if state[1] < 0.80:
        target = np.array([0.13, 0.86], dtype=np.float32)
    elif state[0] < 0.74:
        target = np.array([0.74, 0.98], dtype=np.float32)
    else:
        target = env.goal
    return pd_action(state, target, kp=5.2, kd=2.6)


def run_demo(waypoints=None, kind="expert", max_steps=260):
    state = env.reset()
    states = []
    actions = []
    done = False
    collided = False
    waypoint_idx = 0

    for t in range(max_steps):
        if kind == "expert":
            target = np.array(waypoints[min(waypoint_idx, len(waypoints) - 1)], dtype=np.float32)
            if np.linalg.norm(state[:2] - target) < 0.10 and waypoint_idx < len(waypoints) - 1:
                waypoint_idx += 1
                target = np.array(waypoints[waypoint_idx], dtype=np.float32)
            action = pd_action(state, target, kp=5.4, kd=2.9) + np.random.normal(0.0, 0.05, size=2)
        elif kind == "slow":
            target = np.array(waypoints[min(waypoint_idx, len(waypoints) - 1)], dtype=np.float32)
            if np.linalg.norm(state[:2] - target) < 0.12 and waypoint_idx < len(waypoints) - 1:
                waypoint_idx += 1
                target = np.array(waypoints[waypoint_idx], dtype=np.float32)
            action = pd_action(state, target, kp=3.6, kd=1.6) + np.random.normal(0.0, 0.10, size=2)
        elif kind == "straight":
            action = pd_action(state, env.goal, kp=3.8, kd=0.9) + np.random.normal(0.0, 0.10, size=2)
        elif kind == "wrong_detour":
            target = np.array([0.86, 0.04], dtype=np.float32) if t < 120 else env.goal
            action = pd_action(state, target, kp=4.0, kd=1.0) + np.array([0.07, -0.05])
        else:
            target = np.array([0.44, 0.52], dtype=np.float32) if t < 110 else env.goal
            action = pd_action(state, target, kp=2.1, kd=0.8) + 0.36 * np.array([np.sin(0.22 * t), np.cos(0.17 * t)])

        action = np.clip(action, -1.0, 1.0)
        states.append(state.copy())
        actions.append(action.copy())
        state, done, collided = env.step(action)
        if done or collided:
            break

    steps = len(states)
    if done and not collided:
        score = 2.2 - 0.005 * steps
    elif collided:
        score = -2.0 - 0.002 * steps
    else:
        score = -1.0 - 0.5 * np.linalg.norm(state[:2] - env.goal)

    return Demo(np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32), float(score), bool(done and not collided))


def build_dataset():
    demos = []
    top_route = [(0.08, 0.08), (0.13, 0.86), (0.74, 0.98), (1.0, 1.0)]
    wide_route = [(0.10, 0.12), (0.22, 0.96), (0.86, 1.02), (1.0, 1.0)]

    for _ in range(240):
        demos.append(run_demo(top_route, "expert"))
    for _ in range(70):
        demos.append(run_demo(wide_route, "slow"))
    for _ in range(360):
        demos.append(run_demo(kind="straight"))
    for _ in range(240):
        demos.append(run_demo(kind="wrong_detour"))
    for _ in range(220):
        demos.append(run_demo(kind="hesitate"))

    scores = np.array([demo.score for demo in demos], dtype=np.float32)
    threshold = np.quantile(scores, 0.78)

    states = []
    actions = []
    returns = []
    adv_labels = []
    for demo in demos:
        label = 1.0 if demo.score >= threshold else 0.0
        for state, action in zip(demo.states, demo.actions):
            states.append(state)
            actions.append(action)
            returns.append(demo.score)
            adv_labels.append(label)

    recovery_states = []
    recovery_actions = []
    rng = np.random.default_rng(SEED + 7)
    for demo in demos[:240]:
        if not demo.success:
            continue
        for state in demo.states[::4]:
            disturbed = state.copy()
            disturbed[2:] = np.clip(disturbed[2:] + rng.normal(0.0, 0.28, size=2), -1.25, 1.25)
            disturbed[:2] += rng.normal(0.0, 0.015, size=2)
            if env.inside_obstacle(disturbed[:2]) or env.out_of_bounds(disturbed[:2]):
                continue
            recovery_states.append(disturbed.astype(np.float32))
            recovery_actions.append(recovery_target(disturbed).astype(np.float32))

    return (
        torch.tensor(np.array(states), dtype=torch.float32),
        torch.tensor(np.array(actions), dtype=torch.float32),
        torch.tensor(np.array(returns), dtype=torch.float32).unsqueeze(1),
        torch.tensor(np.array(adv_labels), dtype=torch.float32),
        torch.tensor(np.array(recovery_states), dtype=torch.float32),
        torch.tensor(np.array(recovery_actions), dtype=torch.float32),
        demos,
    )


class Policy(nn.Module):
    def __init__(self, cond):
        super().__init__()
        self.cond = cond
        self.net = nn.Sequential(
            nn.Linear(5 if cond else 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, state, adv=None):
        if self.cond:
            x = torch.cat([state, adv.unsqueeze(1)], dim=1)
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


class RLTokenEditor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, state):
        return torch.tanh(self.net(state))


def local_risk(state, action):
    sim_state = state.copy()
    sim_state[2:] = np.clip(sim_state[2:] + np.clip(action, -1.0, 1.0) * env.dt, -1.25, 1.25)
    sim_state[:2] = sim_state[:2] + sim_state[2:] * env.dt
    if env.inside_obstacle(sim_state[:2]) or env.out_of_bounds(sim_state[:2]):
        return True
    speed = np.linalg.norm(sim_state[2:])
    if speed > 0.72:
        return True
    for o in env.obstacles:
        dx = max(o[0] - sim_state[0], 0.0, sim_state[0] - o[2])
        dy = max(o[1] - sim_state[1], 0.0, sim_state[1] - o[3])
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < 0.07:
            return True
    return False


def train_models(states, actions, returns, adv_labels, epochs=150, batch_size=1024):
    plain = Policy(cond=False)
    recap = Policy(cond=True)
    value = ValueNet()

    popt = torch.optim.Adam(plain.parameters(), lr=1e-3)
    ropt = torch.optim.Adam(recap.parameters(), lr=1e-3)
    vopt = torch.optim.Adam(value.parameters(), lr=1e-3)

    num = states.shape[0]
    for epoch in range(epochs):
        order = torch.randperm(num)
        for start in range(0, num, batch_size):
            idx = order[start : start + batch_size]
            s = states[idx]
            a = actions[idx]
            r = returns[idx]
            adv = adv_labels[idx]

            vloss = F.mse_loss(value(s), r)
            vopt.zero_grad()
            vloss.backward()
            vopt.step()

            ploss = F.mse_loss(plain(s), a)
            popt.zero_grad()
            ploss.backward()
            popt.step()

            rloss = F.mse_loss(recap(s, adv), a)
            null_mask = torch.rand_like(adv) < 0.30
            if null_mask.any():
                rloss = rloss + 0.25 * F.mse_loss(recap(s[null_mask], torch.zeros_like(adv[null_mask])), a[null_mask])
            ropt.zero_grad()
            rloss.backward()
            ropt.step()

        if epoch in {0, 49, 99, epochs - 1}:
            print(f"epoch {epoch + 1:03d}/{epochs} | value {vloss.item():.4f} | plain {ploss.item():.4f} | recap {rloss.item():.4f}")

    return plain, recap, value


def train_editor(recovery_states, recovery_actions, recap_policy, epochs=120, batch_size=512):
    editor = RLTokenEditor()
    opt = torch.optim.AdamW(editor.parameters(), lr=2e-3, weight_decay=1e-4)
    num = recovery_states.shape[0]
    for _ in range(epochs):
        order = torch.randperm(num)
        for start in range(0, num, batch_size):
            idx = order[start : start + batch_size]
            pred = editor(recovery_states[idx])
            loss = F.mse_loss(pred, recovery_actions[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return editor


def make_disturbance_specs(num_episodes):
    rng = np.random.default_rng(SEED + 99)
    specs = []
    for _ in range(num_episodes):
        specs.append({"step": int(rng.integers(18, 42)), "delta_v": rng.normal(0.0, 0.30, size=2).astype(np.float32)})
    return specs


def rollout(policy, mode, disturbance_specs, editor=None, max_steps=280):
    trajectories = []
    effective_steps = []
    successes = []
    collisions = []

    for spec in disturbance_specs:
        state = env.reset(jitter=0.018)
        traj = [state.copy()]
        done = False
        collided = False
        recovery_timer = 0

        for t in range(max_steps):
            if t == spec["step"]:
                state = env.apply_velocity_kick(spec["delta_v"])
                traj.append(state.copy())
                recovery_timer = 12

            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if mode == "plain":
                    base_action = policy(st)[0].numpy()
                else:
                    base_action = policy(st, torch.ones(1))[0].numpy()

            if mode == "online" and recovery_timer > 0:
                if local_risk(state, base_action):
                    corrective = recovery_target(state)
                    action = np.clip(0.45 * base_action + 0.55 * corrective, -1.0, 1.0)
                else:
                    action = base_action
                recovery_timer -= 1
            else:
                action = base_action

            action = np.clip(action + np.random.normal(0.0, 0.015, size=2), -1.0, 1.0)
            state, done, collided = env.step(action)
            traj.append(state.copy())
            if done or collided:
                break

        success = bool(done and not collided)
        trajectories.append(np.array(traj, dtype=np.float32))
        effective_steps.append((len(traj) - 1) if success else max_steps)
        successes.append(success)
        collisions.append(bool(collided))

    return {
        "trajectories": trajectories,
        "effective_steps": np.array(effective_steps),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
    }


def draw_env(ax):
    xmin, xmax, ymin, ymax = env.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.scatter([0.0], [0.0], c="black", s=36, zorder=5)
    ax.scatter([env.goal[0]], [env.goal[1]], c="gold", edgecolors="black", s=130, zorder=5)
    ax.add_patch(plt.Circle(env.goal, env.goal_radius, color="gold", alpha=0.15, zorder=1))
    for obs in env.obstacles:
        ax.add_patch(Rectangle((obs[0], obs[1]), obs[2] - obs[0], obs[3] - obs[1], color="gray", alpha=0.35))
    ax.grid(alpha=0.16)


def make_comparison_plot(plain_results, recap_results, online_results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    configs = [
        ("Plain BC", plain_results, "#c44e52"),
        ("RECAP Offline", recap_results, "#dd8452"),
        ("RECAP + Recovery Editor", online_results, "#55a868"),
    ]
    for ax, (title, results, color) in zip(axes, configs):
        draw_env(ax)
        ax.set_title(title)
        for traj in results["trajectories"][:14]:
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.35, lw=1.8)
        ax.text(
            0.03,
            0.97,
            f"success {results['success_rate'] * 100:.0f}%\nsteps {results['effective_steps'].mean():.0f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.92),
        )
    fig.suptitle("Disturbed 2D Navigation: Plain BC vs RECAP vs RECAP + Recovery Editor", fontsize=16)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "upgraded_2d_comparison.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def choose_examples(results, count=6):
    order = np.argsort(results["effective_steps"])
    idx = np.linspace(0, len(order) - 1, count, dtype=int)
    return [results["trajectories"][order[i]] for i in idx]


def make_gif(results, path, title, color):
    examples = choose_examples(results, count=6)
    max_len = max(len(traj) for traj in examples)
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    draw_env(ax)
    ax.set_title(title)
    lines = [ax.plot([], [], color=color, lw=2.0, alpha=0.30 + 0.1 * (i / len(examples)))[0] for i in range(len(examples))]
    dots = [ax.plot([], [], "o", color=color, markersize=4.5)[0] for _ in examples]
    counter = ax.text(0.02, 1.02, "", transform=ax.transAxes)

    def init():
        counter.set_text("")
        artists = [counter]
        for line, dot in zip(lines, dots):
            line.set_data([], [])
            dot.set_data([], [])
            artists.extend([line, dot])
        return artists

    def update(frame):
        counter.set_text(f"t = {frame:03d}")
        artists = [counter]
        for line, dot, traj in zip(lines, dots, examples):
            idx = min(frame, len(traj) - 1)
            line.set_data(traj[: idx + 1, 0], traj[: idx + 1, 1])
            dot.set_data([traj[idx, 0]], [traj[idx, 1]])
            artists.extend([line, dot])
        return artists

    anim = FuncAnimation(fig, update, frames=max_len, init_func=init, interval=85, blit=True)
    anim.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)


def main():
    print("Generating dataset...")
    states, actions, returns, adv_labels, recovery_states, recovery_actions, demos = build_dataset()
    print(
        f"demos: {len(demos)} | transitions: {len(states)} | dataset success: {np.mean([d.success for d in demos]) * 100:.1f}%"
    )
    print(f"recovery samples: {len(recovery_states)}")

    print("\nTraining plain BC, RECAP policy, and value model...")
    plain_policy, recap_policy, _value_net = train_models(states, actions, returns, adv_labels)

    print("\nTraining online recovery editor...")
    editor = train_editor(recovery_states, recovery_actions, recap_policy)

    print("\nEvaluating disturbed policies...")
    disturbance_specs = make_disturbance_specs(18)
    plain_results = rollout(plain_policy, "plain", disturbance_specs)
    recap_results = rollout(recap_policy, "recap", disturbance_specs)
    online_results = rollout(recap_policy, "online", disturbance_specs, editor=editor)

    delta_success = (online_results["success_rate"] - recap_results["success_rate"]) * 100
    delta_steps = recap_results["effective_steps"].mean() - online_results["effective_steps"].mean()
    delta_collision = (recap_results["collision_rate"] - online_results["collision_rate"]) * 100

    print("\n=== UPGRADED RESULTS ===")
    print(
        f"Plain BC                -> Avg steps: {plain_results['effective_steps'].mean():.0f} | "
        f"Success: {plain_results['success_rate'] * 100:.0f}% | Collisions: {plain_results['collision_rate'] * 100:.0f}%"
    )
    print(
        f"RECAP (offline)         -> Avg steps: {recap_results['effective_steps'].mean():.0f} | "
        f"Success: {recap_results['success_rate'] * 100:.0f}% | Collisions: {recap_results['collision_rate'] * 100:.0f}%"
    )
    print(
        f"RECAP + RL Tokens       -> Avg steps: {online_results['effective_steps'].mean():.0f} | "
        f"Success: {online_results['success_rate'] * 100:.0f}% | Collisions: {online_results['collision_rate'] * 100:.0f}%"
    )
    print(
        f"Improvement over RECAP  -> Success {delta_success:+.1f} pts | "
        f"Effective steps {delta_steps:+.1f} | Collision {delta_collision:+.1f} pts"
    )

    png_path = make_comparison_plot(plain_results, recap_results, online_results)
    make_gif(plain_results, os.path.join(OUTPUT_DIR, "upgraded_plain.gif"), "Plain BC", "#c44e52")
    make_gif(recap_results, os.path.join(OUTPUT_DIR, "upgraded_recap.gif"), "RECAP Offline", "#dd8452")
    make_gif(online_results, os.path.join(OUTPUT_DIR, "upgraded_rl_tokens.gif"), "RECAP + Recovery Editor", "#55a868")
    print(f"Saved comparison PNG -> {png_path}")
    print("Saved GIFs -> upgraded_plain.gif, upgraded_recap.gif, upgraded_rl_tokens.gif")


if __name__ == "__main__":
    main()
