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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

SEED = 31
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")


class DroneHeistEnv3D:
    def __init__(self):
        self.dt = 0.075
        self.max_acc = 1.0
        self.max_vel = 1.25
        self.start = np.array([0.06, 0.10, 0.10], dtype=np.float32)
        self.artifact = np.array([0.16, 0.82, 0.16], dtype=np.float32)
        self.uplink = np.array([0.72, 0.86, 0.92], dtype=np.float32)
        self.extract = np.array([0.98, 0.22, 0.90], dtype=np.float32)
        self.artifact_radius = 0.09
        self.uplink_radius = 0.10
        self.extract_radius = 0.10
        self.bounds = (-0.10, 1.10, -0.08, 1.08, -0.02, 1.08)
        self.obstacles = [
            [0.34, 0.10, 0.00, 0.63, 0.76, 0.68],
            [0.44, 0.54, 0.34, 0.92, 0.80, 0.88],
            [0.58, 0.18, 0.00, 0.76, 0.42, 0.46],
            [0.82, 0.38, 0.18, 0.96, 0.72, 0.74],
        ]
        self.state = None

    def reset(self, jitter=0.0):
        pos_noise = np.random.normal(0.0, jitter, size=3)
        vel_noise = np.random.normal(0.0, 0.35 * jitter, size=3)
        self.state = np.zeros(8, dtype=np.float32)
        self.state[:3] = self.start + pos_noise.astype(np.float32)
        self.state[3:6] = vel_noise.astype(np.float32)
        self.state[6] = 0.0
        self.state[7] = 0.0
        return self.state.copy()

    def inside_obstacle(self, pos):
        return any(
            box[0] <= pos[0] <= box[3]
            and box[1] <= pos[1] <= box[4]
            and box[2] <= pos[2] <= box[5]
            for box in self.obstacles
        )

    def out_of_bounds(self, pos):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        return pos[0] < xmin or pos[0] > xmax or pos[1] < ymin or pos[1] > ymax or pos[2] < zmin or pos[2] > zmax

    def step(self, action):
        action = np.clip(action, -self.max_acc, self.max_acc)
        self.state[3:6] = np.clip(self.state[3:6] + action * self.dt, -self.max_vel, self.max_vel)
        self.state[:3] = self.state[:3] + self.state[3:6] * self.dt
        collided = self.inside_obstacle(self.state[:3]) or self.out_of_bounds(self.state[:3])
        if self.state[6] < 0.5 and np.linalg.norm(self.state[:3] - self.artifact) < self.artifact_radius:
            self.state[6] = 1.0
        if self.state[6] > 0.5 and self.state[7] < 0.5 and np.linalg.norm(self.state[:3] - self.uplink) < self.uplink_radius:
            self.state[7] = 1.0
        done = bool(self.state[6] > 0.5 and self.state[7] > 0.5 and np.linalg.norm(self.state[:3] - self.extract) < self.extract_radius)
        return self.state.copy(), done, collided

    def apply_velocity_kick(self, delta_v):
        self.state[3:6] = np.clip(self.state[3:6] + delta_v.astype(np.float32), -self.max_vel, self.max_vel)
        return self.state.copy()


env = DroneHeistEnv3D()


@dataclass
class Demo:
    states: np.ndarray
    actions: np.ndarray
    score: float
    success: bool
    style: float


def pd_action(state, target, kp=5.2, kd=2.7):
    pos_err = target - state[:3]
    vel_err = -state[3:6]
    return np.clip(kp * pos_err + kd * vel_err, -1.0, 1.0)


def eval_score(state, success, collided, steps):
    artifact = float(state[6] > 0.5)
    uplink = float(state[7] > 0.5)
    if success:
        return 4.0 - 0.007 * steps
    if collided:
        return -2.6 + 0.55 * artifact + 0.65 * uplink - 0.002 * steps
    return -1.2 + 0.8 * artifact + 0.9 * uplink - 0.45 * np.linalg.norm(state[:3] - env.extract)


def run_demo(waypoints=None, label="expert", style=0.0, max_steps=300):
    state = env.reset()
    states = []
    actions = []
    waypoint_idx = 0
    done = False
    collided = False

    for t in range(max_steps):
        if label == "expert":
            target = np.array(waypoints[min(waypoint_idx, len(waypoints) - 1)], dtype=np.float32)
            if np.linalg.norm(state[:3] - target) < 0.12 and waypoint_idx < len(waypoints) - 1:
                waypoint_idx += 1
                target = np.array(waypoints[waypoint_idx], dtype=np.float32)
            action = pd_action(state, target, kp=5.8, kd=3.2) + np.random.normal(0.0, 0.05, size=3)
        elif label == "safe":
            target = np.array(waypoints[min(waypoint_idx, len(waypoints) - 1)], dtype=np.float32)
            if np.linalg.norm(state[:3] - target) < 0.14 and waypoint_idx < len(waypoints) - 1:
                waypoint_idx += 1
                target = np.array(waypoints[waypoint_idx], dtype=np.float32)
            action = pd_action(state, target, kp=4.4, kd=2.3) + np.random.normal(0.0, 0.07, size=3)
        elif label == "rush_extract":
            action = pd_action(state, env.extract, kp=3.4, kd=1.0) + np.array([0.05, -0.02, 0.05])
        elif label == "artifact_then_crash":
            target = env.artifact if state[6] < 0.5 else np.array([0.60, 0.66, 0.42], dtype=np.float32)
            action = pd_action(state, target, kp=4.4, kd=1.2)
        else:
            target = env.artifact if state[6] < 0.5 else (env.uplink if state[7] < 0.5 else np.array([0.86, 0.36, 0.82], dtype=np.float32))
            action = pd_action(state, target, kp=2.9, kd=0.8) + 0.22 * np.array([np.sin(t * 0.24), np.cos(t * 0.18), np.sin(t * 0.16)])

        action = np.clip(action, -1.0, 1.0)
        states.append(state.copy())
        actions.append(action.copy())
        state, done, collided = env.step(action)
        if done or collided:
            break

    return Demo(
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32),
        eval_score(state, done and not collided, collided, len(states)),
        bool(done and not collided),
        style,
    )


def build_dataset():
    demos = []
    fast_route = [
        (0.10, 0.18, 0.12),
        (0.16, 0.82, 0.18),
        (0.28, 0.96, 0.36),
        (0.70, 0.88, 0.94),
        (0.90, 0.54, 1.00),
        (0.98, 0.22, 0.90),
    ]
    safe_route = [
        (0.08, 0.20, 0.12),
        (0.18, 0.88, 0.20),
        (0.24, 1.00, 0.52),
        (0.62, 0.98, 0.98),
        (0.90, 0.46, 1.02),
        (0.98, 0.22, 0.90),
    ]

    for _ in range(180):
        demos.append(run_demo(fast_route, "expert", style=0.0))
    for _ in range(120):
        demos.append(run_demo(safe_route, "safe", style=1.0))
    for _ in range(220):
        demos.append(run_demo(label="rush_extract", style=0.0))
    for _ in range(180):
        demos.append(run_demo(label="artifact_then_crash", style=0.0))
    for _ in range(160):
        demos.append(run_demo(label="orbit", style=0.0))

    scores = np.array([demo.score for demo in demos], dtype=np.float32)
    threshold = np.quantile(scores, 0.78)

    states = []
    actions = []
    returns = []
    adv_labels = []
    styles = []
    for demo in demos:
        label = 1.0 if demo.score >= threshold else 0.0
        for state, action in zip(demo.states, demo.actions):
            states.append(state)
            actions.append(action)
            returns.append(demo.score)
            adv_labels.append(label)
            styles.append(demo.style if label > 0.5 else 0.0)

    return (
        torch.tensor(np.array(states), dtype=torch.float32),
        torch.tensor(np.array(actions), dtype=torch.float32),
        torch.tensor(np.array(returns), dtype=torch.float32).unsqueeze(1),
        torch.tensor(np.array(adv_labels), dtype=torch.float32),
        torch.tensor(np.array(styles), dtype=torch.float32),
        demos,
    )


class LatentPolicy(nn.Module):
    def __init__(self, conditioned=True):
        super().__init__()
        self.conditioned = conditioned
        in_dim = 10 if conditioned else 8
        self.net = nn.Sequential(
            nn.Linear(in_dim, 192),
            nn.ReLU(),
            nn.Linear(192, 192),
            nn.ReLU(),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, state, adv=None, style=None):
        if self.conditioned:
            if adv is None or style is None:
                raise ValueError("Conditioned policy requires adv and style.")
            x = torch.cat([state, adv.unsqueeze(1), style.unsqueeze(1)], dim=1)
        else:
            x = state
        return torch.tanh(self.net(x))


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state):
        return self.net(state)


def train_models(states, actions, returns, adv_labels, styles, epochs=120, batch_size=1024):
    states = states.to(DEVICE)
    actions = actions.to(DEVICE)
    returns = returns.to(DEVICE)
    adv_labels = adv_labels.to(DEVICE)
    styles = styles.to(DEVICE)

    plain = LatentPolicy(conditioned=False).to(DEVICE)
    recap = LatentPolicy(conditioned=True).to(DEVICE)
    value_net = ValueNet().to(DEVICE)

    popt = torch.optim.AdamW(plain.parameters(), lr=1e-3, weight_decay=1e-4)
    ropt = torch.optim.AdamW(recap.parameters(), lr=1e-3, weight_decay=1e-4)
    vopt = torch.optim.AdamW(value_net.parameters(), lr=1e-3, weight_decay=1e-4)

    num = states.shape[0]
    for epoch in range(epochs):
        order = torch.randperm(num)
        for start in range(0, num, batch_size):
            idx = order[start : start + batch_size]
            s = states[idx]
            a = actions[idx]
            ret = returns[idx]
            adv = adv_labels[idx]
            sty = styles[idx]

            ploss = F.mse_loss(plain(s), a)
            popt.zero_grad()
            ploss.backward()
            popt.step()

            rloss = F.mse_loss(recap(s, adv, sty), a)
            null_mask = torch.rand_like(adv) < 0.30
            if null_mask.any():
                rloss = rloss + 0.25 * F.mse_loss(recap(s[null_mask], torch.zeros_like(adv[null_mask]), torch.zeros_like(sty[null_mask])), a[null_mask])
            ropt.zero_grad()
            rloss.backward()
            ropt.step()

            vloss = F.mse_loss(value_net(s), ret)
            vopt.zero_grad()
            vloss.backward()
            vopt.step()

        if epoch in {0, 39, 79, epochs - 1}:
            print(
                f"epoch {epoch + 1:03d}/{epochs} | plain {ploss.item():.4f} | "
                f"recap-latent {rloss.item():.4f} | value {vloss.item():.4f}"
            )
    return plain, recap, value_net


def inside_obstacle_state(pos):
    return any(
        box[0] <= pos[0] <= box[3]
        and box[1] <= pos[1] <= box[4]
        and box[2] <= pos[2] <= box[5]
        for box in env.obstacles
    )


def out_of_bounds_state(pos):
    xmin, xmax, ymin, ymax, zmin, zmax = env.bounds
    return pos[0] < xmin or pos[0] > xmax or pos[1] < ymin or pos[1] > ymax or pos[2] < zmin or pos[2] > zmax


def step_imagined(state, action):
    next_state = state.copy()
    act = np.clip(action, -env.max_acc, env.max_acc).astype(np.float32)
    next_state[3:6] = np.clip(next_state[3:6] + act * env.dt, -env.max_vel, env.max_vel)
    next_state[:3] = next_state[:3] + next_state[3:6] * env.dt
    collided = inside_obstacle_state(next_state[:3]) or out_of_bounds_state(next_state[:3])
    if next_state[6] < 0.5 and np.linalg.norm(next_state[:3] - env.artifact) < env.artifact_radius:
        next_state[6] = 1.0
    if next_state[6] > 0.5 and next_state[7] < 0.5 and np.linalg.norm(next_state[:3] - env.uplink) < env.uplink_radius:
        next_state[7] = 1.0
    done = bool(next_state[6] > 0.5 and next_state[7] > 0.5 and np.linalg.norm(next_state[:3] - env.extract) < env.extract_radius)
    return next_state, done, collided


def stage_reward(prev_state, next_state, done, collided):
    reward = -0.03
    if next_state[6] > prev_state[6]:
        reward += 0.85
    if next_state[7] > prev_state[7]:
        reward += 1.10
    if done:
        reward += 2.2
    if collided:
        reward -= 2.8
    reward -= 0.10 * np.linalg.norm(next_state[:3] - env.extract)
    return reward


def imagined_return(policy, value_net, init_state, style, horizon=26):
    state = init_state.copy()
    total = 0.0
    done = False
    collided = False
    for step in range(horizon):
        st = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action = policy(
                st,
                torch.ones(1, device=DEVICE),
                torch.tensor([style], dtype=torch.float32, device=DEVICE),
            )[0].detach().cpu().numpy()
        next_state, done, collided = step_imagined(state, action)
        total += (0.96 ** step) * stage_reward(state, next_state, done, collided)
        state = next_state
        if done or collided:
            break
    if not done and not collided:
        st = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            bootstrap = value_net(st)[0, 0].item()
        total += (0.96 ** horizon) * 0.40 * bootstrap
    return total


def style_bias(state, style):
    speed = np.linalg.norm(state[3:6])
    has_artifact = state[6] > 0.5
    has_uplink = state[7] > 0.5
    bias = 0.0
    if not has_uplink and (speed > 0.55 or (has_artifact and state[2] < 0.72)):
        bias += 0.12 if style == 1.0 else -0.02
    if has_uplink:
        bias += 0.14 if style == 0.0 else -0.08
    return bias


def choose_counterfactual_style(policy, value_net, state, default_style):
    scores = {}
    for style in (0.0, 1.0):
        score = imagined_return(policy, value_net, state, style)
        score += style_bias(state, style)
        if style == default_style:
            score += 0.02
        scores[style] = score
    best_style = max(scores, key=scores.get)
    return best_style, scores


def make_disturbance_specs(num_episodes):
    rng = np.random.default_rng(SEED + 200)
    return [{"step": int(rng.integers(22, 58)), "delta_v": rng.normal(0.0, 0.26, size=3).astype(np.float32)} for _ in range(num_episodes)]


def rollout(policy, mode, disturbance_specs, value_net=None, max_steps=300):
    trajectories = []
    effective_steps = []
    success = []
    collisions = []
    artifact = []
    uplink = []
    total_planner_interventions = 0

    for spec in disturbance_specs:
        state = env.reset(jitter=0.018)
        traj = [state.copy()]
        done = False
        collided = False
        style = 0.0
        recovery_timer = 0
        planner_interventions = 0

        for t in range(max_steps):
            if t == spec["step"]:
                state = env.apply_velocity_kick(spec["delta_v"])
                traj.append(state.copy())
                recovery_timer = 14

            st = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                if mode == "plain":
                    action = policy(st)[0].detach().cpu().numpy()
                else:
                    if mode == "planner" and recovery_timer > 0 and value_net is not None:
                        new_style, scores = choose_counterfactual_style(policy, value_net, state, default_style=style)
                        if scores[new_style] > scores.get(style, -1e9) + 0.01:
                            style = new_style
                            planner_interventions += int(new_style != 0.0)
                        recovery_timer -= 1
                    action = policy(
                        st,
                        torch.ones(1, device=DEVICE),
                        torch.tensor([style], dtype=torch.float32, device=DEVICE),
                    )[0].detach().cpu().numpy()

            action = np.clip(action + np.random.normal(0.0, 0.012, size=3), -1.0, 1.0)
            state, done, collided = env.step(action)
            traj.append(state.copy())
            if done or collided:
                break

        ok = bool(done and not collided)
        trajectories.append(np.array(traj, dtype=np.float32))
        effective_steps.append((len(traj) - 1) if ok else max_steps)
        success.append(ok)
        collisions.append(bool(collided))
        artifact.append(bool(state[6] > 0.5))
        uplink.append(bool(state[7] > 0.5))
        total_planner_interventions += planner_interventions

    return {
        "trajectories": trajectories,
        "effective_steps": np.array(effective_steps),
        "success_rate": float(np.mean(success)),
        "collision_rate": float(np.mean(collisions)),
        "artifact_rate": float(np.mean(artifact)),
        "uplink_rate": float(np.mean(uplink)),
        "planner_interventions": total_planner_interventions if mode == "planner" else 0,
    }


def add_box(ax, box, color="#c64b59", alpha=0.18):
    x0, y0, z0, x1, y1, z1 = box
    vertices = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ]
    )
    faces = [
        [vertices[i] for i in [0, 1, 2, 3]],
        [vertices[i] for i in [4, 5, 6, 7]],
        [vertices[i] for i in [0, 1, 5, 4]],
        [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [1, 2, 6, 5]],
        [vertices[i] for i in [0, 3, 7, 4]],
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors=color, linewidths=0.4, alpha=alpha))


def style_3d(ax, title):
    xmin, xmax, ymin, ymax, zmin, zmax = env.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=23, azim=-56)
    ax.scatter(*env.start, c="black", s=24)
    ax.scatter(*env.artifact, c="#ff9d2e", s=60)
    ax.scatter(*env.uplink, c="#6bd0ff", s=70)
    ax.scatter(*env.extract, c="#f6d54a", edgecolors="black", s=90)
    for box in env.obstacles:
        add_box(ax, box)


def make_comparison_plot(plain_results, recap_results, planner_results):
    fig = plt.figure(figsize=(17, 6))
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    configs = [
        ("Plain BC", plain_results, "#c44e52"),
        ("RECAP Fixed Style", recap_results, "#dd8452"),
        ("RECAP + Counterfactual Planner", planner_results, "#4c9f70"),
    ]
    for ax, (title, results, color) in zip(axes, configs):
        style_3d(ax, title)
        for traj in results["trajectories"][:10]:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, alpha=0.35, lw=1.7)
        ax.text2D(
            0.03, 0.97,
            f"succ {results['success_rate'] * 100:.0f}%\nart {results['artifact_rate'] * 100:.0f}%\nuplink {results['uplink_rate'] * 100:.0f}%",
            transform=ax.transAxes, va="top"
        )
    fig.suptitle("Disturbed 3D Drone Heist: RECAP vs Counterfactual Planner", fontsize=16)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "counterfactual_3d_comparison.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def choose_examples(results, count=5):
    order = np.argsort(results["effective_steps"])
    idx = np.linspace(0, len(order) - 1, count, dtype=int)
    return [results["trajectories"][order[i]] for i in idx]


def make_3d_gif(results, path, title, color):
    examples = choose_examples(results, count=5)
    max_len = max(len(traj) for traj in examples)
    fig = plt.figure(figsize=(7.8, 7.0))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    style_3d(ax, title)
    lines = [ax.plot([], [], [], color=color, lw=2.2, alpha=0.34 + 0.10 * (i / len(examples)))[0] for i in range(len(examples))]
    dots = [ax.plot([], [], [], "o", color=color, markersize=5)[0] for _ in examples]
    counter = ax.text2D(0.02, 0.98, "", transform=ax.transAxes)

    def init():
        counter.set_text("")
        artists = [counter]
        for line, dot in zip(lines, dots):
            line.set_data([], [])
            line.set_3d_properties([])
            dot.set_data([], [])
            dot.set_3d_properties([])
            artists.extend([line, dot])
        return artists

    def update(frame):
        counter.set_text(f"t = {frame:03d}")
        ax.view_init(elev=24 + 2 * np.sin(frame * 0.04), azim=-56 + 0.12 * frame)
        artists = [counter]
        for line, dot, traj in zip(lines, dots, examples):
            idx = min(frame, len(traj) - 1)
            line.set_data(traj[: idx + 1, 0], traj[: idx + 1, 1])
            line.set_3d_properties(traj[: idx + 1, 2])
            dot.set_data([traj[idx, 0]], [traj[idx, 1]])
            dot.set_3d_properties([traj[idx, 2]])
            artists.extend([line, dot])
        return artists

    anim = FuncAnimation(fig, update, frames=max_len, init_func=init, interval=75, blit=False)
    anim.save(path, writer=PillowWriter(fps=14))
    plt.close(fig)


def main():
    print(f"Using device: {DEVICE}")
    print("Generating counterfactual-planning dataset...")
    states, actions, returns, adv_labels, styles, demos = build_dataset()
    print(f"demos: {len(demos)} | transitions: {len(states)} | dataset success: {np.mean([d.success for d in demos]) * 100:.1f}%")

    print("\nTraining baseline, latent-conditioned RECAP policy, and value model...")
    plain_policy, latent_policy, value_net = train_models(states, actions, returns, adv_labels, styles)

    print("\nEvaluating disturbed policies...")
    specs = make_disturbance_specs(20)
    plain_results = rollout(plain_policy, "plain", specs)
    recap_results = rollout(latent_policy, "recap", specs)
    planner_results = rollout(latent_policy, "planner", specs, value_net=value_net)

    delta_success = (planner_results["success_rate"] - recap_results["success_rate"]) * 100
    delta_steps = recap_results["effective_steps"].mean() - planner_results["effective_steps"].mean()
    delta_collision = (recap_results["collision_rate"] - planner_results["collision_rate"]) * 100
    delta_artifact = (planner_results["artifact_rate"] - recap_results["artifact_rate"]) * 100
    delta_uplink = (planner_results["uplink_rate"] - recap_results["uplink_rate"]) * 100

    print("\n=== COUNTERFACTUAL PLANNER RESULTS ===")
    print(
        f"Plain BC                -> Avg steps: {plain_results['effective_steps'].mean():.0f} | "
        f"Success: {plain_results['success_rate'] * 100:.0f}% | Artifact: {plain_results['artifact_rate'] * 100:.0f}% | "
        f"Uplink: {plain_results['uplink_rate'] * 100:.0f}% | Collisions: {plain_results['collision_rate'] * 100:.0f}%"
    )
    print(
        f"RECAP fixed-style       -> Avg steps: {recap_results['effective_steps'].mean():.0f} | "
        f"Success: {recap_results['success_rate'] * 100:.0f}% | Artifact: {recap_results['artifact_rate'] * 100:.0f}% | "
        f"Uplink: {recap_results['uplink_rate'] * 100:.0f}% | Collisions: {recap_results['collision_rate'] * 100:.0f}%"
    )
    print(
        f"RECAP + planner         -> Avg steps: {planner_results['effective_steps'].mean():.0f} | "
        f"Success: {planner_results['success_rate'] * 100:.0f}% | Artifact: {planner_results['artifact_rate'] * 100:.0f}% | "
        f"Uplink: {planner_results['uplink_rate'] * 100:.0f}% | Collisions: {planner_results['collision_rate'] * 100:.0f}%"
    )
    print(
        f"Improvement over RECAP  -> Success {delta_success:+.1f} pts | Artifact {delta_artifact:+.1f} pts | "
        f"Uplink {delta_uplink:+.1f} pts | Effective steps {delta_steps:+.1f} | Collision {delta_collision:+.1f} pts"
    )
    print(f"Planner safe-style interventions: {planner_results['planner_interventions']}")

    png_path = make_comparison_plot(plain_results, recap_results, planner_results)
    make_3d_gif(plain_results, os.path.join(OUTPUT_DIR, "counterfactual_plain_3d.gif"), "Plain BC 3D", "#c44e52")
    make_3d_gif(recap_results, os.path.join(OUTPUT_DIR, "counterfactual_recap_3d.gif"), "RECAP Fixed Style 3D", "#dd8452")
    make_3d_gif(planner_results, os.path.join(OUTPUT_DIR, "counterfactual_planner_3d.gif"), "RECAP + Counterfactual Planner 3D", "#4c9f70")
    print(f"Saved comparison PNG -> {png_path}")
    print("Saved GIFs -> counterfactual_plain_3d.gif, counterfactual_recap_3d.gif, counterfactual_planner_3d.gif")


if __name__ == "__main__":
    main()
