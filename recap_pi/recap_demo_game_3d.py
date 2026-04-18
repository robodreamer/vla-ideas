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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

SEED = 7
torch.manual_seed(SEED)
np.random.seed(SEED)


def get_device():
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


DEVICE = get_device()


class KeyDoorEnv3D:
    def __init__(self):
        self.dt = 0.075
        self.max_acc = 1.0
        self.max_vel = 1.2
        self.start = np.array([0.06, 0.08, 0.10], dtype=np.float32)
        self.key = np.array([0.18, 0.84, 0.18], dtype=np.float32)
        self.portal = np.array([0.98, 0.98, 0.94], dtype=np.float32)
        self.key_radius = 0.09
        self.portal_radius = 0.10
        self.bounds = (-0.10, 1.12, -0.08, 1.12, -0.02, 1.08)
        self.obstacles = [
            [0.36, 0.18, 0.00, 0.68, 0.84, 0.72],   # central tower
            [0.52, 0.48, 0.30, 0.92, 0.78, 0.84],   # roof blocking low key->portal path
            [0.72, 0.02, 0.00, 0.90, 0.34, 0.68],   # wrong-side wall
        ]
        self.state = None

    def reset(self, jitter_scale=0.0):
        pos_noise = np.random.normal(0.0, jitter_scale, size=3)
        vel_noise = np.random.normal(0.0, jitter_scale * 0.4, size=3)
        self.state = np.zeros(7, dtype=np.float32)
        self.state[:3] = self.start + pos_noise.astype(np.float32)
        self.state[3:6] = vel_noise.astype(np.float32)
        self.state[6] = 0.0
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
        return (
            pos[0] < xmin
            or pos[0] > xmax
            or pos[1] < ymin
            or pos[1] > ymax
            or pos[2] < zmin
            or pos[2] > zmax
        )

    def step(self, action):
        action = np.clip(action, -self.max_acc, self.max_acc)
        self.state[3:6] = np.clip(self.state[3:6] + action * self.dt, -self.max_vel, self.max_vel)
        self.state[:3] = self.state[:3] + self.state[3:6] * self.dt

        collided = self.inside_obstacle(self.state[:3]) or self.out_of_bounds(self.state[:3])
        picked_key = False
        if self.state[6] < 0.5 and np.linalg.norm(self.state[:3] - self.key) < self.key_radius:
            self.state[6] = 1.0
            picked_key = True

        done = bool(self.state[6] > 0.5 and np.linalg.norm(self.state[:3] - self.portal) < self.portal_radius)
        return self.state.copy(), done, collided, picked_key


env = KeyDoorEnv3D()


@dataclass
class Demo:
    states: np.ndarray
    actions: np.ndarray
    score: float
    success: bool
    got_key: bool
    steps: int
    label: str


def pd_action(state, target, kp=5.2, kd=2.8):
    pos_err = target - state[:3]
    vel_err = -state[3:6]
    return np.clip(kp * pos_err + kd * vel_err, -1.0, 1.0)


def run_waypoint_demo(waypoints, label, noise=0.0, kp=5.2, kd=2.8, max_steps=240):
    state = env.reset()
    states = []
    actions = []
    waypoint_idx = 0
    got_key = False
    success = False
    collided = False

    for _ in range(max_steps):
        target = np.array(waypoints[min(waypoint_idx, len(waypoints) - 1)], dtype=np.float32)
        if np.linalg.norm(state[:3] - target) < 0.12 and waypoint_idx < len(waypoints) - 1:
            waypoint_idx += 1
            target = np.array(waypoints[waypoint_idx], dtype=np.float32)

        act = pd_action(state, target, kp=kp, kd=kd)
        if noise > 0.0:
            act = np.clip(act + np.random.normal(0.0, noise, size=3), -1.0, 1.0)

        states.append(state.copy())
        actions.append(act.copy())
        state, success, collided, picked_key = env.step(act)
        got_key = got_key or picked_key or state[6] > 0.5
        if success or collided:
            break

    if success:
        score = 3.0 - 0.006 * len(states)
    elif collided:
        score = -2.2 + 0.4 * float(got_key) - 0.002 * len(states)
    else:
        portal_dist = np.linalg.norm(state[:3] - env.portal)
        score = -1.0 + 0.7 * float(got_key) - 0.45 * portal_dist

    return Demo(
        states=np.array(states, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        score=float(score),
        success=bool(success and not collided),
        got_key=bool(got_key),
        steps=len(states),
        label=label,
    )


def run_bad_demo(kind, max_steps=260):
    state = env.reset()
    states = []
    actions = []
    got_key = False
    success = False
    collided = False

    for t in range(max_steps):
        if kind == "portal_rush":
            target = env.portal
            act = pd_action(state, target, kp=3.6, kd=1.1)
            act += np.array([0.05, 0.00, 0.04])
        elif kind == "low_key_then_crash":
            if state[6] < 0.5:
                target = env.key
                act = pd_action(state, target, kp=4.8, kd=1.9)
            else:
                target = np.array([0.74, 0.64, 0.40], dtype=np.float32)
                act = pd_action(state, target, kp=4.4, kd=1.2)
        elif kind == "hover_key":
            target = env.key + np.array([0.10 * np.sin(t * 0.18), 0.0, 0.06 * np.cos(t * 0.22)], dtype=np.float32)
            act = pd_action(state, target, kp=2.4, kd=0.7)
        elif kind == "wrong_side_arc":
            if t < 70:
                target = np.array([0.86, 0.18, 0.22], dtype=np.float32)
            else:
                target = np.array([0.94, 0.74, 0.42], dtype=np.float32) if state[6] > 0.5 else env.portal
            act = pd_action(state, target, kp=3.8, kd=1.0)
        else:
            if state[6] < 0.5:
                target = np.array([0.14, 0.76, 0.16], dtype=np.float32)
            else:
                target = np.array([0.56, 0.86, 0.52], dtype=np.float32)
            act = pd_action(state, target, kp=3.0, kd=0.8)
            act += 0.25 * np.array([np.sin(t * 0.23), np.cos(t * 0.19), np.sin(t * 0.17)])

        act = np.clip(act + np.random.normal(0.0, 0.10, size=3), -1.0, 1.0)
        states.append(state.copy())
        actions.append(act.copy())
        state, success, collided, picked_key = env.step(act)
        got_key = got_key or picked_key or state[6] > 0.5
        if success or collided:
            break

    if success:
        score = 1.0 - 0.008 * len(states)
    elif collided:
        score = -2.4 + 0.3 * float(got_key) - 0.002 * len(states)
    else:
        portal_dist = np.linalg.norm(state[:3] - env.portal)
        score = -1.4 + 0.5 * float(got_key) - 0.35 * portal_dist

    return Demo(
        states=np.array(states, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        score=float(score),
        success=bool(success and not collided),
        got_key=bool(got_key),
        steps=len(states),
        label=kind,
    )


def build_dataset():
    demos = []

    good_route = [
        (0.10, 0.20, 0.12),
        (0.16, 0.86, 0.18),
        (0.34, 0.94, 0.80),
        (0.82, 0.96, 0.98),
        (0.98, 0.98, 0.94),
    ]
    wide_route = [
        (0.08, 0.24, 0.10),
        (0.18, 0.88, 0.20),
        (0.28, 1.00, 0.88),
        (0.90, 1.00, 1.00),
        (0.98, 0.98, 0.94),
    ]

    for _ in range(260):
        demos.append(run_waypoint_demo(good_route, label="expert_route", noise=0.05, kp=5.8, kd=3.2))
    for _ in range(70):
        demos.append(run_waypoint_demo(wide_route, label="safe_route", noise=0.08, kp=4.8, kd=2.6))
    for _ in range(420):
        demos.append(run_bad_demo("portal_rush"))
    for _ in range(280):
        demos.append(run_bad_demo("low_key_then_crash"))
    for _ in range(220):
        demos.append(run_bad_demo("hover_key"))
    for _ in range(240):
        demos.append(run_bad_demo("wrong_side_arc"))
    for _ in range(180):
        demos.append(run_bad_demo("orbit"))

    scores = np.array([demo.score for demo in demos], dtype=np.float32)
    threshold = np.quantile(scores, 0.80)

    states = []
    actions = []
    conds = []
    returns = []
    for demo in demos:
        cond = 1.0 if demo.score >= threshold else 0.0
        for state, action in zip(demo.states, demo.actions):
            states.append(state)
            actions.append(action)
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
        in_dim = 8 if conditioned else 7
        self.net = nn.Sequential(
            nn.Linear(in_dim, 160),
            nn.ReLU(),
            nn.Linear(160, 160),
            nn.ReLU(),
            nn.Linear(160, 96),
            nn.ReLU(),
            nn.Linear(96, 3),
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
            nn.Linear(7, 160),
            nn.ReLU(),
            nn.Linear(160, 160),
            nn.ReLU(),
            nn.Linear(160, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )

    def forward(self, state):
        return self.net(state)


def train_models(states, actions, conds, returns, epochs=150, batch_size=1024):
    states = states.to(DEVICE)
    actions = actions.to(DEVICE)
    conds = conds.to(DEVICE)
    returns = returns.to(DEVICE)

    plain = Policy(conditioned=False).to(DEVICE)
    recap = Policy(conditioned=True).to(DEVICE)
    value = ValueNet().to(DEVICE)

    popt = torch.optim.Adam(plain.parameters(), lr=1e-3)
    ropt = torch.optim.Adam(recap.parameters(), lr=1e-3)
    vopt = torch.optim.Adam(value.parameters(), lr=1e-3)

    num = states.shape[0]
    for epoch in range(epochs):
        order = torch.randperm(num, device=DEVICE)
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

            rloss = F.mse_loss(recap(s, c), a)
            null_mask = torch.rand_like(c) < 0.20
            if null_mask.any():
                rloss = rloss + 0.25 * F.mse_loss(recap(s[null_mask], torch.zeros_like(c[null_mask])), a[null_mask])
            ropt.zero_grad()
            rloss.backward()
            ropt.step()

        if epoch in {0, 39, 79, 119, epochs - 1}:
            print(
                f"epoch {epoch + 1:03d}/{epochs} | "
                f"value {vloss.item():.4f} | plain {ploss.item():.4f} | recap {rloss.item():.4f}"
            )

    return plain, recap, value


def rollout(policy, conditioned, episodes=40, max_steps=240):
    trajectories = []
    effective_steps = []
    raw_steps = []
    successes = []
    collisions = []
    key_rates = []
    path_lengths = []

    for _ in range(episodes):
        state = env.reset(jitter_scale=0.016)
        traj = [state.copy()]
        collided = False
        success = False
        got_key = bool(state[6] > 0.5)

        for _ in range(max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                if conditioned:
                    action = policy(state_tensor, torch.ones(1, device=DEVICE))[0].detach().cpu().numpy()
                else:
                    action = policy(state_tensor)[0].detach().cpu().numpy()
            action = np.clip(action + np.random.normal(0.0, 0.012, size=3), -1.0, 1.0)
            state, success, collided, picked_key = env.step(action)
            got_key = got_key or picked_key or state[6] > 0.5
            traj.append(state.copy())
            if success or collided:
                break

        traj = np.array(traj, dtype=np.float32)
        step_count = len(traj) - 1
        path_len = float(np.sum(np.linalg.norm(traj[1:, :3] - traj[:-1, :3], axis=1)))
        trajectories.append(traj)
        raw_steps.append(step_count)
        effective_steps.append(step_count if success and not collided else max_steps)
        successes.append(bool(success and not collided))
        collisions.append(bool(collided))
        key_rates.append(bool(got_key))
        path_lengths.append(path_len)

    return {
        "trajectories": trajectories,
        "raw_steps": np.array(raw_steps),
        "effective_steps": np.array(effective_steps),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "key_rate": float(np.mean(key_rates)),
        "path_length": float(np.mean(path_lengths)),
    }


def add_box_faces(ax, box, color, alpha=0.14):
    x0, y0, z0, x1, y1, z1 = box
    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
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
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors=color, linewidths=0.6, alpha=alpha))


def style_3d_axis(ax, title):
    xmin, xmax, ymin, ymax, zmin, zmax = env.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.view_init(elev=24, azim=-58)
    ax.scatter(*env.start, c="black", s=36)
    ax.scatter(*env.key, c="#f28e2b", s=80)
    ax.scatter(*env.portal, c="#eeca3b", edgecolors="black", s=110)
    for box in env.obstacles:
        add_box_faces(ax, box, "#b23a48")


def projection_rectangles(plane):
    rects = []
    for box in env.obstacles:
        if plane == "xy":
            rects.append((box[0], box[1], box[3] - box[0], box[4] - box[1]))
        elif plane == "xz":
            rects.append((box[0], box[2], box[3] - box[0], box[5] - box[2]))
        else:
            rects.append((box[1], box[2], box[4] - box[1], box[5] - box[2]))
    return rects


def style_projection(ax, plane, title):
    xmin, xmax, ymin, ymax, zmin, zmax = env.bounds
    if plane == "xy":
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        start = env.start[[0, 1]]
        key = env.key[[0, 1]]
        portal = env.portal[[0, 1]]
        xlabel, ylabel = "x", "y"
    elif plane == "xz":
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(zmin, zmax)
        start = env.start[[0, 2]]
        key = env.key[[0, 2]]
        portal = env.portal[[0, 2]]
        xlabel, ylabel = "x", "z"
    else:
        ax.set_xlim(ymin, ymax)
        ax.set_ylim(zmin, zmax)
        start = env.start[[1, 2]]
        key = env.key[[1, 2]]
        portal = env.portal[[1, 2]]
        xlabel, ylabel = "y", "z"

    for rect in projection_rectangles(plane):
        ax.add_patch(Rectangle((rect[0], rect[1]), rect[2], rect[3], color="#b23a48", alpha=0.22))
    ax.scatter(*start, c="black", s=28)
    ax.scatter(*key, c="#f28e2b", s=52)
    ax.scatter(*portal, c="#eeca3b", edgecolors="black", s=72)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.18)
    ax.set_aspect("equal")


def choose_examples(results, count=6):
    order = np.argsort(results["effective_steps"])
    indices = np.linspace(0, len(order) - 1, count, dtype=int)
    return [results["trajectories"][order[idx]] for idx in indices]


def make_comparison_plot(plain_results, recap_results):
    fig = plt.figure(figsize=(18, 11))
    ax_plain_3d = fig.add_subplot(2, 3, 1, projection="3d")
    ax_recap_3d = fig.add_subplot(2, 3, 2, projection="3d")
    ax_bar = fig.add_subplot(2, 3, 3)
    ax_xy = fig.add_subplot(2, 3, 4)
    ax_xz = fig.add_subplot(2, 3, 5)
    ax_yz = fig.add_subplot(2, 3, 6)

    style_3d_axis(ax_plain_3d, "Plain Imitation")
    style_3d_axis(ax_recap_3d, "RECAP Conditioned")

    for traj in plain_results["trajectories"][:14]:
        ax_plain_3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="#4c78a8", alpha=0.32, lw=1.5)
    for traj in recap_results["trajectories"][:14]:
        ax_recap_3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="#54a24b", alpha=0.42, lw=1.7)

    metrics = ["Eff. steps", "Success %", "Key pickup %", "Path len"]
    plain_vals = [
        plain_results["effective_steps"].mean(),
        plain_results["success_rate"] * 100,
        plain_results["key_rate"] * 100,
        plain_results["path_length"],
    ]
    recap_vals = [
        recap_results["effective_steps"].mean(),
        recap_results["success_rate"] * 100,
        recap_results["key_rate"] * 100,
        recap_results["path_length"],
    ]
    x = np.arange(len(metrics))
    width = 0.34
    ax_bar.bar(x - width / 2, plain_vals, width, label="Plain", color="#4c78a8")
    ax_bar.bar(x + width / 2, recap_vals, width, label="RECAP", color="#54a24b")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics, rotation=16)
    ax_bar.set_title("Episode Metrics")
    ax_bar.legend()

    for plane, ax, title in [("xy", ax_xy, "Top View"), ("xz", ax_xz, "Side View x-z"), ("yz", ax_yz, "Side View y-z")]:
        style_projection(ax, plane, title)
        for traj in plain_results["trajectories"][:10]:
            if plane == "xy":
                ax.plot(traj[:, 0], traj[:, 1], color="#4c78a8", alpha=0.26, lw=1.5)
            elif plane == "xz":
                ax.plot(traj[:, 0], traj[:, 2], color="#4c78a8", alpha=0.26, lw=1.5)
            else:
                ax.plot(traj[:, 1], traj[:, 2], color="#4c78a8", alpha=0.26, lw=1.5)
        for traj in recap_results["trajectories"][:10]:
            if plane == "xy":
                ax.plot(traj[:, 0], traj[:, 1], color="#54a24b", alpha=0.34, lw=1.6)
            elif plane == "xz":
                ax.plot(traj[:, 0], traj[:, 2], color="#54a24b", alpha=0.34, lw=1.6)
            else:
                ax.plot(traj[:, 1], traj[:, 2], color="#54a24b", alpha=0.34, lw=1.6)

    fig.suptitle("3D Key-Then-Portal Task: Plain BC vs RECAP Conditioning", fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "game_3d_comparison.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_rollout_gif(results, path, title, color):
    examples = choose_examples(results, count=6)
    max_len = max(len(traj) for traj in examples)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6))
    plane_defs = [("xy", axes[0], "Top"), ("xz", axes[1], "x-z"), ("yz", axes[2], "y-z")]
    for plane, ax, plane_title in plane_defs:
        style_projection(ax, plane, plane_title)

    counter = fig.text(0.50, 0.96, "", ha="center", fontsize=12)
    fig.suptitle(title, y=1.02, fontsize=16)

    lines = []
    dots = []
    for plane, ax, _ in plane_defs:
        plane_lines = []
        plane_dots = []
        for idx, _ in enumerate(examples):
            alpha = 0.32 + 0.10 * (idx / len(examples))
            plane_lines.append(ax.plot([], [], color=color, lw=2.0, alpha=alpha)[0])
            plane_dots.append(ax.plot([], [], "o", color=color, markersize=4.4)[0])
        lines.append(plane_lines)
        dots.append(plane_dots)

    def init():
        counter.set_text("")
        artists = [counter]
        for plane_lines, plane_dots in zip(lines, dots):
            for line, dot in zip(plane_lines, plane_dots):
                line.set_data([], [])
                dot.set_data([], [])
                artists.extend([line, dot])
        return artists

    def update(frame):
        counter.set_text(f"t = {frame:03d}")
        artists = [counter]
        for plane_idx, (plane, _, _) in enumerate(plane_defs):
            for traj_idx, traj in enumerate(examples):
                idx = min(frame, len(traj) - 1)
                if plane == "xy":
                    xdata, ydata = traj[: idx + 1, 0], traj[: idx + 1, 1]
                    px, py = traj[idx, 0], traj[idx, 1]
                elif plane == "xz":
                    xdata, ydata = traj[: idx + 1, 0], traj[: idx + 1, 2]
                    px, py = traj[idx, 0], traj[idx, 2]
                else:
                    xdata, ydata = traj[: idx + 1, 1], traj[: idx + 1, 2]
                    px, py = traj[idx, 1], traj[idx, 2]
                lines[plane_idx][traj_idx].set_data(xdata, ydata)
                dots[plane_idx][traj_idx].set_data([px], [py])
                artists.extend([lines[plane_idx][traj_idx], dots[plane_idx][traj_idx]])
        return artists

    anim = FuncAnimation(fig, update, frames=max_len, init_func=init, interval=90, blit=True)
    anim.save(path, writer=PillowWriter(fps=11))
    plt.close(fig)


def make_rollout_3d_gif(results, path, title, color):
    examples = choose_examples(results, count=5)
    max_len = max(len(traj) for traj in examples)

    fig = plt.figure(figsize=(7.6, 7.0))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    style_3d_axis(ax, title)

    lines = [ax.plot([], [], [], color=color, lw=2.2, alpha=0.34 + 0.10 * (idx / len(examples)))[0] for idx in range(len(examples))]
    dots = [ax.plot([], [], [], "o", color=color, markersize=5)[0] for _ in examples]
    counter = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, fontsize=12)

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
        ax.view_init(elev=24, azim=-58 + 0.20 * frame)
        artists = [counter]
        for line, dot, traj in zip(lines, dots, examples):
            idx = min(frame, len(traj) - 1)
            line.set_data(traj[: idx + 1, 0], traj[: idx + 1, 1])
            line.set_3d_properties(traj[: idx + 1, 2])
            dot.set_data([traj[idx, 0]], [traj[idx, 1]])
            dot.set_3d_properties([traj[idx, 2]])
            artists.extend([line, dot])
        return artists

    anim = FuncAnimation(fig, update, frames=max_len, init_func=init, interval=100, blit=False)
    anim.save(path, writer=PillowWriter(fps=10))
    plt.close(fig)


def main():
    print(f"Using device: {DEVICE}")
    print("Generating 3D mixed-quality gameplay demonstrations...")
    states, actions, conds, returns, demos = build_dataset()
    demo_success = np.mean([demo.success for demo in demos])
    demo_key_rate = np.mean([demo.got_key for demo in demos])
    print(
        f"dataset demos: {len(demos)} | transitions: {len(states)} | "
        f"key pickups: {demo_key_rate * 100:.1f}% | successes: {demo_success * 100:.1f}%"
    )

    print("\nTraining plain imitation and RECAP-conditioned policies...")
    plain_policy, recap_policy, _value_net = train_models(states, actions, conds, returns)

    print("\nEvaluating policies...")
    plain_results = rollout(plain_policy, conditioned=False)
    recap_results = rollout(recap_policy, conditioned=True)

    plain_steps = plain_results["effective_steps"].mean()
    recap_steps = recap_results["effective_steps"].mean()

    print("\n=== RESULTS ===")
    print(
        f"Plain Imitation  -> Avg steps: {plain_steps:.0f} | "
        f"Success: {plain_results['success_rate'] * 100:.0f}% | "
        f"Key pickup: {plain_results['key_rate'] * 100:.0f}% | "
        f"Collisions: {plain_results['collision_rate'] * 100:.0f}% | "
        f"Raw term steps: {plain_results['raw_steps'].mean():.0f}"
    )
    print(
        f"RECAP Advantage  -> Avg steps: {recap_steps:.0f} | "
        f"Success: {recap_results['success_rate'] * 100:.0f}% | "
        f"Key pickup: {recap_results['key_rate'] * 100:.0f}% | "
        f"Collisions: {recap_results['collision_rate'] * 100:.0f}% | "
        f"Raw term steps: {recap_results['raw_steps'].mean():.0f}"
    )
    print(f"Duration improvement: {plain_steps / recap_steps:.2f}x faster")
    print(
        f"Success improvement: "
        f"{recap_results['success_rate'] / max(plain_results['success_rate'], 1e-6):.2f}x higher"
    )

    print("\nSaving visuals...")
    make_comparison_plot(plain_results, recap_results)
    make_rollout_gif(
        plain_results,
        os.path.join(OUTPUT_DIR, "game_3d_plain.gif"),
        "Plain Imitation: 3D Key-Portal Rollouts",
        "#4c78a8",
    )
    make_rollout_gif(
        recap_results,
        os.path.join(OUTPUT_DIR, "game_3d_recap.gif"),
        "RECAP Conditioned: 3D Key-Portal Rollouts",
        "#54a24b",
    )
    make_rollout_3d_gif(
        plain_results,
        os.path.join(OUTPUT_DIR, "game_3d_plain_3d.gif"),
        "Plain Imitation 3D",
        "#4c78a8",
    )
    make_rollout_3d_gif(
        recap_results,
        os.path.join(OUTPUT_DIR, "game_3d_recap_3d.gif"),
        "RECAP Conditioned 3D",
        "#54a24b",
    )
    print(f"Saved {os.path.join(OUTPUT_DIR, 'game_3d_comparison.png')}")
    print(f"Saved {os.path.join(OUTPUT_DIR, 'game_3d_plain.gif')}")
    print(f"Saved {os.path.join(OUTPUT_DIR, 'game_3d_recap.gif')}")
    print(f"Saved {os.path.join(OUTPUT_DIR, 'game_3d_plain_3d.gif')}")
    print(f"Saved {os.path.join(OUTPUT_DIR, 'game_3d_recap_3d.gif')}")


if __name__ == "__main__":
    main()
