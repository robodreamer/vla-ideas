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
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

SEED = 11
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

COLORS = {
    "plain": "#4c78a8",
    "recap": "#4daf4a",
    "artifact": "#ff9d2e",
    "uplink": "#6bd0ff",
    "extract": "#f6d54a",
    "hazard": "#c64b59",
    "floor": "#eaf1f8",
    "grid": "#ced8e3",
    "bg": "#f7fbff",
}


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

    def reset(self, jitter_scale=0.0):
        pos_noise = np.random.normal(0.0, jitter_scale, size=3)
        vel_noise = np.random.normal(0.0, jitter_scale * 0.35, size=3)
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
        grabbed_artifact = False
        activated_uplink = False

        if self.state[6] < 0.5 and np.linalg.norm(self.state[:3] - self.artifact) < self.artifact_radius:
            self.state[6] = 1.0
            grabbed_artifact = True

        if self.state[6] > 0.5 and self.state[7] < 0.5 and np.linalg.norm(self.state[:3] - self.uplink) < self.uplink_radius:
            self.state[7] = 1.0
            activated_uplink = True

        done = bool(self.state[6] > 0.5 and self.state[7] > 0.5 and np.linalg.norm(self.state[:3] - self.extract) < self.extract_radius)
        return self.state.copy(), done, collided, grabbed_artifact, activated_uplink


env = DroneHeistEnv3D()


@dataclass
class Demo:
    states: np.ndarray
    actions: np.ndarray
    score: float
    success: bool
    got_artifact: bool
    got_uplink: bool
    steps: int
    label: str


def pd_action(state, target, kp=5.2, kd=2.7):
    pos_err = target - state[:3]
    vel_err = -state[3:6]
    return np.clip(kp * pos_err + kd * vel_err, -1.0, 1.0)


def eval_demo_score(state, steps, success, collided):
    got_artifact = state[6] > 0.5
    got_uplink = state[7] > 0.5
    extract_dist = np.linalg.norm(state[:3] - env.extract)
    progress_bonus = 0.75 * float(got_artifact) + 0.95 * float(got_uplink)
    if success:
        return 4.0 - 0.007 * steps
    if collided:
        return -2.6 + 0.55 * float(got_artifact) + 0.65 * float(got_uplink) - 0.002 * steps
    return -1.1 + progress_bonus - 0.45 * extract_dist


def run_waypoint_demo(waypoints, label, noise=0.0, kp=5.2, kd=2.7, max_steps=260):
    state = env.reset()
    states = []
    actions = []
    waypoint_idx = 0
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
        state, success, collided, _, _ = env.step(act)
        if success or collided:
            break

    return Demo(
        states=np.array(states, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        score=float(eval_demo_score(state, len(states), success, collided)),
        success=bool(success and not collided),
        got_artifact=bool(state[6] > 0.5),
        got_uplink=bool(state[7] > 0.5),
        steps=len(states),
        label=label,
    )


def run_bad_demo(kind, max_steps=280):
    state = env.reset()
    states = []
    actions = []
    success = False
    collided = False

    for t in range(max_steps):
        if kind == "rush_extract":
            target = env.extract
            act = pd_action(state, target, kp=3.4, kd=1.0)
            act += np.array([0.05, -0.02, 0.05])
        elif kind == "artifact_then_crash":
            if state[6] < 0.5:
                target = env.artifact
                act = pd_action(state, target, kp=4.9, kd=1.9)
            else:
                target = np.array([0.58, 0.66, 0.42], dtype=np.float32)
                act = pd_action(state, target, kp=4.4, kd=1.1)
        elif kind == "camp_uplink":
            if state[6] < 0.5:
                target = env.artifact
            else:
                target = env.uplink + np.array([0.10 * np.sin(t * 0.22), 0.06 * np.cos(t * 0.19), 0.04 * np.sin(t * 0.17)], dtype=np.float32)
            act = pd_action(state, target, kp=2.5, kd=0.7)
        elif kind == "wrong_side_swoop":
            if t < 85:
                target = np.array([0.88, 0.76, 0.26], dtype=np.float32)
            elif state[6] < 0.5:
                target = env.artifact
            else:
                target = np.array([0.92, 0.48, 0.48], dtype=np.float32)
            act = pd_action(state, target, kp=3.7, kd=1.0)
        else:
            if state[6] < 0.5:
                target = np.array([0.18, 0.84, 0.20], dtype=np.float32)
            elif state[7] < 0.5:
                target = np.array([0.50, 0.94, 0.62], dtype=np.float32)
            else:
                target = np.array([0.82, 0.30, 0.80], dtype=np.float32)
            act = pd_action(state, target, kp=3.0, kd=0.8)
            act += 0.23 * np.array([np.sin(t * 0.24), np.cos(t * 0.20), np.sin(t * 0.16)])

        act = np.clip(act + np.random.normal(0.0, 0.10, size=3), -1.0, 1.0)
        states.append(state.copy())
        actions.append(act.copy())
        state, success, collided, _, _ = env.step(act)
        if success or collided:
            break

    return Demo(
        states=np.array(states, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        score=float(eval_demo_score(state, len(states), success, collided)),
        success=bool(success and not collided),
        got_artifact=bool(state[6] > 0.5),
        got_uplink=bool(state[7] > 0.5),
        steps=len(states),
        label=kind,
    )


def build_dataset():
    demos = []

    heist_route = [
        (0.10, 0.18, 0.12),
        (0.16, 0.82, 0.18),
        (0.28, 0.96, 0.36),
        (0.70, 0.88, 0.94),
        (0.90, 0.54, 1.00),
        (0.98, 0.22, 0.90),
    ]
    wide_heist_route = [
        (0.08, 0.22, 0.10),
        (0.18, 0.86, 0.20),
        (0.26, 0.98, 0.42),
        (0.64, 0.98, 0.98),
        (0.90, 0.46, 1.02),
        (0.98, 0.22, 0.90),
    ]

    for _ in range(300):
        demos.append(run_waypoint_demo(heist_route, "expert_heist", noise=0.05, kp=5.8, kd=3.2))
    for _ in range(90):
        demos.append(run_waypoint_demo(wide_heist_route, "safe_heist", noise=0.08, kp=4.9, kd=2.6))
    for _ in range(420):
        demos.append(run_bad_demo("rush_extract"))
    for _ in range(300):
        demos.append(run_bad_demo("artifact_then_crash"))
    for _ in range(240):
        demos.append(run_bad_demo("camp_uplink"))
    for _ in range(220):
        demos.append(run_bad_demo("wrong_side_swoop"))
    for _ in range(180):
        demos.append(run_bad_demo("jammed_orbit"))

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
        in_dim = 9 if conditioned else 8
        self.net = nn.Sequential(
            nn.Linear(in_dim, 192),
            nn.ReLU(),
            nn.Linear(192, 192),
            nn.ReLU(),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
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
            nn.Linear(8, 192),
            nn.ReLU(),
            nn.Linear(192, 192),
            nn.ReLU(),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
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


def rollout(policy, conditioned, episodes=42, max_steps=260):
    trajectories = []
    effective_steps = []
    raw_steps = []
    successes = []
    collisions = []
    artifact_rate = []
    uplink_rate = []
    path_lengths = []

    for _ in range(episodes):
        state = env.reset(jitter_scale=0.016)
        traj = [state.copy()]
        collided = False
        success = False

        for _ in range(max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                if conditioned:
                    action = policy(state_tensor, torch.ones(1, device=DEVICE))[0].detach().cpu().numpy()
                else:
                    action = policy(state_tensor)[0].detach().cpu().numpy()
            action = np.clip(action + np.random.normal(0.0, 0.010, size=3), -1.0, 1.0)
            state, success, collided, _, _ = env.step(action)
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
        artifact_rate.append(bool(traj[-1, 6] > 0.5))
        uplink_rate.append(bool(traj[-1, 7] > 0.5))
        path_lengths.append(path_len)

    return {
        "trajectories": trajectories,
        "raw_steps": np.array(raw_steps),
        "effective_steps": np.array(effective_steps),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "artifact_rate": float(np.mean(artifact_rate)),
        "uplink_rate": float(np.mean(uplink_rate)),
        "path_length": float(np.mean(path_lengths)),
    }


def add_box_faces(ax, box, color, alpha=0.18):
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
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors=color, linewidths=0.5, alpha=alpha))


def draw_floor_plane(ax):
    xmin, xmax, ymin, ymax, _, _ = env.bounds
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 2), np.linspace(ymin, ymax, 2))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color=COLORS["floor"], alpha=0.30, shade=False, linewidth=0)


def draw_goal_halo(ax, center, radius, color):
    u = np.linspace(0, 2 * np.pi, 40)
    x = center[0] + radius * np.cos(u)
    y = center[1] + radius * np.sin(u)
    z = np.full_like(u, center[2])
    ax.plot(x, y, z, color=color, lw=1.6, alpha=0.85)


def style_3d_axis(ax, title):
    xmin, xmax, ymin, ymax, zmin, zmax = env.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_facecolor(COLORS["bg"])
    ax.xaxis.set_pane_color((0.93, 0.96, 0.99, 0.68))
    ax.yaxis.set_pane_color((0.93, 0.96, 0.99, 0.68))
    ax.zaxis.set_pane_color((0.93, 0.96, 0.99, 0.24))
    ax.grid(alpha=0.10)
    ax.view_init(elev=24, azim=-56)
    draw_floor_plane(ax)
    for box in env.obstacles:
        add_box_faces(ax, box, COLORS["hazard"])
    draw_goal_halo(ax, env.artifact, env.artifact_radius, COLORS["artifact"])
    draw_goal_halo(ax, env.uplink, env.uplink_radius, COLORS["uplink"])
    draw_goal_halo(ax, env.extract, env.extract_radius, COLORS["extract"])
    ax.scatter(*env.start, c="black", s=30, depthshade=False)
    ax.scatter(*env.artifact, c=COLORS["artifact"], s=70, depthshade=False)
    ax.scatter(*env.uplink, c=COLORS["uplink"], s=85, depthshade=False)
    ax.scatter(*env.extract, c=COLORS["extract"], edgecolors="black", s=100, depthshade=False)
    ax.text(*env.artifact, " artifact", color="#9b5c00", fontsize=9)
    ax.text(*env.uplink, " uplink", color="#155b74", fontsize=9)
    ax.text(*env.extract, " extract", color="#6c5a00", fontsize=9)


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


def add_projection_halo(ax, center, radius, color):
    ax.add_patch(Circle((center[0], center[1]), radius, edgecolor=color, facecolor="none", linewidth=1.4, alpha=0.85))


def style_projection(ax, plane, title):
    xmin, xmax, ymin, ymax, zmin, zmax = env.bounds
    ax.set_facecolor("#fbfdff")
    if plane == "xy":
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        start = env.start[[0, 1]]
        artifact = env.artifact[[0, 1]]
        uplink = env.uplink[[0, 1]]
        extract = env.extract[[0, 1]]
        xlabel, ylabel = "x", "y"
    elif plane == "xz":
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(zmin, zmax)
        start = env.start[[0, 2]]
        artifact = env.artifact[[0, 2]]
        uplink = env.uplink[[0, 2]]
        extract = env.extract[[0, 2]]
        xlabel, ylabel = "x", "z"
    else:
        ax.set_xlim(ymin, ymax)
        ax.set_ylim(zmin, zmax)
        start = env.start[[1, 2]]
        artifact = env.artifact[[1, 2]]
        uplink = env.uplink[[1, 2]]
        extract = env.extract[[1, 2]]
        xlabel, ylabel = "y", "z"

    for rect in projection_rectangles(plane):
        ax.add_patch(Rectangle((rect[0], rect[1]), rect[2], rect[3], color=COLORS["hazard"], alpha=0.18))
    ax.scatter(*start, c="black", s=26)
    ax.scatter(*artifact, c=COLORS["artifact"], s=48)
    ax.scatter(*uplink, c=COLORS["uplink"], s=54)
    ax.scatter(*extract, c=COLORS["extract"], edgecolors="black", s=60)
    add_projection_halo(ax, artifact, env.artifact_radius, COLORS["artifact"])
    add_projection_halo(ax, uplink, env.uplink_radius, COLORS["uplink"])
    add_projection_halo(ax, extract, env.extract_radius, COLORS["extract"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.15, color=COLORS["grid"])
    ax.set_aspect("equal")


def smooth_trajectory(traj, factor=4):
    if len(traj) < 2:
        return traj.copy()
    pieces = []
    ease = np.linspace(0.0, 1.0, factor, endpoint=False)
    ease = ease * ease * (3.0 - 2.0 * ease)
    for idx in range(len(traj) - 1):
        start = traj[idx]
        end = traj[idx + 1]
        segment = start[None, :] * (1.0 - ease[:, None]) + end[None, :] * ease[:, None]
        pieces.append(segment)
    pieces.append(traj[-1][None, :])
    return np.concatenate(pieces, axis=0)


def choose_examples(results, count=6):
    order = np.argsort(results["effective_steps"])
    indices = np.linspace(0, len(order) - 1, count, dtype=int)
    return [results["trajectories"][order[idx]] for idx in indices]


def make_comparison_plot(plain_results, recap_results):
    fig = plt.figure(figsize=(18, 11), facecolor="white")
    ax_plain_3d = fig.add_subplot(2, 3, 1, projection="3d")
    ax_recap_3d = fig.add_subplot(2, 3, 2, projection="3d")
    ax_bar = fig.add_subplot(2, 3, 3)
    ax_xy = fig.add_subplot(2, 3, 4)
    ax_xz = fig.add_subplot(2, 3, 5)
    ax_yz = fig.add_subplot(2, 3, 6)

    style_3d_axis(ax_plain_3d, "Plain Imitation")
    style_3d_axis(ax_recap_3d, "RECAP Conditioned")

    for traj in plain_results["trajectories"][:14]:
        smooth = smooth_trajectory(traj)
        ax_plain_3d.plot(smooth[:, 0], smooth[:, 1], smooth[:, 2], color=COLORS["plain"], alpha=0.24, lw=1.8)
    for traj in recap_results["trajectories"][:14]:
        smooth = smooth_trajectory(traj)
        ax_recap_3d.plot(smooth[:, 0], smooth[:, 1], smooth[:, 2], color=COLORS["recap"], alpha=0.34, lw=2.0)

    metrics = ["Eff. steps", "Success %", "Artifact %", "Uplink %", "Path len"]
    plain_vals = [
        plain_results["effective_steps"].mean(),
        plain_results["success_rate"] * 100,
        plain_results["artifact_rate"] * 100,
        plain_results["uplink_rate"] * 100,
        plain_results["path_length"],
    ]
    recap_vals = [
        recap_results["effective_steps"].mean(),
        recap_results["success_rate"] * 100,
        recap_results["artifact_rate"] * 100,
        recap_results["uplink_rate"] * 100,
        recap_results["path_length"],
    ]
    x = np.arange(len(metrics))
    width = 0.34
    ax_bar.bar(x - width / 2, plain_vals, width, label="Plain", color=COLORS["plain"], alpha=0.90)
    ax_bar.bar(x + width / 2, recap_vals, width, label="RECAP", color=COLORS["recap"], alpha=0.90)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics, rotation=16)
    ax_bar.set_title("Episode Metrics", fontsize=13)
    ax_bar.grid(axis="y", alpha=0.12)
    ax_bar.legend(frameon=False)

    for plane, ax, title in [("xy", ax_xy, "Top View"), ("xz", ax_xz, "Side View x-z"), ("yz", ax_yz, "Side View y-z")]:
        style_projection(ax, plane, title)
        for traj in plain_results["trajectories"][:10]:
            smooth = smooth_trajectory(traj)
            if plane == "xy":
                ax.plot(smooth[:, 0], smooth[:, 1], color=COLORS["plain"], alpha=0.22, lw=1.5)
            elif plane == "xz":
                ax.plot(smooth[:, 0], smooth[:, 2], color=COLORS["plain"], alpha=0.22, lw=1.5)
            else:
                ax.plot(smooth[:, 1], smooth[:, 2], color=COLORS["plain"], alpha=0.22, lw=1.5)
        for traj in recap_results["trajectories"][:10]:
            smooth = smooth_trajectory(traj)
            if plane == "xy":
                ax.plot(smooth[:, 0], smooth[:, 1], color=COLORS["recap"], alpha=0.30, lw=1.8)
            elif plane == "xz":
                ax.plot(smooth[:, 0], smooth[:, 2], color=COLORS["recap"], alpha=0.30, lw=1.8)
            else:
                ax.plot(smooth[:, 1], smooth[:, 2], color=COLORS["recap"], alpha=0.30, lw=1.8)

    fig.suptitle("3D Drone Heist: Artifact -> Uplink -> Extraction", fontsize=18, y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "game_3d_comparison.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_rollout_gif(results, path, title, color):
    examples = [smooth_trajectory(traj, factor=4) for traj in choose_examples(results, count=6)]
    max_len = max(len(traj) for traj in examples)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8), facecolor="white")
    plane_defs = [("xy", axes[0], "Top"), ("xz", axes[1], "x-z"), ("yz", axes[2], "y-z")]
    for plane, ax, plane_title in plane_defs:
        style_projection(ax, plane, plane_title)

    counter = fig.text(0.50, 0.95, "", ha="center", fontsize=12)
    fig.suptitle(title, y=1.01, fontsize=16)

    lines = []
    dots = []
    for _, ax, _ in plane_defs:
        plane_lines = []
        plane_dots = []
        for idx in range(len(examples)):
            alpha = 0.30 + 0.10 * (idx / len(examples))
            plane_lines.append(ax.plot([], [], color=color, lw=2.1, alpha=alpha, solid_capstyle="round")[0])
            plane_dots.append(ax.plot([], [], "o", color=color, markersize=4.6)[0])
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
        counter.set_text(f"t = {frame / 4.0:05.1f}")
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

    anim = FuncAnimation(fig, update, frames=max_len, init_func=init, interval=55, blit=True)
    anim.save(path, writer=PillowWriter(fps=18))
    plt.close(fig)


def make_rollout_3d_gif(results, path, title, color):
    examples = [smooth_trajectory(traj, factor=4) for traj in choose_examples(results, count=5)]
    max_len = max(len(traj) for traj in examples)

    fig = plt.figure(figsize=(7.8, 7.1), facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    style_3d_axis(ax, title)

    lines = [
        ax.plot([], [], [], color=color, lw=2.4, alpha=0.34 + 0.10 * (idx / len(examples)), solid_capstyle="round")[0]
        for idx in range(len(examples))
    ]
    dots = [ax.plot([], [], [], "o", color=color, markersize=5.2)[0] for _ in examples]
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
        counter.set_text(f"t = {frame / 4.0:05.1f}")
        ax.view_init(elev=24 + 2.5 * np.sin(frame * 0.035), azim=-56 + 0.11 * frame)
        artists = [counter]
        for line, dot, traj in zip(lines, dots, examples):
            idx = min(frame, len(traj) - 1)
            line.set_data(traj[: idx + 1, 0], traj[: idx + 1, 1])
            line.set_3d_properties(traj[: idx + 1, 2])
            dot.set_data([traj[idx, 0]], [traj[idx, 1]])
            dot.set_3d_properties([traj[idx, 2]])
            artists.extend([line, dot])
        return artists

    anim = FuncAnimation(fig, update, frames=max_len, init_func=init, interval=60, blit=False)
    anim.save(path, writer=PillowWriter(fps=16))
    plt.close(fig)


def main():
    print(f"Using device: {DEVICE}")
    print("Generating 3D mixed-quality drone-heist demonstrations...")
    states, actions, conds, returns, demos = build_dataset()
    demo_success = np.mean([demo.success for demo in demos])
    demo_artifact = np.mean([demo.got_artifact for demo in demos])
    demo_uplink = np.mean([demo.got_uplink for demo in demos])
    print(
        f"dataset demos: {len(demos)} | transitions: {len(states)} | "
        f"artifact grabs: {demo_artifact * 100:.1f}% | uplinks: {demo_uplink * 100:.1f}% | "
        f"successes: {demo_success * 100:.1f}%"
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
        f"Artifact: {plain_results['artifact_rate'] * 100:.0f}% | "
        f"Uplink: {plain_results['uplink_rate'] * 100:.0f}% | "
        f"Collisions: {plain_results['collision_rate'] * 100:.0f}% | "
        f"Raw term steps: {plain_results['raw_steps'].mean():.0f}"
    )
    print(
        f"RECAP Advantage  -> Avg steps: {recap_steps:.0f} | "
        f"Success: {recap_results['success_rate'] * 100:.0f}% | "
        f"Artifact: {recap_results['artifact_rate'] * 100:.0f}% | "
        f"Uplink: {recap_results['uplink_rate'] * 100:.0f}% | "
        f"Collisions: {recap_results['collision_rate'] * 100:.0f}% | "
        f"Raw term steps: {recap_results['raw_steps'].mean():.0f}"
    )
    print(f"Duration improvement: {plain_steps / recap_steps:.2f}x faster")
    print(f"Success improvement: {recap_results['success_rate'] / max(plain_results['success_rate'], 1e-6):.2f}x higher")

    print("\nSaving visuals...")
    make_comparison_plot(plain_results, recap_results)
    make_rollout_gif(
        plain_results,
        os.path.join(OUTPUT_DIR, "game_3d_plain.gif"),
        "Plain Imitation: Drone Heist",
        COLORS["plain"],
    )
    make_rollout_gif(
        recap_results,
        os.path.join(OUTPUT_DIR, "game_3d_recap.gif"),
        "RECAP Conditioned: Drone Heist",
        COLORS["recap"],
    )
    make_rollout_3d_gif(
        plain_results,
        os.path.join(OUTPUT_DIR, "game_3d_plain_3d.gif"),
        "Plain Imitation 3D",
        COLORS["plain"],
    )
    make_rollout_3d_gif(
        recap_results,
        os.path.join(OUTPUT_DIR, "game_3d_recap_3d.gif"),
        "RECAP Conditioned 3D",
        COLORS["recap"],
    )
    print(f"Saved {os.path.join(OUTPUT_DIR, 'game_3d_comparison.png')}")
    print(f"Saved {os.path.join(OUTPUT_DIR, 'game_3d_plain.gif')}")
    print(f"Saved {os.path.join(OUTPUT_DIR, 'game_3d_recap.gif')}")
    print(f"Saved {os.path.join(OUTPUT_DIR, 'game_3d_plain_3d.gif')}")
    print(f"Saved {os.path.join(OUTPUT_DIR, 'game_3d_recap_3d.gif')}")


if __name__ == "__main__":
    main()
