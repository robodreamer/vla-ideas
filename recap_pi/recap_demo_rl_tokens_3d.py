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

SEED = 23
torch.manual_seed(SEED)
np.random.seed(SEED)


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


def pd_action(state, target, kp=5.2, kd=2.7):
    pos_err = target - state[:3]
    vel_err = -state[3:6]
    return np.clip(kp * pos_err + kd * vel_err, -1.0, 1.0)


def staged_target(state):
    if state[6] < 0.5:
        return env.artifact
    if state[7] < 0.5:
        return env.uplink
    return env.extract


def recovery_target(state):
    target = staged_target(state)
    if state[6] > 0.5 and state[7] < 0.5:
        mid = np.array([0.66, 0.92, 0.96], dtype=np.float32)
        if np.linalg.norm(state[:3] - mid) > 0.18:
            target = mid
    if state[7] > 0.5:
        mid = np.array([0.92, 0.46, 0.98], dtype=np.float32)
        if np.linalg.norm(state[:3] - mid) > 0.16:
            target = mid
    return pd_action(state, target, kp=5.4, kd=2.8)


def eval_score(state, success, collided, steps):
    artifact = float(state[6] > 0.5)
    uplink = float(state[7] > 0.5)
    if success:
        return 4.0 - 0.007 * steps
    if collided:
        return -2.6 + 0.55 * artifact + 0.65 * uplink - 0.002 * steps
    return -1.2 + 0.8 * artifact + 0.9 * uplink - 0.45 * np.linalg.norm(state[:3] - env.extract)


def run_demo(waypoints=None, kind="expert", max_steps=280):
    state = env.reset()
    states = []
    actions = []
    done = False
    collided = False
    waypoint_idx = 0

    for t in range(max_steps):
        if kind == "expert":
            target = np.array(waypoints[min(waypoint_idx, len(waypoints) - 1)], dtype=np.float32)
            if np.linalg.norm(state[:3] - target) < 0.12 and waypoint_idx < len(waypoints) - 1:
                waypoint_idx += 1
                target = np.array(waypoints[waypoint_idx], dtype=np.float32)
            action = pd_action(state, target, kp=5.8, kd=3.2) + np.random.normal(0.0, 0.05, size=3)
        elif kind == "safe":
            target = np.array(waypoints[min(waypoint_idx, len(waypoints) - 1)], dtype=np.float32)
            if np.linalg.norm(state[:3] - target) < 0.14 and waypoint_idx < len(waypoints) - 1:
                waypoint_idx += 1
                target = np.array(waypoints[waypoint_idx], dtype=np.float32)
            action = pd_action(state, target, kp=4.9, kd=2.5) + np.random.normal(0.0, 0.08, size=3)
        elif kind == "rush_extract":
            action = pd_action(state, env.extract, kp=3.4, kd=1.0) + np.array([0.05, -0.02, 0.05])
        elif kind == "artifact_then_crash":
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

    return Demo(np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32), eval_score(state, done and not collided, collided, len(states)), bool(done and not collided))


def build_dataset():
    demos = []
    good_route = [
        (0.10, 0.18, 0.12),
        (0.16, 0.82, 0.18),
        (0.28, 0.96, 0.36),
        (0.70, 0.88, 0.94),
        (0.90, 0.54, 1.00),
        (0.98, 0.22, 0.90),
    ]
    safe_route = [
        (0.08, 0.22, 0.10),
        (0.18, 0.86, 0.20),
        (0.26, 0.98, 0.42),
        (0.64, 0.98, 0.98),
        (0.90, 0.46, 1.02),
        (0.98, 0.22, 0.90),
    ]

    for _ in range(180):
        demos.append(run_demo(good_route, "expert"))
    for _ in range(60):
        demos.append(run_demo(safe_route, "safe"))
    for _ in range(220):
        demos.append(run_demo(kind="rush_extract"))
    for _ in range(180):
        demos.append(run_demo(kind="artifact_then_crash"))
    for _ in range(160):
        demos.append(run_demo(kind="orbit"))

    scores = np.array([demo.score for demo in demos], dtype=np.float32)
    threshold = np.quantile(scores, 0.80)

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

    return (
        torch.tensor(np.array(states), dtype=torch.float32),
        torch.tensor(np.array(actions), dtype=torch.float32),
        torch.tensor(np.array(returns), dtype=torch.float32).unsqueeze(1),
        torch.tensor(np.array(adv_labels), dtype=torch.float32),
        demos,
    )


class Policy(nn.Module):
    def __init__(self, cond):
        super().__init__()
        self.cond = cond
        self.net = nn.Sequential(
            nn.Linear(9 if cond else 8, 160),
            nn.ReLU(),
            nn.Linear(160, 160),
            nn.ReLU(),
            nn.Linear(160, 96),
            nn.ReLU(),
            nn.Linear(96, 3),
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
            nn.Linear(8, 160),
            nn.ReLU(),
            nn.Linear(160, 160),
            nn.ReLU(),
            nn.Linear(160, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )

    def forward(self, state):
        return self.net(state)


class GateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 96),
            nn.ReLU(),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(1)


def train_models(states, actions, returns, adv_labels, epochs=120, batch_size=1024):
    plain = Policy(False)
    recap = Policy(True)
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

        if epoch in {0, 39, 79, epochs - 1}:
            print(f"epoch {epoch + 1:03d}/{epochs} | value {vloss.item():.4f} | plain {ploss.item():.4f} | recap {rloss.item():.4f}")
    return plain, recap, value


def local_risk(state, action):
    sim = state.copy()
    sim[3:6] = np.clip(sim[3:6] + np.clip(action, -1.0, 1.0) * env.dt, -env.max_vel, env.max_vel)
    sim[:3] = sim[:3] + sim[3:6] * env.dt
    if env.inside_obstacle(sim[:3]) or env.out_of_bounds(sim[:3]):
        return True
    if np.linalg.norm(sim[3:6]) > 0.78:
        return True
    return False


def simulate_from_state(recap_policy, init_state, intervene=False, max_steps=180):
    state = init_state.copy()
    recovery_timer = 22 if intervene else 0
    done = False
    collided = False
    steps = 0

    for _ in range(max_steps):
        st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            base_action = recap_policy(st, torch.ones(1))[0].numpy()
        if intervene and recovery_timer > 0 and local_risk(state, base_action):
            corrective = recovery_target(state)
            action = np.clip(0.20 * base_action + 0.80 * corrective, -1.0, 1.0)
            recovery_timer -= 1
        else:
            action = base_action
        state, done, collided = env.step(action)
        steps += 1
        if done or collided:
            break
    return eval_score(state, done and not collided, collided, steps), bool(done and not collided)


def build_intervention_dataset(recap_policy, demos):
    rng = np.random.default_rng(SEED + 101)
    disturbed_states = []
    labels = []

    successful = [demo for demo in demos if demo.success]
    for demo in successful[:140]:
        for state in demo.states[::5]:
            disturbed = state.copy()
            disturbed[3:6] = np.clip(disturbed[3:6] + rng.normal(0.0, 0.24, size=3), -env.max_vel, env.max_vel)
            disturbed[:3] += rng.normal(0.0, 0.02, size=3)
            if env.inside_obstacle(disturbed[:3]) or env.out_of_bounds(disturbed[:3]):
                continue

            base_score, base_success = simulate_from_state(recap_policy, disturbed, intervene=False)
            oracle_score, oracle_success = simulate_from_state(recap_policy, disturbed, intervene=True)
            useful = (oracle_success and not base_success) or (oracle_score > base_score + 0.18)
            disturbed_states.append(disturbed.astype(np.float32))
            labels.append(1.0 if useful else 0.0)

    return torch.tensor(np.array(disturbed_states), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.float32)


def train_gate(states, labels, epochs=80, batch_size=256):
    gate = GateNet()
    opt = torch.optim.AdamW(gate.parameters(), lr=2e-3, weight_decay=1e-4)
    num = states.shape[0]
    pos_weight = (len(labels) - labels.sum()) / labels.sum().clamp(min=1.0)

    for _ in range(epochs):
        order = torch.randperm(num)
        for start in range(0, num, batch_size):
            idx = order[start : start + batch_size]
            logits = gate(states[idx])
            loss = F.binary_cross_entropy_with_logits(logits, labels[idx], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return gate


def make_disturbance_specs(num_episodes):
    rng = np.random.default_rng(SEED + 100)
    return [{"step": int(rng.integers(22, 58)), "delta_v": rng.normal(0.0, 0.26, size=3).astype(np.float32)} for _ in range(num_episodes)]


def rollout(policy, mode, disturbance_specs, gate=None, max_steps=300):
    trajectories = []
    effective_steps = []
    success = []
    collisions = []
    artifact = []
    uplink = []

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
                recovery_timer = 22

            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if mode == "plain":
                    base_action = policy(st)[0].numpy()
                else:
                    base_action = policy(st, torch.ones(1))[0].numpy()

            if mode == "online" and recovery_timer > 0:
                gate_on = True
                if gate is not None:
                    with torch.no_grad():
                        gate_on = torch.sigmoid(gate(st))[0].item() > 0.55
                if gate_on and local_risk(state, base_action):
                    corrective = recovery_target(state)
                    action = np.clip(0.20 * base_action + 0.80 * corrective, -1.0, 1.0)
                    recovery_timer -= 1
                else:
                    action = base_action
            else:
                action = base_action

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

    return {
        "trajectories": trajectories,
        "effective_steps": np.array(effective_steps),
        "success_rate": float(np.mean(success)),
        "collision_rate": float(np.mean(collisions)),
        "artifact_rate": float(np.mean(artifact)),
        "uplink_rate": float(np.mean(uplink)),
    }


def add_box(ax, box, color="#c64b59", alpha=0.18):
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


def make_comparison_plot(plain_results, recap_results, online_results):
    fig = plt.figure(figsize=(17, 6))
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    configs = [
        ("Plain BC", plain_results, "#c44e52"),
        ("RECAP Offline", recap_results, "#dd8452"),
        ("RECAP + Recovery Editor", online_results, "#55a868"),
    ]
    for ax, (title, results, color) in zip(axes, configs):
        style_3d(ax, title)
        for traj in results["trajectories"][:10]:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, alpha=0.35, lw=1.7)
        ax.text2D(
            0.03,
            0.97,
            f"succ {results['success_rate'] * 100:.0f}%\nart {results['artifact_rate'] * 100:.0f}%\nuplink {results['uplink_rate'] * 100:.0f}%",
            transform=ax.transAxes,
            va="top",
        )
    fig.suptitle("Disturbed 3D Drone Heist: Plain vs RECAP vs RECAP + Online Recovery", fontsize=16)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "upgraded_3d_comparison.png")
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
    print("Generating 3D dataset...")
    states, actions, returns, adv_labels, demos = build_dataset()
    print(f"demos: {len(demos)} | transitions: {len(states)} | dataset success: {np.mean([d.success for d in demos]) * 100:.1f}%")

    print("\nTraining models...")
    plain_policy, recap_policy, _value_net = train_models(states, actions, returns, adv_labels)

    print("\nBuilding intervention labels and training gate...")
    gate_states, gate_labels = build_intervention_dataset(recap_policy, demos)
    print(f"gate samples: {len(gate_states)} | positive intervene labels: {gate_labels.mean().item() * 100:.1f}%")
    gate = train_gate(gate_states, gate_labels)

    print("\nEvaluating disturbed 3D policies...")
    specs = make_disturbance_specs(16)
    plain_results = rollout(plain_policy, "plain", specs)
    recap_results = rollout(recap_policy, "recap", specs)
    online_results = rollout(recap_policy, "online", specs, gate=gate)

    delta_success = (online_results["success_rate"] - recap_results["success_rate"]) * 100
    delta_steps = recap_results["effective_steps"].mean() - online_results["effective_steps"].mean()
    delta_collision = (recap_results["collision_rate"] - online_results["collision_rate"]) * 100
    delta_artifact = (online_results["artifact_rate"] - recap_results["artifact_rate"]) * 100
    delta_uplink = (online_results["uplink_rate"] - recap_results["uplink_rate"]) * 100

    print("\n=== 3D RL TOKENS RESULTS ===")
    print(
        f"Plain BC                -> Avg steps: {plain_results['effective_steps'].mean():.0f} | "
        f"Success: {plain_results['success_rate'] * 100:.0f}% | Artifact: {plain_results['artifact_rate'] * 100:.0f}% | "
        f"Uplink: {plain_results['uplink_rate'] * 100:.0f}% | Collisions: {plain_results['collision_rate'] * 100:.0f}%"
    )
    print(
        f"RECAP (offline)         -> Avg steps: {recap_results['effective_steps'].mean():.0f} | "
        f"Success: {recap_results['success_rate'] * 100:.0f}% | Artifact: {recap_results['artifact_rate'] * 100:.0f}% | "
        f"Uplink: {recap_results['uplink_rate'] * 100:.0f}% | Collisions: {recap_results['collision_rate'] * 100:.0f}%"
    )
    print(
        f"RECAP + RL Tokens       -> Avg steps: {online_results['effective_steps'].mean():.0f} | "
        f"Success: {online_results['success_rate'] * 100:.0f}% | Artifact: {online_results['artifact_rate'] * 100:.0f}% | "
        f"Uplink: {online_results['uplink_rate'] * 100:.0f}% | Collisions: {online_results['collision_rate'] * 100:.0f}%"
    )
    print(
        f"Improvement over RECAP  -> Success {delta_success:+.1f} pts | Artifact {delta_artifact:+.1f} pts | "
        f"Uplink {delta_uplink:+.1f} pts | Effective steps {delta_steps:+.1f} | Collision {delta_collision:+.1f} pts"
    )

    png_path = make_comparison_plot(plain_results, recap_results, online_results)
    make_3d_gif(plain_results, os.path.join(OUTPUT_DIR, "upgraded_3d_plain.gif"), "Plain BC 3D", "#c44e52")
    make_3d_gif(recap_results, os.path.join(OUTPUT_DIR, "upgraded_3d_recap.gif"), "RECAP Offline 3D", "#dd8452")
    make_3d_gif(online_results, os.path.join(OUTPUT_DIR, "upgraded_3d_rl_tokens.gif"), "RECAP + Recovery Editor 3D", "#55a868")
    print(f"Saved comparison PNG -> {png_path}")
    print("Saved GIFs -> upgraded_3d_plain.gif, upgraded_3d_recap.gif, upgraded_3d_rl_tokens.gif")


if __name__ == "__main__":
    main()
