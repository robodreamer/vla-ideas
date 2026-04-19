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


def train_models(states, actions, returns, adv_labels, styles, epochs=120, batch_size=1024):
    states = states.to(DEVICE)
    actions = actions.to(DEVICE)
    adv_labels = adv_labels.to(DEVICE)
    styles = styles.to(DEVICE)

    plain = LatentPolicy(conditioned=False).to(DEVICE)
    recap = LatentPolicy(conditioned=True).to(DEVICE)

    popt = torch.optim.Adam(plain.parameters(), lr=1e-3)
    ropt = torch.optim.Adam(recap.parameters(), lr=1e-3)

    num = states.shape[0]
    for epoch in range(epochs):
        order = torch.randperm(num)
        for start in range(0, num, batch_size):
            idx = order[start : start + batch_size]
            s = states[idx]
            a = actions[idx]
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

        if epoch in {0, 39, 79, epochs - 1}:
            print(f"epoch {epoch + 1:03d}/{epochs} | plain {ploss.item():.4f} | recap-latent {rloss.item():.4f}")
    return plain, recap


def local_risk(state, action):
    sim = state.copy()
    sim[3:6] = np.clip(sim[3:6] + np.clip(action, -1.0, 1.0) * env.dt, -env.max_vel, env.max_vel)
    sim[:3] = sim[:3] + sim[3:6] * env.dt
    if env.inside_obstacle(sim[:3]) or env.out_of_bounds(sim[:3]):
        return True
    if np.linalg.norm(sim[3:6]) > 0.78:
        return True
    return False


def simulate_from_state(policy, init_state, switch_style=False, max_steps=180):
    state = init_state.copy()
    style = 0.0
    recovery_timer = 18 if switch_style else 0
    done = False
    collided = False
    steps = 0
    for _ in range(max_steps):
        st = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            base_action = policy(
                st,
                torch.ones(1, device=DEVICE),
                torch.tensor([style], dtype=torch.float32, device=DEVICE),
            )[0].detach().cpu().numpy()
        if switch_style and recovery_timer > 0 and local_risk(state, base_action):
            style = 1.0
            recovery_timer -= 1
            with torch.no_grad():
                action = policy(
                    st,
                    torch.ones(1, device=DEVICE),
                    torch.tensor([style], dtype=torch.float32, device=DEVICE),
                )[0].detach().cpu().numpy()
        else:
            action = base_action
        state, done, collided = env.step(action)
        steps += 1
        if done or collided:
            break
    return eval_score(state, done and not collided, collided, steps), bool(done and not collided)


def build_gate_dataset(policy, demos):
    rng = np.random.default_rng(SEED + 123)
    disturbed_states = []
    labels = []
    successful = [demo for demo in demos if demo.success]
    for demo in successful[:160]:
        for state in demo.states[::5]:
            disturbed = state.copy()
            disturbed[3:6] = np.clip(disturbed[3:6] + rng.normal(0.0, 0.24, size=3), -env.max_vel, env.max_vel)
            disturbed[:3] += rng.normal(0.0, 0.018, size=3)
            if env.inside_obstacle(disturbed[:3]) or env.out_of_bounds(disturbed[:3]):
                continue
            base_score, base_success = simulate_from_state(policy, disturbed, switch_style=False)
            adapt_score, adapt_success = simulate_from_state(policy, disturbed, switch_style=True)
            useful = (adapt_success and not base_success) or (adapt_score > base_score + 0.20)
            disturbed_states.append(disturbed.astype(np.float32))
            labels.append(1.0 if useful else 0.0)
    return torch.tensor(np.array(disturbed_states), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.float32)


def train_gate(states, labels, epochs=80, batch_size=256):
    states = states.to(DEVICE)
    labels = labels.to(DEVICE)
    gate = GateNet().to(DEVICE)
    opt = torch.optim.AdamW(gate.parameters(), lr=2e-3, weight_decay=1e-4)
    pos = labels.sum().clamp(min=1.0)
    pos_weight = (len(labels) - pos) / pos
    num = states.shape[0]
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
    rng = np.random.default_rng(SEED + 200)
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
        style = 0.0
        recovery_timer = 0

        for t in range(max_steps):
            if t == spec["step"]:
                state = env.apply_velocity_kick(spec["delta_v"])
                traj.append(state.copy())
                recovery_timer = 18

            st = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                if mode == "plain":
                    action = policy(st)[0].detach().cpu().numpy()
                else:
                    if mode == "latent" and recovery_timer > 0:
                        gate_on = True
                        if gate is not None:
                            gate_on = torch.sigmoid(gate(st))[0].item() > 0.55
                        preview = policy(
                            st,
                            torch.ones(1, device=DEVICE),
                            torch.tensor([style], dtype=torch.float32, device=DEVICE),
                        )[0].detach().cpu().numpy()
                        if gate_on and local_risk(state, preview):
                            style = 1.0
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


def make_comparison_plot(plain_results, recap_results, latent_results):
    fig = plt.figure(figsize=(17, 6))
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    configs = [
        ("Plain BC", plain_results, "#c44e52"),
        ("RECAP Fixed Style", recap_results, "#dd8452"),
        ("RECAP + Latent Adapt", latent_results, "#55a868"),
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
    fig.suptitle("Disturbed 3D Drone Heist: RECAP vs Latent Strategy Adaptation", fontsize=16)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "latent_adapt_3d_comparison.png")
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
    print("Generating latent-adaptation dataset...")
    states, actions, returns, adv_labels, styles, demos = build_dataset()
    print(f"demos: {len(demos)} | transitions: {len(states)} | dataset success: {np.mean([d.success for d in demos]) * 100:.1f}%")

    print("\nTraining baseline and latent-conditioned RECAP policies...")
    plain_policy, latent_policy = train_models(states, actions, returns, adv_labels, styles)

    print("\nBuilding gate labels and training latent adaptation gate...")
    gate_states, gate_labels = build_gate_dataset(latent_policy, demos)
    print(f"gate samples: {len(gate_states)} | positive style-switch labels: {gate_labels.mean().item() * 100:.1f}%")
    gate = train_gate(gate_states, gate_labels)

    print("\nEvaluating disturbed policies...")
    specs = make_disturbance_specs(16)
    plain_results = rollout(plain_policy, "plain", specs)
    recap_results = rollout(latent_policy, "recap", specs)
    latent_results = rollout(latent_policy, "latent", specs, gate=gate)

    delta_success = (latent_results["success_rate"] - recap_results["success_rate"]) * 100
    delta_steps = recap_results["effective_steps"].mean() - latent_results["effective_steps"].mean()
    delta_collision = (recap_results["collision_rate"] - latent_results["collision_rate"]) * 100
    delta_artifact = (latent_results["artifact_rate"] - recap_results["artifact_rate"]) * 100
    delta_uplink = (latent_results["uplink_rate"] - recap_results["uplink_rate"]) * 100

    print("\n=== LATENT ADAPTATION RESULTS ===")
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
        f"RECAP + latent adapt    -> Avg steps: {latent_results['effective_steps'].mean():.0f} | "
        f"Success: {latent_results['success_rate'] * 100:.0f}% | Artifact: {latent_results['artifact_rate'] * 100:.0f}% | "
        f"Uplink: {latent_results['uplink_rate'] * 100:.0f}% | Collisions: {latent_results['collision_rate'] * 100:.0f}%"
    )
    print(
        f"Improvement over RECAP  -> Success {delta_success:+.1f} pts | Artifact {delta_artifact:+.1f} pts | "
        f"Uplink {delta_uplink:+.1f} pts | Effective steps {delta_steps:+.1f} | Collision {delta_collision:+.1f} pts"
    )

    png_path = make_comparison_plot(plain_results, recap_results, latent_results)
    make_3d_gif(plain_results, os.path.join(OUTPUT_DIR, "latent_plain_3d.gif"), "Plain BC 3D", "#c44e52")
    make_3d_gif(recap_results, os.path.join(OUTPUT_DIR, "latent_recap_3d.gif"), "RECAP Fixed Style 3D", "#dd8452")
    make_3d_gif(latent_results, os.path.join(OUTPUT_DIR, "latent_adapt_3d.gif"), "RECAP + Latent Adapt 3D", "#55a868")
    print(f"Saved comparison PNG -> {png_path}")
    print("Saved GIFs -> latent_plain_3d.gif, latent_recap_3d.gif, latent_adapt_3d.gif")


if __name__ == "__main__":
    main()
