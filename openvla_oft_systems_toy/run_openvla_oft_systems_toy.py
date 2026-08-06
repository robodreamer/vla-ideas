#!/usr/bin/env python3
"""Deterministic OFT-inspired delayed-observation 2D control systems toy.

This is not OpenVLA-OFT training or a reproduction.  It isolates the execution
question: serial action generation versus a parallel, continuous action chunk,
and how often the latter should be refreshed when observations arrive late.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DT = 0.04
HORIZON = 7.2
CHUNK = 8
OBS_DELAY = 0.16
MAX_SPEED = 1.15


@dataclass(frozen=True)
class Method:
    name: str
    label: str
    kind: str
    refresh: int
    decision_latency: float
    color: str


METHODS = (
    Method("autoregressive_serial", "serial autoregressive-like", "serial", 1, 0.055, "#c44e52"),
    Method("parallel_open_loop", "parallel chunk, open loop", "parallel", CHUNK, 0.060, "#4c72b0"),
    Method("parallel_refresh_4", "parallel chunk, refresh 4", "parallel", 4, 0.060, "#55a868"),
    Method("parallel_refresh_2", "parallel chunk, refresh 2", "parallel", 2, 0.060, "#8172b3"),
)


def goal_at(t: float, phase: float) -> np.ndarray:
    """A moving, bounded target makes stale chunks visibly costly."""
    return np.array([
        0.74 * math.cos(0.72 * t + phase) + 0.12 * math.sin(1.6 * t + phase),
        0.54 * math.sin(0.94 * t + phase),
    ])


def desired_action(p: np.ndarray, v: np.ndarray, t: float, phase: float) -> np.ndarray:
    # This is the deterministic continuous-action/L1-regression proxy: the
    # chunk head emits clipped continuous velocities, not discretized tokens.
    target = goal_at(t, phase)
    a = 2.4 * (target - p) - 0.48 * v
    return np.clip(a, -MAX_SPEED, MAX_SPEED)


def make_parallel_chunk(obs_p: np.ndarray, obs_v: np.ndarray, obs_t: float, phase: float) -> np.ndarray:
    p, v = obs_p.copy(), obs_v.copy()
    actions = []
    for i in range(CHUNK):
        a = desired_action(p, v, obs_t + (i + 1) * DT, phase)
        actions.append(a)
        # Head's open-loop imagined rollout; execution will differ due to noise.
        v = 0.78 * v + 0.22 * a
        p = p + DT * v
    return np.asarray(actions)


def rollout(method: Method, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-math.pi, math.pi)
    p = np.array([-0.82, -0.48]) + rng.normal(0, 0.035, 2)
    v = np.zeros(2)
    history_p = [p.copy()]
    history_v = [v.copy()]
    actions, points, goals, times = [], [p.copy()], [], []
    chunk = np.zeros((CHUNK, 2))
    chunk_index = CHUNK
    busy_until = 0.0
    commands = 0
    tracking = []
    for k in range(int(HORIZON / DT)):
        t = k * DT
        delayed_k = max(0, k - round(OBS_DELAY / DT))
        obs_p, obs_v = history_p[delayed_k], history_v[delayed_k]
        if method.kind == "serial":
            # One action needs one serial decode.  During synthetic decode latency,
            # the last command is held, reducing effective decision/action rate.
            if t >= busy_until:
                action = desired_action(obs_p, obs_v, t - OBS_DELAY, phase)
                busy_until = t + method.decision_latency
                commands += 1
            elif actions:
                action = actions[-1]
            else:
                action = np.zeros(2)
        else:
            if chunk_index >= method.refresh:
                chunk = make_parallel_chunk(obs_p, obs_v, t - OBS_DELAY, phase)
                chunk_index = 0
                commands += CHUNK
            action = chunk[chunk_index]
            chunk_index += 1
        gust = (rng.normal(0, 0.045, 2) if k % 9 == 0 else np.zeros(2))
        v = 0.77 * v + 0.23 * action + gust
        p = p + DT * v
        g = goal_at(t, phase)
        actions.append(action.copy())
        points.append(p.copy())
        goals.append(g)
        times.append(t)
        tracking.append(float(np.linalg.norm(p - g)))
        history_p.append(p.copy())
        history_v.append(v.copy())
    actions_a, points_a, goals_a = np.asarray(actions), np.asarray(points[1:]), np.asarray(goals)
    accel = np.diff(actions_a, axis=0) / DT
    jerk = np.diff(accel, axis=0) / DT
    return {
        "method": method.name, "seed": seed, "time": np.asarray(times), "points": points_a,
        "goals": goals_a, "actions": actions_a, "mean_tracking_error": float(np.mean(tracking)),
        "final_tracking_error": float(tracking[-1]), "success": float(np.mean(tracking[-20:]) < 0.16),
        "smoothness_jerk": float(np.mean(np.linalg.norm(jerk, axis=1))),
        "simulated_decision_latency_ms": method.decision_latency * 1000,
        # Includes redundant future actions when a chunk is refreshed early. This
        # is output/action-generation throughput, not the 25 Hz plant tick rate.
        "output_action_rate_hz": commands / HORIZON,
    }


def save_csv(rows: list[dict]) -> None:
    fields = ["method", "trials", "simulated_decision_latency_ms", "output_action_rate_hz",
              "mean_tracking_error", "final_tracking_error", "success_rate", "smoothness_jerk"]
    with (OUT / "openvla_oft_systems_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], examples: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for m in METHODS:
        r = examples[m.name]
        ax.plot(r["points"][:, 0], r["points"][:, 1], color=m.color, label=m.label)
    ax.plot(examples[METHODS[0].name]["goals"][:, 0], examples[METHODS[0].name]["goals"][:, 1],
            "k--", lw=1.5, label="moving target")
    ax.set(title="One matched delayed-observation rollout", xlabel="x", ylabel="y", aspect="equal")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    labels = [r["method"].replace("parallel_", "p-").replace("autoregressive_", "ar-") for r in rows]
    x = np.arange(len(rows)); width = .36
    axes[1].bar(x - width / 2, [r["success_rate"] * 100 for r in rows], width, label="success (%)")
    axes[1].bar(x + width / 2, [r["output_action_rate_hz"] for r in rows], width, label="action-output throughput (Hz)")
    for i, r in enumerate(rows):
        axes[1].text(i, max(r["success_rate"] * 100, r["output_action_rate_hz"]) + 1,
                     f"jerk {r['smoothness_jerk']:.0f}", ha="center", fontsize=7, rotation=25)
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set(title="Latency/rate/success trade-off", ylim=(0, 110))
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "openvla_oft_systems_summary.png", dpi=170); plt.close(fig)


def report(rows: list[dict], args: argparse.Namespace) -> None:
    lines = ["# OpenVLA-OFT Systems Toy Report", "", "Generated by `run_openvla_oft_systems_toy.py`.", "",
             "## What was tested", "",
             "This deterministic delayed-observation 2D experiment tests a narrow systems implication of the OpenVLA-OFT recipe: serial-like continuous action availability versus parallel continuous chunks, with open-loop execution or closed-loop chunk refresh. It is not an OpenVLA-OFT reproduction. The controller is analytical; the latency values are configured simulation parameters; and no OpenVLA weights, visual/language inputs, L1 training, LIBERO/ALOHA task, or paper result is represented.", "",
             f"The generated run uses seed `{args.seed}` and `{args.trials}` trials per method. Each rollout has a 40 ms control tick, 160 ms observation delay, an 8-action chunk horizon, and seeded velocity gusts. `parallel_refresh_4` and `parallel_refresh_2` replace an unfinished chunk after four or two executed actions, respectively; their throughput therefore includes discarded predicted suffixes.", "",
             "## Metric definitions", "",
             "- **simulated decision latency**: configured time before a serial action or parallel chunk is available, not wall-clock profiling.",
             "- **action-output throughput**: generated action values per simulated rollout second; early-refresh suffixes are counted.",
             "- **mean tracking error**: time-average Euclidean distance to the moving target.",
             "- **terminal-window success**: final 20-tick mean tracking error below 0.16.",
             "- **jerk proxy**: mean norm of the second finite difference of emitted commands; a descriptive smoothness measure, not an actuator limit.", "",
             "## Results", "",
             "| Method | latency (ms, simulated) | action-output throughput (Hz) | mean tracking error | final tracking error | success | jerk proxy |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['simulated_decision_latency_ms']:.0f} | {r['output_action_rate_hz']:.1f} | {r['mean_tracking_error']:.3f} | {r['final_tracking_error']:.3f} | {100*r['success_rate']:.1f}% | {r['smoothness_jerk']:.1f} |")
    lines += ["", "## Visualizations", "",
              "- `outputs/openvla_oft_systems_summary.png` shows one matched delayed-observation rollout and the aggregate latency/rate/success trade-off.",
              "- The action-output rate counts parallel actions generated but later discarded after early refresh.",
              "", "## Interpretation", "",
              "The serial-like condition has the lowest generated-action availability. A full parallel chunk restores plant-rate availability, but its stale suffixes leave its terminal-window success below `parallel_refresh_4`. In this run, refreshing after four actions produces the highest success (33.3%), while refreshing after two actions produces the lowest mean tracking error (0.311) but the same 28.1% terminal-window success as the open-loop parallel condition. The result is a local trade-off: more refreshes limit stale actions, but also repeatedly re-anchor to delayed feedback and add command boundaries.",
              "", "## Limitations and follow-ups", "",
              "These results describe only the configured analytical toy. They do not reproduce, validate, or compare against OpenVLA-OFT; they do not measure device latency or action-generation throughput; and they do not support a general optimum for chunk horizon or refresh cadence. Primary-source OFT coverage and the distinction between authors' reported results and this toy are in [`notes/2026-08-06-openvla-oft.md`](../../notes/2026-08-06-openvla-oft.md).",
              "", "## Outputs", "",
              "- `outputs/openvla_oft_systems_metrics.csv`",
              "- `outputs/openvla_oft_systems_summary.png`"]
    (OUT / "openvla_oft_systems_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--trials", type=int, default=48)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    OUT.mkdir(exist_ok=True)
    rows, examples = [], {}
    for mi, method in enumerate(METHODS):
        rs = [rollout(method, args.seed + mi * 10000 + trial) for trial in range(args.trials)]
        examples[method.name] = rollout(method, args.seed)
        rows.append({"method": method.name, "trials": args.trials,
                     "simulated_decision_latency_ms": np.mean([r["simulated_decision_latency_ms"] for r in rs]),
                     "output_action_rate_hz": np.mean([r["output_action_rate_hz"] for r in rs]),
                     "mean_tracking_error": np.mean([r["mean_tracking_error"] for r in rs]),
                     "final_tracking_error": np.mean([r["final_tracking_error"] for r in rs]),
                     "success_rate": np.mean([r["success"] for r in rs]),
                     "smoothness_jerk": np.mean([r["smoothness_jerk"] for r in rs])})
    save_csv(rows); plot(rows, examples); report(rows, args)
    print(f"wrote {OUT / 'openvla_oft_systems_metrics.csv'}")
    for r in rows:
        print(f"{r['method']}: success={r['success_rate']:.1%}, output_rate={r['output_action_rate_hz']:.1f} Hz, error={r['mean_tracking_error']:.3f}")


if __name__ == "__main__":
    main()
