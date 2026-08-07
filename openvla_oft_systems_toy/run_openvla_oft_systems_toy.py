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
    by_name = {row["method"]: row for row in rows}
    refresh4 = by_name["parallel_refresh_4"]
    refresh2 = by_name["parallel_refresh_2"]
    open_loop = by_name["parallel_open_loop"]
    serial = by_name["autoregressive_serial"]
    lines = [
        "# OpenVLA-OFT Systems Toy Report", "",
        "Generated by `run_openvla_oft_systems_toy.py`; the values below are generated from the accompanying CSV, not hand-maintained.", "",
        "## What was tested", "",
        "This deterministic 2D point-mass experiment tests one deployment-side implication of the OpenVLA-OFT action interface: serial one-action availability versus parallel continuous action chunks, and the consequence of consuming or replacing an unfinished chunk under delayed feedback. It is not an OpenVLA-OFT reproduction. No OpenVLA weights, image/language inputs, LoRA, learned action head, L1 optimization, LIBERO/ALOHA task, robot dynamics, or paper result is represented locally.", "",
        f"Each of the `{args.trials}` trials per method uses seed family `{args.seed}`, a 40 ms plant tick (180 ticks / 7.2 s), a 160 ms state-observation delay, and an 8-action chunk. The controller emits clipped continuous velocities `clip(2.4 * (goal - position) - 0.48 * velocity, -1.15, 1.15)`. The plant applies `v_next = 0.77 v + 0.23 u + gust`, then `p_next = p + 0.04 v_next`; a seeded per-axis Gaussian velocity gust (standard deviation 0.045) occurs every ninth tick. A parallel chunk rolls a separate imagined model forward from the delayed state, so its suffix diverges from the disturbed plant.", "",
        "The comparators are `autoregressive_serial` (one action at a time; the prior action is held during a 55 ms synthetic busy period), `parallel_open_loop` (execute all eight predicted actions), `parallel_refresh_4` (execute four and discard the suffix), and `parallel_refresh_2` (execute two and discard the suffix). The parallel rows record a 60 ms configured interface-latency annotation, but the current script makes a parallel chunk usable immediately on a refresh tick: it does not simulate a parallel busy clock. That asymmetry is a boundary of the toy, not a hardware comparison.", "",
        "## Metric definitions", "",
        "- **configured decision latency**: 55 ms is the serial busy period; 60 ms is recorded metadata for parallel rows. Neither is a wall-clock measurement, and only the serial value gates action execution.",
        "- **action-output throughput**: generated action values divided by 7.2 simulated seconds. Every parallel generation counts eight actions, including suffix values discarded after refresh; this measures generated output work/availability, not useful executed actions or the fixed 25 Hz plant rate.",
        "- **mean tracking error**: the rollout mean Euclidean distance from point mass to moving target across all 180 ticks.",
        "- **final tracking error**: Euclidean target distance at the last tick.",
        "- **terminal-window success**: a rollout scores 1 when its last 20 ticks (0.8 s) have mean tracking error below 0.16; table success is the average over trials.",
        "- **jerk proxy**: rollout mean norm of the second finite difference of emitted commands, divided by `dt²`; it is descriptive command variation, not an actuator limit.", "",
        "## Results", "",
        "| Method | latency (ms, configured) | action-output throughput (actions/s) | mean tracking error | final tracking error | terminal-window success | jerk proxy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(f"| {r['method']} | {r['simulated_decision_latency_ms']:.0f} | {r['output_action_rate_hz']:.1f} | {r['mean_tracking_error']:.3f} | {r['final_tracking_error']:.3f} | {100*r['success_rate']:.1f}% | {r['smoothness_jerk']:.1f} |")
    lines += [
        "", "## Visualizations", "",
        "- `openvla_oft_systems_summary.png` has a left panel with one matched-seed rollout for each method against the moving target. It illustrates geometry; it is not a representative-statistics claim.",
        "- Its right panel juxtaposes aggregate terminal-window success and generated action-output throughput, and labels the bar groups with rounded jerk proxies. The unlike units share a panel only as a compact trade-off view; use the table for exact values.",
        "- In both the figure and table, early-refresh output rate includes actions generated but not executed.",
        "", "## Interpretation", "",
        f"The serial condition produces `{serial['output_action_rate_hz']:.1f}` actions/s because its held-command semantics turn a 55 ms busy period into decisions on alternating 40 ms ticks. It has the largest mean error ({serial['mean_tracking_error']:.3f}) and jerk proxy ({serial['smoothness_jerk']:.2f}). Parallel open loop reaches `{open_loop['output_action_rate_hz']:.1f}` generated actions/s and lower jerk ({open_loop['smoothness_jerk']:.2f}), but its eight-step suffix remains conditioned on stale state and it reaches only `{100 * open_loop['success_rate']:.1f}%` terminal-window success.", "",
        f"Refresh-4 is the best terminal compromise in this run: `{100 * refresh4['success_rate']:.1f}%` success, `{refresh4['mean_tracking_error']:.3f}` mean error, and `{refresh4['smoothness_jerk']:.2f}` jerk. Refresh-2 limits stale-suffix exposure further and therefore has the lowest mean error (`{refresh2['mean_tracking_error']:.3f}`), but it re-anchors every two ticks to a four-tick-old observation, creates more command boundaries, and doubles generated output work versus refresh-4 (`{refresh2['output_action_rate_hz']:.1f}` versus `{refresh4['output_action_rate_hz']:.1f}` actions/s). Its final error and jerk are slightly worse, and its `{100 * refresh2['success_rate']:.1f}%` terminal-window success matches open loop. Thus average error, end-of-rollout occupancy, smoothness, and generated work select different refresh periods here; four is not a universal optimum.",
        "", "## Connection to other vla-ideas experiments", "",
        "- **`async_chunking_compare`** studies when a next chunk should be planned under inference delay. This toy isolates the execution consequence of retaining versus discarding a stale suffix; an async planner could supply a latency-compensated state estimate at refresh.",
        "- **`turbo_vla_direct_control`** studies frequent refresh for direct vision+language-to-action chunks. Here the visual/language learning is intentionally removed, leaving the related question of how a parallel continuous output head is consumed.",
        "- **`bspline_action_parameterization`** changes a chunk's representation and its smoothness. This toy holds a dense eight-action representation fixed and changes refresh timing; a deployment could combine a compact continuous representation with a refresh schedule.",
        "- **`path_consistent_safety_filtering`** preserves proposed geometry while changing timing for safety. Early refresh changes the suffix proposal itself, so a real stack may need both a refresh policy and a path-consistent execution layer.",
        "", "## Limitations and follow-ups", "",
        "- The toy omits OpenVLA-OFT training and all task/model components; its analytical controller does not validate the paper's benchmark or hardware claims. The paper/project/repository and the separation from this evidence are documented in [`notes/2026-08-06-openvla-oft.md`](../../notes/2026-08-06-openvla-oft.md).",
        "- Serial 55 ms changes actual held-command behavior, whereas parallel 60 ms is only recorded metadata. Implement an event-driven scheduler with sampled latency, queueing, deadlines, and fallback behavior for both paths before comparing end-to-end deployment latency.",
        "- Export paired per-trial records and sweep observation age, horizon, gust magnitude, and refresh period with uncertainty intervals; this would test where the refresh-4/refresh-2 terminal crossover occurs.",
        "- Compare naive delayed-state refresh with latency-compensated state prediction, then move to a constrained arm task with a learned continuous chunk head, randomized observations/instructions, and a capacity-matched serial baseline.",
        "", "## Outputs", "",
        "- `openvla_oft_systems_metrics.csv`: exact aggregate rows for this invocation.",
        "- `openvla_oft_systems_summary.png`: matched rollout and aggregate trade-off visualization.",
        "- `openvla_oft_systems_report.md`: this generated report.",
    ]
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
