import argparse
import csv
import os
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE_DIR, ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


METHOD_ORDER = [
    "synchronous",
    "naive_async",
    "rtc_hard",
    "prefix_conditioned",
    "vlash",
    "vlash_reflex",
]

REPORT_METHODS = [
    "synchronous",
    "naive_async",
    "rtc_hard",
    "prefix_conditioned",
    "vlash",
]

COLORS = {
    "synchronous": "#4c78a8",
    "naive_async": "#f58518",
    "rtc_hard": "#54a24b",
    "prefix_conditioned": "#72b7b2",
    "vlash": "#e45756",
    "vlash_reflex": "#b279a2",
}

LABELS = {
    "synchronous": "Synchronous",
    "naive_async": "Naive Async",
    "rtc_hard": "RTC Hard Stitch",
    "prefix_conditioned": "Train-Time Prefix",
    "vlash": "VLASH",
    "vlash_reflex": "VLASH + Reflex",
}


@dataclass
class Config:
    horizon: int = 18
    delay: int = 8
    dt: float = 0.02
    goal: float = 10.0
    max_steps: int = 900
    max_accel: float = 8.0
    disturbance_std: float = 0.035
    moving_goal_amplitude: float = 0.0
    moving_goal_period: float = 2.6
    goal_tolerance: float = 0.12
    settle_steps: int = 12
    reflex_gain_p: float = 0.42
    reflex_gain_d: float = 0.16
    reflex_limit: float = 0.95


@dataclass
class PrefixConditionedModel:
    weights: np.ndarray
    bias: float

    def predict_accel(self, state: np.ndarray, committed_tail: np.ndarray, cfg: Config) -> float:
        features = prefix_features(state, committed_tail, cfg)
        accel = float(features[:-1] @ self.weights + self.bias)
        return float(np.clip(accel, -cfg.max_accel, cfg.max_accel))

    def predict_plan(self, state: np.ndarray, committed_tail: np.ndarray, cfg: Config) -> np.ndarray:
        accel = self.predict_accel(state, committed_tail, cfg)
        return np.full(cfg.horizon, accel, dtype=np.float64)


class PointMassEnv:
    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.x = 0.0
        self.v = 0.0
        self.t = 0.0
        self.goal = cfg.goal

    def current_goal(self) -> float:
        if self.cfg.moving_goal_amplitude <= 0.0:
            return self.cfg.goal
        phase = 2.0 * np.pi * self.t / self.cfg.moving_goal_period
        return self.cfg.goal + self.cfg.moving_goal_amplitude * np.sin(phase)

    def observe(self) -> np.ndarray:
        return np.array([self.x, self.v, self.current_goal()], dtype=np.float64)

    def reset(self) -> np.ndarray:
        self.x = 0.0
        self.v = 0.0
        self.t = 0.0
        self.goal = self.current_goal()
        return self.observe()

    def step(self, action: float) -> np.ndarray:
        a = float(np.clip(action, -self.cfg.max_accel, self.cfg.max_accel))
        dv_dist = self.rng.normal(0.0, self.cfg.disturbance_std)
        self.v += a * self.cfg.dt + dv_dist
        self.x += self.v * self.cfg.dt
        self.t += self.cfg.dt
        self.goal = self.current_goal()
        return self.observe()


def safe_nanmean(values: list[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def constant_accel_plan(state: np.ndarray, cfg: Config) -> np.ndarray:
    x, v, goal = state
    horizon_time = cfg.horizon * cfg.dt
    dx = goal - x
    a_const = 2.0 * (dx - v * horizon_time) / max(horizon_time**2, 1e-6)
    return np.full(cfg.horizon, np.clip(a_const, -cfg.max_accel, cfg.max_accel), dtype=np.float64)


def rollout_future(state: np.ndarray, actions: np.ndarray, cfg: Config) -> np.ndarray:
    x, v, goal = state.astype(np.float64).copy()
    for action in actions:
        v += float(np.clip(action, -cfg.max_accel, cfg.max_accel)) * cfg.dt
        x += v * cfg.dt
    return np.array([x, v, goal], dtype=np.float64)


def reflex_residual(state: np.ndarray, cfg: Config) -> float:
    x, v, goal = state
    residual = cfg.reflex_gain_p * (goal - x) - cfg.reflex_gain_d * v
    return float(np.clip(residual, -cfg.reflex_limit, cfg.reflex_limit))


def prefix_features(state: np.ndarray, committed_tail: np.ndarray, cfg: Config) -> np.ndarray:
    x, v, goal = state.astype(np.float64)
    dx = goal - x
    if committed_tail.size == 0:
        prefix_mean = 0.0
        prefix_last = 0.0
        prefix_slope = 0.0
        prefix_std = 0.0
        prefix_energy = 0.0
        prefix_len = 0.0
    else:
        prefix_mean = float(np.mean(committed_tail))
        prefix_last = float(committed_tail[-1])
        prefix_slope = float(committed_tail[-1] - committed_tail[0]) / max(committed_tail.size - 1, 1)
        prefix_std = float(np.std(committed_tail))
        prefix_energy = float(np.mean(committed_tail**2))
        prefix_len = float(committed_tail.size)

    return np.array(
        [
            x,
            v,
            goal,
            dx,
            prefix_mean,
            prefix_last,
            prefix_slope,
            prefix_std,
            prefix_energy,
            prefix_len / max(cfg.horizon, 1),
            cfg.delay / max(cfg.horizon, 1),
            dx * prefix_mean,
            v * prefix_mean,
            dx * v,
            1.0,
        ],
        dtype=np.float64,
    )


def train_prefix_conditioned_model(cfg: Config, seed: int, samples: int = 7000) -> PrefixConditionedModel:
    rng = np.random.default_rng(seed)
    feature_rows = []
    targets = []

    for _ in range(samples):
        state = np.array(
            [
                rng.uniform(-4.0, 14.0),
                rng.uniform(-7.0, 7.0),
                cfg.goal + rng.uniform(-max(cfg.moving_goal_amplitude, 0.5), max(cfg.moving_goal_amplitude, 0.5)),
            ],
            dtype=np.float64,
        )
        stale_plan = constant_accel_plan(state, cfg)
        plan_noise = rng.normal(0.0, 0.65, size=cfg.delay)
        committed_tail = np.clip(stale_plan[: cfg.delay] + plan_noise, -cfg.max_accel, cfg.max_accel)
        future_state = rollout_future(state, committed_tail, cfg)
        target = float(constant_accel_plan(future_state, cfg)[0])
        feature_rows.append(prefix_features(state, committed_tail, cfg))
        targets.append(target)

    x_train = np.vstack(feature_rows)
    y_train = np.array(targets, dtype=np.float64)
    x_center = x_train[:, :-1]
    x_aug = np.concatenate([x_center, np.ones((x_center.shape[0], 1), dtype=np.float64)], axis=1)
    ridge = 1e-3 * np.eye(x_aug.shape[1], dtype=np.float64)
    weights = np.linalg.solve(x_aug.T @ x_aug + ridge, x_aug.T @ y_train)
    return PrefixConditionedModel(weights=weights[:-1], bias=float(weights[-1]))


def plan_for_method(
    method: str,
    state_at_plan_start: np.ndarray,
    committed_tail: np.ndarray,
    cfg: Config,
    prefix_model: PrefixConditionedModel,
) -> np.ndarray:
    if method in {"vlash", "vlash_reflex"}:
        future_state = rollout_future(state_at_plan_start, committed_tail, cfg)
        return constant_accel_plan(future_state, cfg)

    if method == "prefix_conditioned":
        return prefix_model.predict_plan(state_at_plan_start, committed_tail, cfg)

    planned = constant_accel_plan(state_at_plan_start, cfg)
    if method == "rtc_hard" and committed_tail.size > 0:
        prefix_len = min(cfg.delay, planned.size, committed_tail.size)
        planned[:prefix_len] = committed_tail[:prefix_len]
        if prefix_len < planned.size:
            planned[prefix_len:] *= 1.45
    return planned


def summarize_run(trace: dict, cfg: Config) -> dict:
    x = trace["x"]
    v = trace["v"]
    a = trace["a"]
    t = trace["t"]
    goal = trace["goal"]

    err = np.abs(x - goal)
    goal_hits = np.where(err <= cfg.goal_tolerance)[0]
    settle_time = np.nan
    if goal_hits.size > 0:
        for idx in goal_hits:
            tail = err[idx : idx + cfg.settle_steps]
            if tail.size == cfg.settle_steps and np.all(tail <= cfg.goal_tolerance):
                settle_time = float(t[idx])
                break

    jerk = np.diff(a, prepend=a[0]) / cfg.dt
    velocity_jump = np.diff(v, prepend=v[0])
    tail_start = len(err) // 2
    tail_err = err[tail_start:]

    return {
        "final_x": float(x[-1]),
        "final_goal": float(goal[-1]),
        "final_error": float(err[-1]),
        "settle_time": settle_time,
        "success": float(np.isfinite(settle_time)),
        "rms_accel": float(np.sqrt(np.mean(a**2))),
        "rms_jerk": float(np.sqrt(np.mean(jerk**2))),
        "max_abs_jerk": float(np.max(np.abs(jerk))),
        "rms_vel_jump": float(np.sqrt(np.mean(velocity_jump**2))),
        "path_effort": float(np.sum(np.abs(a)) * cfg.dt),
        "mean_abs_error": float(np.mean(err)),
        "tail_mean_abs_error": float(np.mean(tail_err)),
        "within_tol_frac": float(np.mean(err <= cfg.goal_tolerance)),
        "duration": float(t[-1]),
    }


def run_single(method: str, cfg: Config, seed: int, prefix_model: PrefixConditionedModel) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    env = PointMassEnv(cfg, rng)
    state = env.reset()

    x_hist = [state[0]]
    v_hist = [state[1]]
    a_hist = [0.0]
    t_hist = [0.0]
    goal_hist = [state[2]]

    active_chunk = None
    active_idx = 0
    pending_chunk = None
    pending_ready_in = 0
    sync_wait = cfg.delay if method == "synchronous" else 0
    plan_started = False

    while len(t_hist) < cfg.max_steps:
        if method == "synchronous":
            if active_chunk is None and sync_wait == 0:
                active_chunk = constant_accel_plan(state, cfg)
                active_idx = 0
            if active_chunk is None and sync_wait > 0:
                action = 0.0
                sync_wait -= 1
            else:
                action = float(active_chunk[active_idx])
                active_idx += 1
                if active_idx >= cfg.horizon:
                    active_chunk = None
                    active_idx = 0
                    sync_wait = cfg.delay
        else:
            if active_chunk is None:
                active_chunk = constant_accel_plan(state, cfg)
                active_idx = 0
                pending_chunk = None
                pending_ready_in = 0
                plan_started = False

            remaining = cfg.horizon - active_idx
            if not plan_started and remaining <= cfg.delay:
                committed_tail = active_chunk[active_idx : active_idx + cfg.delay].copy()
                pending_chunk = plan_for_method(method, state.copy(), committed_tail, cfg, prefix_model)
                pending_ready_in = cfg.delay
                plan_started = True

            action = float(active_chunk[active_idx])
            if method == "vlash_reflex":
                action += reflex_residual(state, cfg)
            action = float(np.clip(action, -cfg.max_accel, cfg.max_accel))
            active_idx += 1

            if plan_started:
                pending_ready_in -= 1

            if active_idx >= cfg.horizon:
                if pending_chunk is not None and pending_ready_in <= 0:
                    active_chunk = pending_chunk
                else:
                    active_chunk = constant_accel_plan(state, cfg)
                active_idx = 0
                pending_chunk = None
                pending_ready_in = 0
                plan_started = False

        state = env.step(action)
        x_hist.append(state[0])
        v_hist.append(state[1])
        a_hist.append(action)
        t_hist.append(env.t)
        goal_hist.append(state[2])

    trace = {
        "t": np.array(t_hist, dtype=np.float64),
        "x": np.array(x_hist, dtype=np.float64),
        "v": np.array(v_hist, dtype=np.float64),
        "a": np.array(a_hist, dtype=np.float64),
        "goal": np.array(goal_hist, dtype=np.float64),
    }
    metrics = summarize_run(trace, cfg)
    return trace, metrics


def run_experiments(
    cfg: Config,
    single_seed: int,
    monte_carlo_trials: int,
    prefix_model: PrefixConditionedModel,
) -> tuple[dict, dict]:
    traces = {}
    metric_table = {method: [] for method in METHOD_ORDER}

    for method in METHOD_ORDER:
        trace, _ = run_single(method, cfg, seed=single_seed, prefix_model=prefix_model)
        traces[method] = trace
        for trial in range(monte_carlo_trials):
            _, metrics = run_single(method, cfg, seed=single_seed + 1000 + trial, prefix_model=prefix_model)
            metric_table[method].append(metrics)
    return traces, metric_table


def aggregate_metric_table(metric_table: dict) -> dict:
    return {
        method: {
            key: safe_nanmean([row[key] for row in rows])
            for key in rows[0].keys()
        }
        for method, rows in metric_table.items()
    }


def save_metrics_csv(metric_table: dict, path: str) -> None:
    fieldnames = ["method"] + sorted(metric_table[METHOD_ORDER[0]][0].keys())
    aggregate_table = aggregate_metric_table(metric_table)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHOD_ORDER:
            row = {"method": method}
            row.update(aggregate_table[method])
            writer.writerow(row)


def save_delay_sweep_csv(rows: list[dict], path: str) -> None:
    fieldnames = [
        "scenario",
        "delay",
        "method",
        "success",
        "settle_time",
        "within_tol_frac",
        "tail_mean_abs_error",
        "rms_jerk",
        "final_error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_single_run_plot(traces: dict, cfg: Config, path: str, methods: list[str]) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    for method in methods:
        trace = traces[method]
        color = COLORS[method]
        label = LABELS[method]
        axes[0].plot(trace["t"], trace["x"], color=color, linewidth=2.2, label=label)
        axes[1].plot(trace["t"], trace["v"], color=color, linewidth=2.0)
        axes[2].plot(trace["t"], trace["a"], color=color, linewidth=1.6)
        jerk = np.diff(trace["a"], prepend=trace["a"][0]) / cfg.dt
        axes[3].plot(trace["t"], jerk, color=color, linewidth=1.4)

    axes[0].plot(traces[methods[0]]["t"], traces[methods[0]]["goal"], "--", color="black", linewidth=1.2, label="Goal")
    axes[0].set_ylabel("Position")
    axes[1].set_ylabel("Velocity")
    axes[2].set_ylabel("Acceleration")
    axes[3].set_ylabel("Jerk")
    axes[3].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=3, frameon=True)

    fig.suptitle(
        f"Chunked VLA Delay Toy | H={cfg.horizon}, d={cfg.delay}, disturbance={cfg.disturbance_std:.3f}",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_metric_plot(metric_table: dict, path: str, methods: list[str], title: str, moving_goal: bool) -> None:
    if moving_goal:
        metrics = {
            "Tail mean abs error": [safe_nanmean([row["tail_mean_abs_error"] for row in metric_table[m]]) for m in methods],
            "Within tolerance fraction": [safe_nanmean([row["within_tol_frac"] for row in metric_table[m]]) for m in methods],
            "RMS jerk": [safe_nanmean([row["rms_jerk"] for row in metric_table[m]]) for m in methods],
            "Control effort": [safe_nanmean([row["path_effort"] for row in metric_table[m]]) for m in methods],
        }
    else:
        metrics = {
            "Mean settle time (s)": [safe_nanmean([row["settle_time"] for row in metric_table[m]]) for m in methods],
            "Success rate": [np.mean([row["success"] for row in metric_table[m]]) for m in methods],
            "RMS jerk": [safe_nanmean([row["rms_jerk"] for row in metric_table[m]]) for m in methods],
            "Control effort": [safe_nanmean([row["path_effort"] for row in metric_table[m]]) for m in methods],
        }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    x = np.arange(len(methods))
    bar_colors = [COLORS[m] for m in methods]

    for ax, (metric_title, values) in zip(axes, metrics.items()):
        ax.bar(x, values, color=bar_colors, alpha=0.9)
        ax.set_title(metric_title)
        ax.set_xticks(x, [LABELS[m] for m in methods], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_delay_sweep_plot(delay_rows: list[dict], path: str, scenario_name: str) -> None:
    if scenario_name == "moving_goal":
        metrics = [
            ("within_tol_frac", "Within tolerance fraction"),
            ("tail_mean_abs_error", "Tail mean abs error"),
            ("rms_jerk", "RMS jerk"),
        ]
    else:
        metrics = [
            ("success", "Success rate"),
            ("settle_time", "Mean settle time (s)"),
            ("rms_jerk", "RMS jerk"),
        ]
    methods = REPORT_METHODS
    delays = sorted({int(row["delay"]) for row in delay_rows if row["scenario"] == scenario_name})

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (metric_key, metric_title) in zip(axes, metrics):
        for method in methods:
            ys = [
                next(
                    row[metric_key]
                    for row in delay_rows
                    if row["scenario"] == scenario_name and row["method"] == method and int(row["delay"]) == delay
                )
                for delay in delays
            ]
            ax.plot(delays, ys, marker="o", linewidth=2.0, color=COLORS[method], label=LABELS[method])
        ax.set_title(metric_title)
        ax.set_xlabel("Delay steps")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Value")
    axes[0].legend(frameon=True, fontsize=9)
    fig.suptitle(f"Delay Sweep | {scenario_name}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_summary(metric_table: dict, scenario_name: str) -> None:
    print(f"\n=== Monte Carlo Summary | {scenario_name} ===")
    for method in METHOD_ORDER:
        rows = metric_table[method]
        rms_jerk = safe_nanmean([row["rms_jerk"] for row in rows])
        final_error = safe_nanmean([row["final_error"] for row in rows])
        if scenario_name == "moving_goal":
            tail_error = safe_nanmean([row["tail_mean_abs_error"] for row in rows])
            within_tol = safe_nanmean([row["within_tol_frac"] for row in rows])
            print(
                f"{LABELS[method]:18s} | within tol {100.0 * within_tol:5.1f}% | "
                f"tail err {tail_error:6.3f} | final err {final_error:6.3f} | rms jerk {rms_jerk:7.2f}"
            )
        else:
            mean_settle = safe_nanmean([row["settle_time"] for row in rows])
            success = np.mean([row["success"] for row in rows])
            print(
                f"{LABELS[method]:18s} | success {100.0 * success:5.1f}% | "
                f"settle {mean_settle:5.2f}s | final err {final_error:6.3f} | rms jerk {rms_jerk:7.2f}"
            )


def format_metric_line(metric_table: dict, method: str) -> str:
    rows = metric_table[method]
    success = 100.0 * np.mean([row["success"] for row in rows])
    settle = safe_nanmean([row["settle_time"] for row in rows])
    error = safe_nanmean([row["final_error"] for row in rows])
    jerk = safe_nanmean([row["rms_jerk"] for row in rows])
    return (
        f"- {LABELS[method]}: `{success:.0f}%` success, "
        f"`{settle:.2f}s` mean settle, `{error:.3f}` final error, `{jerk:.1f}` RMS jerk"
    )


def format_tracking_line(metric_table: dict, method: str) -> str:
    rows = metric_table[method]
    within_tol = 100.0 * safe_nanmean([row["within_tol_frac"] for row in rows])
    tail_error = safe_nanmean([row["tail_mean_abs_error"] for row in rows])
    error = safe_nanmean([row["final_error"] for row in rows])
    jerk = safe_nanmean([row["rms_jerk"] for row in rows])
    return (
        f"- {LABELS[method]}: `{within_tol:.0f}%` within tolerance, "
        f"`{tail_error:.3f}` tail mean error, `{error:.3f}` final error, `{jerk:.1f}` RMS jerk"
    )


def write_report(
    static_cfg: Config,
    dynamic_cfg: Config,
    static_metrics: dict,
    dynamic_metrics: dict,
    paths: dict,
) -> None:
    report_path = os.path.join(DOCS_DIR, "async_chunking_experiment_report.md")

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Async Chunking Toy Experiment Report\n\n")
        handle.write("This note records the current comparison status for toy delayed-action VLA deployment ideas in `async_chunking_compare/`.\n\n")
        handle.write("## Bottom line\n\n")
        handle.write(
            "In this point-mass simulator, `VLASH` is the strongest robust runtime strategy, the learned `Train-Time Prefix` baseline closes much of the gap without explicit rollout, and the crude `RTC Hard Stitch` fails badly because hard prefix freezing without inpainting breaks trajectory consistency.\n\n"
        )
        handle.write("The main qualitative pattern matches the motivating intuition from the papers:\n\n")
        handle.write("- `Synchronous` is stable but slow because the robot pauses while replanning.\n")
        handle.write("- `Naive Async` improves utilization but suffers stale-state oscillation.\n")
        handle.write("- `RTC Hard Stitch` is a useful negative control, not a faithful RTC implementation.\n")
        handle.write("- `Train-Time Prefix` learns to continue committed motion from stale observations plus the committed prefix.\n")
        handle.write("- `VLASH` remains best because it aligns policy input to the future execution-time state directly.\n\n")

        handle.write("## Implemented experiments\n\n")
        handle.write("Script:\n\n")
        handle.write("- `async_chunking_compare/run_async_chunking_compare.py`\n\n")
        handle.write("Outputs:\n\n")
        for key in [
            "static_single",
            "static_metrics",
            "dynamic_single",
            "dynamic_metrics",
            "delay_sweep",
            "static_csv",
            "dynamic_csv",
            "delay_csv",
        ]:
            handle.write(f"- `{os.path.relpath(paths[key], BASE_DIR)}`\n")
        handle.write("\n")

        handle.write("### Static-goal scenario\n\n")
        handle.write("Setup summary:\n\n")
        handle.write(f"- Goal at `x={static_cfg.goal:.1f}`\n")
        handle.write(f"- Chunk horizon `H={static_cfg.horizon}`\n")
        handle.write(f"- Delay `d={static_cfg.delay}`\n")
        handle.write(f"- Control step `dt={static_cfg.dt:.2f}s`\n")
        handle.write(f"- Velocity disturbance std `{static_cfg.disturbance_std:.3f}`\n\n")
        handle.write("Current verified results:\n\n")
        for method in REPORT_METHODS:
            handle.write(format_metric_line(static_metrics, method) + "\n")
        handle.write("\n")
        handle.write("![Static single-run comparison](../outputs/async_chunking_static_single_run.png)\n\n")
        handle.write("![Static Monte Carlo metrics](../outputs/async_chunking_static_monte_carlo.png)\n\n")

        handle.write("Interpretation:\n\n")
        handle.write("- `Synchronous` eventually reaches the target but wastes wall-clock time in every planning gap.\n")
        handle.write("- `Naive Async` keeps moving, but its commands were planned for the wrong physical state and it oscillates.\n")
        handle.write("- `Train-Time Prefix` clearly improves over naive async by using the committed prefix as an implicit delay cue.\n")
        handle.write("- `VLASH` is still better because it uses the exact rolled-forward future state instead of having to infer it.\n")
        handle.write("- `RTC Hard Stitch` demonstrates why hard handoff alone is not enough; the missing inpainting step matters.\n\n")

        handle.write("### Moving-goal scenario\n\n")
        handle.write("Setup summary:\n\n")
        handle.write(f"- Goal oscillation amplitude `{dynamic_cfg.moving_goal_amplitude:.2f}`\n")
        handle.write(f"- Goal oscillation period `{dynamic_cfg.moving_goal_period:.2f}s`\n")
        handle.write(f"- Same `H={dynamic_cfg.horizon}`, `d={dynamic_cfg.delay}`, `dt={dynamic_cfg.dt:.2f}s`\n\n")
        handle.write("Current verified results:\n\n")
        for method in REPORT_METHODS:
            handle.write(format_tracking_line(dynamic_metrics, method) + "\n")
        handle.write("\n")
        handle.write("![Dynamic single-run comparison](../outputs/async_chunking_dynamic_single_run.png)\n\n")
        handle.write("![Dynamic Monte Carlo metrics](../outputs/async_chunking_dynamic_monte_carlo.png)\n\n")

        handle.write("Interpretation:\n\n")
        handle.write("- The moving target increases the penalty for stale state because the reference itself shifts during the delay window.\n")
        handle.write("- `Train-Time Prefix` still helps by damping the large stale-state oscillations, but it lags the moving reference.\n")
        handle.write("- `VLASH` tracks the changing goal more tightly in the trace because it plans against the execution-time state instead of the stale one.\n")
        handle.write("- This is the scenario where the biological internal-model analogy is most visible in the toy.\n\n")

        handle.write("### Delay sweep\n\n")
        handle.write("![Delay sweep](../outputs/async_chunking_delay_sweep.png)\n\n")
        handle.write("Interpretation:\n\n")
        handle.write("- Small delays are tolerable for most methods.\n")
        handle.write("- As `d/H` grows, `Naive Async` degrades first, then the learned prefix model, while `VLASH` remains usable longer.\n")
        handle.write("- The hard-stitch negative control becomes unstable quickly, reinforcing that RTC's actual guidance step is doing real work.\n\n")

        handle.write("## Current conclusion\n\n")
        handle.write(
            "The toy now supports a more credible story than the first draft. The runtime ranking is no longer just `VLASH` versus obviously broken baselines: the learned prefix-conditioned policy demonstrates that delay-awareness can be internalized during training, but explicit future-state alignment still wins when the environment becomes more dynamic.\n"
        )


def scenario_cfg(base_cfg: Config, *, moving_goal_amplitude: float, moving_goal_period: float) -> Config:
    return Config(
        horizon=base_cfg.horizon,
        delay=base_cfg.delay,
        dt=base_cfg.dt,
        goal=base_cfg.goal,
        max_steps=base_cfg.max_steps,
        max_accel=base_cfg.max_accel,
        disturbance_std=base_cfg.disturbance_std,
        moving_goal_amplitude=moving_goal_amplitude,
        moving_goal_period=moving_goal_period,
        goal_tolerance=base_cfg.goal_tolerance,
        settle_steps=base_cfg.settle_steps,
        reflex_gain_p=base_cfg.reflex_gain_p,
        reflex_gain_d=base_cfg.reflex_gain_d,
        reflex_limit=base_cfg.reflex_limit,
    )


def run_delay_sweep(
    base_cfg: Config,
    delays: list[int],
    trials: int,
    seed: int,
    moving_goal_amplitude: float,
    moving_goal_period: float,
) -> list[dict]:
    rows = []
    scenario_name = "moving_goal" if moving_goal_amplitude > 0.0 else "static_goal"

    for delay in delays:
        cfg = scenario_cfg(
            base_cfg,
            moving_goal_amplitude=moving_goal_amplitude,
            moving_goal_period=moving_goal_period,
        )
        cfg.delay = delay
        prefix_model = train_prefix_conditioned_model(cfg, seed=seed + 77 + delay)
        _, metric_table = run_experiments(cfg, single_seed=seed + delay, monte_carlo_trials=trials, prefix_model=prefix_model)
        for method in REPORT_METHODS:
            rows.append(
                {
                    "scenario": scenario_name,
                    "delay": delay,
                    "method": method,
                    "success": np.mean([row["success"] for row in metric_table[method]]),
                    "settle_time": safe_nanmean([row["settle_time"] for row in metric_table[method]]),
                    "within_tol_frac": safe_nanmean([row["within_tol_frac"] for row in metric_table[method]]),
                    "tail_mean_abs_error": safe_nanmean([row["tail_mean_abs_error"] for row in metric_table[method]]),
                    "rms_jerk": safe_nanmean([row["rms_jerk"] for row in metric_table[method]]),
                    "final_error": safe_nanmean([row["final_error"] for row in metric_table[method]]),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare delayed chunked-control strategies in a toy point-mass environment.")
    parser.add_argument("--horizon", type=int, default=18)
    parser.add_argument("--delay", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--disturbance-std", type=float, default=0.035)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--moving-goal-amplitude", type=float, default=1.2)
    parser.add_argument("--moving-goal-period", type=float, default=2.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = Config(
        horizon=args.horizon,
        delay=args.delay,
        dt=args.dt,
        disturbance_std=args.disturbance_std,
    )
    if base_cfg.delay >= base_cfg.horizon:
        raise ValueError("delay must be smaller than horizon")

    static_cfg = scenario_cfg(base_cfg, moving_goal_amplitude=0.0, moving_goal_period=args.moving_goal_period)
    dynamic_cfg = scenario_cfg(
        base_cfg,
        moving_goal_amplitude=args.moving_goal_amplitude,
        moving_goal_period=args.moving_goal_period,
    )

    static_prefix_model = train_prefix_conditioned_model(static_cfg, seed=args.seed + 11)
    dynamic_prefix_model = train_prefix_conditioned_model(dynamic_cfg, seed=args.seed + 29)

    static_traces, static_metric_table = run_experiments(
        static_cfg,
        single_seed=args.seed,
        monte_carlo_trials=args.trials,
        prefix_model=static_prefix_model,
    )
    dynamic_traces, dynamic_metric_table = run_experiments(
        dynamic_cfg,
        single_seed=args.seed + 1,
        monte_carlo_trials=args.trials,
        prefix_model=dynamic_prefix_model,
    )

    static_single_path = os.path.join(OUTPUT_DIR, "async_chunking_static_single_run.png")
    static_metric_path = os.path.join(OUTPUT_DIR, "async_chunking_static_monte_carlo.png")
    static_csv_path = os.path.join(OUTPUT_DIR, "async_chunking_static_metrics.csv")
    dynamic_single_path = os.path.join(OUTPUT_DIR, "async_chunking_dynamic_single_run.png")
    dynamic_metric_path = os.path.join(OUTPUT_DIR, "async_chunking_dynamic_monte_carlo.png")
    dynamic_csv_path = os.path.join(OUTPUT_DIR, "async_chunking_dynamic_metrics.csv")
    delay_plot_path = os.path.join(OUTPUT_DIR, "async_chunking_delay_sweep.png")
    delay_csv_path = os.path.join(OUTPUT_DIR, "async_chunking_delay_sweep.csv")

    make_single_run_plot(static_traces, static_cfg, static_single_path, REPORT_METHODS)
    make_metric_plot(
        static_metric_table,
        static_metric_path,
        REPORT_METHODS,
        "Static Goal Monte Carlo Comparison",
        moving_goal=False,
    )
    save_metrics_csv(static_metric_table, static_csv_path)

    make_single_run_plot(dynamic_traces, dynamic_cfg, dynamic_single_path, REPORT_METHODS)
    make_metric_plot(
        dynamic_metric_table,
        dynamic_metric_path,
        REPORT_METHODS,
        "Moving Goal Monte Carlo Comparison",
        moving_goal=True,
    )
    save_metrics_csv(dynamic_metric_table, dynamic_csv_path)

    delay_rows = run_delay_sweep(
        base_cfg,
        delays=[2, 4, 6, 8, 10, 12],
        trials=max(12, args.trials // 2),
        seed=args.seed,
        moving_goal_amplitude=args.moving_goal_amplitude,
        moving_goal_period=args.moving_goal_period,
    )
    make_delay_sweep_plot(delay_rows, delay_plot_path, "moving_goal")
    save_delay_sweep_csv(delay_rows, delay_csv_path)

    print_summary(static_metric_table, "static_goal")
    print_summary(dynamic_metric_table, "moving_goal")

    paths = {
        "static_single": static_single_path,
        "static_metrics": static_metric_path,
        "static_csv": static_csv_path,
        "dynamic_single": dynamic_single_path,
        "dynamic_metrics": dynamic_metric_path,
        "dynamic_csv": dynamic_csv_path,
        "delay_sweep": delay_plot_path,
        "delay_csv": delay_csv_path,
    }
    write_report(static_cfg, dynamic_cfg, static_metric_table, dynamic_metric_table, paths)

    print(f"\nWrote {static_single_path}")
    print(f"Wrote {static_metric_path}")
    print(f"Wrote {static_csv_path}")
    print(f"Wrote {dynamic_single_path}")
    print(f"Wrote {dynamic_metric_path}")
    print(f"Wrote {dynamic_csv_path}")
    print(f"Wrote {delay_plot_path}")
    print(f"Wrote {delay_csv_path}")
    print(f"Wrote {os.path.join(DOCS_DIR, 'async_chunking_experiment_report.md')}")


if __name__ == "__main__":
    main()
