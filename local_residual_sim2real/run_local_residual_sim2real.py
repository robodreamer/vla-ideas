#!/usr/bin/env python3
"""Deterministic local-residual sim-to-real adaptation and safety toy.

A biased prior plans short throw/catch-like transitions. Online observations update one of
four correction rules: global model replacement, global residual fine-tuning, nearest-memory
correction, or a regularized local residual with an uncertainty gate. Every method is evaluated
with and without a mutually reachable safe-set filter.

This is an adaptation/safety mechanism probe, not juggling, a VLA, or a reproduction of
"Rapid On-Robot Learning for Dynamic Manipulation Skills: Robot Juggling".
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
from dataclasses import asdict, dataclass
from typing import Any

BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("global_replacement", "residual_finetune", "nearest_memory", "local_residual")
SAFE_OPTIONS = (False, True)
LABELS = {
    "global_replacement": "Global replacement",
    "residual_finetune": "Residual fine-tune",
    "nearest_memory": "Nearest-memory",
    "local_residual": "Local residual + gate",
}
COLORS = {
    "global_replacement": "#e45756",
    "residual_finetune": "#f58518",
    "nearest_memory": "#72b7b2",
    "local_residual": "#54a24b",
}


@dataclass(frozen=True)
class Config:
    interactions: int = 240
    eval_every: int = 20
    seeds: tuple[int, ...] = (41, 42, 43, 44, 45)
    chain_length: int = 5
    train_chains_per_checkpoint: int = 36
    eval_chains: int = 100
    command_limit: float = 1.35
    safe_command_limit: float = 0.82
    hard_state_radius: float = 0.68
    catch_radius: float = 0.50
    safe_radius: float = 0.60
    recovery_limit: float = 1.05
    replacement_reg: float = 2e-3
    residual_reg: float = 0.035
    local_reg: float = 0.16
    memory_width: float = 0.72
    nearest_k: int = 9
    gate_neighbors: float = 7.0
    gate_distance: float = 0.92
    action_iterations: int = 3
    min_fit_samples: int = 6
    ablation_interactions: int = 12


B_PRIOR = np.array([[0.88, 0.19], [-0.13, 0.76]], dtype=np.float64)
F_PRIOR = np.array([[0.28, 0.15], [-0.10, 0.34]], dtype=np.float64)
C_PRIOR = np.array([[0.13, -0.07], [0.05, 0.10]], dtype=np.float64)
ADAPT_CENTER = np.array([0.48, -0.32], dtype=np.float64)
SOURCE_CENTER = np.array([-0.58, 0.43], dtype=np.float64)
OOD_CENTER = np.array([1.28, 0.92], dtype=np.float64)


def prior_base(state: np.ndarray, context: np.ndarray) -> np.ndarray:
    return F_PRIOR @ state + C_PRIOR @ context


def prior_next(state: np.ndarray, context: np.ndarray, action: np.ndarray) -> np.ndarray:
    return prior_base(state, context) + B_PRIOR @ action


def sim2real_gate(context: np.ndarray) -> float:
    d = context - ADAPT_CENTER
    return float(math.exp(-0.5 * float(d @ d) / (0.67**2)))


def true_residual(state: np.ndarray, context: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Localized dynamics gap plus a weak unmodeled nonlinear term."""
    gate = sim2real_gate(context)
    local_bias = np.array(
        [0.42 + 0.18 * context[0] - 0.10 * context[1], -0.31 + 0.09 * context[0] + 0.16 * context[1]],
        dtype=np.float64,
    )
    delta_b = np.array([[0.15, -0.08], [0.07, 0.12]], dtype=np.float64)
    local = local_bias + delta_b @ action + np.array(
        [0.09 * math.sin(1.4 * action[0]), -0.07 * math.sin(1.6 * action[1])], dtype=np.float64
    )
    weak_global = np.array(
        [0.018 * state[0] * context[1], -0.016 * state[1] * context[0]], dtype=np.float64
    )
    return gate * local + weak_global


def true_next(
    state: np.ndarray,
    context: np.ndarray,
    action: np.ndarray,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    out = prior_next(state, context, action) + true_residual(state, context, action)
    if noise is not None:
        out = out + noise
    return out


def features(state: np.ndarray, context: np.ndarray, action: np.ndarray) -> np.ndarray:
    return np.array(
        [
            1.0,
            state[0], state[1],
            context[0], context[1],
            action[0], action[1],
            context[0] * action[0], context[1] * action[1],
            state[0] * action[1], state[1] * action[0],
        ],
        dtype=np.float64,
    )


def distance_features(state: np.ndarray, context: np.ndarray, action: np.ndarray) -> np.ndarray:
    return np.concatenate([0.65 * state, context, 0.45 * action])


def ridge(x: np.ndarray, y: np.ndarray, reg: float, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        xw, yw = x, y
    else:
        root = np.sqrt(np.maximum(weights, 0.0))[:, None]
        xw, yw = x * root, y * root
    penalty = reg * np.eye(x.shape[1], dtype=np.float64)
    penalty[0, 0] *= 0.15
    return np.linalg.solve(xw.T @ xw + penalty, xw.T @ yw)


class Adapter:
    def __init__(
        self,
        method: str,
        cfg: Config,
        *,
        memory_width: float | None = None,
        local_reg: float | None = None,
    ) -> None:
        self.method = method
        self.cfg = cfg
        self.memory_width = cfg.memory_width if memory_width is None else memory_width
        self.local_reg = cfg.local_reg if local_reg is None else local_reg
        self._phi: list[np.ndarray] = []
        self._near: list[np.ndarray] = []
        self._state: list[np.ndarray] = []
        self._context: list[np.ndarray] = []
        self._action: list[np.ndarray] = []
        self._observed: list[np.ndarray] = []
        self._residual: list[np.ndarray] = []
        self._global_weights: np.ndarray | None = None

    @property
    def samples(self) -> int:
        return len(self._observed)

    def update(self, state: np.ndarray, context: np.ndarray, action: np.ndarray, observed: np.ndarray) -> None:
        self._phi.append(features(state, context, action))
        self._near.append(distance_features(state, context, action))
        self._state.append(state.copy())
        self._context.append(context.copy())
        self._action.append(action.copy())
        self._observed.append(observed.copy())
        self._residual.append(observed - prior_next(state, context, action))
        if self.method in ("global_replacement", "residual_finetune") and self.samples >= self.cfg.min_fit_samples:
            x = np.vstack(self._phi)
            if self.method == "global_replacement":
                self._global_weights = ridge(x, np.vstack(self._observed), self.cfg.replacement_reg)
            else:
                self._global_weights = ridge(x, np.vstack(self._residual), self.cfg.residual_reg)

    def correction(self, state: np.ndarray, context: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
        if self.samples < self.cfg.min_fit_samples:
            return np.zeros(2, dtype=np.float64), 0.0
        phi = features(state, context, action)
        if self.method == "global_replacement":
            assert self._global_weights is not None
            full = phi @ self._global_weights
            return full - prior_next(state, context, action), 1.0
        if self.method == "residual_finetune":
            assert self._global_weights is not None
            return phi @ self._global_weights, 1.0

        query = distance_features(state, context, action)
        near = np.vstack(self._near)
        d2 = np.sum((near - query[None, :]) ** 2, axis=1)
        residuals = np.vstack(self._residual)
        if self.method == "nearest_memory":
            k = min(self.cfg.nearest_k, self.samples)
            idx = np.argpartition(d2, k - 1)[:k]
            weights = np.exp(-0.5 * d2[idx] / max(self.memory_width**2, 1e-9)) + 1e-9
            return np.average(residuals[idx], axis=0, weights=weights), 1.0

        weights = np.exp(-0.5 * d2 / max(self.memory_width**2, 1e-9))
        effective = float(np.sum(weights))
        # Kernel-local constant residual with ridge shrinkage toward the frozen prior.
        # This is the zero-order local ridge estimator: lambda behaves as prior pseudo-counts.
        estimate = np.sum(weights[:, None] * residuals, axis=0) / (effective + self.local_reg)
        nearest = float(math.sqrt(float(np.min(d2))))
        count_gate = effective / (effective + self.cfg.gate_neighbors)
        distance_gate = math.exp(-0.5 * (nearest / self.cfg.gate_distance) ** 2)
        gate = float(np.clip(count_gate * distance_gate, 0.0, 1.0))
        return gate * estimate, gate

    def predict(self, state: np.ndarray, context: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
        correction, confidence = self.correction(state, context, action)
        return prior_next(state, context, action) + correction, confidence


def nominal_action(state: np.ndarray, context: np.ndarray) -> np.ndarray:
    return np.linalg.solve(B_PRIOR, -prior_base(state, context))


def proposed_action(adapter: Adapter, state: np.ndarray, context: np.ndarray, cfg: Config) -> np.ndarray:
    action = nominal_action(state, context)
    for _ in range(cfg.action_iterations):
        correction, _ = adapter.correction(state, context, action)
        action = np.linalg.solve(B_PRIOR, -(prior_base(state, context) + correction))
    return action


def normalized_radius(state: np.ndarray) -> float:
    return float(math.sqrt((state[0] / 0.95) ** 2 + (state[1] / 0.82) ** 2))


def predicted_mutual_margin(adapter: Adapter, state: np.ndarray, context: np.ndarray, action: np.ndarray, cfg: Config) -> float:
    predicted, _ = adapter.predict(state, context, action)
    catch_margin = cfg.safe_radius - normalized_radius(predicted)
    recovery = nominal_action(predicted, -0.35 * context)
    recovery_margin = cfg.recovery_limit - float(np.max(np.abs(recovery)))
    command_margin = cfg.safe_command_limit - float(np.max(np.abs(action)))
    return float(min(catch_margin, recovery_margin / cfg.recovery_limit, command_margin / cfg.safe_command_limit))


def safe_filter(adapter: Adapter, state: np.ndarray, context: np.ndarray, candidate: np.ndarray, cfg: Config) -> tuple[np.ndarray, bool]:
    anchor = nominal_action(state, context)
    anchor = np.clip(anchor, -cfg.command_limit, cfg.command_limit)
    if predicted_mutual_margin(adapter, state, context, candidate, cfg) >= 0.0:
        return candidate, False
    lo, hi = 0.0, 1.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        trial = anchor + mid * (candidate - anchor)
        if predicted_mutual_margin(adapter, state, context, trial, cfg) >= 0.0:
            lo = mid
        else:
            hi = mid
    return anchor + lo * (candidate - anchor), True


def context_for(rng: np.random.Generator, split: str) -> np.ndarray:
    if split == "adapt":
        return ADAPT_CENTER + rng.normal(0.0, [0.31, 0.27])
    if split == "edge":
        return ADAPT_CENTER + rng.normal(0.0, [0.62, 0.55])
    if split == "source":
        return SOURCE_CENTER + rng.normal(0.0, [0.25, 0.22])
    if split == "ood":
        return OOD_CENTER + rng.normal(0.0, [0.28, 0.25])
    raise ValueError(split)


def evaluate(
    adapter: Adapter,
    safe: bool,
    cfg: Config,
    seed: int,
    split: str,
    chains: int | None = None,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_chains = cfg.eval_chains if chains is None else chains
    chain_successes: list[float] = []
    catch_successes: list[float] = []
    control_errors: list[float] = []
    prediction_errors: list[float] = []
    margins: list[float] = []
    unsafe_commands = 0
    raw_unsafe_proposals = 0
    violations = 0
    filter_interventions = 0
    confidences: list[float] = []
    correction_norms: list[float] = []
    for _ in range(n_chains):
        state = rng.normal(0.0, [0.11, 0.09])
        chain_ok = True
        for _step in range(cfg.chain_length):
            context = context_for(rng, split)
            candidate = proposed_action(adapter, state, context, cfg)
            raw_unsafe = predicted_mutual_margin(adapter, state, context, candidate, cfg) < 0.0
            raw_unsafe_proposals += int(raw_unsafe)
            if safe:
                action, intervened = safe_filter(adapter, state, context, candidate, cfg)
                filter_interventions += int(intervened)
            else:
                action = candidate
            action = np.clip(action, -cfg.command_limit, cfg.command_limit)
            unsafe_commands += int(predicted_mutual_margin(adapter, state, context, action, cfg) < -1e-8)
            predicted, confidence = adapter.predict(state, context, action)
            actual = true_next(state, context, action)
            pred_error = float(np.linalg.norm(predicted - actual))
            control_error = float(np.linalg.norm(actual))
            radius = normalized_radius(actual)
            margin = cfg.catch_radius - radius
            hard_violation = radius > cfg.hard_state_radius
            caught = radius <= cfg.catch_radius
            violations += int(hard_violation)
            chain_ok = chain_ok and caught and not hard_violation
            catch_successes.append(float(caught))
            control_errors.append(control_error)
            prediction_errors.append(pred_error)
            margins.append(margin)
            confidences.append(confidence)
            correction_norms.append(float(np.linalg.norm(predicted - prior_next(state, context, action))))
            state = actual + rng.normal(0.0, [0.025, 0.022])
        chain_successes.append(float(chain_ok))
    transitions = n_chains * cfg.chain_length
    return {
        "chain_success": float(np.mean(chain_successes)),
        "catch_success": float(np.mean(catch_successes)),
        "control_error": float(np.mean(control_errors)),
        "prediction_error": float(np.mean(prediction_errors)),
        "transition_margin": float(np.mean(margins)),
        "p10_transition_margin": float(np.quantile(margins, 0.10)),
        "unsafe_command_rate": unsafe_commands / transitions,
        "raw_unsafe_proposal_rate": raw_unsafe_proposals / transitions,
        "violation_rate": violations / transitions,
        "filter_intervention_rate": filter_interventions / transitions,
        "mean_confidence": float(np.mean(confidences)),
        "mean_correction_norm": float(np.mean(correction_norms)),
    }


def train_one(
    method: str,
    safe: bool,
    seed: int,
    cfg: Config,
    *,
    interactions: int | None = None,
    memory_width: float | None = None,
    local_reg: float | None = None,
) -> tuple[Adapter, list[dict[str, Any]]]:
    budget = cfg.interactions if interactions is None else interactions
    adapter = Adapter(method, cfg, memory_width=memory_width, local_reg=local_reg)
    rng = np.random.default_rng(seed * 1009 + 17)
    curve: list[dict[str, Any]] = []

    def checkpoint(at: int) -> None:
        metrics = evaluate(adapter, safe, cfg, seed * 5003 + at + 71, "adapt", cfg.train_chains_per_checkpoint)
        curve.append({"method": method, "safe_set": int(safe), "seed": seed, "interactions": at, **metrics})

    checkpoint(0)
    state = rng.normal(0.0, [0.12, 0.10])
    for interaction in range(1, budget + 1):
        context = context_for(rng, "adapt")
        candidate = proposed_action(adapter, state, context, cfg)
        if safe:
            action, _ = safe_filter(adapter, state, context, candidate, cfg)
        else:
            action = candidate
        action = np.clip(action, -cfg.command_limit, cfg.command_limit)
        noise = rng.normal(0.0, [0.028, 0.024])
        observed = true_next(state, context, action, noise)
        adapter.update(state, context, action, observed)
        state = observed + rng.normal(0.0, [0.025, 0.022])
        if normalized_radius(state) > cfg.hard_state_radius or interaction % cfg.chain_length == 0:
            state = rng.normal(0.0, [0.12, 0.10])
        if interaction % cfg.eval_every == 0 or interaction == budget:
            checkpoint(interaction)
    return adapter, curve


def auc_from_curve(rows: list[dict[str, Any]], budget: int) -> float:
    x = np.array([row["interactions"] for row in rows], dtype=np.float64)
    y = np.array([row["chain_success"] for row in rows], dtype=np.float64)
    return float(np.trapezoid(y, x) / budget)


def first_threshold(rows: list[dict[str, Any]], threshold: float = 0.70) -> float | None:
    for row in rows:
        if row["chain_success"] >= threshold:
            return float(row["interactions"])
    return None


def run_main(cfg: Config) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    curves: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []
    for method in METHODS:
        for safe in SAFE_OPTIONS:
            for seed in cfg.seeds:
                adapter, curve = train_one(method, safe, seed, cfg)
                curves.extend(curve)
                adapt = evaluate(adapter, safe, cfg, seed * 8191 + 1, "adapt")
                source = evaluate(adapter, safe, cfg, seed * 8191 + 2, "source")
                ood = evaluate(adapter, safe, cfg, seed * 8191 + 3, "ood")
                prior_adapter = Adapter(method, cfg)
                source_prior = evaluate(prior_adapter, safe, cfg, seed * 8191 + 2, "source")
                conditions.append(
                    {
                        "method": method,
                        "safe_set": int(safe),
                        "seed": seed,
                        "samples": adapter.samples,
                        "interaction_auc": auc_from_curve(curve, cfg.interactions),
                        "interactions_to_70": first_threshold(curve),
                        **{f"adapt_{k}": v for k, v in adapt.items()},
                        **{f"source_{k}": v for k, v in source.items()},
                        **{f"ood_{k}": v for k, v in ood.items()},
                        "source_chain_retention": source["chain_success"] - source_prior["chain_success"],
                        "source_control_retention": source_prior["control_error"] - source["control_error"],
                    }
                )
    sanity = sanity_checks(cfg)
    return conditions, curves, sanity


def sanity_checks(cfg: Config) -> dict[str, Any]:
    state = np.array([0.10, -0.06])
    context = ADAPT_CENTER.copy()
    action = nominal_action(state, context)
    repeat_a = true_next(state, context, action)
    repeat_b = true_next(state, context, action)
    prior_err = float(np.linalg.norm(prior_next(state, context, action) - repeat_a))
    source_context = SOURCE_CENTER.copy()
    source_action = nominal_action(state, source_context)
    source_err = float(np.linalg.norm(prior_next(state, source_context, source_action) - true_next(state, source_context, source_action)))
    adapter = Adapter("local_residual", cfg)
    rng = np.random.default_rng(123)
    for _ in range(24):
        s = rng.normal(0.0, [0.12, 0.1])
        c = ADAPT_CENTER + rng.normal(0.0, [0.2, 0.2])
        a = nominal_action(s, c)
        adapter.update(s, c, a, true_next(s, c, a))
    local_corr, local_gate = adapter.correction(state, context, action)
    far_corr, far_gate = adapter.correction(state, OOD_CENTER, nominal_action(state, OOD_CENTER))
    risky_state = np.array([0.7, -0.55])
    risky_context = ADAPT_CENTER + np.array([0.25, -0.22])
    risky = proposed_action(adapter, risky_state, risky_context, cfg) + np.array([2.4, -2.1])
    filtered, intervened = safe_filter(adapter, risky_state, risky_context, risky, cfg)
    checks = {
        "deterministic_transition": bool(np.array_equal(repeat_a, repeat_b)),
        "biased_prior_has_local_gap": prior_err > 0.25,
        "source_prior_gap_is_smaller": source_err < 0.08,
        "local_adapter_learns_nonzero_correction": float(np.linalg.norm(local_corr)) > 0.08,
        "uncertainty_gate_is_local": local_gate > far_gate,
        "far_correction_is_suppressed": float(np.linalg.norm(far_corr)) < float(np.linalg.norm(local_corr)),
        "safe_filter_intervenes_on_risky_case": bool(intervened),
        "safe_filter_returns_finite_action": bool(np.all(np.isfinite(filtered))),
    }
    return {
        "all_passed": bool(all(checks.values())),
        "checks": checks,
        "diagnostics": {
            "adapt_prior_error": prior_err,
            "source_prior_error": source_err,
            "local_gate": local_gate,
            "far_gate": far_gate,
            "risky_action_max_abs": float(np.max(np.abs(risky))),
            "filtered_action_max_abs": float(np.max(np.abs(filtered))),
        },
    }


def run_ablations(cfg: Config) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seeds = cfg.seeds[:3]
    settings: list[tuple[str, float, float, bool]] = []
    for width in (0.18, 0.35, 0.72, 1.20, 2.00):
        settings.append(("memory_width", width, cfg.local_reg, True))
    for reg in (0.01, 0.16, 1.50, 8.00, 40.00):
        settings.append(("regularization", cfg.memory_width, reg, True))
    settings.extend(
        [
            ("safe_set", cfg.memory_width, cfg.local_reg, False),
            ("safe_set", cfg.memory_width, cfg.local_reg, True),
        ]
    )
    for kind, width, reg, safe in settings:
        for seed in seeds:
            adapter, curve = train_one(
                "local_residual",
                safe,
                seed + 200,
                cfg,
                interactions=cfg.ablation_interactions,
                memory_width=width,
                local_reg=reg,
            )
            metrics = evaluate(adapter, safe, cfg, seed * 3571 + 811, "edge")
            rows.append(
                {
                    "ablation": kind,
                    "value": int(safe) if kind == "safe_set" else (width if kind == "memory_width" else reg),
                    "safe_set": int(safe),
                    "seed": seed,
                    "interactions": cfg.ablation_interactions,
                    "interaction_auc": auc_from_curve(curve, cfg.ablation_interactions),
                    **metrics,
                }
            )
    return rows


def aggregate(rows: list[dict[str, Any]], group_keys: tuple[str, ...], metric_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        record = {k: v for k, v in zip(group_keys, key)}
        for metric in metric_keys:
            values = np.array([m[metric] for m in members if m.get(metric) is not None], dtype=np.float64)
            record[f"{metric}_mean"] = float(np.mean(values)) if values.size else None
            record[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0 if values.size else None
            record[f"{metric}_sem"] = float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0 if values.size else None
            record[f"{metric}_n"] = int(values.size)
        output.append(record)
    return output


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_learning(curves: list[dict[str, Any]], out: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for method in METHODS:
        for safe in SAFE_OPTIONS:
            subset = [r for r in curves if r["method"] == method and bool(r["safe_set"]) == safe]
            xs = sorted({int(r["interactions"]) for r in subset})
            means = []
            sems = []
            violations = []
            for x in xs:
                rows_x = [r for r in subset if r["interactions"] == x]
                chain = np.array([r["chain_success"] for r in rows_x])
                means.append(float(np.mean(chain)))
                sems.append(float(np.std(chain, ddof=1) / math.sqrt(len(chain))) if len(chain) > 1 else 0.0)
                violations.append(float(np.mean([r["violation_rate"] for r in rows_x])))
            style = "-" if safe else "--"
            label = f"{LABELS[method]} {'+ safe set' if safe else 'unconstrained'}"
            axes[0].plot(xs, means, style, color=COLORS[method], label=label, linewidth=2)
            axes[0].fill_between(xs, np.array(means) - sems, np.array(means) + sems, color=COLORS[method], alpha=0.10)
            axes[1].plot(xs, violations, style, color=COLORS[method], label=label, linewidth=2)
    axes[0].set(title="Online chain success", xlabel="real-like transitions", ylabel="5-skill chain success", ylim=(-0.02, 1.02))
    axes[1].set(title="Safety violations during evaluation", xlabel="real-like transitions", ylabel="violation rate", ylim=(-0.002, None))
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2, loc="lower right")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_tradeoffs(summary: list[dict[str, Any]], out: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    x = np.arange(len(METHODS))
    width = 0.36
    for j, safe in enumerate(SAFE_OPTIONS):
        rows = [next(r for r in summary if r["method"] == m and bool(r["safe_set"]) == safe) for m in METHODS]
        offset = (j - 0.5) * width
        axes[0].bar(x + offset, [r["adapt_chain_success_mean"] for r in rows], width, label="safe set" if safe else "unconstrained", alpha=0.85)
        axes[1].bar(x + offset, [r["adapt_prediction_error_mean"] for r in rows], width, alpha=0.85)
        axes[2].bar(x + offset, [r["source_chain_retention_mean"] for r in rows], width, alpha=0.85)
    short = ["Replace", "Residual", "Memory", "Local"]
    for ax in axes:
        ax.set_xticks(x, short)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set(title="Final adaptation", ylabel="chain success", ylim=(0, 1))
    axes[1].set(title="Model accuracy", ylabel="prediction error")
    axes[2].set(title="Prior retention", ylabel="source chain success delta")
    axes[0].legend(fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_ablations(rows: list[dict[str, Any]], out: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), constrained_layout=True)
    specs = (("memory_width", "Memory width"), ("regularization", "Local regularization"), ("safe_set", "Safe-set constraint"))
    for ax, (kind, title) in zip(axes, specs):
        subset = [r for r in rows if r["ablation"] == kind]
        values = sorted({r["value"] for r in subset})
        success = [np.mean([r["chain_success"] for r in subset if r["value"] == v]) for v in values]
        violation = [np.mean([r["violation_rate"] for r in subset if r["value"] == v]) for v in values]
        ax.plot(range(len(values)), success, "o-", color="#54a24b", label="chain success")
        ax.plot(range(len(values)), violation, "s--", color="#e45756", label="violation rate")
        labels = ["off" if kind == "safe_set" and v == 0 else "on" if kind == "safe_set" else f"{v:g}" for v in values]
        ax.set_xticks(range(len(values)), labels)
        ax.set(title=title, ylim=(-0.02, 1.02))
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run one seed and a 40-interaction budget.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    cfg = Config()
    if args.smoke:
        cfg = Config(
            interactions=40,
            eval_every=10,
            seeds=(41,),
            train_chains_per_checkpoint=12,
            eval_chains=20,
            ablation_interactions=30,
        )
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    conditions, curves, sanity = run_main(cfg)
    ablations = run_ablations(cfg)
    if not sanity["all_passed"]:
        failed = [name for name, ok in sanity["checks"].items() if not ok]
        raise RuntimeError(f"sanity checks failed: {failed}")

    metric_keys = (
        "interaction_auc", "interactions_to_70", "adapt_chain_success", "adapt_catch_success",
        "adapt_control_error", "adapt_prediction_error", "adapt_transition_margin",
        "adapt_p10_transition_margin", "adapt_unsafe_command_rate", "adapt_raw_unsafe_proposal_rate", "adapt_violation_rate",
        "adapt_filter_intervention_rate", "source_chain_success", "source_control_error",
        "ood_chain_success", "ood_control_error", "ood_prediction_error",
        "source_chain_retention", "source_control_retention",
    )
    summary = aggregate(conditions, ("method", "safe_set"), metric_keys)
    ablation_summary = aggregate(
        ablations,
        ("ablation", "value", "safe_set"),
        ("interaction_auc", "chain_success", "control_error", "prediction_error", "transition_margin", "violation_rate", "filter_intervention_rate"),
    )

    write_csv(out / "condition_metrics.csv", conditions)
    write_csv(out / "learning_curve.csv", curves)
    write_csv(out / "summary_metrics.csv", summary)
    write_csv(out / "ablation_metrics.csv", ablations)
    write_csv(out / "ablation_summary.csv", ablation_summary)
    (out / "sanity_check.json").write_text(json.dumps(sanity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = {
        "scope": "Adaptation/safety mechanism probe; not juggling, a VLA, or paper reproduction.",
        "config": asdict(cfg),
        "methods": list(METHODS),
        "safe_set": "Predicted catch ellipse, command bound, and one-step recovery-command bound; candidate projected toward prior action.",
        "sanity": sanity,
        "summary": summary,
        "ablation_summary": ablation_summary,
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_learning(curves, out / "learning_curves.png")
    plot_tradeoffs(summary, out / "method_tradeoffs.png")
    plot_ablations(ablations, out / "ablations.png")
    print(f"wrote deterministic outputs to {out}")
    print(f"sanity: {len(sanity['checks'])}/{len(sanity['checks'])} passed")


if __name__ == "__main__":
    main()
