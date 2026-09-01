#!/usr/bin/env python3
"""Deterministic causal-interaction-memory toy inspired by Zeva.

This package is a mechanism probe, not a reproduction of Zeva, not a robot
benchmark, and not evidence that a causal model over hidden physics has been
identified. The toy creates related manipulation tasks whose hidden mass,
friction, and joint-stiffness properties induce correlated phase-specific force
corrections. Methods differ only in how they store, retrieve, consolidate, and
update interaction memory across related tasks and retries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PHASES = ("lift", "slide", "turn")
PHASE_LABELS = {"lift": "Lift", "slide": "Slide", "turn": "Turn"}
METHODS = (
    "raw_history",
    "transition_retrieval",
    "phase_dual_memory",
    "online_finetune",
)
METHOD_LABELS = {
    "raw_history": "Raw history",
    "transition_retrieval": "Transition retrieval",
    "phase_dual_memory": "Phase-aware dual memory",
    "online_finetune": "Online fine-tune",
}
METHOD_COLORS = {
    "raw_history": "#7f8c8d",
    "transition_retrieval": "#4c78a8",
    "phase_dual_memory": "#59a14f",
    "online_finetune": "#e15759",
}
ABLATION_LABELS = {
    "full": "full dual memory",
    "no_fast": "no fast retry memory",
    "no_slow": "no slow consolidation",
    "no_cross": "no cross-phase retrieval",
    "no_phase": "no phase-aware prior",
}
PHASE_CONTEXT_INDEX = {
    0: np.array([0, 2], dtype=int),
    1: np.array([0, 1, 2], dtype=int),
    2: np.array([2, 3], dtype=int),
}
BASE_TOLERANCES = np.array([0.060, 0.065, 0.070], dtype=float)
CORRECTION_MIX = np.array(
    [
        [0.48, 0.14, 0.06],
        [0.22, 0.54, 0.10],
        [0.08, 0.18, 0.62],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class Config:
    seed: int = 29
    trials: int = 240
    support_tasks: int = 4
    max_attempts: int = 4
    sample_efficiency_trials: int = 160
    robustness_trials: int = 160
    ablation_trials: int = 180
    support_sweep: tuple[int, ...] = (0, 2, 4, 8, 12)
    robustness_levels: tuple[float, ...] = (0.85, 1.0, 1.15, 1.30, 1.50)
    main_severities: tuple[float, ...] = (0.90, 1.00, 1.20, 1.45)
    support_severity: float = 1.0
    hidden_scale: float = 0.50
    hidden_task_std: float = 0.08
    support_context_low: float = 0.12
    support_context_high: float = 0.88
    support_context_shift: float = 0.12
    observation_noise_scale: float = 0.0
    fine_tune_ridge: float = 0.18
    k_context: int = 4
    k_transition: int = 3


@dataclass
class InteractionTask:
    trial: int
    split: str
    severity: float
    context: np.ndarray
    hidden: np.ndarray
    correction: np.ndarray


@dataclass
class AttemptOutcome:
    success: bool
    completed_phases: int
    transitions: List[Dict[str, object]]
    attempt_phase_error: float


class BaseMethod:
    name: str = "base"

    def __init__(self, support_tasks: Sequence[InteractionTask], config: Config) -> None:
        self.support_tasks = list(support_tasks)
        self.config = config

    def predict(self, task: InteractionTask) -> np.ndarray:
        raise NotImplementedError

    def update(self, task: InteractionTask, outcome: AttemptOutcome) -> None:
        del task, outcome

    def update_count(self) -> int:
        return 0


class RawHistoryMethod(BaseMethod):
    name = "raw_history"

    def __init__(self, support_tasks: Sequence[InteractionTask], config: Config) -> None:
        super().__init__(support_tasks, config)
        self.prefix_estimates: List[tuple[int, float]] = []

    def predict(self, task: InteractionTask) -> np.ndarray:
        if not self.support_tasks:
            base = np.zeros(3, dtype=float)
        else:
            best_task = self.support_tasks[0]
            best_distance = float("inf")
            for support in self.support_tasks:
                distance = float(np.linalg.norm(task.context - support.context))
                if self.prefix_estimates:
                    current = np.array([value for _, value in self.prefix_estimates], dtype=float)
                    matched = np.array([support.correction[phase] for phase, _ in self.prefix_estimates], dtype=float)
                    distance += 1.0 * float(np.linalg.norm(current - matched))
                if distance < best_distance:
                    best_distance = distance
                    best_task = support
            base = best_task.correction.copy()
        if self.prefix_estimates:
            shared_bias = float(np.mean([value for _, value in self.prefix_estimates]))
            base = 0.65 * base + 0.35 * shared_bias
        return np.clip(base, -0.70, 0.70)

    def update(self, task: InteractionTask, outcome: AttemptOutcome) -> None:
        del task
        for row in outcome.transitions:
            self.prefix_estimates.append((int(row["phase_index"]), float(row["estimate"])))
        self.prefix_estimates = self.prefix_estimates[-6:]


class TransitionRetrievalMethod(BaseMethod):
    name = "transition_retrieval"

    def __init__(self, support_tasks: Sequence[InteractionTask], config: Config) -> None:
        super().__init__(support_tasks, config)
        self.observed_phase_estimates: Dict[int, float] = {}

    def _context_prior(self, task: InteractionTask, phase_index: int) -> float:
        rows = []
        phase_dims = PHASE_CONTEXT_INDEX[phase_index]
        for support in self.support_tasks:
            distance = float(np.linalg.norm(support.context[phase_dims] - task.context[phase_dims]))
            rows.append((distance, float(support.correction[phase_index])))
        if not rows:
            return 0.0
        rows.sort(key=lambda item: item[0])
        rows = rows[: self.config.k_transition]
        weights = np.array([1.0 / (distance + 0.08) ** 2 for distance, _ in rows], dtype=float)
        values = np.array([value for _, value in rows], dtype=float)
        return float(np.dot(weights, values) / np.sum(weights))

    def predict(self, task: InteractionTask) -> np.ndarray:
        prediction = np.zeros(3, dtype=float)
        for phase_index in range(3):
            if phase_index in self.observed_phase_estimates:
                prediction[phase_index] = self.observed_phase_estimates[phase_index]
            else:
                prediction[phase_index] = self._context_prior(task, phase_index)
        return np.clip(prediction, -0.70, 0.70)

    def update(self, task: InteractionTask, outcome: AttemptOutcome) -> None:
        del task
        for row in outcome.transitions:
            phase_index = int(row["phase_index"])
            previous = self.observed_phase_estimates.get(phase_index, 0.0)
            self.observed_phase_estimates[phase_index] = 0.35 * previous + 0.65 * float(row["estimate"])


class PhaseDualMemoryMethod(BaseMethod):
    name = "phase_dual_memory"

    def __init__(
        self,
        support_tasks: Sequence[InteractionTask],
        config: Config,
        *,
        use_fast: bool = True,
        use_slow: bool = True,
        use_cross_phase: bool = True,
        use_phase_aware: bool = True,
    ) -> None:
        super().__init__(support_tasks, config)
        self.use_fast = use_fast
        self.use_slow = use_slow
        self.use_cross_phase = use_cross_phase
        self.use_phase_aware = use_phase_aware
        self.fast_cache: Dict[int, float] = {}
        if self.support_tasks:
            correction_matrix = np.stack([task.correction for task in self.support_tasks], axis=0)
            self.global_mean = np.mean(correction_matrix, axis=0)
            cov = np.cov(correction_matrix.T) if len(correction_matrix) > 1 else np.eye(3) * 0.05
            self.global_cov = np.asarray(cov, dtype=float) + np.eye(3) * 0.012
        else:
            self.global_mean = np.zeros(3, dtype=float)
            self.global_cov = np.eye(3, dtype=float) * 0.05

    def _phase_prior(self, task: InteractionTask) -> np.ndarray:
        prior = np.zeros(3, dtype=float)
        if not self.support_tasks:
            return prior
        for phase_index in range(3):
            rows = []
            dims = PHASE_CONTEXT_INDEX[phase_index] if self.use_phase_aware else np.arange(4)
            for support in self.support_tasks:
                distance = float(np.linalg.norm(support.context[dims] - task.context[dims]))
                if not self.use_phase_aware:
                    target_value = float(np.mean(support.correction))
                else:
                    target_value = float(support.correction[phase_index])
                rows.append((distance, target_value))
            rows.sort(key=lambda item: item[0])
            rows = rows[: self.config.k_context]
            weights = np.array([1.0 / (distance + 0.06) ** 2 for distance, _ in rows], dtype=float)
            values = np.array([value for _, value in rows], dtype=float)
            prior[phase_index] = float(np.dot(weights, values) / np.sum(weights))
        return prior

    def predict(self, task: InteractionTask) -> np.ndarray:
        phase_prior = self._phase_prior(task)
        if self.use_slow:
            mean = 0.58 * phase_prior + 0.42 * self.global_mean
            base_cov = self.global_cov
        else:
            mean = np.zeros(3, dtype=float)
            base_cov = np.eye(3, dtype=float) * 0.05
        mean = mean.copy()
        if not self.use_fast or not self.fast_cache:
            return np.clip(mean, -0.70, 0.70)
        observed_indices = np.array(sorted(self.fast_cache), dtype=int)
        observed_values = np.array([self.fast_cache[index] for index in observed_indices], dtype=float)
        cov = base_cov if self.use_cross_phase else np.diag(np.diag(base_cov))
        innovation_cov = cov[np.ix_(observed_indices, observed_indices)] + np.eye(len(observed_indices)) * 0.02
        kalman_gain = cov[:, observed_indices] @ np.linalg.inv(innovation_cov)
        posterior = mean + kalman_gain @ (observed_values - mean[observed_indices])
        for phase_index, value in self.fast_cache.items():
            posterior[phase_index] = 0.72 * value + 0.28 * posterior[phase_index]
        return np.clip(posterior, -0.70, 0.70)

    def update(self, task: InteractionTask, outcome: AttemptOutcome) -> None:
        del task
        if not self.use_fast:
            return
        for row in outcome.transitions:
            phase_index = int(row["phase_index"])
            estimate = float(row["estimate"])
            if phase_index in self.fast_cache:
                self.fast_cache[phase_index] = 0.45 * self.fast_cache[phase_index] + 0.55 * estimate
            else:
                self.fast_cache[phase_index] = estimate


class OnlineFineTuneMethod(BaseMethod):
    name = "online_finetune"

    def __init__(self, support_tasks: Sequence[InteractionTask], config: Config) -> None:
        super().__init__(support_tasks, config)
        self.current_observations: Dict[int, float] = {}
        self.refits = 0
        self.design_matrix: List[np.ndarray] = []
        self.targets: List[float] = []
        self.feature_dim = len(self._feature_vector(0, np.zeros(4, dtype=float), {}))
        for support in self.support_tasks:
            correction = support.correction
            partial_histories = [
                {},
                {0: float(correction[0])},
                {0: float(correction[0]), 1: float(correction[1])},
                {1: float(correction[1])},
                {2: float(correction[2])},
            ]
            for partial in partial_histories:
                for phase_index in range(3):
                    self.design_matrix.append(self._feature_vector(phase_index, support.context, partial))
                    self.targets.append(float(correction[phase_index]))
        self._refit()

    def _feature_vector(
        self,
        phase_index: int,
        context: np.ndarray,
        observations: Mapping[int, float],
    ) -> np.ndarray:
        mask = np.array([1.0 if phase in observations else 0.0 for phase in range(3)], dtype=float)
        observed_values = np.array([observations.get(phase, 0.0) for phase in range(3)], dtype=float)
        features: List[float] = []
        features.extend(np.eye(3)[phase_index].tolist())
        features.extend(context.tolist())
        features.extend(observed_values.tolist())
        features.extend(mask.tolist())
        features.extend([context[0] * mask[0], context[1] * mask[1], context[3] * mask[2], context[2] * np.sum(mask)])
        features.extend([0.6 * observed_values[0], observed_values[1], 1.2 * observed_values[2]])
        return np.asarray(features, dtype=float)

    def _refit(self) -> None:
        if self.design_matrix:
            x = np.stack(self.design_matrix, axis=0)
            y = np.asarray(self.targets, dtype=float)
            ridge = self.config.fine_tune_ridge * np.eye(self.feature_dim)
            self.weights = np.linalg.solve(x.T @ x + ridge, x.T @ y)
        else:
            self.weights = np.zeros(self.feature_dim, dtype=float)
        self.refits += 1

    def predict(self, task: InteractionTask) -> np.ndarray:
        prediction = []
        for phase_index in range(3):
            feature = self._feature_vector(phase_index, task.context, self.current_observations)
            prediction.append(float(feature @ self.weights))
        return np.clip(np.asarray(prediction, dtype=float), -0.70, 0.70)

    def update(self, task: InteractionTask, outcome: AttemptOutcome) -> None:
        for row in outcome.transitions:
            self.current_observations[int(row["phase_index"])] = float(row["estimate"])
        for row in outcome.transitions:
            phase_index = int(row["phase_index"])
            estimate = float(row["estimate"])
            self.design_matrix.append(self._feature_vector(phase_index, task.context, self.current_observations))
            self.targets.append(estimate)
        self._refit()

    def update_count(self) -> int:
        return self.refits


def method_factory(name: str, support_tasks: Sequence[InteractionTask], config: Config) -> BaseMethod:
    if name == "raw_history":
        return RawHistoryMethod(support_tasks, config)
    if name == "transition_retrieval":
        return TransitionRetrievalMethod(support_tasks, config)
    if name == "phase_dual_memory":
        return PhaseDualMemoryMethod(support_tasks, config)
    if name == "online_finetune":
        return OnlineFineTuneMethod(support_tasks, config)
    raise ValueError(name)


def ablation_factory(setting: str, support_tasks: Sequence[InteractionTask], config: Config) -> BaseMethod:
    if setting == "full":
        return PhaseDualMemoryMethod(support_tasks, config)
    if setting == "no_fast":
        return PhaseDualMemoryMethod(support_tasks, config, use_fast=False)
    if setting == "no_slow":
        return PhaseDualMemoryMethod(support_tasks, config, use_slow=False)
    if setting == "no_cross":
        return PhaseDualMemoryMethod(support_tasks, config, use_cross_phase=False)
    if setting == "no_phase":
        return PhaseDualMemoryMethod(support_tasks, config, use_phase_aware=False)
    raise ValueError(setting)


def correction_from_hidden(hidden: np.ndarray, context: np.ndarray) -> np.ndarray:
    payload, travel, precision, lever = context
    correction = CORRECTION_MIX @ hidden
    correction[0] += 0.12 * (payload - 0.5) * hidden[0] + 0.05 * (precision - 0.5) * hidden[1]
    correction[1] += 0.10 * (travel - 0.5) * hidden[1] + 0.08 * (payload - 0.5) * hidden[0]
    correction[2] += 0.12 * (lever - 0.5) * hidden[2] + 0.06 * (precision - 0.5) * hidden[1]
    return np.clip(correction, -0.70, 0.70)


def generate_family_hidden(rng: np.random.Generator, config: Config) -> np.ndarray:
    return rng.uniform(-config.hidden_scale, config.hidden_scale, size=3)


def generate_task(
    rng: np.random.Generator,
    config: Config,
    family_hidden: np.ndarray,
    trial: int,
    split: str,
    severity: float,
) -> InteractionTask:
    context = rng.uniform(config.support_context_low, config.support_context_high, size=4)
    if severity > 1.0:
        context = np.clip(
            context + rng.normal(0.0, config.support_context_shift * (severity - 1.0), size=4),
            0.0,
            1.0,
        )
    hidden = np.clip(
        family_hidden + rng.normal(0.0, config.hidden_task_std * severity, size=3),
        -0.65,
        0.65,
    )
    correction = correction_from_hidden(hidden, context)
    return InteractionTask(trial=trial, split=split, severity=severity, context=context, hidden=hidden, correction=correction)


def observed_correction_estimate(task: InteractionTask, phase_index: int, predicted: float, rng: np.random.Generator, config: Config) -> float:
    true_correction = float(task.correction[phase_index])
    error = predicted - true_correction
    phase_context_dim = PHASE_CONTEXT_INDEX[phase_index][0]
    bias = (0.26 + 0.04 * phase_index) * error
    bias += 0.03 * (float(task.context[phase_context_dim]) - 0.5)
    bias += 0.02 * math.tanh(1.8 * error)
    if config.observation_noise_scale > 0.0:
        bias += float(rng.normal(0.0, config.observation_noise_scale))
    return float(np.clip(true_correction + bias, -0.70, 0.70))


def execute_attempt(
    task: InteractionTask,
    predicted_correction: np.ndarray,
    config: Config,
    rng: np.random.Generator,
) -> AttemptOutcome:
    tolerances = BASE_TOLERANCES / math.sqrt(task.severity)
    transitions: List[Dict[str, object]] = []
    completed_phases = 0
    executed_errors: List[float] = []
    for phase_index, phase_name in enumerate(PHASES):
        true_correction = float(task.correction[phase_index])
        predicted = float(predicted_correction[phase_index])
        error = predicted - true_correction
        progress = 1.0 + 1.35 * error if error <= 0.0 else 1.0 + 0.95 * error - 1.10 * error * error
        drift = max(0.0, error - 0.02) * 2.60
        stall = max(0.0, -error - 0.02) * 2.30
        impedance = 0.70 + 0.50 * true_correction + 0.05 * phase_index + 0.06 * (float(task.context[0]) - 0.5)
        estimate = observed_correction_estimate(task, phase_index, predicted, rng, config)
        success = abs(error) <= tolerances[phase_index] and drift < 0.16 and stall < 0.16
        transitions.append(
            {
                "phase": phase_name,
                "phase_index": phase_index,
                "predicted": predicted,
                "target": true_correction,
                "estimate": estimate,
                "progress": progress,
                "drift": drift,
                "stall": stall,
                "impedance": impedance,
                "error": error,
                "success": int(success),
            }
        )
        executed_errors.append(abs(error))
        if success:
            completed_phases += 1
        else:
            break
    return AttemptOutcome(
        success=completed_phases == len(PHASES),
        completed_phases=completed_phases,
        transitions=transitions,
        attempt_phase_error=float(np.mean(executed_errors)) if executed_errors else float("nan"),
    )


def split_for_trial(index: int) -> tuple[str, float]:
    mapping = {
        0: ("near", 0.90),
        1: ("matched", 1.00),
        2: ("shifted", 1.20),
        3: ("robust", 1.45),
    }
    return mapping[index % 4]


def evaluate_single_method(
    method: BaseMethod,
    task: InteractionTask,
    config: Config,
    seed_offset: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed_offset)
    solved = False
    first_attempt_success = 0
    attempt_rows: List[AttemptOutcome] = []
    solve_attempt = config.max_attempts + 1
    best_completed_phases = 0
    for attempt_index in range(config.max_attempts):
        prediction = method.predict(task)
        outcome = execute_attempt(task, prediction, config, rng)
        attempt_rows.append(outcome)
        best_completed_phases = max(best_completed_phases, outcome.completed_phases)
        if attempt_index == 0:
            first_attempt_success = int(outcome.success)
        if outcome.success:
            solved = True
            solve_attempt = attempt_index + 1
            break
        method.update(task, outcome)
    retry_curve = [0] * config.max_attempts
    if solved:
        for index in range(solve_attempt - 1, config.max_attempts):
            retry_curve[index] = 1
    attempt1_error = attempt_rows[0].attempt_phase_error if attempt_rows else float("nan")
    final_error = attempt_rows[-1].attempt_phase_error if attempt_rows else float("nan")
    return {
        "success": int(solved),
        "first_attempt_success": first_attempt_success,
        "solve_attempt": solve_attempt,
        "best_completed_phases": best_completed_phases,
        "attempt1_completed_phases": attempt_rows[0].completed_phases if attempt_rows else 0,
        "attempt1_phase_error": attempt1_error,
        "final_attempt_phase_error": final_error,
        "retry_gain": int(solved) - first_attempt_success,
        "retry_curve": retry_curve,
        "updates": method.update_count(),
    }


def run_main_eval(config: Config, methods: Sequence[str] = METHODS) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    master_rng = np.random.default_rng(config.seed)
    for trial in range(config.trials):
        split, severity = split_for_trial(trial)
        family_hidden = generate_family_hidden(master_rng, config)
        support_tasks = [
            generate_task(master_rng, config, family_hidden, trial, "support", config.support_severity)
            for _ in range(config.support_tasks)
        ]
        target_task = generate_task(master_rng, config, family_hidden, trial, split, severity)
        for method_index, method_name in enumerate(methods):
            method = method_factory(method_name, support_tasks, config)
            metrics = evaluate_single_method(method, target_task, config, config.seed + 10000 * trial + 101 * method_index)
            row = {
                "trial": trial,
                "split": split,
                "severity": severity,
                "support_tasks": config.support_tasks,
                "method": method_name,
                **{key: value for key, value in metrics.items() if key != "retry_curve"},
            }
            for attempt_index, solved_by_attempt in enumerate(metrics["retry_curve"], start=1):
                row[f"solved_by_attempt_{attempt_index}"] = solved_by_attempt
            rows.append(row)
    return rows


def run_sample_efficiency(config: Config) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for support_tasks in config.support_sweep:
        sweep_rng = np.random.default_rng(config.seed + 5000 + 97 * support_tasks)
        for method_index, method_name in enumerate(METHODS):
            per_trial: List[Dict[str, object]] = []
            for trial in range(config.sample_efficiency_trials):
                family_hidden = generate_family_hidden(sweep_rng, config)
                support = [
                    generate_task(sweep_rng, config, family_hidden, trial, "support", config.support_severity)
                    for _ in range(support_tasks)
                ]
                target = generate_task(sweep_rng, config, family_hidden, trial, "sample_efficiency", 1.0)
                method = method_factory(method_name, support, config)
                metrics = evaluate_single_method(method, target, config, config.seed + 200000 + 1000 * support_tasks + 97 * trial + method_index)
                per_trial.append(metrics)
            rows.append(
                {
                    "support_tasks": support_tasks,
                    "method": method_name,
                    "episodes": len(per_trial),
                    "success_rate": nanmean(float(metrics["success"]) for metrics in per_trial),
                    "transfer_rate": nanmean(float(metrics["first_attempt_success"]) for metrics in per_trial),
                    "mean_attempts": nanmean(float(metrics["solve_attempt"]) for metrics in per_trial),
                    "retry_gain": nanmean(float(metrics["retry_gain"]) for metrics in per_trial),
                }
            )
    return rows


def run_robustness(config: Config) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for severity in config.robustness_levels:
        sweep_rng = np.random.default_rng(config.seed + 9000 + int(100 * severity))
        for method_index, method_name in enumerate(METHODS):
            per_trial: List[Dict[str, object]] = []
            for trial in range(config.robustness_trials):
                family_hidden = generate_family_hidden(sweep_rng, config)
                support = [
                    generate_task(sweep_rng, config, family_hidden, trial, "support", config.support_severity)
                    for _ in range(config.support_tasks)
                ]
                target = generate_task(sweep_rng, config, family_hidden, trial, "robustness", severity)
                method = method_factory(method_name, support, config)
                metrics = evaluate_single_method(method, target, config, config.seed + 400000 + 97 * trial + method_index + int(100 * severity))
                per_trial.append(metrics)
            rows.append(
                {
                    "severity": severity,
                    "method": method_name,
                    "episodes": len(per_trial),
                    "success_rate": nanmean(float(metrics["success"]) for metrics in per_trial),
                    "transfer_rate": nanmean(float(metrics["first_attempt_success"]) for metrics in per_trial),
                    "mean_attempts": nanmean(float(metrics["solve_attempt"]) for metrics in per_trial),
                }
            )
    return rows


def run_ablations(config: Config) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    settings = ("full", "no_fast", "no_slow", "no_cross", "no_phase")
    ablation_rng = np.random.default_rng(config.seed + 13000)
    tasks: List[tuple[list[InteractionTask], InteractionTask]] = []
    for trial in range(config.ablation_trials):
        split, severity = split_for_trial(trial)
        family_hidden = generate_family_hidden(ablation_rng, config)
        support = [
            generate_task(ablation_rng, config, family_hidden, trial, "support", config.support_severity)
            for _ in range(config.support_tasks)
        ]
        target = generate_task(ablation_rng, config, family_hidden, trial, split, severity)
        tasks.append((support, target))
    for setting_index, setting in enumerate(settings):
        per_trial: List[Dict[str, object]] = []
        for trial, (support, target) in enumerate(tasks):
            method = ablation_factory(setting, support, config)
            metrics = evaluate_single_method(method, target, config, config.seed + 600000 + 997 * trial + setting_index)
            per_trial.append(metrics)
        rows.append(
            {
                "setting": setting,
                "episodes": len(per_trial),
                "success_rate": nanmean(float(metrics["success"]) for metrics in per_trial),
                "transfer_rate": nanmean(float(metrics["first_attempt_success"]) for metrics in per_trial),
                "attempt2_rate": nanmean(float(metrics["retry_curve"][1]) for metrics in per_trial),
                "attempt3_rate": nanmean(float(metrics["retry_curve"][2]) for metrics in per_trial),
                "mean_attempts": nanmean(float(metrics["solve_attempt"]) for metrics in per_trial),
            }
        )
    return rows


def run_sanity_checks(config: Config) -> Dict[str, object]:
    checks: Dict[str, object] = {}
    rng = np.random.default_rng(config.seed + 17000)
    family_hidden = generate_family_hidden(rng, config)
    support = [generate_task(rng, config, family_hidden, 0, "support", config.support_severity) for _ in range(6)]
    target = generate_task(rng, config, family_hidden, 0, "matched", 1.0)

    dual = PhaseDualMemoryMethod(support, config)
    prior = dual.predict(target)
    first_outcome = execute_attempt(target, prior, config, np.random.default_rng(config.seed + 17001))
    dual.update(target, first_outcome)
    posterior = dual.predict(target)
    checks["dual_memory_cross_phase_update"] = bool(
        np.mean(np.abs(posterior - target.correction)) < np.mean(np.abs(prior - target.correction))
    )

    retrieval = TransitionRetrievalMethod(support, config)
    retrieval_prior = retrieval.predict(target)
    retrieval_outcome = execute_attempt(target, retrieval_prior, config, np.random.default_rng(config.seed + 17002))
    retrieval.update(target, retrieval_outcome)
    retrieval_posterior = retrieval.predict(target)
    first_failed_phase = retrieval_outcome.transitions[-1]["phase_index"]
    checks["transition_retrieval_improves_seen_phase"] = bool(
        abs(retrieval_posterior[first_failed_phase] - target.correction[first_failed_phase])
        < abs(retrieval_prior[first_failed_phase] - target.correction[first_failed_phase])
    )

    zero_support_cfg = replace_config(config, support_tasks=0, trials=120)
    four_support_cfg = replace_config(config, support_tasks=4, trials=120)
    zero_rows = run_main_eval(zero_support_cfg, methods=["phase_dual_memory"])
    four_rows = run_main_eval(four_support_cfg, methods=["phase_dual_memory"])
    zero_rate = nanmean(float(row["success"]) for row in zero_rows)
    four_rate = nanmean(float(row["success"]) for row in four_rows)
    checks["support_memory_improves_dual_success"] = bool(four_rate > zero_rate + 0.20)

    main_once = run_main_eval(replace_config(config, trials=40), methods=["phase_dual_memory"])
    main_twice = run_main_eval(replace_config(config, trials=40), methods=["phase_dual_memory"])
    checks["deterministic_repeat"] = bool(main_once == main_twice)

    rows = run_main_eval(replace_config(config, trials=120))
    overall = summarize(rows)
    by_method = {row["method"]: row for row in overall if row["split"] == "all"}
    checks["dual_best_overall_retry_success"] = bool(
        by_method["phase_dual_memory"]["success_rate"] > by_method["transition_retrieval"]["success_rate"]
        > by_method["raw_history"]["success_rate"]
    )

    hard_rows = [row for row in rows if row["split"] == "robust"]
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in hard_rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    checks["dual_most_robust_on_hard_split"] = bool(
        nanmean(float(row["success"]) for row in grouped["phase_dual_memory"])
        > nanmean(float(row["success"]) for row in grouped["online_finetune"])
    )

    checks["all_pass"] = bool(all(value for key, value in checks.items() if key != "all_pass"))
    return checks


def replace_config(config: Config, **updates: object) -> Config:
    values = config.__dict__.copy()
    values.update(updates)
    return Config(**values)


def nanmean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.nanmean(array)) if np.any(np.isfinite(array)) else float("nan")


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    for method_name in METHODS:
        method_rows = [row for row in rows if row["method"] == method_name]
        for split in ("all", "near", "matched", "shifted", "robust"):
            selected = method_rows if split == "all" else [row for row in method_rows if row["split"] == split]
            if not selected:
                continue
            summary_rows.append(
                {
                    "method": method_name,
                    "split": split,
                    "episodes": len(selected),
                    "success_rate": nanmean(float(row["success"]) for row in selected),
                    "transfer_rate": nanmean(float(row["first_attempt_success"]) for row in selected),
                    "retry_gain": nanmean(float(row["retry_gain"]) for row in selected),
                    "mean_attempts": nanmean(float(row["solve_attempt"]) for row in selected),
                    "attempt1_completed_phases": nanmean(float(row["attempt1_completed_phases"]) for row in selected),
                    "best_completed_phases": nanmean(float(row["best_completed_phases"]) for row in selected),
                    "attempt1_phase_error": nanmean(float(row["attempt1_phase_error"]) for row in selected),
                    "final_attempt_phase_error": nanmean(float(row["final_attempt_phase_error"]) for row in selected),
                }
            )
    return summary_rows


def retry_curve(rows: Sequence[Dict[str, object]], max_attempts: int) -> List[Dict[str, object]]:
    curve_rows: List[Dict[str, object]] = []
    for method_name in METHODS:
        method_rows = [row for row in rows if row["method"] == method_name]
        for attempt in range(1, max_attempts + 1):
            curve_rows.append(
                {
                    "method": method_name,
                    "attempt": attempt,
                    "solved_rate": nanmean(float(row[f"solved_by_attempt_{attempt}"]) for row in method_rows),
                }
            )
    return curve_rows


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return None if not math.isfinite(numeric) else numeric
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_headline(summary_rows: Sequence[Dict[str, object]], output: Path) -> None:
    overall = {row["method"]: row for row in summary_rows if row["split"] == "all"}
    x = np.arange(len(METHODS))
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))
    panels = [
        ("transfer_rate", "First-attempt transfer", (0.0, 1.0)),
        ("success_rate", "Retry success within 4 attempts", (0.0, 1.0)),
        ("retry_gain", "Retry gain over first attempt", (0.0, 1.0)),
        ("mean_attempts", "Attempts to solve (lower)", None),
    ]
    for axis, (key, title, ylim) in zip(axes.flat, panels):
        values = [float(overall[method][key]) for method in METHODS]
        axis.bar(x, values, color=[METHOD_COLORS[method] for method in METHODS])
        axis.set_xticks(x, [METHOD_LABELS[method] for method in METHODS], rotation=18, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if ylim is not None:
            axis.set_ylim(*ylim)
    fig.suptitle("Causal-interaction-memory toy: headline transfer and retry metrics", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_retry_curves(curve_rows: Sequence[Dict[str, object]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.6, 4.8))
    for method in METHODS:
        selected = [row for row in curve_rows if row["method"] == method]
        axis.plot(
            [int(row["attempt"]) for row in selected],
            [float(row["solved_rate"]) for row in selected],
            marker="o",
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.set_xlabel("Retry budget")
    axis.set_ylabel("Solved fraction")
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_sample_efficiency(rows: Sequence[Dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        supports = [int(row["support_tasks"]) for row in selected]
        axes[0].plot(supports, [float(row["success_rate"]) for row in selected], marker="o", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axes[1].plot(supports, [float(row["transfer_rate"]) for row in selected], marker="o", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    axes[0].set_title("Final success vs. related-task memory")
    axes[1].set_title("First-attempt transfer vs. related-task memory")
    for axis in axes:
        axis.set_xlabel("Support tasks per family")
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Rate")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_robustness(rows: Sequence[Dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        severities = [float(row["severity"]) for row in selected]
        axes[0].plot(severities, [float(row["success_rate"]) for row in selected], marker="o", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axes[1].plot(severities, [float(row["transfer_rate"]) for row in selected], marker="o", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    axes[0].set_title("Retry success under stronger hidden shifts")
    axes[1].set_title("Transfer under stronger hidden shifts")
    for axis in axes:
        axis.set_xlabel("Severity multiplier")
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Rate")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_ablations(rows: Sequence[Dict[str, object]], output: Path) -> None:
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    axes[0].bar(x, [float(row["success_rate"]) for row in rows], color="#59a14f")
    axes[0].set_xticks(x, [ABLATION_LABELS[str(row["setting"])] for row in rows], rotation=20, ha="right")
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_title("Memory ablations: final retry success")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - 0.18, [float(row["transfer_rate"]) for row in rows], width=0.36, color="#4c78a8", label="transfer")
    axes[1].bar(x + 0.18, [float(row["attempt2_rate"]) for row in rows], width=0.36, color="#e15759", label="solved by attempt 2")
    axes[1].set_xticks(x, [ABLATION_LABELS[str(row["setting"])] for row in rows], rotation=20, ha="right")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Memory ablations: transfer and fast adaptation")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--trials", type=int, default=240)
    parser.add_argument("--support-tasks", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    config = Config(seed=args.seed, trials=args.trials, support_tasks=args.support_tasks, max_attempts=args.max_attempts)
    if args.smoke:
        config = replace_config(
            config,
            trials=min(config.trials, 40),
            support_tasks=min(config.support_tasks, 3),
            sample_efficiency_trials=48,
            robustness_trials=48,
            ablation_trials=64,
            support_sweep=(0, 2, 4),
            robustness_levels=(0.85, 1.15, 1.50),
        )

    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    main_rows = run_main_eval(config)
    summary_rows = summarize(main_rows)
    retry_rows = retry_curve(main_rows, config.max_attempts)
    sample_rows = run_sample_efficiency(config)
    robustness_rows = run_robustness(config)
    ablation_rows = run_ablations(config)
    sanity = run_sanity_checks(config)
    if not sanity["all_pass"]:
        raise RuntimeError(f"sanity checks failed: {sanity}")

    overall = {row["method"]: row for row in summary_rows if row["split"] == "all"}
    claims = {
        "dual_memory_best_overall_success": bool(
            overall["phase_dual_memory"]["success_rate"]
            == max(float(row["success_rate"]) for row in overall.values())
        ),
        "dual_memory_best_robustness_at_1_50": bool(
            next(
                row
                for row in robustness_rows
                if row["method"] == "phase_dual_memory"
                and abs(float(row["severity"]) - 1.50) < 1e-9
            )["success_rate"]
            > max(
                float(row["success_rate"])
                for row in robustness_rows
                if row["method"] != "phase_dual_memory"
                and abs(float(row["severity"]) - 1.50) < 1e-9
            )
        ),
        "retrieval_beats_raw_history_on_retry_gain": bool(
            overall["transition_retrieval"]["retry_gain"] > overall["raw_history"]["retry_gain"]
        ),
        "full_dual_ablation_beats_no_cross": bool(
            next(row for row in ablation_rows if row["setting"] == "full")["success_rate"]
            > next(row for row in ablation_rows if row["setting"] == "no_cross")["success_rate"]
        ),
    }
    runtime_seconds = time.perf_counter() - start

    write_csv(args.output_dir / "trial_metrics.csv", main_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "retry_curves.csv", retry_rows)
    write_csv(args.output_dir / "sample_efficiency.csv", sample_rows)
    write_csv(args.output_dir / "robustness.csv", robustness_rows)
    write_csv(args.output_dir / "ablations.csv", ablation_rows)
    with (args.output_dir / "sanity_checks.json").open("w") as handle:
        json.dump(json_ready(sanity), handle, indent=2, sort_keys=True)

    metrics = {
        "config": json_ready(
            config.__dict__
            | {
                "output_dir": str(args.output_dir.relative_to(BASE_DIR))
                if args.output_dir.is_relative_to(BASE_DIR)
                else str(args.output_dir),
                "smoke": args.smoke,
            }
        ),
        "summary": summary_rows,
        "retry_curves": retry_rows,
        "sample_efficiency": sample_rows,
        "robustness": robustness_rows,
        "ablations": ablation_rows,
        "sanity_checks": sanity,
        "claims_supported_by_this_run": claims,
        "runtime_seconds": runtime_seconds,
    }
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(json_ready(metrics), handle, indent=2, sort_keys=True)

    plot_headline(summary_rows, args.output_dir / "headline_metrics.png")
    plot_retry_curves(retry_rows, args.output_dir / "retry_curves.png")
    plot_sample_efficiency(sample_rows, args.output_dir / "sample_efficiency.png")
    plot_robustness(robustness_rows, args.output_dir / "robustness.png")
    plot_ablations(ablation_rows, args.output_dir / "ablations.png")

    print(
        json.dumps(
            json_ready(
                {
                    "smoke": args.smoke,
                    "sanity": sanity,
                    "claims": claims,
                    "runtime_seconds": runtime_seconds,
                    "overall": [overall[method] for method in METHODS],
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
