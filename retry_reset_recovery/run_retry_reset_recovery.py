#!/usr/bin/env python3
"""Deterministic 2D retry/reset recovery-data toy inspired by FLARE.

This is a mechanism probe, not a VLA or a reproduction of the paper.  It uses
locally weighted behavior cloning over scripted demonstrations so data support
and skill organization remain inspectable.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors


METHODS = [
    "success_only_bc",
    "generic_recovery_bc",
    "perturb_bridge_bc",
    "monolithic_resets",
    "monitor_gated_reset_skills",
]
METHOD_LABELS = {
    "success_only_bc": "Success-only BC",
    "generic_recovery_bc": "Generic recovery",
    "perturb_bridge_bc": "Perturb + bridge",
    "monolithic_resets": "Monolithic resets",
    "monitor_gated_reset_skills": "Monitor-gated skills",
}
STATUS = {"ready": 0, "toppled": 1, "dropped": 2, "wedged": 3, "placed": 4}
FAILURE_TYPES = ("toppled", "dropped", "wedged")
COLORS = {
    "success_only_bc": "#7f8c8d",
    "generic_recovery_bc": "#d99058",
    "perturb_bridge_bc": "#4c78a8",
    "monolithic_resets": "#b279a2",
    "monitor_gated_reset_skills": "#59a14f",
}


@dataclass
class ObjectState:
    pos: np.ndarray
    goal: np.ndarray
    stage: np.ndarray
    status: str = "ready"


@dataclass
class World:
    robot: np.ndarray
    objects: List[ObjectState]
    held: int = -1
    active: int = 0
    step: int = 0
    compounding_failures: int = 0
    last_worsen_step: int = -99


@dataclass
class FailureEvent:
    kind: str
    object_index: int
    trigger: str
    magnitude: float = 0.0
    injected: bool = False
    injected_step: int = -1
    recovered_step: int = -1


@dataclass
class Scenario:
    trial: int
    split: str
    scene_seed: int
    stage_jitter: np.ndarray
    goal_jitter: np.ndarray
    robot_start: np.ndarray
    events: List[FailureEvent]


@dataclass
class Rollout:
    metrics: Dict[str, float | int | str]
    trajectory: List[Dict[str, object]] = field(default_factory=list)


class WeightedKNNPolicy:
    """A compact locally weighted BC policy with fixed observation scaling."""

    def __init__(self, k: int = 9, name: str = "policy") -> None:
        self.k = k
        self.name = name
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.nn: Optional[NearestNeighbors] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "WeightedKNNPolicy":
        if len(x) == 0:
            raise ValueError(f"empty dataset for {self.name}")
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.nn = NearestNeighbors(n_neighbors=min(self.k, len(self.x)), algorithm="kd_tree")
        self.nn.fit(self.x)
        return self

    def predict(self, observation: np.ndarray) -> np.ndarray:
        if self.nn is None or self.y is None:
            raise RuntimeError(f"{self.name} is not fitted")
        distances, indices = self.nn.kneighbors(observation[None, :], return_distance=True)
        d = distances[0]
        idx = indices[0]
        weights = 1.0 / (d + 0.035) ** 2
        action = np.sum(self.y[idx] * weights[:, None], axis=0) / np.sum(weights)
        action[:2] = np.clip(action[:2], -0.115, 0.115)
        action[2:] = np.clip(action[2:], -1.0, 1.0)
        return action


def clip_norm(v: np.ndarray, max_norm: float = 0.11) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= max_norm or n < 1e-12:
        return v.copy()
    return v * (max_norm / n)


def base_stage(index: int) -> np.ndarray:
    return np.array([0.17 + 0.03 * index, 0.22 + 0.28 * index], dtype=float)


def base_goal(index: int) -> np.ndarray:
    return np.array([0.82 - 0.03 * index, 0.22 + 0.28 * index], dtype=float)


def make_world(
    stage_jitter: Optional[np.ndarray] = None,
    goal_jitter: Optional[np.ndarray] = None,
    robot_start: Optional[np.ndarray] = None,
) -> World:
    sj = np.zeros((3, 2)) if stage_jitter is None else np.asarray(stage_jitter, dtype=float)
    gj = np.zeros((3, 2)) if goal_jitter is None else np.asarray(goal_jitter, dtype=float)
    objects = []
    for i in range(3):
        stage = base_stage(i) + sj[i]
        goal = base_goal(i) + gj[i]
        objects.append(ObjectState(pos=stage.copy(), goal=goal, stage=stage, status="ready"))
    start = np.array([0.50, 0.08]) if robot_start is None else np.asarray(robot_start, dtype=float)
    return World(robot=start.copy(), objects=objects)


def copy_world(world: World) -> World:
    return copy.deepcopy(world)


def active_target(world: World) -> np.ndarray:
    if world.active >= len(world.objects):
        return np.array([0.50, 0.08])
    obj = world.objects[world.active]
    return obj.goal if world.held == world.active else obj.pos


def encode_observation(world: World) -> np.ndarray:
    """Encode full state, then apply a fixed metric that emphasizes robot pose.

    This intentionally makes trajectory-monotonic pose/state correlations useful
    to nearest-neighbor BC unless the data explicitly break them.
    """
    active_onehot = np.zeros(4)
    active_onehot[min(world.active, 3)] = 1.0
    values: List[float] = [world.robot[0], world.robot[1], float(world.held >= 0), (world.held + 1) / 4.0]
    scales: List[float] = [3.2, 3.2, 1.1, 0.7]
    values.extend(active_onehot.tolist())
    scales.extend([0.75] * 4)
    for obj in world.objects:
        values.extend([obj.pos[0], obj.pos[1], obj.goal[0], obj.goal[1], STATUS[obj.status] / 4.0])
        scales.extend([0.95, 0.95, 0.45, 0.45, 0.48])
    # Relative geometry is visible but deliberately lower weighted than robot pose.
    if world.active < 3:
        obj = world.objects[world.active]
        rel_obj = obj.pos - world.robot
        rel_goal = obj.goal - world.robot
        values.extend([rel_obj[0], rel_obj[1], rel_goal[0], rel_goal[1]])
    else:
        values.extend([0.0, 0.0, 0.0, 0.0])
    scales.extend([0.62, 0.62, 0.42, 0.42])
    return np.asarray(values) * np.asarray(scales)


def task_oracle(world: World) -> np.ndarray:
    if world.active >= 3:
        return np.array([0.0, 0.0, -1.0, 0.0])
    obj = world.objects[world.active]
    if world.held == world.active:
        delta = obj.goal - world.robot
        if np.linalg.norm(delta) < 0.075:
            return np.array([0.0, 0.0, -1.0, 0.0])
        return np.r_[clip_norm(delta), 1.0, 0.0]
    delta = obj.pos - world.robot
    if np.linalg.norm(delta) < 0.072:
        return np.array([0.0, 0.0, 1.0, 0.0])
    return np.r_[clip_norm(delta), -1.0, 0.0]


def generic_recovery_oracle(world: World) -> np.ndarray:
    home = np.array([0.50, 0.08])
    delta = home - world.robot
    if np.linalg.norm(delta) > 0.075:
        return np.r_[clip_norm(delta), -1.0, 0.0]
    return task_oracle(world)


def reset_oracle(world: World, target: int) -> np.ndarray:
    obj = world.objects[target]
    if obj.status in ("ready", "placed"):
        home = np.array([0.50, 0.08])
        if np.linalg.norm(home - world.robot) < 0.06:
            return np.array([0.0, 0.0, -1.0, 1.0])
        return np.r_[clip_norm(home - world.robot), -1.0, 1.0]
    if world.held == target:
        delta = obj.stage - world.robot
        if np.linalg.norm(delta) < 0.08:
            return np.array([0.0, 0.0, -1.0, 1.0])
        return np.r_[clip_norm(delta), 1.0, 1.0]
    delta = obj.pos - world.robot
    if np.linalg.norm(delta) < 0.078:
        return np.array([0.0, 0.0, 1.0, 1.0])
    return np.r_[clip_norm(delta), -1.0, 1.0]


def worsen_invalid(world: World, index: int) -> None:
    if world.step - world.last_worsen_step < 10:
        return
    obj = world.objects[index]
    old = obj.status
    if old == "toppled":
        obj.status = "dropped"
        obj.pos = np.clip(obj.pos + np.array([0.10, -0.07]), [0.06, 0.08], [0.94, 0.92])
    elif old == "dropped":
        obj.status = "wedged"
        obj.pos = np.array([0.06, np.clip(obj.pos[1], 0.10, 0.90)])
    else:
        return
    world.compounding_failures += 1
    world.last_worsen_step = world.step


def advance(world: World, action: np.ndarray) -> Dict[str, object]:
    action = np.asarray(action, dtype=float)
    world.step += 1
    world.robot = np.clip(world.robot + clip_norm(action[:2], 0.12), [0.03, 0.04], [0.97, 0.96])
    reset_score = float(action[3])
    grip = float(action[2])
    info: Dict[str, object] = {"restored": -1, "placed": -1, "worsened": -1}

    if world.held >= 0:
        world.objects[world.held].pos = world.robot.copy()

    if grip > 0.28 and world.held < 0 and world.active < 3:
        candidates = sorted(range(3), key=lambda i: float(np.linalg.norm(world.objects[i].pos - world.robot)))
        nearest = candidates[0]
        obj = world.objects[nearest]
        if np.linalg.norm(obj.pos - world.robot) < 0.095:
            if obj.status == "ready":
                world.held = nearest
            elif reset_score > 0.42:
                world.held = nearest
            else:
                old = obj.status
                worsen_invalid(world, nearest)
                if obj.status != old:
                    info["worsened"] = nearest

    if grip < -0.28 and world.held >= 0:
        idx = world.held
        obj = world.objects[idx]
        world.held = -1
        if reset_score > 0.42 and np.linalg.norm(world.robot - obj.stage) < 0.14:
            obj.status = "ready"
            obj.pos = obj.stage.copy()
            info["restored"] = idx
        elif obj.status == "ready" and np.linalg.norm(world.robot - obj.goal) < 0.13 and idx == world.active:
            obj.status = "placed"
            obj.pos = obj.goal.copy()
            world.active += 1
            info["placed"] = idx
        else:
            obj.status = "dropped"
            obj.pos = world.robot.copy()
            world.compounding_failures += 1
            world.last_worsen_step = world.step
            info["worsened"] = idx
    return info


def inject_event(world: World, event: FailureEvent, rng: np.random.Generator) -> bool:
    if event.injected or world.active != event.object_index:
        return False
    condition = False
    if event.trigger == "pregrasp":
        condition = world.held < 0 and world.step >= 4 + 22 * event.object_index
    elif event.trigger == "holding":
        condition = world.held == event.object_index
    elif event.trigger == "post_approach":
        obj = world.objects[event.object_index]
        condition = world.held < 0 and np.linalg.norm(world.robot - obj.pos) < 0.18
    if not condition:
        return False

    idx = event.object_index
    obj = world.objects[idx]
    if event.kind == "id_pose":
        angle = rng.uniform(-math.pi, math.pi)
        offset = event.magnitude * np.array([math.cos(angle), math.sin(angle)])
        world.robot = np.clip(world.robot + offset, [0.04, 0.05], [0.96, 0.95])
    elif event.kind == "toppled":
        obj.status = "toppled"
    elif event.kind == "dropped":
        if world.held == idx:
            world.held = -1
        obj.status = "dropped"
        offset = event.magnitude * np.array([rng.choice([-1.0, 1.0]), rng.uniform(-0.65, 0.65)])
        obj.pos = np.clip(obj.pos + offset, [0.08, 0.10], [0.92, 0.90])
    elif event.kind == "wedged":
        if world.held == idx:
            world.held = -1
        obj.status = "wedged"
        obj.pos = np.array([0.055, np.clip(obj.pos[1] + rng.uniform(-0.08, 0.08), 0.10, 0.90)])
    else:
        raise ValueError(event.kind)
    event.injected = True
    event.injected_step = world.step
    return True


def make_training_data(seed: int, demos: int, perturb_radius: float) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    success_x: List[np.ndarray] = []
    success_y: List[np.ndarray] = []
    pb_x: List[np.ndarray] = []
    pb_y: List[np.ndarray] = []
    generic_x: List[np.ndarray] = []
    generic_y: List[np.ndarray] = []
    reset_x: Dict[int, List[np.ndarray]] = {i: [] for i in range(3)}
    reset_y: Dict[int, List[np.ndarray]] = {i: [] for i in range(3)}

    snapshots: List[World] = []
    for demo in range(demos):
        sj = rng.normal(0.0, 0.018, size=(3, 2))
        gj = rng.normal(0.0, 0.018, size=(3, 2))
        start = np.array([0.50, 0.08]) + rng.normal(0.0, 0.018, size=2)
        world = make_world(sj, gj, start)
        for _ in range(150):
            if world.active >= 3:
                break
            action = task_oracle(world)
            obs = encode_observation(world)
            success_x.append(obs)
            success_y.append(action)
            pb_x.append(obs)
            pb_y.append(action)
            generic_x.append(obs)
            generic_y.append(action)
            if world.held < 0 and world.step % 4 == 0:
                snapshots.append(copy_world(world))
            advance(world, action)

    # Perturb+bridge data preserves the environment state but broadens robot pose.
    snapshot_stride = max(1, len(snapshots) // (demos * 9))
    for snap in snapshots[::snapshot_stride]:
        for _ in range(3):
            angle = rng.uniform(-math.pi, math.pi)
            radius = perturb_radius * math.sqrt(rng.uniform(0.08, 1.0))
            pert = copy_world(snap)
            pert.robot = np.clip(pert.robot + radius * np.array([math.cos(angle), math.sin(angle)]), [0.04, 0.05], [0.96, 0.95])
            # Direct bridge to the next task-required pose.
            for _ in range(9):
                action = task_oracle(pert)
                pb_x.append(encode_observation(pert))
                pb_y.append(action)
                if np.linalg.norm(active_target(pert) - pert.robot) < 0.08:
                    break
                advance(pert, action)

            # Generic recovery returns to a global home independent of task phase.
            generic = copy_world(snap)
            generic.robot = pert.robot.copy()
            for _ in range(12):
                action = generic_recovery_oracle(generic)
                generic_x.append(encode_observation(generic))
                generic_y.append(action)
                advance(generic, action)
                if np.linalg.norm(generic.robot - np.array([0.50, 0.08])) < 0.07:
                    break

    # Object-centric reset demonstrations. Tail data intentionally remains in the
    # monolithic mixture; a gate would stop a reset skill once the state is valid.
    reset_demos_per_type = max(8, demos // 2)
    for idx in range(3):
        for failure in FAILURE_TYPES:
            for _ in range(reset_demos_per_type):
                sj = rng.normal(0.0, 0.025, size=(3, 2))
                gj = rng.normal(0.0, 0.02, size=(3, 2))
                world = make_world(sj, gj, rng.uniform([0.08, 0.08], [0.92, 0.92]))
                world.active = idx
                for j in range(idx):
                    world.objects[j].status = "placed"
                    world.objects[j].pos = world.objects[j].goal.copy()
                obj = world.objects[idx]
                obj.status = failure
                if failure == "toppled":
                    obj.pos = obj.stage + rng.normal(0.0, 0.035, size=2)
                elif failure == "dropped":
                    obj.pos = np.clip(obj.stage + rng.normal(0.0, 0.14, size=2), [0.08, 0.10], [0.92, 0.90])
                else:
                    obj.pos = np.array([0.055, np.clip(obj.stage[1] + rng.normal(0.0, 0.07), 0.10, 0.90)])
                restored_at = None
                for t in range(60):
                    action = reset_oracle(world, idx)
                    reset_x[idx].append(encode_observation(world))
                    reset_y[idx].append(action)
                    info = advance(world, action)
                    if info["restored"] == idx and restored_at is None:
                        restored_at = t
                    if restored_at is not None and t - restored_at >= 6:
                        break

    def arr(xs: Sequence[np.ndarray]) -> np.ndarray:
        return np.asarray(xs, dtype=float)

    success = (arr(success_x), arr(success_y))
    pb = (arr(pb_x), arr(pb_y))
    generic = (arr(generic_x), arr(generic_y))
    resets = {i: (arr(reset_x[i]), arr(reset_y[i])) for i in range(3)}
    mono_x = np.concatenate([pb[0]] + [resets[i][0] for i in range(3)], axis=0)
    mono_y = np.concatenate([pb[1]] + [resets[i][1] for i in range(3)], axis=0)
    return {
        "success": success,
        "perturb_bridge": pb,
        "generic": generic,
        "resets": resets,
        "monolithic": (mono_x, mono_y),
        "coverage": {
            "success_robot_variance": conditional_robot_variance(success[0]),
            "perturb_bridge_robot_variance": conditional_robot_variance(pb[0]),
            "generic_robot_variance": conditional_robot_variance(generic[0]),
            "counts": {
                "success": len(success[0]),
                "perturb_bridge": len(pb[0]),
                "generic": len(generic[0]),
                "monolithic": len(mono_x),
                "reset_per_object": {str(i): len(resets[i][0]) for i in range(3)},
            },
        },
    }


def conditional_robot_variance(x: np.ndarray) -> float:
    # Active-stage one-hot occupies scaled columns 4:8. Average pose variance per stage.
    stages = np.argmax(x[:, 4:8], axis=1)
    vals = []
    robot_unscaled = x[:, :2] / 3.2
    for s in range(3):
        rows = robot_unscaled[stages == s]
        if len(rows) > 2:
            vals.append(float(np.trace(np.cov(rows.T))))
    return float(np.mean(vals)) if vals else 0.0


def train_policies(data: Dict[str, object]) -> Dict[str, object]:
    policies: Dict[str, object] = {}
    for key in ("success", "generic", "perturb_bridge", "monolithic"):
        x, y = data[key]  # type: ignore[misc]
        policies[key] = WeightedKNNPolicy(k=9, name=key).fit(x, y)
    resets: Dict[int, WeightedKNNPolicy] = {}
    for i in range(3):
        x, y = data["resets"][i]  # type: ignore[index]
        resets[i] = WeightedKNNPolicy(k=7, name=f"reset_{i}").fit(x, y)
    policies["resets"] = resets
    return policies


def generate_scenarios(seed: int, trials: int, id_magnitude: float = 0.34) -> List[Scenario]:
    rng = np.random.default_rng(seed)
    scenarios: List[Scenario] = []
    kinds = ["clean", "id", "toppled", "dropped", "wedged", "mixed", "mixed", "hard"]
    for trial in range(trials):
        kind = kinds[trial % len(kinds)]
        scene_seed = int(rng.integers(0, 2**31 - 1))
        sj = rng.normal(0.0, 0.024 if kind != "hard" else 0.04, size=(3, 2))
        gj = rng.normal(0.0, 0.024 if kind != "hard" else 0.04, size=(3, 2))
        robot_start = np.array([0.50, 0.08]) + rng.normal(0.0, 0.035, size=2)
        events: List[FailureEvent] = []
        if kind == "id":
            events = [FailureEvent("id_pose", 0, "post_approach", id_magnitude)]
        elif kind in FAILURE_TYPES:
            trigger = "holding" if kind == "dropped" else "post_approach"
            events = [FailureEvent(kind, 1, trigger, 0.18)]
        elif kind == "mixed":
            ft = FAILURE_TYPES[(trial // len(kinds)) % 3]
            events = [
                FailureEvent("id_pose", 0, "post_approach", id_magnitude),
                FailureEvent(ft, 1, "holding" if ft == "dropped" else "post_approach", 0.18),
            ]
        elif kind == "hard":
            events = [
                FailureEvent("id_pose", 0, "post_approach", id_magnitude * 1.25),
                FailureEvent("dropped", 1, "holding", 0.24),
                FailureEvent("wedged", 2, "post_approach", 0.20),
            ]
        split = "robust" if kind in ("mixed", "hard") else kind
        scenarios.append(Scenario(trial, split, scene_seed, sj, gj, robot_start, events))
    return scenarios


def monitor_decision(
    world: World,
    rng: np.random.Generator,
    tpr: float,
    fpr: float,
    object_accuracy: float,
) -> Tuple[Optional[int], bool, bool]:
    invalid = [i for i, obj in enumerate(world.objects) if obj.status in FAILURE_TYPES]
    if invalid:
        if rng.random() > tpr:
            return None, False, True
        if rng.random() < object_accuracy:
            target = invalid[0]
        else:
            target = int(rng.integers(0, 3))
        return target, target not in invalid, False
    if rng.random() < fpr:
        return int(rng.integers(0, 3)), True, False
    return None, False, False


def rollout(
    scenario: Scenario,
    method: str,
    policies: Dict[str, object],
    max_steps: int,
    monitor_cfg: Dict[str, float],
    record: bool = False,
    disabled_reset_types: Sequence[str] = (),
) -> Rollout:
    rng = np.random.default_rng(scenario.scene_seed + 1009 * METHODS.index(method))
    world = make_world(scenario.stage_jitter, scenario.goal_jitter, scenario.robot_start)
    events = copy.deepcopy(scenario.events)
    reset_active: Optional[int] = None
    reset_calls = 0
    false_resets = 0
    false_reset_latched = False
    monitor_misses = 0
    recovery_durations: List[int] = []
    id_recovered = 0
    ood_recovered = 0
    injected_id = 0
    injected_ood = 0
    progress_at_event: Dict[int, int] = {}
    trajectory: List[Dict[str, object]] = []

    policy_map = {
        "success_only_bc": policies["success"],
        "generic_recovery_bc": policies["generic"],
        "perturb_bridge_bc": policies["perturb_bridge"],
        "monolithic_resets": policies["monolithic"],
    }

    for _ in range(max_steps):
        injected_now: List[str] = []
        for ei, event in enumerate(events):
            if inject_event(world, event, rng):
                progress_at_event[ei] = world.active
                injected_now.append(event.kind)
                if event.kind == "id_pose":
                    injected_id += 1
                else:
                    injected_ood += 1

        if method == "monitor_gated_reset_skills":
            if reset_active is not None:
                # A real gate checks whether the environment is task-valid again.
                if world.objects[reset_active].status in ("ready", "placed"):
                    reset_active = None
                else:
                    action = policies["resets"][reset_active].predict(encode_observation(world))  # type: ignore[index]
            if reset_active is None:
                target, false_positive, missed = monitor_decision(
                    world,
                    rng,
                    monitor_cfg["tpr"],
                    monitor_cfg["fpr"],
                    monitor_cfg["object_accuracy"],
                )
                monitor_misses += int(missed)
                if target is not None:
                    reset_calls += 1
                    false_resets += int(false_positive)
                    reset_active = target
                if reset_active is not None and world.objects[reset_active].status in disabled_reset_types:
                    reset_active = None
                if reset_active is not None:
                    action = policies["resets"][reset_active].predict(encode_observation(world))  # type: ignore[index]
                else:
                    action = policies["perturb_bridge"].predict(encode_observation(world))  # type: ignore[union-attr]
        else:
            action = policy_map[method].predict(encode_observation(world))  # type: ignore[union-attr]

        reset_signal = float(action[3]) > 0.42
        all_valid = all(obj.status not in FAILURE_TYPES for obj in world.objects)
        if method == "monolithic_resets":
            if reset_signal and all_valid and not false_reset_latched:
                false_resets += 1
                reset_calls += 1
                false_reset_latched = True
            elif not reset_signal:
                false_reset_latched = False
            elif reset_signal and not all_valid and not false_reset_latched:
                reset_calls += 1
                false_reset_latched = True

        info = advance(world, action)

        for ei, event in enumerate(events):
            if not event.injected or event.recovered_step >= 0:
                continue
            recovered = False
            if event.kind == "id_pose":
                # Retry is counted when the same task stage makes concrete progress.
                recovered = world.active > progress_at_event.get(ei, world.active) or world.held == event.object_index
            else:
                recovered = world.objects[event.object_index].status in ("ready", "placed")
            if recovered:
                event.recovered_step = world.step
                recovery_durations.append(world.step - event.injected_step)
                if event.kind == "id_pose":
                    id_recovered += 1
                else:
                    ood_recovered += 1

        if record:
            trajectory.append(
                {
                    "step": world.step,
                    "robot": world.robot.copy(),
                    "object_pos": [obj.pos.copy() for obj in world.objects],
                    "status": [obj.status for obj in world.objects],
                    "active": world.active,
                    "held": world.held,
                    "reset_active": -1 if reset_active is None else reset_active,
                    "reset_signal": float(action[3]),
                    "injected": injected_now,
                    "placed": int(info["placed"]),
                    "restored": int(info["restored"]),
                }
            )
        if world.active >= 3:
            break

    completion = world.active / 3.0
    success = int(world.active >= 3)
    injected_total = injected_id + injected_ood
    unrecovered = sum(1 for e in events if e.injected and e.recovered_step < 0)
    metrics: Dict[str, float | int | str] = {
        "trial": scenario.trial,
        "split": scenario.split,
        "method": method,
        "success": success,
        "task_completion": completion,
        "steps": world.step,
        "injected_failures": injected_total,
        "injected_id": injected_id,
        "injected_ood": injected_ood,
        "recovered_failures": injected_total - unrecovered,
        "id_recovery_rate": id_recovered / injected_id if injected_id else float("nan"),
        "ood_recovery_rate": ood_recovered / injected_ood if injected_ood else float("nan"),
        "mean_recovery_time": float(np.mean(recovery_durations)) if recovery_durations else float("nan"),
        "false_resets": false_resets,
        "reset_calls": reset_calls,
        "monitor_misses": monitor_misses,
        "compounding_failures": world.compounding_failures,
        "robust_success": success if scenario.split == "robust" else float("nan"),
    }
    return Rollout(metrics, trajectory)


def nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for method in METHODS:
        mrows = [r for r in rows if r["method"] == method]
        for split in ["all", "robust", "id", "toppled", "dropped", "wedged", "clean"]:
            selected = mrows if split == "all" else [r for r in mrows if r["split"] == split]
            if not selected:
                continue
            summaries.append(
                {
                    "method": method,
                    "split": split,
                    "episodes": len(selected),
                    "success_rate": nanmean(float(r["success"]) for r in selected),
                    "task_completion": nanmean(float(r["task_completion"]) for r in selected),
                    "mean_steps": nanmean(float(r["steps"]) for r in selected),
                    "mean_recovery_time": nanmean(float(r["mean_recovery_time"]) for r in selected),
                    "id_recovery_rate": nanmean(float(r["id_recovery_rate"]) for r in selected),
                    "ood_recovery_rate": nanmean(float(r["ood_recovery_rate"]) for r in selected),
                    "false_resets_per_episode": nanmean(float(r["false_resets"]) for r in selected),
                    "compounding_failures_per_episode": nanmean(float(r["compounding_failures"]) for r in selected),
                    "reset_calls_per_episode": nanmean(float(r["reset_calls"]) for r in selected),
                }
            )
    return summaries


def run_eval(
    scenarios: Sequence[Scenario],
    policies: Dict[str, object],
    max_steps: int,
    monitor_cfg: Dict[str, float],
    methods: Sequence[str] = METHODS,
    disabled_reset_types: Sequence[str] = (),
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for method in methods:
        for scenario in scenarios:
            rows.append(
                rollout(
                    scenario,
                    method,
                    policies,
                    max_steps,
                    monitor_cfg,
                    disabled_reset_types=disabled_reset_types,
                ).metrics
            )
    return rows


def run_sanity_checks(data: Dict[str, object], policies: Dict[str, object], seed: int) -> Dict[str, object]:
    checks: Dict[str, object] = {}

    w = make_world()
    w.step = 4
    before = [(o.pos.copy(), o.status) for o in w.objects]
    e = FailureEvent("id_pose", 0, "pregrasp", 0.32)
    injected = inject_event(w, e, np.random.default_rng(seed))
    environment_unchanged = all(np.allclose(p, o.pos) and s == o.status for (p, s), o in zip(before, w.objects))
    checks["id_pose_preserves_environment"] = bool(injected and environment_unchanged)

    reset_results = {}
    for failure in FAILURE_TYPES:
        rw = make_world(robot_start=np.array([0.75, 0.12]))
        rw.objects[0].status = failure
        if failure == "dropped":
            rw.objects[0].pos += np.array([0.16, 0.10])
        elif failure == "wedged":
            rw.objects[0].pos = np.array([0.055, rw.objects[0].stage[1]])
        for _ in range(70):
            advance(rw, reset_oracle(rw, 0))
            if rw.objects[0].status == "ready":
                break
        reset_results[failure] = rw.objects[0].status == "ready"
    checks["oracle_resets_all_failure_types"] = reset_results

    clean = Scenario(0, "clean", seed + 11, np.zeros((3, 2)), np.zeros((3, 2)), np.array([0.50, 0.08]), [])
    clean_roll = rollout(clean, "success_only_bc", policies, 180, {"tpr": 1.0, "fpr": 0.0, "object_accuracy": 1.0})
    checks["success_bc_completes_nominal"] = bool(clean_roll.metrics["success"] == 1)

    coverage = data["coverage"]  # type: ignore[assignment]
    checks["perturb_bridge_increases_pose_coverage"] = bool(
        coverage["perturb_bridge_robot_variance"] > 1.25 * coverage["success_robot_variance"]  # type: ignore[index]
    )

    mw = make_world()
    target, fp, miss = monitor_decision(mw, np.random.default_rng(seed), 1.0, 0.0, 1.0)
    mw.objects[1].status = "toppled"
    target_bad, fp_bad, miss_bad = monitor_decision(mw, np.random.default_rng(seed), 1.0, 0.0, 1.0)
    checks["oracle_monitor_gate"] = bool(target is None and not fp and not miss and target_bad == 1 and not fp_bad and not miss_bad)

    scenario = generate_scenarios(seed + 33, 1)[0]
    cfg = {"tpr": 0.91, "fpr": 0.035, "object_accuracy": 0.88}
    a = rollout(scenario, "monitor_gated_reset_skills", policies, 210, cfg).metrics
    b = rollout(scenario, "monitor_gated_reset_skills", policies, 210, cfg).metrics
    checks["deterministic_repeat"] = all(
        (math.isnan(float(a[k])) and math.isnan(float(b[k]))) or a[k] == b[k]
        for k in ("success", "task_completion", "steps", "false_resets", "compounding_failures", "mean_recovery_time")
    )
    checks["all_pass"] = bool(
        checks["id_pose_preserves_environment"]
        and all(reset_results.values())
        and checks["success_bc_completes_nominal"]
        and checks["perturb_bridge_increases_pose_coverage"]
        and checks["oracle_monitor_gate"]
        and checks["deterministic_repeat"]
    )
    return checks


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return None if not math.isfinite(x) else x
    if isinstance(value, np.integer):
        return int(value)
    return value


def plot_method_summary(summary: List[Dict[str, object]], output: Path) -> None:
    rows = {str(r["method"]): r for r in summary if r["split"] == "all"}
    x = np.arange(len(METHODS))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
    panels = [
        ("success_rate", "Task success", (0, 1.02)),
        ("mean_recovery_time", "Recovery time (steps, lower)", None),
        ("false_resets_per_episode", "False resets / episode (lower)", None),
        ("compounding_failures_per_episode", "Compounding failures / episode (lower)", None),
    ]
    for ax, (key, title, ylim) in zip(axes.flat, panels):
        vals = [float(rows[m][key]) for m in METHODS]
        ax.bar(x, vals, color=[COLORS[m] for m in METHODS])
        ax.set_title(title)
        ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS], rotation=24, ha="right")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Retry/reset recovery-data toy: paired evaluation", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_robustness(robustness: List[Dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for method in METHODS:
        rows = sorted([r for r in robustness if r["sweep"] == "id_magnitude" and r["method"] == method], key=lambda r: float(r["level"]))
        axes[0].plot([float(r["level"]) for r in rows], [float(r["success_rate"]) for r in rows], marker="o", label=METHOD_LABELS[method], color=COLORS[method])
    axes[0].set_xlabel("ID robot-pose perturbation magnitude")
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    failure_types = ["toppled", "dropped", "wedged", "robust"]
    width = 0.16
    x = np.arange(len(failure_types))
    for j, method in enumerate(METHODS):
        vals = []
        for ft in failure_types:
            row = next(r for r in robustness if r["sweep"] == "failure_type" and r["method"] == method and r["level"] == ft)
            vals.append(float(row["success_rate"]))
        axes[1].bar(x + (j - 2) * width, vals, width=width, color=COLORS[method], label=METHOD_LABELS[method])
    axes[1].set_xticks(x, failure_types)
    axes[1].set_ylim(0, 1.03)
    axes[1].set_ylabel("Success rate")
    axes[1].set_title("OOD and mixed-failure robustness")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_ablations(ablations: List[Dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    monitor_rows = [r for r in ablations if r["ablation"] == "monitor_quality"]
    labels = [str(r["setting"]) for r in monitor_rows]
    x = np.arange(len(labels))
    success_bars = axes[0].bar(x - 0.18, [float(r["success_rate"]) for r in monitor_rows], width=0.36, label="success", color="#59a14f")
    false_ax = axes[0].twinx()
    false_bars = false_ax.bar(x + 0.18, [float(r["false_resets_per_episode"]) for r in monitor_rows], width=0.36, label="false resets", color="#e15759")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylim(0, 1.03)
    axes[0].set_ylabel("Success rate")
    false_ax.set_ylabel("False resets / episode")
    axes[0].set_title("Monitor-gate quality")
    axes[0].legend([success_bars, false_bars], ["success", "false resets"], loc="upper left")
    axes[0].grid(axis="y", alpha=0.25)

    skill_rows = [r for r in ablations if r["ablation"] == "missing_reset_skill"]
    axes[1].bar(np.arange(len(skill_rows)), [float(r["success_rate"]) for r in skill_rows], color="#4c78a8")
    axes[1].set_xticks(np.arange(len(skill_rows)), [str(r["setting"]) for r in skill_rows], rotation=22, ha="right")
    axes[1].set_ylim(0, 1.03)
    axes[1].set_title("Reset-library ablation")
    axes[1].set_ylabel("Success rate")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_data_coverage(data: Dict[str, object], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), sharex=True, sharey=True)
    for ax, key, title in zip(
        axes,
        ("success", "generic", "perturb_bridge"),
        ("Success-only", "Generic recovery", "Perturb + bridge"),
    ):
        x, _ = data[key]  # type: ignore[misc]
        pts = x[:: max(1, len(x) // 900), :2] / 3.2
        stage = np.argmax(x[:: max(1, len(x) // 900), 4:8], axis=1)
        ax.scatter(pts[:, 0], pts[:, 1], c=stage, cmap="viridis", s=7, alpha=0.45)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("robot x")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("robot y")
    fig.suptitle("Robot-pose support by task stage (colors)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_representative(rollouts: Dict[str, Rollout], scenario: Scenario, output: Path) -> None:
    fig, axes = plt.subplots(1, len(METHODS), figsize=(18, 3.8), sharex=True, sharey=True)
    for ax, method in zip(axes, METHODS):
        traj = rollouts[method].trajectory
        robot = np.array([t["robot"] for t in traj])
        if len(robot):
            active = np.array([t["active"] for t in traj])
            ax.scatter(robot[:, 0], robot[:, 1], c=active, cmap="viridis", s=6, alpha=0.55)
            ax.plot(robot[:, 0], robot[:, 1], lw=0.7, color=COLORS[method], alpha=0.7)
        for i in range(3):
            stage = base_stage(i) + scenario.stage_jitter[i]
            goal = base_goal(i) + scenario.goal_jitter[i]
            ax.scatter(*stage, marker="o", s=45, facecolors="none", edgecolors="black")
            ax.scatter(*goal, marker="*", s=80, color="gold", edgecolors="black", linewidths=0.4)
        m = rollouts[method].metrics
        ax.set_title(f"{METHOD_LABELS[method]}\nS={int(m['success'])}, comp={int(m['compounding_failures'])}", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("y")
    for ax in axes:
        ax.set_xlabel("x")
    fig.suptitle("Representative hard rollout (circles: staging, stars: goals)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--train-demos", type=int, default=48)
    parser.add_argument("--max-steps", type=int, default=230)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()

    if args.smoke:
        args.trials = min(args.trials, 16)
        args.train_demos = min(args.train_demos, 10)
        args.max_steps = min(args.max_steps, 180)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = make_training_data(args.seed, args.train_demos, perturb_radius=0.43)
    policies = train_policies(data)
    sanity = run_sanity_checks(data, policies, args.seed)
    if not sanity["all_pass"]:
        raise RuntimeError(f"sanity checks failed: {sanity}")

    monitor_cfg = {"tpr": 0.97, "fpr": 0.015, "object_accuracy": 0.96}
    scenarios = generate_scenarios(args.seed + 101, args.trials, id_magnitude=0.34)
    trial_rows = run_eval(scenarios, policies, args.max_steps, monitor_cfg)
    summary = summarize(trial_rows)

    # Robustness: paired ID-only scenarios at increasing pose deviations.
    robustness: List[Dict[str, object]] = []
    sweep_trials = 12 if args.smoke else 32
    for mag in (0.0, 0.18, 0.34, 0.50, 0.66):
        id_scenarios = generate_scenarios(args.seed + 300 + int(100 * mag), sweep_trials, id_magnitude=mag)
        for s in id_scenarios:
            s.events = [FailureEvent("id_pose", 0, "post_approach", mag)] if mag > 0 else []
            s.split = "id_sweep"
        rows = run_eval(id_scenarios, policies, args.max_steps, monitor_cfg)
        for method in METHODS:
            selected = [r for r in rows if r["method"] == method]
            robustness.append({"sweep": "id_magnitude", "level": mag, "method": method, "episodes": len(selected), "success_rate": nanmean(float(r["success"]) for r in selected), "task_completion": nanmean(float(r["task_completion"]) for r in selected)})
    for split in ("toppled", "dropped", "wedged", "robust"):
        for method in METHODS:
            selected = [r for r in trial_rows if r["method"] == method and r["split"] == split]
            robustness.append({"sweep": "failure_type", "level": split, "method": method, "episodes": len(selected), "success_rate": nanmean(float(r["success"]) for r in selected), "task_completion": nanmean(float(r["task_completion"]) for r in selected)})

    # Monitor and skill-library ablations on failure-rich scenarios.
    ablations: List[Dict[str, object]] = []
    ablation_scenarios = [s for s in scenarios if s.split in ("robust", "toppled", "dropped", "wedged")]
    monitor_settings = [
        ("oracle gate", {"tpr": 1.0, "fpr": 0.0, "object_accuracy": 1.0}),
        ("default monitor", monitor_cfg),
        ("weak detector", {"tpr": 0.65, "fpr": 0.08, "object_accuracy": 0.70}),
        ("over-triggering", {"tpr": 0.96, "fpr": 0.22, "object_accuracy": 0.88}),
    ]
    for name, cfg in monitor_settings:
        rows = run_eval(ablation_scenarios, policies, args.max_steps, cfg, methods=["monitor_gated_reset_skills"])
        ablations.append({"ablation": "monitor_quality", "setting": name, "episodes": len(rows), "success_rate": nanmean(float(r["success"]) for r in rows), "task_completion": nanmean(float(r["task_completion"]) for r in rows), "false_resets_per_episode": nanmean(float(r["false_resets"]) for r in rows), "compounding_failures_per_episode": nanmean(float(r["compounding_failures"]) for r in rows)})
    for missing in ("none", "toppled", "dropped", "wedged"):
        disabled = () if missing == "none" else (missing,)
        rows = run_eval(ablation_scenarios, policies, args.max_steps, {"tpr": 1.0, "fpr": 0.0, "object_accuracy": 1.0}, methods=["monitor_gated_reset_skills"], disabled_reset_types=disabled)
        ablations.append({"ablation": "missing_reset_skill", "setting": f"missing {missing}" if missing != "none" else "full library", "episodes": len(rows), "success_rate": nanmean(float(r["success"]) for r in rows), "task_completion": nanmean(float(r["task_completion"]) for r in rows), "false_resets_per_episode": nanmean(float(r["false_resets"]) for r in rows), "compounding_failures_per_episode": nanmean(float(r["compounding_failures"]) for r in rows)})

    # Representative hard scenario for paired visual inspection.
    hard = next((s for s in scenarios if len(s.events) == 3), scenarios[-1])
    representative = {m: rollout(hard, m, policies, args.max_steps, monitor_cfg, record=True) for m in METHODS}

    write_csv(args.output_dir / "trial_metrics.csv", trial_rows)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "robustness.csv", robustness)
    write_csv(args.output_dir / "ablations.csv", ablations)
    with (args.output_dir / "sanity_checks.json").open("w") as f:
        json.dump(json_ready(sanity), f, indent=2, sort_keys=True)

    overall = {str(r["method"]): r for r in summary if r["split"] == "all"}
    claims = {
        "perturb_bridge_beats_success_only_on_id": bool(
            next(r for r in robustness if r["sweep"] == "id_magnitude" and r["level"] == 0.5 and r["method"] == "perturb_bridge_bc")["success_rate"]
            > next(r for r in robustness if r["sweep"] == "id_magnitude" and r["level"] == 0.5 and r["method"] == "success_only_bc")["success_rate"]
        ),
        "gated_skills_best_overall_success": bool(overall["monitor_gated_reset_skills"]["success_rate"] == max(float(v["success_rate"]) for v in overall.values())),
        "gating_reduces_false_resets_vs_monolithic": bool(overall["monitor_gated_reset_skills"]["false_resets_per_episode"] < overall["monolithic_resets"]["false_resets_per_episode"]),
        "gating_reduces_compounding_vs_monolithic": bool(overall["monitor_gated_reset_skills"]["compounding_failures_per_episode"] < overall["monolithic_resets"]["compounding_failures_per_episode"]),
    }
    metrics = {
        "config": vars(args) | {"output_dir": str(args.output_dir), "monitor": monitor_cfg},
        "data_coverage": data["coverage"],
        "sanity_checks": sanity,
        "claims_supported_by_this_run": claims,
        "summary": summary,
        "robustness": robustness,
        "ablations": ablations,
        "representative_trial": hard.trial,
    }
    with (args.output_dir / "metrics.json").open("w") as f:
        json.dump(json_ready(metrics), f, indent=2, sort_keys=True)

    plot_method_summary(summary, args.output_dir / "method_summary.png")
    plot_robustness(robustness, args.output_dir / "robustness_sweeps.png")
    plot_ablations(ablations, args.output_dir / "ablations.png")
    plot_data_coverage(data, args.output_dir / "data_coverage.png")
    plot_representative(representative, hard, args.output_dir / "representative_rollout.png")

    print(json.dumps(json_ready({"smoke": args.smoke, "sanity": sanity, "claims": claims, "overall": list(overall.values())}), indent=2))


if __name__ == "__main__":
    main()
