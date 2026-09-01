#!/usr/bin/env python3
"""Deterministic staged manipulation-chain diagnostics inspired by Behavior-Skill.

This package is a toy mechanism probe, not a reproduction of Behavior-Skill or a
robot benchmark. It compares end-to-end rollouts with independently reset
skill tests, adds reset-realism ablations, and ranks where additional nominal
skill demos or perturbed-reset recovery data provide the largest marginal value.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
DOCS_DIR = BASE_DIR / "docs"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHANNELS = ("pose", "grasp", "joint", "tool", "precision")
PROFILE_ORDER = ("exact", "pose_jitter", "realistic", "offnominal")
PROFILE_LABELS = {
    "exact": "Exact reset",
    "pose_jitter": "Pose jitter",
    "realistic": "Realistic reset",
    "offnominal": "Off-nominal start",
}
SEMANTIC_FAMILIES = {
    "MoveTo": "navigation",
    "PickUpFrom": "contact-rich",
    "OpenDoor": "articulation",
    "PlaceIn": "precise placement",
    "CloseDoor": "articulation",
    "Pour": "tool use",
    "PlaceOn": "placement",
    "SweepSurface": "contact-rich",
    "Attach": "tool use",
    "Release": "release",
}
SKILL_ORDER = [
    "MoveTo",
    "PickUpFrom",
    "OpenDoor",
    "PlaceIn",
    "CloseDoor",
    "Pour",
    "PlaceOn",
    "SweepSurface",
    "Attach",
    "Release",
]
SKILL_LABELS = {
    "MoveTo": "Move To",
    "PickUpFrom": "Pick Up From",
    "OpenDoor": "Open Door",
    "PlaceIn": "Place In",
    "CloseDoor": "Close Door",
    "Pour": "Pour",
    "PlaceOn": "Place On",
    "SweepSurface": "Sweep Surface",
    "Attach": "Attach",
    "Release": "Release",
}
COLORS = {
    "navigation": "#4c78a8",
    "contact-rich": "#f58518",
    "articulation": "#e45756",
    "precise placement": "#72b7b2",
    "tool use": "#b279a2",
    "placement": "#54a24b",
    "release": "#9d755d",
}


@dataclass(frozen=True)
class SkillTypeSpec:
    name: str
    family: str
    nominal_steps: int
    base_capability: float
    demo_gain: float
    recovery_gain: float
    mismatch_weights: Mapping[str, float]
    valid_threshold: float
    chain_sensitivity: float
    default_demands: Mapping[str, float]


@dataclass(frozen=True)
class StepTemplate:
    skill_type: str
    instruction: str
    preconditions: Tuple[str, ...]
    success: Tuple[str, ...]
    focus_object: str


@dataclass(frozen=True)
class TaskBlueprint:
    name: str
    short_name: str
    steps: Tuple[StepTemplate, ...]
    task_bias: Mapping[str, float]


@dataclass(frozen=True)
class SkillInstance:
    trajectory_id: int
    task_name: str
    task_short_name: str
    skill_index: int
    total_skills: int
    skill_type: str
    skill_label: str
    family: str
    instruction: str
    focus_object: str
    preconditions: Tuple[str, ...]
    success_conditions: Tuple[str, ...]
    nominal_steps: int
    horizon_steps: int
    demands: Mapping[str, float]
    affinity: float
    threshold: float
    reset_signature: Mapping[str, float]


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: int
    task_name: str
    task_short_name: str
    skills: Tuple[SkillInstance, ...]


@dataclass(frozen=True)
class DataPlan:
    name: str
    nominal_boosts: Mapping[str, float] = field(default_factory=dict)
    recovery_boosts: Mapping[str, float] = field(default_factory=dict)

    def nominal(self, skill_type: str) -> float:
        return float(self.nominal_boosts.get(skill_type, 0.0))

    def recovery(self, skill_type: str) -> float:
        return float(self.recovery_boosts.get(skill_type, 0.0))

    def with_nominal(self, skill_type: str, amount: float, name: Optional[str] = None) -> "DataPlan":
        updated = dict(self.nominal_boosts)
        updated[skill_type] = updated.get(skill_type, 0.0) + amount
        return DataPlan(name or f"{self.name}+demo:{skill_type}", updated, dict(self.recovery_boosts))

    def with_recovery(self, skill_type: str, amount: float, name: Optional[str] = None) -> "DataPlan":
        updated = dict(self.recovery_boosts)
        updated[skill_type] = updated.get(skill_type, 0.0) + amount
        return DataPlan(name or f"{self.name}+recovery:{skill_type}", dict(self.nominal_boosts), updated)


@dataclass(frozen=True)
class SkillOutcome:
    success: int
    margin: float
    steps_used: int
    valid_preconditions: int
    precondition_gap: float
    mismatch_total: float


PROFILE_SCALES: Mapping[str, Mapping[str, float]] = {
    "exact": {"pose": 0.00, "grasp": 0.00, "joint": 0.00, "tool": 0.00, "precision": 0.00},
    "pose_jitter": {"pose": 0.08, "grasp": 0.02, "joint": 0.03, "tool": 0.01, "precision": 0.05},
    "realistic": {"pose": 0.15, "grasp": 0.08, "joint": 0.09, "tool": 0.08, "precision": 0.12},
    "offnominal": {"pose": 0.24, "grasp": 0.16, "joint": 0.18, "tool": 0.16, "precision": 0.19},
}


def build_skill_specs() -> Dict[str, SkillTypeSpec]:
    return {
        "MoveTo": SkillTypeSpec(
            "MoveTo",
            "navigation",
            nominal_steps=9,
            base_capability=0.89,
            demo_gain=0.06,
            recovery_gain=0.03,
            mismatch_weights={"pose": 0.70, "grasp": 0.05, "joint": 0.02, "tool": 0.02, "precision": 0.32},
            valid_threshold=0.48,
            chain_sensitivity=0.38,
            default_demands={"precision": 0.16, "contact": 0.05, "articulation": 0.02, "tool": 0.02, "clutter": 0.14},
        ),
        "PickUpFrom": SkillTypeSpec(
            "PickUpFrom",
            "contact-rich",
            nominal_steps=12,
            base_capability=0.58,
            demo_gain=0.11,
            recovery_gain=0.08,
            mismatch_weights={"pose": 0.88, "grasp": 0.95, "joint": 0.05, "tool": 0.06, "precision": 0.52},
            valid_threshold=0.42,
            chain_sensitivity=0.72,
            default_demands={"precision": 0.24, "contact": 0.36, "articulation": 0.02, "tool": 0.04, "clutter": 0.16},
        ),
        "OpenDoor": SkillTypeSpec(
            "OpenDoor",
            "articulation",
            nominal_steps=11,
            base_capability=0.46,
            demo_gain=0.10,
            recovery_gain=0.08,
            mismatch_weights={"pose": 0.78, "grasp": 0.22, "joint": 1.02, "tool": 0.04, "precision": 0.62},
            valid_threshold=0.38,
            chain_sensitivity=0.78,
            default_demands={"precision": 0.28, "contact": 0.32, "articulation": 0.44, "tool": 0.02, "clutter": 0.12},
        ),
        "PlaceIn": SkillTypeSpec(
            "PlaceIn",
            "precise placement",
            nominal_steps=13,
            base_capability=0.32,
            demo_gain=0.15,
            recovery_gain=0.12,
            mismatch_weights={"pose": 0.95, "grasp": 0.88, "joint": 0.12, "tool": 0.06, "precision": 1.08},
            valid_threshold=0.34,
            chain_sensitivity=0.96,
            default_demands={"precision": 0.38, "contact": 0.28, "articulation": 0.12, "tool": 0.05, "clutter": 0.14},
        ),
        "CloseDoor": SkillTypeSpec(
            "CloseDoor",
            "articulation",
            nominal_steps=10,
            base_capability=0.40,
            demo_gain=0.09,
            recovery_gain=0.08,
            mismatch_weights={"pose": 0.70, "grasp": 0.12, "joint": 0.86, "tool": 0.04, "precision": 0.56},
            valid_threshold=0.38,
            chain_sensitivity=0.68,
            default_demands={"precision": 0.24, "contact": 0.24, "articulation": 0.40, "tool": 0.02, "clutter": 0.10},
        ),
        "Pour": SkillTypeSpec(
            "Pour",
            "tool use",
            nominal_steps=14,
            base_capability=0.28,
            demo_gain=0.14,
            recovery_gain=0.13,
            mismatch_weights={"pose": 0.84, "grasp": 0.86, "joint": 0.08, "tool": 1.08, "precision": 0.96},
            valid_threshold=0.33,
            chain_sensitivity=1.00,
            default_demands={"precision": 0.34, "contact": 0.38, "articulation": 0.02, "tool": 0.48, "clutter": 0.10},
        ),
        "PlaceOn": SkillTypeSpec(
            "PlaceOn",
            "placement",
            nominal_steps=10,
            base_capability=0.57,
            demo_gain=0.09,
            recovery_gain=0.07,
            mismatch_weights={"pose": 0.74, "grasp": 0.62, "joint": 0.04, "tool": 0.05, "precision": 0.56},
            valid_threshold=0.41,
            chain_sensitivity=0.66,
            default_demands={"precision": 0.20, "contact": 0.18, "articulation": 0.02, "tool": 0.03, "clutter": 0.12},
        ),
        "SweepSurface": SkillTypeSpec(
            "SweepSurface",
            "contact-rich",
            nominal_steps=13,
            base_capability=0.52,
            demo_gain=0.10,
            recovery_gain=0.09,
            mismatch_weights={"pose": 0.72, "grasp": 0.72, "joint": 0.04, "tool": 0.24, "precision": 0.54},
            valid_threshold=0.40,
            chain_sensitivity=0.72,
            default_demands={"precision": 0.20, "contact": 0.34, "articulation": 0.04, "tool": 0.18, "clutter": 0.16},
        ),
        "Attach": SkillTypeSpec(
            "Attach",
            "tool use",
            nominal_steps=15,
            base_capability=0.24,
            demo_gain=0.16,
            recovery_gain=0.14,
            mismatch_weights={"pose": 1.02, "grasp": 0.74, "joint": 0.26, "tool": 0.72, "precision": 1.16},
            valid_threshold=0.30,
            chain_sensitivity=1.05,
            default_demands={"precision": 0.42, "contact": 0.42, "articulation": 0.18, "tool": 0.22, "clutter": 0.08},
        ),
        "Release": SkillTypeSpec(
            "Release",
            "release",
            nominal_steps=8,
            base_capability=0.82,
            demo_gain=0.05,
            recovery_gain=0.05,
            mismatch_weights={"pose": 0.30, "grasp": 0.44, "joint": 0.02, "tool": 0.03, "precision": 0.24},
            valid_threshold=0.52,
            chain_sensitivity=0.30,
            default_demands={"precision": 0.10, "contact": 0.10, "articulation": 0.02, "tool": 0.02, "clutter": 0.06},
        ),
    }


def build_task_blueprints() -> Tuple[TaskBlueprint, ...]:
    return (
        TaskBlueprint(
            name="Store mug in cabinet",
            short_name="mug_cabinet",
            steps=(
                StepTemplate("MoveTo", "move to the mug", ("robot starts near home base",), ("robot is aligned with the mug",), "mug"),
                StepTemplate("PickUpFrom", "pick up the mug from the counter", ("robot is aligned with the mug", "gripper is empty"), ("mug is held securely",), "mug"),
                StepTemplate("MoveTo", "carry the mug to the cabinet", ("mug is held securely",), ("robot is aligned with the cabinet handle",), "cabinet"),
                StepTemplate("OpenDoor", "open the cabinet door", ("robot is aligned with the cabinet handle", "mug is held securely"), ("cabinet door is open enough for insertion",), "cabinet"),
                StepTemplate("PlaceIn", "place the mug inside the cabinet", ("mug is held securely", "cabinet door is open enough for insertion"), ("mug is inside the cabinet",), "cabinet"),
                StepTemplate("CloseDoor", "close the cabinet door", ("mug is inside the cabinet",), ("cabinet door is closed",), "cabinet"),
            ),
            task_bias={"precision": 0.05, "contact": 0.03, "articulation": 0.05, "tool": 0.00, "clutter": 0.02},
        ),
        TaskBlueprint(
            name="Microwave the popcorn bag",
            short_name="popcorn_microwave",
            steps=(
                StepTemplate("MoveTo", "move to the popcorn bag", ("robot starts near home base",), ("robot is aligned with the popcorn bag",), "popcorn bag"),
                StepTemplate("PickUpFrom", "pick up the popcorn bag", ("robot is aligned with the popcorn bag", "gripper is empty"), ("popcorn bag is held securely",), "popcorn bag"),
                StepTemplate("MoveTo", "carry the popcorn bag to the microwave", ("popcorn bag is held securely",), ("robot is aligned with the microwave handle",), "microwave"),
                StepTemplate("OpenDoor", "open the microwave door", ("robot is aligned with the microwave handle", "popcorn bag is held securely"), ("microwave door is open enough for insertion",), "microwave"),
                StepTemplate("PlaceIn", "place the popcorn bag inside the microwave", ("popcorn bag is held securely", "microwave door is open enough for insertion"), ("popcorn bag is inside the microwave",), "microwave"),
                StepTemplate("CloseDoor", "close the microwave door", ("popcorn bag is inside the microwave",), ("microwave door is closed",), "microwave"),
            ),
            task_bias={"precision": 0.06, "contact": 0.03, "articulation": 0.08, "tool": 0.00, "clutter": 0.03},
        ),
        TaskBlueprint(
            name="Serve water into a bowl",
            short_name="serve_water",
            steps=(
                StepTemplate("MoveTo", "move to the cup", ("robot starts near home base",), ("robot is aligned with the cup",), "cup"),
                StepTemplate("PickUpFrom", "pick up the cup", ("robot is aligned with the cup", "gripper is empty"), ("cup is held securely",), "cup"),
                StepTemplate("MoveTo", "carry the cup to the bowl", ("cup is held securely",), ("robot is aligned with the bowl",), "bowl"),
                StepTemplate("Pour", "pour water from the cup into the bowl", ("cup is held securely", "robot is aligned with the bowl"), ("enough water has been transferred to the bowl",), "bowl"),
                StepTemplate("PlaceOn", "place the cup back on the coaster", ("cup is held securely",), ("cup is resting on the coaster",), "coaster"),
                StepTemplate("Release", "release the cup", ("cup is resting on the coaster", "cup is held securely"), ("gripper is open and the cup stays on the coaster",), "cup"),
            ),
            task_bias={"precision": 0.06, "contact": 0.05, "articulation": 0.00, "tool": 0.09, "clutter": 0.02},
        ),
        TaskBlueprint(
            name="Wipe the table and return the sponge",
            short_name="wipe_table",
            steps=(
                StepTemplate("MoveTo", "move to the sponge", ("robot starts near home base",), ("robot is aligned with the sponge",), "sponge"),
                StepTemplate("PickUpFrom", "pick up the sponge", ("robot is aligned with the sponge", "gripper is empty"), ("sponge is held securely",), "sponge"),
                StepTemplate("SweepSurface", "wipe the marked table area", ("sponge is held securely",), ("table debris is cleared from the marked area",), "table"),
                StepTemplate("MoveTo", "carry the sponge to the tray", ("sponge is held securely",), ("robot is aligned with the tray",), "tray"),
                StepTemplate("PlaceOn", "place the sponge on the tray", ("sponge is held securely",), ("sponge is resting on the tray",), "tray"),
                StepTemplate("Release", "release the sponge", ("sponge is resting on the tray", "sponge is held securely"), ("gripper is open and the sponge stays on the tray",), "sponge"),
            ),
            task_bias={"precision": 0.04, "contact": 0.08, "articulation": 0.00, "tool": 0.02, "clutter": 0.05},
        ),
        TaskBlueprint(
            name="Attach the camera to the tripod",
            short_name="camera_tripod",
            steps=(
                StepTemplate("MoveTo", "move to the camera", ("robot starts near home base",), ("robot is aligned with the camera",), "camera"),
                StepTemplate("PickUpFrom", "pick up the camera", ("robot is aligned with the camera", "gripper is empty"), ("camera is held securely",), "camera"),
                StepTemplate("MoveTo", "carry the camera to the tripod", ("camera is held securely",), ("robot is aligned with the tripod mount",), "tripod"),
                StepTemplate("Attach", "attach the camera to the tripod", ("camera is held securely", "robot is aligned with the tripod mount"), ("camera is locked onto the tripod mount",), "tripod"),
                StepTemplate("Release", "release the camera", ("camera is locked onto the tripod mount", "camera is held securely"), ("gripper is open and the camera stays attached",), "camera"),
            ),
            task_bias={"precision": 0.10, "contact": 0.08, "articulation": 0.02, "tool": 0.12, "clutter": 0.01},
        ),
    )


SPECS = build_skill_specs()
BLUEPRINTS = build_task_blueprints()


def sanitize(text: str) -> str:
    return text.lower().replace(" ", "_").replace("/", "_")


def build_dataset(seed: int, trajectories_per_task: int) -> List[Trajectory]:
    rng = np.random.default_rng(seed)
    trajectories: List[Trajectory] = []
    tid = 0
    for blueprint in BLUEPRINTS:
        for _ in range(trajectories_per_task):
            skills: List[SkillInstance] = []
            n = len(blueprint.steps)
            task_noise = {k: float(v + rng.normal(0.0, 0.018)) for k, v in blueprint.task_bias.items()}
            for idx, step in enumerate(blueprint.steps):
                spec = SPECS[step.skill_type]
                late = idx / max(1, n - 1)
                demands: Dict[str, float] = {}
                for key, base in spec.default_demands.items():
                    jitter = float(rng.normal(0.0, 0.028))
                    demands[key] = max(0.02, base + task_noise.get(key, 0.0) + 0.06 * late + jitter)
                demands["late_stage"] = 0.12 + 0.62 * late
                affinity = float(rng.normal(0.0, 0.055))
                threshold = float(rng.normal(0.0, 0.05))
                reset_signature = {
                    "pose": float(np.clip(rng.normal(1.0 + 0.12 * late, 0.20), 0.55, 1.65)),
                    "grasp": float(np.clip(rng.normal(1.0 + 0.16 * demands["contact"], 0.20), 0.55, 1.75)),
                    "joint": float(np.clip(rng.normal(1.0 + 0.22 * demands["articulation"], 0.22), 0.45, 1.80)),
                    "tool": float(np.clip(rng.normal(1.0 + 0.22 * demands["tool"], 0.20), 0.45, 1.75)),
                    "precision": float(np.clip(rng.normal(1.0 + 0.18 * demands["precision"], 0.18), 0.55, 1.80)),
                }
                nominal_steps = spec.nominal_steps + int(round(4.0 * demands["precision"] + 2.5 * demands["contact"] + 1.5 * demands["tool"]))
                skills.append(
                    SkillInstance(
                        trajectory_id=tid,
                        task_name=blueprint.name,
                        task_short_name=blueprint.short_name,
                        skill_index=idx,
                        total_skills=n,
                        skill_type=step.skill_type,
                        skill_label=SKILL_LABELS[step.skill_type],
                        family=spec.family,
                        instruction=step.instruction,
                        focus_object=step.focus_object,
                        preconditions=step.preconditions,
                        success_conditions=step.success,
                        nominal_steps=nominal_steps,
                        horizon_steps=2 * nominal_steps,
                        demands=demands,
                        affinity=affinity,
                        threshold=threshold,
                        reset_signature=reset_signature,
                    )
                )
            trajectories.append(Trajectory(tid, blueprint.name, blueprint.short_name, tuple(skills)))
            tid += 1
    return trajectories


def weighted_sum(values: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return float(sum(float(values.get(k, 0.0)) * float(w) for k, w in weights.items()))


def mismatch_total(mismatch: Mapping[str, float]) -> float:
    return float(sum(float(v) for v in mismatch.values()))


def demand_cost(instance: SkillInstance) -> float:
    d = instance.demands
    return float(
        0.62 * d["precision"]
        + 0.52 * d["contact"]
        + 0.46 * d["articulation"]
        + 0.48 * d["tool"]
        + 0.26 * d["clutter"]
        + 0.18 * d["late_stage"]
    )


def build_profile_mismatch(instance: SkillInstance, profile: str) -> Dict[str, float]:
    scales = PROFILE_SCALES[profile]
    mismatch: Dict[str, float] = {}
    for channel in CHANNELS:
        mismatch[channel] = float(scales[channel] * instance.reset_signature[channel])
    return mismatch


def build_full_mismatch(instance: SkillInstance, drift: Mapping[str, float]) -> Dict[str, float]:
    mismatch: Dict[str, float] = {}
    for channel in CHANNELS:
        scale = 1.0 + 0.20 * float(instance.demands["late_stage"])
        if channel == "precision":
            scale += 0.20 * float(instance.demands["precision"])
        mismatch[channel] = float(drift.get(channel, 0.0) * scale)
    return mismatch


def precondition_gap(spec: SkillTypeSpec, mismatch: Mapping[str, float]) -> float:
    score = weighted_sum(mismatch, spec.mismatch_weights)
    return max(0.0, score - spec.valid_threshold)


def evaluate_instance(instance: SkillInstance, plan: DataPlan, mismatch: Mapping[str, float]) -> SkillOutcome:
    spec = SPECS[instance.skill_type]
    gap = precondition_gap(spec, mismatch)
    valid = int(gap <= 1e-12)
    mis_total = mismatch_total(mismatch)
    capability = (
        spec.base_capability
        + spec.demo_gain * plan.nominal(instance.skill_type)
        + spec.recovery_gain * plan.recovery(instance.skill_type) * (0.35 + 1.35 * mis_total + 0.85 * gap)
    )
    score = capability + instance.affinity - demand_cost(instance) - weighted_sum(mismatch, spec.mismatch_weights) - 1.05 * gap - instance.threshold
    success = int(score >= 0.0)
    extra = max(0.0, -score)
    steps = instance.nominal_steps + int(round(3.0 * mis_total + 4.0 * gap + 5.0 * extra))
    if not success:
        steps = instance.horizon_steps
    steps = int(min(instance.horizon_steps, max(2, steps)))
    return SkillOutcome(success, float(score), steps, valid, float(gap), float(mis_total))


def update_chain_drift(drift: MutableMapping[str, float], instance: SkillInstance) -> Dict[str, float]:
    spec = SPECS[instance.skill_type]
    d = instance.demands
    next_drift = dict(drift)
    next_drift["pose"] = 0.52 * next_drift.get("pose", 0.0) + 0.025 + 0.022 * d["precision"] + 0.014 * d["clutter"]
    next_drift["precision"] = 0.55 * next_drift.get("precision", 0.0) + 0.018 + 0.040 * d["precision"] + 0.016 * spec.chain_sensitivity
    if instance.skill_type == "MoveTo":
        next_drift["pose"] *= 0.72
        next_drift["precision"] *= 0.88
    if instance.skill_type in ("PickUpFrom", "PlaceIn", "PlaceOn", "Pour", "Attach", "Release", "SweepSurface"):
        next_drift["grasp"] = 0.58 * next_drift.get("grasp", 0.0) + 0.028 + 0.038 * d["contact"]
    else:
        next_drift["grasp"] = 0.42 * next_drift.get("grasp", 0.0)
    if instance.skill_type in ("OpenDoor", "CloseDoor"):
        next_drift["joint"] = 0.62 * next_drift.get("joint", 0.0) + 0.030 + 0.048 * d["articulation"]
    else:
        next_drift["joint"] = 0.48 * next_drift.get("joint", 0.0)
    if instance.skill_type in ("Pour", "Attach", "SweepSurface"):
        next_drift["tool"] = 0.60 * next_drift.get("tool", 0.0) + 0.024 + 0.060 * d["tool"]
    else:
        next_drift["tool"] = 0.45 * next_drift.get("tool", 0.0)
    return {k: float(v) for k, v in next_drift.items()}


def evaluate_full_rollouts(dataset: Sequence[Trajectory], plan: DataPlan) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    trajectory_rows: List[Dict[str, object]] = []
    skill_rows: List[Dict[str, object]] = []
    for traj in dataset:
        drift = {channel: 0.0 for channel in CHANNELS}
        attempted = 0
        executed_successes = 0
        stopped_at: Optional[int] = None
        total_steps = 0
        for instance in traj.skills:
            mismatch = build_full_mismatch(instance, drift)
            result = evaluate_instance(instance, plan, mismatch)
            attempted += 1
            total_steps += result.steps_used
            executed_successes += result.success
            skill_rows.append(
                {
                    "trajectory_id": traj.trajectory_id,
                    "task_name": traj.task_name,
                    "task_short_name": traj.task_short_name,
                    "skill_index": instance.skill_index,
                    "skill_type": instance.skill_type,
                    "family": instance.family,
                    "instruction": instance.instruction,
                    "profile": "full_rollout",
                    "attempted": 1,
                    "executed": 1,
                    "success": result.success,
                    "valid_preconditions": result.valid_preconditions,
                    "precondition_gap": result.precondition_gap,
                    "margin": result.margin,
                    "steps_used": result.steps_used,
                    "mismatch_total": result.mismatch_total,
                }
            )
            if result.success:
                drift = update_chain_drift(drift, instance)
            else:
                stopped_at = instance.skill_index
                break
        if attempted < len(traj.skills):
            for instance in traj.skills[attempted:]:
                skill_rows.append(
                    {
                        "trajectory_id": traj.trajectory_id,
                        "task_name": traj.task_name,
                        "task_short_name": traj.task_short_name,
                        "skill_index": instance.skill_index,
                        "skill_type": instance.skill_type,
                        "family": instance.family,
                        "instruction": instance.instruction,
                        "profile": "full_rollout",
                        "attempted": 0,
                        "executed": 0,
                        "success": 0,
                        "valid_preconditions": 0,
                        "precondition_gap": float("nan"),
                        "margin": float("nan"),
                        "steps_used": 0,
                        "mismatch_total": float("nan"),
                    }
                )
        trajectory_rows.append(
            {
                "trajectory_id": traj.trajectory_id,
                "task_name": traj.task_name,
                "task_short_name": traj.task_short_name,
                "task_success": int(executed_successes == len(traj.skills)),
                "executed_skills": attempted,
                "total_skills": len(traj.skills),
                "executed_fraction": attempted / len(traj.skills),
                "successful_skill_fraction": executed_successes / len(traj.skills),
                "stopped_at_skill_index": -1 if stopped_at is None else stopped_at,
                "total_steps": total_steps,
            }
        )
    return trajectory_rows, skill_rows


def evaluate_reset_profiles(dataset: Sequence[Trajectory], plan: DataPlan, profiles: Sequence[str]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    skill_rows: List[Dict[str, object]] = []
    trajectory_rows: List[Dict[str, object]] = []
    for profile in profiles:
        grouped: Dict[int, List[Dict[str, object]]] = defaultdict(list)
        for traj in dataset:
            for instance in traj.skills:
                mismatch = build_profile_mismatch(instance, profile)
                result = evaluate_instance(instance, plan, mismatch)
                row = {
                    "trajectory_id": traj.trajectory_id,
                    "task_name": traj.task_name,
                    "task_short_name": traj.task_short_name,
                    "skill_index": instance.skill_index,
                    "skill_type": instance.skill_type,
                    "family": instance.family,
                    "instruction": instance.instruction,
                    "profile": profile,
                    "attempted": 1,
                    "executed": 1,
                    "success": result.success,
                    "valid_preconditions": result.valid_preconditions,
                    "precondition_gap": result.precondition_gap,
                    "margin": result.margin,
                    "steps_used": result.steps_used,
                    "mismatch_total": result.mismatch_total,
                }
                skill_rows.append(row)
                grouped[traj.trajectory_id].append(row)
        for traj in dataset:
            rows = grouped[traj.trajectory_id]
            trajectory_rows.append(
                {
                    "trajectory_id": traj.trajectory_id,
                    "task_name": traj.task_name,
                    "task_short_name": traj.task_short_name,
                    "profile": profile,
                    "tscr": float(np.mean([float(r["success"]) for r in rows])) if rows else 0.0,
                    "valid_reset_fraction": float(np.mean([float(r["valid_preconditions"]) for r in rows])) if rows else 0.0,
                    "mean_steps": float(np.mean([float(r["steps_used"]) for r in rows])) if rows else 0.0,
                }
            )
    return skill_rows, trajectory_rows


def mean_or_nan(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")


def summarize_full_trajectories(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    all_rows = list(rows)
    out = [
        {
            "scope": "overall",
            "task_name": "all",
            "task_success_rate": mean_or_nan(float(r["task_success"]) for r in all_rows),
            "executed_fraction": mean_or_nan(float(r["executed_fraction"]) for r in all_rows),
            "successful_skill_fraction": mean_or_nan(float(r["successful_skill_fraction"]) for r in all_rows),
            "mean_steps": mean_or_nan(float(r["total_steps"]) for r in all_rows),
            "episodes": len(all_rows),
        }
    ]
    for task_name in sorted({str(r["task_name"]) for r in all_rows}):
        selected = [r for r in all_rows if r["task_name"] == task_name]
        out.append(
            {
                "scope": "task",
                "task_name": task_name,
                "task_success_rate": mean_or_nan(float(r["task_success"]) for r in selected),
                "executed_fraction": mean_or_nan(float(r["executed_fraction"]) for r in selected),
                "successful_skill_fraction": mean_or_nan(float(r["successful_skill_fraction"]) for r in selected),
                "mean_steps": mean_or_nan(float(r["total_steps"]) for r in selected),
                "episodes": len(selected),
            }
        )
    return out


def summarize_reset_trajectories(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    all_rows = list(rows)
    for profile in profiles_present(all_rows):
        selected = [r for r in all_rows if r["profile"] == profile]
        out.append(
            {
                "scope": "overall",
                "task_name": "all",
                "profile": profile,
                "tscr": mean_or_nan(float(r["tscr"]) for r in selected),
                "valid_reset_fraction": mean_or_nan(float(r["valid_reset_fraction"]) for r in selected),
                "mean_steps": mean_or_nan(float(r["mean_steps"]) for r in selected),
                "episodes": len(selected),
            }
        )
        for task_name in sorted({str(r["task_name"]) for r in selected}):
            task_rows = [r for r in selected if r["task_name"] == task_name]
            out.append(
                {
                    "scope": "task",
                    "task_name": task_name,
                    "profile": profile,
                    "tscr": mean_or_nan(float(r["tscr"]) for r in task_rows),
                    "valid_reset_fraction": mean_or_nan(float(r["valid_reset_fraction"]) for r in task_rows),
                    "mean_steps": mean_or_nan(float(r["mean_steps"]) for r in task_rows),
                    "episodes": len(task_rows),
                }
            )
    return out


def profiles_present(rows: Sequence[Mapping[str, object]]) -> List[str]:
    seen = []
    for profile in PROFILE_ORDER:
        if any(r.get("profile") == profile for r in rows):
            seen.append(profile)
    return seen


def summarize_skill_metrics(full_skill_rows: Sequence[Mapping[str, object]], reset_skill_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    by_skill_type = [
        st
        for st in SKILL_ORDER
        if any(r["skill_type"] == st for r in full_skill_rows)
        or any(r["skill_type"] == st for r in reset_skill_rows)
    ]
    for skill_type in by_skill_type:
        full_rows = [r for r in full_skill_rows if r["skill_type"] == skill_type]
        full_attempted = [r for r in full_rows if int(r["attempted"]) == 1]
        row: Dict[str, object] = {
            "skill_type": skill_type,
            "skill_label": SKILL_LABELS[skill_type],
            "family": SEMANTIC_FAMILIES[skill_type],
            "occurrences": len(full_rows),
            "full_attempt_rate": mean_or_nan(float(r["attempted"]) for r in full_rows),
            "full_conditional_success": mean_or_nan(float(r["success"]) for r in full_attempted) if full_attempted else float("nan"),
            "full_unexecuted_rate": 1.0 - mean_or_nan(float(r["attempted"]) for r in full_rows),
        }
        for profile in profiles_present(reset_skill_rows):
            selected = [r for r in reset_skill_rows if r["skill_type"] == skill_type and r["profile"] == profile]
            row[f"{profile}_stsr"] = mean_or_nan(float(r["success"]) for r in selected)
            row[f"{profile}_valid_rate"] = mean_or_nan(float(r["valid_preconditions"]) for r in selected)
            row[f"{profile}_mean_gap"] = mean_or_nan(float(r["precondition_gap"]) for r in selected)
        out.append(row)
    return out


def summarize_family_metrics(reset_skill_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for family in sorted({str(r["family"]) for r in reset_skill_rows}):
        row: Dict[str, object] = {"family": family}
        for profile in profiles_present(reset_skill_rows):
            selected = [r for r in reset_skill_rows if r["family"] == family and r["profile"] == profile]
            row[f"{profile}_stsr"] = mean_or_nan(float(r["success"]) for r in selected)
        out.append(row)
    return out


def rank_desc(pairs: Sequence[Tuple[str, float]]) -> Dict[str, float]:
    """Return descending average ranks, assigning equal values equal ranks."""
    values = {name: float(value) for name, value in pairs}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"Ranking inputs must be finite: {pairs}")
    ordered_values = sorted(set(values.values()), reverse=True)
    ranks_by_value: Dict[float, float] = {}
    position = 1
    for value in ordered_values:
        count = sum(candidate == value for candidate in values.values())
        ranks_by_value[value] = 0.5 * (position + position + count - 1)
        position += count
    return {name: ranks_by_value[value] for name, value in values.items()}


def spearman_from_pairs(pairs_a: Sequence[Tuple[str, float]], pairs_b: Sequence[Tuple[str, float]]) -> float:
    names = [name for name, _ in pairs_a]
    if set(names) != {name for name, _ in pairs_b} or len(names) < 2:
        return float("nan")
    rank_a = rank_desc(pairs_a)
    rank_b = rank_desc(pairs_b)
    a = np.asarray([rank_a[name] for name in names], dtype=float)
    b = np.asarray([rank_b[name] for name in names], dtype=float)
    a -= np.mean(a)
    b -= np.mean(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else float("nan")


def evaluate_baseline_and_allocations(dataset: Sequence[Trajectory], baseline: DataPlan) -> Dict[str, object]:
    full_traj, full_skills = evaluate_full_rollouts(dataset, baseline)
    reset_skills, reset_traj = evaluate_reset_profiles(dataset, baseline, PROFILE_ORDER)
    full_summary = summarize_full_trajectories(full_traj)
    reset_summary = summarize_reset_trajectories(reset_traj)
    skill_summary = summarize_skill_metrics(full_skills, reset_skills)
    family_summary = summarize_family_metrics(reset_skills)

    baseline_overall_full = next(r for r in full_summary if r["scope"] == "overall")
    baseline_overall_exact = next(r for r in reset_summary if r["scope"] == "overall" and r["profile"] == "exact")
    baseline_overall_realistic = next(r for r in reset_summary if r["scope"] == "overall" and r["profile"] == "realistic")

    skill_by_type = {str(r["skill_type"]): r for r in skill_summary}
    demo_full_pairs = []
    for st in skill_by_type:
        conditional = float(skill_by_type[st]["full_conditional_success"])
        if not math.isfinite(conditional):
            conditional = 0.0
        demo_full_pairs.append(
            (st, float(skill_by_type[st]["full_attempt_rate"]) * (1.0 - conditional))
        )
    demo_reset_pairs = [(st, 1.0 - float(skill_by_type[st]["exact_stsr"])) for st in skill_by_type]
    recovery_full_pairs = [(st, float(skill_by_type[st]["full_attempt_rate"]) * (float(skill_by_type[st]["exact_stsr"]) - float(skill_by_type[st]["realistic_stsr"]))) for st in skill_by_type]
    recovery_reset_pairs = [(st, float(skill_by_type[st]["exact_stsr"]) - float(skill_by_type[st]["realistic_stsr"])) for st in skill_by_type]

    allocation_rows: List[Dict[str, object]] = []
    actual_demo_pairs: List[Tuple[str, float]] = []
    actual_recovery_pairs: List[Tuple[str, float]] = []
    for skill_type in skill_by_type:
        demo_plan = baseline.with_nominal(skill_type, 1.0)
        demo_full_traj, _ = evaluate_full_rollouts(dataset, demo_plan)
        _, demo_reset_traj = evaluate_reset_profiles(dataset, demo_plan, ("exact", "realistic"))
        demo_full_overall = summarize_full_trajectories(demo_full_traj)[0]
        demo_full_success = float(demo_full_overall["task_success_rate"])
        demo_exact_tscr = mean_or_nan(float(r["tscr"]) for r in demo_reset_traj if r["profile"] == "exact")
        demo_realistic_tscr = mean_or_nan(float(r["tscr"]) for r in demo_reset_traj if r["profile"] == "realistic")
        demo_value = float(demo_full_overall["successful_skill_fraction"]) - float(
            baseline_overall_full["successful_skill_fraction"]
        )
        actual_demo_pairs.append((skill_type, demo_value))
        allocation_rows.append(
            {
                "intervention": f"demo:{skill_type}",
                "kind": "demo",
                "skill_type": skill_type,
                "skill_label": SKILL_LABELS[skill_type],
                "family": SEMANTIC_FAMILIES[skill_type],
                "baseline_full_attempt_rate": float(skill_by_type[skill_type]["full_attempt_rate"]),
                "baseline_full_conditional_success": float(skill_by_type[skill_type]["full_conditional_success"]),
                "baseline_exact_stsr": float(skill_by_type[skill_type]["exact_stsr"]),
                "baseline_realistic_stsr": float(skill_by_type[skill_type]["realistic_stsr"]),
                "heuristic_full": dict(demo_full_pairs)[skill_type],
                "heuristic_reset": dict(demo_reset_pairs)[skill_type],
                "delta_task_success": demo_full_success - float(baseline_overall_full["task_success_rate"]),
                "delta_chain_completion": demo_value,
                "delta_exact_tscr": demo_exact_tscr - float(baseline_overall_exact["tscr"]),
                "delta_realistic_tscr": demo_realistic_tscr - float(baseline_overall_realistic["tscr"]),
            }
        )

        recovery_plan = baseline.with_recovery(skill_type, 1.0)
        recovery_full_traj, _ = evaluate_full_rollouts(dataset, recovery_plan)
        _, recovery_reset_traj = evaluate_reset_profiles(dataset, recovery_plan, ("exact", "realistic"))
        recovery_full_overall = summarize_full_trajectories(recovery_full_traj)[0]
        recovery_full_success = float(recovery_full_overall["task_success_rate"])
        recovery_exact_tscr = mean_or_nan(float(r["tscr"]) for r in recovery_reset_traj if r["profile"] == "exact")
        recovery_realistic_tscr = mean_or_nan(float(r["tscr"]) for r in recovery_reset_traj if r["profile"] == "realistic")
        recovery_value = float(recovery_full_overall["successful_skill_fraction"]) - float(
            baseline_overall_full["successful_skill_fraction"]
        )
        actual_recovery_pairs.append((skill_type, recovery_value))
        allocation_rows.append(
            {
                "intervention": f"recovery:{skill_type}",
                "kind": "recovery",
                "skill_type": skill_type,
                "skill_label": SKILL_LABELS[skill_type],
                "family": SEMANTIC_FAMILIES[skill_type],
                "baseline_full_attempt_rate": float(skill_by_type[skill_type]["full_attempt_rate"]),
                "baseline_full_conditional_success": float(skill_by_type[skill_type]["full_conditional_success"]),
                "baseline_exact_stsr": float(skill_by_type[skill_type]["exact_stsr"]),
                "baseline_realistic_stsr": float(skill_by_type[skill_type]["realistic_stsr"]),
                "heuristic_full": dict(recovery_full_pairs)[skill_type],
                "heuristic_reset": dict(recovery_reset_pairs)[skill_type],
                "delta_task_success": recovery_full_success - float(baseline_overall_full["task_success_rate"]),
                "delta_chain_completion": recovery_value,
                "delta_exact_tscr": recovery_exact_tscr - float(baseline_overall_exact["tscr"]),
                "delta_realistic_tscr": recovery_realistic_tscr - float(baseline_overall_realistic["tscr"]),
            }
        )

    demo_actual_ranks = rank_desc(actual_demo_pairs)
    recovery_actual_ranks = rank_desc(actual_recovery_pairs)
    demo_full_ranks = rank_desc(demo_full_pairs)
    demo_reset_ranks = rank_desc(demo_reset_pairs)
    recovery_full_ranks = rank_desc(recovery_full_pairs)
    recovery_reset_ranks = rank_desc(recovery_reset_pairs)
    for row in allocation_rows:
        skill_type = str(row["skill_type"])
        if row["kind"] == "demo":
            row["actual_rank"] = demo_actual_ranks[skill_type]
            row["heuristic_full_rank"] = demo_full_ranks[skill_type]
            row["heuristic_reset_rank"] = demo_reset_ranks[skill_type]
        else:
            row["actual_rank"] = recovery_actual_ranks[skill_type]
            row["heuristic_full_rank"] = recovery_full_ranks[skill_type]
            row["heuristic_reset_rank"] = recovery_reset_ranks[skill_type]
        row["full_rank_gap"] = float(row["heuristic_full_rank"]) - float(row["actual_rank"])
        row["reset_rank_gap"] = float(row["heuristic_reset_rank"]) - float(row["actual_rank"])

    ranking_summary = {
        "demo_full_spearman": spearman_from_pairs(demo_full_pairs, actual_demo_pairs),
        "demo_reset_spearman": spearman_from_pairs(demo_reset_pairs, actual_demo_pairs),
        "recovery_full_spearman": spearman_from_pairs(recovery_full_pairs, actual_recovery_pairs),
        "recovery_reset_spearman": spearman_from_pairs(recovery_reset_pairs, actual_recovery_pairs),
        "top_demo_by_chain_completion": [
            {"skill_type": st, "delta_chain_completion": val}
            for st, val in sorted(actual_demo_pairs, key=lambda kv: (-kv[1], kv[0]))[:5]
        ],
        "top_recovery_by_chain_completion": [
            {"skill_type": st, "delta_chain_completion": val}
            for st, val in sorted(actual_recovery_pairs, key=lambda kv: (-kv[1], kv[0]))[:5]
        ],
    }
    return {
        "full_trajectory_rows": full_traj,
        "full_skill_rows": full_skills,
        "reset_skill_rows": reset_skills,
        "reset_trajectory_rows": reset_traj,
        "full_summary": full_summary,
        "reset_summary": reset_summary,
        "skill_summary": skill_summary,
        "family_summary": family_summary,
        "allocation_rows": allocation_rows,
        "ranking_summary": ranking_summary,
    }


def build_skill_catalog(dataset: Sequence[Trajectory]) -> List[Dict[str, object]]:
    by_type: Dict[str, SkillInstance] = {}
    for traj in dataset:
        for skill in traj.skills:
            by_type.setdefault(skill.skill_type, skill)
    rows = []
    for skill_type in SKILL_ORDER:
        if skill_type not in by_type:
            continue
        skill = by_type[skill_type]
        rows.append(
            {
                "skill_type": skill_type,
                "skill_label": skill.skill_label,
                "family": skill.family,
                "example_instruction": skill.instruction,
                "preconditions": " | ".join(skill.preconditions),
                "success_conditions": " | ".join(skill.success_conditions),
                "horizon_rule": "2x nominal demonstration length",
            }
        )
    return rows


def run_sanity_checks(dataset: Sequence[Trajectory], baseline_results: Mapping[str, object], seed: int) -> Dict[str, object]:
    checks: Dict[str, object] = {}
    a = evaluate_baseline_and_allocations(dataset, DataPlan("baseline"))
    b = evaluate_baseline_and_allocations(dataset, DataPlan("baseline"))
    checks["deterministic_repeat"] = bool(
        json.dumps(json_ready(a["ranking_summary"]), sort_keys=True) == json.dumps(json_ready(b["ranking_summary"]), sort_keys=True)
        and json.dumps(json_ready(a["full_summary"]), sort_keys=True) == json.dumps(json_ready(b["full_summary"]), sort_keys=True)
    )
    reset_summary = baseline_results["reset_summary"]
    full_summary = baseline_results["full_summary"]
    skill_summary = baseline_results["skill_summary"]
    exact = next(r for r in reset_summary if r["scope"] == "overall" and r["profile"] == "exact")
    pose = next(r for r in reset_summary if r["scope"] == "overall" and r["profile"] == "pose_jitter")
    realistic = next(r for r in reset_summary if r["scope"] == "overall" and r["profile"] == "realistic")
    offnominal = next(r for r in reset_summary if r["scope"] == "overall" and r["profile"] == "offnominal")
    full = next(r for r in full_summary if r["scope"] == "overall")
    checks["exact_valid_preconditions"] = bool(abs(float(exact["valid_reset_fraction"]) - 1.0) < 1e-12)
    checks["noise_monotone_tscr"] = bool(float(exact["tscr"]) >= float(pose["tscr"]) >= float(realistic["tscr"]) >= float(offnominal["tscr"]))
    checks["noise_monotone_validity"] = bool(
        float(exact["valid_reset_fraction"]) >= float(pose["valid_reset_fraction"]) >= float(realistic["valid_reset_fraction"]) >= float(offnominal["valid_reset_fraction"])
    )
    checks["exact_tscr_exceeds_full_success"] = bool(float(exact["tscr"]) > float(full["task_success_rate"]))
    family_exact = {}
    for row in baseline_results["family_summary"]:
        family_exact[row["family"]] = float(row["exact_stsr"])
    checks["contact_bottlenecks_below_navigation"] = bool(
        family_exact.get("contact-rich", 1.0) < family_exact.get("navigation", 0.0)
        and family_exact.get("tool use", 1.0) < family_exact.get("navigation", 0.0)
        and family_exact.get("articulation", 1.0) < family_exact.get("navigation", 0.0)
    )
    rank = baseline_results["ranking_summary"]
    checks["allocation_gains_nonzero"] = bool(
        float(rank["top_demo_by_chain_completion"][0]["delta_chain_completion"]) > 0.0
        and float(rank["top_recovery_by_chain_completion"][0]["delta_chain_completion"]) > 0.0
    )
    checks["allocation_rankings_are_finite"] = bool(
        all(
            math.isfinite(float(rank[key]))
            for key in (
                "demo_full_spearman",
                "demo_reset_spearman",
                "recovery_full_spearman",
                "recovery_reset_spearman",
            )
        )
    )
    checks["all_pass"] = all(bool(v) for v in checks.values())
    return checks


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json_ready(v) for k, v in row.items()})


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
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def plot_headline(full_summary: Sequence[Mapping[str, object]], reset_summary: Sequence[Mapping[str, object]], output: Path) -> None:
    full = next(r for r in full_summary if r["scope"] == "overall")
    exact = next(r for r in reset_summary if r["scope"] == "overall" and r["profile"] == "exact")
    realistic = next(r for r in reset_summary if r["scope"] == "overall" and r["profile"] == "realistic")
    offnominal = next(r for r in reset_summary if r["scope"] == "overall" and r["profile"] == "offnominal")
    labels = ["Full-task success", "Full executed fraction", "TSCR (exact)", "TSCR (realistic)", "TSCR (off-nominal)"]
    values = [
        float(full["task_success_rate"]),
        float(full["executed_fraction"]),
        float(exact["tscr"]),
        float(realistic["tscr"]),
        float(offnominal["tscr"]),
    ]
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    colors = ["#4c78a8", "#9ecae9", "#59a14f", "#f58518", "#e15759"]
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_title("Behavior-Skill-style diagnostic gap in the toy chain")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_stsr_heatmap(skill_summary: Sequence[Mapping[str, object]], output: Path) -> None:
    rows = [r for r in skill_summary if r["skill_type"] in SKILL_ORDER]
    metrics = np.array(
        [
            [
                float(r["full_attempt_rate"]),
                float(r["full_conditional_success"]),
                float(r["exact_stsr"]),
                float(r["realistic_stsr"]),
                float(r["offnominal_stsr"]),
            ]
            for r in rows
        ]
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    im = ax.imshow(metrics, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_yticks(np.arange(len(rows)), [SKILL_LABELS[str(r["skill_type"])] for r in rows])
    ax.set_xticks(np.arange(5), ["Attempt rate", "Cond. success", "Exact STSR", "Realistic STSR", "Off-nominal STSR"])
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    ax.set_title("Skill-type coverage and success")
    for i in range(metrics.shape[0]):
        for j in range(metrics.shape[1]):
            ax.text(j, i, f"{metrics[i, j]:.2f}", ha="center", va="center", color="white" if metrics[i, j] < 0.60 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_reset_ablations(reset_summary: Sequence[Mapping[str, object]], family_summary: Sequence[Mapping[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    overall_rows = [r for r in reset_summary if r["scope"] == "overall"]
    tscr = [float(next(r for r in overall_rows if r["profile"] == profile)["tscr"]) for profile in PROFILE_ORDER]
    valid = [float(next(r for r in overall_rows if r["profile"] == profile)["valid_reset_fraction"]) for profile in PROFILE_ORDER]
    x = np.arange(len(PROFILE_ORDER))
    axes[0].plot(x, tscr, marker="o", lw=2.0, label="TSCR", color="#4c78a8")
    axes[0].plot(x, valid, marker="s", lw=2.0, label="valid preconditions", color="#e15759")
    axes[0].set_xticks(x, [PROFILE_LABELS[p] for p in PROFILE_ORDER], rotation=18, ha="right")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Reset realism ablation")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    families = ["navigation", "contact-rich", "articulation", "precise placement", "tool use", "placement", "release"]
    realistic_vals = []
    exact_vals = []
    for family in families:
        row = next(r for r in family_summary if r["family"] == family)
        exact_vals.append(float(row["exact_stsr"]))
        realistic_vals.append(float(row["realistic_stsr"]))
    bar_x = np.arange(len(families))
    width = 0.38
    axes[1].bar(bar_x - width / 2, exact_vals, width=width, label="Exact", color="#59a14f")
    axes[1].bar(bar_x + width / 2, realistic_vals, width=width, label="Realistic", color="#f58518")
    axes[1].set_xticks(bar_x, families, rotation=20, ha="right")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("STSR")
    axes[1].set_title("Semantic families under exact vs realistic resets")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_reachability_bias(skill_summary: Sequence[Mapping[str, object]], output: Path) -> None:
    rows = [r for r in skill_summary if r["skill_type"] in SKILL_ORDER]
    x = np.arange(len(rows))
    attempt = [float(r["full_attempt_rate"]) for r in rows]
    exact = [float(r["exact_stsr"]) for r in rows]
    realistic = [float(r["realistic_stsr"]) for r in rows]
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    ax.bar(x, attempt, color="#9ecae9", label="Full-rollout attempt rate")
    ax.plot(x, exact, marker="o", lw=2.0, color="#59a14f", label="Exact-reset STSR")
    ax.plot(x, realistic, marker="s", lw=2.0, color="#e15759", label="Realistic-reset STSR")
    ax.set_xticks(x, [SKILL_LABELS[str(r["skill_type"])] for r in rows], rotation=24, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Reachability bias: later skill types are under-attempted")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_allocation_value(allocation_rows: Sequence[Mapping[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    for ax, kind, title in zip(axes, ("demo", "recovery"), ("Nominal demonstration packs", "Recovery-data packs")):
        rows = sorted([r for r in allocation_rows if r["kind"] == kind], key=lambda r: (-float(r["delta_chain_completion"]), str(r["skill_type"])))
        labels = [SKILL_LABELS[str(r["skill_type"])] for r in rows]
        vals = [float(r["delta_chain_completion"]) for r in rows]
        colors = [COLORS[SEMANTIC_FAMILIES[str(r["skill_type"])] ] for r in rows]
        ax.bar(np.arange(len(rows)), vals, color=colors)
        ax.set_xticks(np.arange(len(rows)), labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Successful-skill fraction lift")
        ax.grid(axis="y", alpha=0.25)
        for i, row in enumerate(rows):
            ax.text(
                i,
                vals[i] + 0.003,
                f"F{float(row['heuristic_full_rank']):g}/R{float(row['heuristic_reset_rank']):g}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    fig.suptitle("Full-chain completion lift with heuristic ranks (F=full rollout, R=reset suite)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def tex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = text
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}\\%"


def write_report_tex(args: argparse.Namespace, baseline_results: Mapping[str, object], output_dir: Path) -> Path:
    full = next(r for r in baseline_results["full_summary"] if r["scope"] == "overall")
    exact = next(r for r in baseline_results["reset_summary"] if r["scope"] == "overall" and r["profile"] == "exact")
    realistic = next(r for r in baseline_results["reset_summary"] if r["scope"] == "overall" and r["profile"] == "realistic")
    offnominal = next(r for r in baseline_results["reset_summary"] if r["scope"] == "overall" and r["profile"] == "offnominal")
    ranking = baseline_results["ranking_summary"]
    skill_rows = {str(r["skill_type"]): r for r in baseline_results["skill_summary"]}
    late_skills = [skill_rows[s] for s in ("PlaceIn", "Pour", "Attach") if s in skill_rows]
    top_demo = ranking["top_demo_by_chain_completion"][0]
    top_recovery = ranking["top_recovery_by_chain_completion"][0]
    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{lmodern}}
\usepackage{{microtype}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{enumitem}}
\usepackage{{graphicx}}
\usepackage{{float}}
\hypersetup{{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black,citecolor=blue!50!black,hypertexnames=false}}
\title{{Skill Reset Diagnostics for Long-Horizon Manipulation Chains\\\large A Deterministic Behavior-Skill-Inspired Toy Study}}
\author{{VLA Ideas}}
\date{{September 1, 2026}}
\begin{{document}}
\maketitle
\begin{{abstract}}
This report presents a deterministic staged-manipulation toy inspired by Behavior-Skill. Five household manipulation chains are decomposed into executable constituent skills with saved intermediate states, skill-specific preconditions, local success conditions, and bounded horizons. The central comparison is between ordinary full rollouts and independently reset skill tests. In the checked run, full-task success is {pct(float(full['task_success_rate']))}, while exact-reset Task Skill Completion Rate (TSCR) is {pct(float(exact['tscr']))}. Under more realistic reset noise, TSCR falls to {pct(float(realistic['tscr']))}; under off-nominal starts it falls further to {pct(float(offnominal['tscr']))}. Contact-rich, articulated, and tool-use skills form persistent bottlenecks, and full-rollout attempt coverage sharply underestimates the importance of later skills. Counterfactual data-allocation sweeps rank targets by improvement in successful skill completion, showing where reset diagnostics help and where reachability-weighted rollout evidence remains necessary.
\end{{abstract}}

\section{{Motivation and Source Mapping}}
Behavior-Skill argues that long-horizon manipulation should be analyzed through constituent skills rather than only end-to-end task completion. Its benchmark saves restorable intermediate states, attaches skill-specific goals, and reports both trajectory-level TSCR and skill-type Skill-Type Success Rate (STSR). This toy keeps those evaluation ideas but replaces VLAs, simulator assets, and BDDL goals with a transparent deterministic reliability model over staged manipulation chains.

\section{{Toy benchmark construction}}
The package instantiates five long-horizon tasks: storing a mug in a cabinet, microwaving popcorn, serving water into a bowl, wiping a table and returning the sponge, and attaching a camera to a tripod. Each trajectory is segmented into semantic skills such as \emph{{Move To}}, \emph{{Pick Up From}}, \emph{{Open Door}}, \emph{{Place In}}, \emph{{Pour}}, \emph{{Attach}}, and \emph{{Release}}. Each skill instance contains:
\begin{{itemize}}[leftmargin=1.5em]
  \item a saved precondition state represented by deterministic latent mismatch channels;
  \item a natural-language skill instruction;
  \item a local success condition; and
  \item a horizon equal to twice the nominal demonstration length.
\end{{itemize}}
Full rollouts stop when a constituent skill fails, leaving later skills unexecuted. Independent skill tests instead restore the skill start state and evaluate the policy directly under exact or noisy resets.

\section{{Checked configuration}}
The checked run used seed {args.seed}, {args.trajectories_per_task} trajectories per task, and one equal-cost counterfactual data pack for each allocation experiment. A nominal demonstration pack raises clean skill capability for one semantic type. A recovery-data pack raises robustness to reset mismatch and chaining drift for one semantic type.

\section{{Headline results}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lrrrr}}
\toprule
Metric & Full rollout & Exact reset & Realistic reset & Off-nominal reset \\
\midrule
Task success / TSCR & {pct(float(full['task_success_rate']))} & {pct(float(exact['tscr']))} & {pct(float(realistic['tscr']))} & {pct(float(offnominal['tscr']))} \\
Executed skill fraction / valid reset fraction & {pct(float(full['executed_fraction']))} & {pct(float(exact['valid_reset_fraction']))} & {pct(float(realistic['valid_reset_fraction']))} & {pct(float(offnominal['valid_reset_fraction']))} \\
Mean steps & {float(full['mean_steps']):.1f} & {float(exact['mean_steps']):.1f} & {float(realistic['mean_steps']):.1f} & {float(offnominal['mean_steps']):.1f} \\
\bottomrule
\end{{tabular}}
\caption{{Aggregate trajectory- and skill-level diagnostics. Full rollouts and reset tests answer different questions.}}
\end{{table}}

Three late, contact-sensitive skills illustrate the reachability gap. In full rollouts, {tex_escape(SKILL_LABELS['PlaceIn'])}, {tex_escape(SKILL_LABELS['Pour'])}, and {tex_escape(SKILL_LABELS['Attach'])} are attempted on only {', '.join(pct(float(r['full_attempt_rate'])) for r in late_skills)} of their occurrences, yet their exact-reset STSRs are {', '.join(pct(float(r['exact_stsr'])) for r in late_skills)} and their realistic-reset STSRs drop to {', '.join(pct(float(r['realistic_stsr'])) for r in late_skills)}. Full rollouts therefore leave much of the bottleneck profile unseen.

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.92\textwidth]{{../outputs/headline_metrics.png}}
  \caption{{Full-task success versus reset-based TSCR under increasingly realistic reset conditions.}}
\end{{figure}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=\textwidth]{{../outputs/reachability_bias.png}}
  \caption{{Attempt coverage from full rollouts is much lower than independently measured success on later skills.}}
\end{{figure}}

\section{{Semantic skill bottlenecks}}
Exact-reset evaluation shows the expected pattern from the source paper's motivation: navigation and release remain easy, while contact-rich, articulated, precise-placement, and tool-use skills are harder. Realistic resets widen the gap further because later skills are more sensitive to pose, grasp, and precision mismatch.

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.92\textwidth]{{../outputs/stsr_heatmap.png}}
  \caption{{Skill-type attempt coverage, conditional success when reached in full rollouts, and STSR under exact and noisy resets.}}
\end{{figure}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=\textwidth]{{../outputs/reset_ablations.png}}
  \caption{{Reset-realism ablations at the overall benchmark level and by semantic family.}}
\end{{figure}}

\section{{Allocation ranking and reachability bias}}
The counterfactual allocation sweep asks where one extra data pack provides the largest lift in the fraction of successfully completed skills across full chains. The best nominal-demonstration target in this run is \emph{{{tex_escape(SKILL_LABELS[str(top_demo['skill_type'])])}}}, which improves chain completion by {pct(float(top_demo['delta_chain_completion']))}. The best recovery-data target is \emph{{{tex_escape(SKILL_LABELS[str(top_recovery['skill_type'])])}}}, which improves chain completion by {pct(float(top_recovery['delta_chain_completion']))}. Spearman correlation with measured allocation value is {float(ranking['demo_full_spearman']):.2f} for the full-rollout demo heuristic versus {float(ranking['demo_reset_spearman']):.2f} for the reset-suite demo heuristic. For recovery allocation the values are {float(ranking['recovery_full_spearman']):.2f} and {float(ranking['recovery_reset_spearman']):.2f}, respectively. In this chain model, the reachability-weighted rollout heuristic is substantially better for demonstrations and clearly better for recovery allocation; the reset suite remains complementary because it exposes capabilities of skills that full rollouts rarely reach.

\begin{{figure}}[H]
  \centering
  \includegraphics[width=\textwidth]{{../outputs/allocation_value.png}}
  \caption{{Lift in successful full-chain skill completion from one extra nominal-demo or recovery-data pack. Text labels show heuristic rank from full rollouts (F) and reset diagnostics (R).}}
\end{{figure}}

\section{{Interpretation}}
This toy reproduces the diagnostic logic rather than the scale of Behavior-Skill. Full rollouts remain important because chaining errors matter, but they are a biased lens for deciding where data collection helps most. Exact-reset skill tests expose constituent capability; noisy-reset ablations expose robustness to imperfect restoration; and the gap between the two indicates where recovery data is likely more valuable than additional clean demonstrations.

\section{{Limitations}}
The package uses a deterministic latent reliability model instead of images, real robot dynamics, BDDL predicates, or learned VLAs. Reset states are abstract mismatch channels, not simulator snapshots. The allocation sweep uses equal-cost synthetic data packs rather than actual retraining. Numbers should therefore be read as mechanism-level evidence only.

\begin{{thebibliography}}{{9}}
\bibitem{{behavior-skill-paper}}
C. Ma, Y. Ma, Z. Ma, M. Zhang, S. Sheng, Y. Zhou, Z. Wang, X. Liu, B. Liu, J. Li, X. Lin, Z. Yang, R. Shi, and Y. Gao.
\emph{{A Fine-Grained Benchmark for Evaluating Vision-Language-Action Policies in Long-Horizon Tasks}}.
arXiv:2608.30536, 2026. \url{{https://arxiv.org/abs/2608.30536}}

\bibitem{{behavior-skill-repo}}
Behavior-Skill project repository. \url{{https://github.com/nubot-nudt/Behavior-Skill}}
\end{{thebibliography}}
\end{{document}}
"""
    docs_dir = output_dir.parent / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    tex_path = docs_dir / "skill_reset_diagnostics_report.tex"
    tex_path.write_text(tex)
    return tex_path


def build_metrics(args: argparse.Namespace, baseline_results: Mapping[str, object], sanity: Mapping[str, object]) -> Dict[str, object]:
    full = next(r for r in baseline_results["full_summary"] if r["scope"] == "overall")
    reset_summary = baseline_results["reset_summary"]
    exact = next(r for r in reset_summary if r["scope"] == "overall" and r["profile"] == "exact")
    realistic = next(r for r in reset_summary if r["scope"] == "overall" and r["profile"] == "realistic")
    offnominal = next(r for r in reset_summary if r["scope"] == "overall" and r["profile"] == "offnominal")
    ranking = baseline_results["ranking_summary"]
    claims = {
        "full_rollout_understates_skill_completion": bool(float(full["task_success_rate"]) + 0.10 < float(exact["tscr"])),
        "reset_realism_reduces_skill_completion": bool(float(exact["tscr"]) > float(realistic["tscr"]) > float(offnominal["tscr"])),
        "reachability_weighted_demo_ranking_is_stronger_here": bool(
            float(ranking["demo_full_spearman"]) >= float(ranking["demo_reset_spearman"])
        ),
        "recovery_allocation_needs_reachability_weighting_here": bool(
            float(ranking["recovery_full_spearman"]) > float(ranking["recovery_reset_spearman"])
        ),
    }
    return {
        "config": {
            "seed": args.seed,
            "trajectories_per_task": args.trajectories_per_task,
            "output_dir": str(args.output_dir.relative_to(BASE_DIR))
            if args.output_dir.is_relative_to(BASE_DIR)
            else str(args.output_dir),
            "intervention_pack": {
                "nominal_demo_pack": "one extra semantic skill demo pack",
                "recovery_data_pack": "one extra perturbed-reset recovery pack",
            },
        },
        "skills": build_skill_catalog(build_dataset(args.seed, args.trajectories_per_task)),
        "full_summary": baseline_results["full_summary"],
        "reset_summary": baseline_results["reset_summary"],
        "skill_summary": baseline_results["skill_summary"],
        "family_summary": baseline_results["family_summary"],
        "allocation_ranking": baseline_results["allocation_rows"],
        "ranking_summary": ranking,
        "sanity_checks": sanity,
        "claims_supported_by_this_run": claims,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--trajectories-per-task", type=int, default=18)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.smoke:
        args.trajectories_per_task = min(args.trajectories_per_task, 6)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(args.seed, args.trajectories_per_task)
    baseline = DataPlan("baseline")
    baseline_results = evaluate_baseline_and_allocations(dataset, baseline)
    sanity = run_sanity_checks(dataset, baseline_results, args.seed)
    if not sanity["all_pass"]:
        raise RuntimeError(f"sanity checks failed: {json.dumps(json_ready(sanity), indent=2, sort_keys=True)}")

    write_csv(args.output_dir / "full_rollouts.csv", baseline_results["full_trajectory_rows"])
    write_csv(args.output_dir / "full_skill_rollouts.csv", baseline_results["full_skill_rows"])
    write_csv(args.output_dir / "skill_eval.csv", baseline_results["reset_skill_rows"])
    write_csv(args.output_dir / "trajectory_reset_summary.csv", baseline_results["reset_trajectory_rows"])
    write_csv(args.output_dir / "full_summary.csv", baseline_results["full_summary"])
    write_csv(args.output_dir / "reset_summary.csv", baseline_results["reset_summary"])
    write_csv(args.output_dir / "stsr_summary.csv", baseline_results["skill_summary"])
    write_csv(args.output_dir / "family_summary.csv", baseline_results["family_summary"])
    write_csv(args.output_dir / "allocation_ranking.csv", baseline_results["allocation_rows"])
    write_csv(args.output_dir / "skill_catalog.csv", build_skill_catalog(dataset))
    with (args.output_dir / "sanity_checks.json").open("w") as f:
        json.dump(json_ready(sanity), f, indent=2, sort_keys=True)

    plot_headline(baseline_results["full_summary"], baseline_results["reset_summary"], args.output_dir / "headline_metrics.png")
    plot_stsr_heatmap(baseline_results["skill_summary"], args.output_dir / "stsr_heatmap.png")
    plot_reset_ablations(baseline_results["reset_summary"], baseline_results["family_summary"], args.output_dir / "reset_ablations.png")
    plot_reachability_bias(baseline_results["skill_summary"], args.output_dir / "reachability_bias.png")
    plot_allocation_value(baseline_results["allocation_rows"], args.output_dir / "allocation_value.png")

    tex_path = write_report_tex(args, baseline_results, args.output_dir)
    metrics = build_metrics(args, baseline_results, sanity)
    with (args.output_dir / "metrics.json").open("w") as f:
        json.dump(json_ready(metrics), f, indent=2, sort_keys=True)

    headline = {
        "full_task_success_rate": next(r for r in baseline_results["full_summary"] if r["scope"] == "overall")["task_success_rate"],
        "exact_tscr": next(r for r in baseline_results["reset_summary"] if r["scope"] == "overall" and r["profile"] == "exact")["tscr"],
        "realistic_tscr": next(r for r in baseline_results["reset_summary"] if r["scope"] == "overall" and r["profile"] == "realistic")["tscr"],
        "offnominal_tscr": next(r for r in baseline_results["reset_summary"] if r["scope"] == "overall" and r["profile"] == "offnominal")["tscr"],
        "report_tex": str(tex_path),
        "ranking": baseline_results["ranking_summary"],
        "sanity": sanity,
    }
    print(json.dumps(json_ready(headline), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
