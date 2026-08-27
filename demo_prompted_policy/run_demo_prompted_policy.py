#!/usr/bin/env python3
"""S1-inspired demonstration-prompted policy toy.

A short 2D demonstration is the in-context task prompt. Five hand-built policies
then execute the task in a separately transformed scene. This is an explanatory
mechanism benchmark, not an S1 implementation or reproduction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
METHODS = [
    "language_prior",
    "nearest_demo_replay",
    "phase_retrieval",
    "full_demo_attention",
    "latent_intent",
]
METHOD_LABELS = {
    "language_prior": "language/task-label prior",
    "nearest_demo_replay": "nearest demo replay",
    "phase_retrieval": "phase retrieval",
    "full_demo_attention": "full-demo attention",
    "latent_intent": "latent-intent matching",
}
METHOD_COLORS = {
    "language_prior": "#777777",
    "nearest_demo_replay": "#d95f02",
    "phase_retrieval": "#7570b3",
    "full_demo_attention": "#1b9e77",
    "latent_intent": "#1f77b4",
}
CANONICAL_OBJECTS = np.array(
    [
        [-0.72, -0.42],
        [-0.18, -0.68],
        [0.58, -0.50],
        [0.72, 0.24],
        [0.12, 0.68],
        [-0.62, 0.44],
    ],
    dtype=float,
)
PRIOR_MOTIF = np.array([0, 1, 2, 3, 4, 5], dtype=int)
MOTIFS = [
    np.array([0, 2, 4, 1, 3, 5], dtype=int),
    np.array([5, 3, 1, 4, 2, 0], dtype=int),
    np.array([1, 4, 0, 3, 5, 2], dtype=int),
    np.array([2, 5, 3, 0, 4, 1], dtype=int),
]
MAX_WORLD_STEP = 0.115
CONTACT_RADIUS = 0.105
CONTACT_DWELL = 2
STALL_LIMIT = 28
MAX_INTERVENTIONS = 2


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    seed: int
    shift_level: int
    mirrored: bool
    corruption: float
    horizon: int
    demo_objects: np.ndarray
    deploy_objects: np.ndarray
    demo_start: np.ndarray
    deploy_start: np.ndarray
    true_sequence: np.ndarray
    demo_path: np.ndarray
    demo_actions: np.ndarray
    embodiment: np.ndarray
    perturbations: Tuple[np.ndarray, np.ndarray]


@dataclass
class Rollout:
    scenario: Scenario
    method: str
    positions: np.ndarray
    commands: np.ndarray
    phases: np.ndarray
    success: bool
    assisted_completion: bool
    progress: float
    tracking_rmse: float
    tracking_progress: float
    interventions: int
    disturbance_events: int
    recoveries: int
    recovery_rate: float
    wrong_contacts: int
    smoothness: float
    command_jerk: float
    steps: int
    final_target_distance: float


@dataclass
class PolicyState:
    internal_phase: int = 0
    demo_index: int = 0
    contact_dwell: int = 0
    last_action: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))


def rotation(theta: float) -> np.ndarray:
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def apply_similarity(points: np.ndarray, theta: float, scale: float, translation: np.ndarray) -> np.ndarray:
    return points @ (scale * rotation(theta)).T + translation


def make_sequence(rng: np.random.Generator, horizon: int) -> np.ndarray:
    motif = MOTIFS[int(rng.integers(0, len(MOTIFS)))].copy()
    offset = int(rng.integers(0, len(motif)))
    motif = np.roll(motif, offset)
    if rng.random() < 0.35:
        motif = motif[::-1]
    seq: List[int] = []
    while len(seq) < horizon:
        block = motif.copy()
        if len(seq) >= 6 and rng.random() < 0.7:
            a, b = rng.choice(len(block), size=2, replace=False)
            block[a], block[b] = block[b], block[a]
        for value in block:
            if not seq or value != seq[-1]:
                seq.append(int(value))
            if len(seq) == horizon:
                break
    return np.asarray(seq, dtype=int)


def smooth_segment(start: np.ndarray, goal: np.ndarray, steps: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, steps, endpoint=False)
    ease = u * u * (3.0 - 2.0 * u)
    delta = goal - start
    perp = np.array([-delta[1], delta[0]])
    perp /= np.linalg.norm(perp) + 1e-9
    arc = 0.055 * np.sin(np.pi * u)[:, None] * perp[None, :]
    return start[None, :] + ease[:, None] * delta[None, :] + arc


def generate_demo(
    rng: np.random.Generator,
    objects: np.ndarray,
    start: np.ndarray,
    sequence: np.ndarray,
    corruption: float,
) -> Tuple[np.ndarray, np.ndarray]:
    points: List[np.ndarray] = [start.copy()]
    current = start.copy()
    for target_id in sequence:
        target = objects[int(target_id)]
        n_move = int(np.clip(np.linalg.norm(target - current) / 0.075, 9, 18))
        segment = smooth_segment(current, target, n_move)
        if corruption > 0.0 and rng.random() < 0.45 + corruption:
            mid = int(rng.integers(max(2, n_move // 4), max(3, 3 * n_move // 4)))
            width = max(2, n_move // 5)
            direction = rng.normal(size=2)
            direction /= np.linalg.norm(direction) + 1e-9
            magnitude = rng.uniform(0.15, 0.42) * corruption / 0.35
            lo, hi = max(0, mid - width), min(n_move, mid + width + 1)
            bump_u = np.linspace(-1.0, 1.0, hi - lo)
            bump = np.cos(0.5 * np.pi * bump_u)[:, None] * direction[None, :] * magnitude
            segment[lo:hi] += bump
        points.extend(segment[1:])
        # A fast two-frame mistake can visually approach a distractor, but does not
        # satisfy the low-speed sustained-contact rule used by latent intent.
        if corruption >= 0.18 and rng.random() < corruption:
            distractors = [j for j in range(len(objects)) if j != int(target_id)]
            wrong = objects[int(rng.choice(distractors))]
            points.append(0.72 * wrong + 0.28 * target)
            points.append(0.86 * wrong + 0.14 * target)
        dwell_noise = rng.normal(0.0, 0.008 + 0.018 * corruption, size=(4, 2))
        dwell = target[None, :] + dwell_noise
        if corruption > 0.0 and rng.random() < 0.4 * corruption:
            dwell[0] += rng.normal(0.0, 0.16, size=2)
        points.extend(dwell)
        current = target.copy()
    path = np.asarray(points, dtype=float)
    actions = np.diff(path, axis=0, prepend=path[:1])
    return path, actions


def make_scene_transform(rng: np.random.Generator, level: int) -> Tuple[float, float, np.ndarray, float]:
    if level == 0:
        theta = rng.uniform(-0.12, 0.12)
        scale = rng.uniform(0.96, 1.04)
        trans = rng.normal(0.0, 0.045, size=2)
        jitter = 0.012
    elif level == 1:
        theta = rng.uniform(-0.75, 0.75)
        scale = rng.uniform(0.84, 1.16)
        trans = rng.uniform(-0.34, 0.34, size=2)
        jitter = 0.065
    else:
        theta = rng.uniform(-1.75, 1.75)
        scale = rng.uniform(0.72, 1.30)
        trans = rng.uniform(-0.55, 0.55, size=2)
        jitter = 0.125
    return theta, scale, trans, jitter


def make_scenario(
    seed: int,
    shift_level: int,
    mirrored: bool,
    corruption: float,
    horizon: int,
    scenario_id: str,
) -> Scenario:
    rng = np.random.default_rng(seed)
    demo_theta = rng.uniform(-0.35, 0.35)
    demo_scale = rng.uniform(0.90, 1.10)
    demo_trans = rng.uniform(-0.18, 0.18, size=2)
    demo_objects = apply_similarity(CANONICAL_OBJECTS, demo_theta, demo_scale, demo_trans)
    demo_objects += rng.normal(0.0, 0.018, size=demo_objects.shape)
    deploy_theta, deploy_scale, deploy_trans, local_jitter = make_scene_transform(rng, shift_level)
    deploy_objects = apply_similarity(CANONICAL_OBJECTS, deploy_theta, deploy_scale, deploy_trans)
    deploy_objects += rng.normal(0.0, local_jitter, size=deploy_objects.shape)
    sequence = make_sequence(rng, horizon)
    demo_start = apply_similarity(np.array([[-0.92, -0.82]]), demo_theta, demo_scale, demo_trans)[0]
    deploy_start = apply_similarity(np.array([[-0.92, -0.82]]), deploy_theta, deploy_scale, deploy_trans)[0]
    deploy_start += rng.normal(0.0, 0.025 + 0.02 * shift_level, size=2)
    demo_path, demo_actions = generate_demo(rng, demo_objects, demo_start, sequence, corruption)
    embodiment = np.diag([-1.0, 1.0]) if mirrored else np.eye(2)
    perturb_scale = 0.20 + 0.055 * shift_level
    perturbations = []
    for _ in range(2):
        v = rng.normal(size=2)
        v /= np.linalg.norm(v) + 1e-9
        perturbations.append(v * rng.uniform(0.75, 1.15) * perturb_scale)
    return Scenario(
        scenario_id=scenario_id,
        seed=seed,
        shift_level=shift_level,
        mirrored=mirrored,
        corruption=float(corruption),
        horizon=horizon,
        demo_objects=demo_objects,
        deploy_objects=deploy_objects,
        demo_start=demo_start,
        deploy_start=deploy_start,
        true_sequence=sequence,
        demo_path=demo_path,
        demo_actions=demo_actions,
        embodiment=embodiment,
        perturbations=(perturbations[0], perturbations[1]),
    )


def fit_similarity(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    src_mean = source.mean(axis=0)
    dst_mean = target.mean(axis=0)
    src = source - src_mean
    dst = target - dst_mean
    u, singular, vt = np.linalg.svd(src.T @ dst)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    scale = float(singular.sum() / (np.sum(src * src) + 1e-9))
    linear = scale * r
    translation = dst_mean - src_mean @ linear.T
    return linear, translation


def transform_points(points: np.ndarray, linear: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points @ linear.T + translation


def normalized_distance_features(path: np.ndarray, objects: np.ndarray) -> np.ndarray:
    scene_scale = np.mean(np.linalg.norm(objects - objects.mean(axis=0), axis=1)) + 1e-9
    return np.linalg.norm(path[:, None, :] - objects[None, :, :], axis=2) / scene_scale


def extract_latent_sequence(path: np.ndarray, objects: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(path[:, None, :] - objects[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    nearest_dist = distances[np.arange(len(path)), nearest]
    speed = np.linalg.norm(np.diff(path, axis=0, prepend=path[:1]), axis=1)
    valid = (nearest_dist < 0.125) & (speed < 0.065)
    events: List[int] = []
    i = 0
    while i < len(path):
        if not valid[i]:
            i += 1
            continue
        j = i + 1
        while j < len(path) and valid[j] and nearest[j] == nearest[i]:
            j += 1
        if j - i >= 2:
            events.append(int(nearest[i]))
        i = j
    deduped: List[int] = []
    for event in events:
        if not deduped or event != deduped[-1]:
            deduped.append(event)
    return np.asarray(deduped, dtype=int)


def cap_action(delta: np.ndarray, max_norm: float = MAX_WORLD_STEP) -> np.ndarray:
    norm = float(np.linalg.norm(delta))
    if norm <= max_norm:
        return delta.copy()
    return delta * (max_norm / (norm + 1e-9))


def command_from_world(desired_world: np.ndarray, embodiment: np.ndarray, aware: bool) -> np.ndarray:
    if aware:
        return np.linalg.solve(embodiment, desired_world)
    return desired_world.copy()


def prepare_policy_data(sc: Scenario) -> Dict[str, np.ndarray]:
    linear, translation = fit_similarity(sc.demo_objects, sc.deploy_objects)
    mapped_path = transform_points(sc.demo_path, linear, translation)
    demo_features = normalized_distance_features(sc.demo_path, sc.demo_objects)
    deploy_scale = np.mean(np.linalg.norm(sc.deploy_objects - sc.deploy_objects.mean(axis=0), axis=1)) + 1e-9
    active_object = np.argmin(
        np.linalg.norm(sc.demo_path[:, None, :] - sc.demo_objects[None, :, :], axis=2), axis=1
    )
    # Object-centered anchors retain local target placement rather than relying only
    # on a single global scene transform.
    rel = sc.demo_path - sc.demo_objects[active_object]
    local_linear = linear / max(math.sqrt(abs(np.linalg.det(linear))), 1e-9)
    anchors = sc.deploy_objects[active_object] + rel @ local_linear.T
    latent_sequence = extract_latent_sequence(sc.demo_path, sc.demo_objects)
    return {
        "linear": linear,
        "translation": translation,
        "mapped_path": mapped_path,
        "demo_features": demo_features,
        "active_object": active_object,
        "anchors": anchors,
        "latent_sequence": latent_sequence,
        "deploy_scale": np.array([deploy_scale]),
    }


def policy_action(
    method: str,
    pos: np.ndarray,
    sc: Scenario,
    data: Dict[str, np.ndarray],
    state: PolicyState,
) -> np.ndarray:
    if method == "language_prior":
        planned = np.resize(PRIOR_MOTIF, sc.horizon)
        idx = min(state.internal_phase, len(planned) - 1)
        target = sc.deploy_objects[int(planned[idx])]
        if np.linalg.norm(pos - target) < CONTACT_RADIUS * 0.9:
            state.contact_dwell += 1
        else:
            state.contact_dwell = 0
        if state.contact_dwell >= CONTACT_DWELL and state.internal_phase < len(planned) - 1:
            state.internal_phase += 1
            state.contact_dwell = 0
            target = sc.deploy_objects[int(planned[state.internal_phase])]
        desired = cap_action(0.72 * (target - pos))
        command = command_from_world(desired, sc.embodiment, aware=True)
    elif method == "nearest_demo_replay":
        path = data["mapped_path"]
        lo = state.demo_index
        hi = min(len(path), lo + 18)
        if lo >= len(path) - 1:
            desired = np.zeros(2)
        else:
            local = path[lo:hi]
            nearest = lo + int(np.argmin(np.linalg.norm(local - pos[None, :], axis=1)))
            state.demo_index = max(state.demo_index, nearest)
            look = min(len(path) - 1, state.demo_index + 4)
            desired = cap_action(0.95 * (path[look] - pos))
            if np.linalg.norm(path[look] - pos) < 0.08:
                state.demo_index = min(len(path) - 1, state.demo_index + 2)
        # Literal replay assumes the demonstration's motor convention. This is the
        # intentionally brittle baseline under a mirrored command mapping.
        command = command_from_world(desired, sc.embodiment, aware=False)
    elif method == "phase_retrieval":
        deploy_scale = float(data["deploy_scale"][0])
        query = np.linalg.norm(pos[None, :] - sc.deploy_objects, axis=1) / deploy_scale
        feats = data["demo_features"]
        lo = max(0, state.demo_index - 2)
        hi = min(len(feats), state.demo_index + 28)
        costs = np.mean((feats[lo:hi] - query[None, :]) ** 2, axis=1)
        picked = lo + int(np.argmin(costs))
        state.demo_index = max(state.demo_index, picked)
        look = min(len(feats) - 1, state.demo_index + 5)
        target = data["anchors"][look]
        desired = cap_action(0.88 * (target - pos))
        command = command_from_world(desired, sc.embodiment, aware=True)
    elif method == "full_demo_attention":
        deploy_scale = float(data["deploy_scale"][0])
        query = np.linalg.norm(pos[None, :] - sc.deploy_objects, axis=1) / deploy_scale
        feats = data["demo_features"]
        path = data["anchors"]
        indices = np.arange(len(feats))
        feature_cost = np.mean((feats - query[None, :]) ** 2, axis=1)
        position_cost = np.sum((path - pos[None, :]) ** 2, axis=1) / (deploy_scale**2)
        backward = np.maximum(0, state.demo_index - indices) / 4.0
        forward = np.maximum(0, indices - state.demo_index - 30) / 10.0
        logits = -(feature_cost / 0.018 + position_cost / 0.035 + backward**2 + forward**2)
        logits -= np.max(logits)
        weights = np.exp(np.clip(logits, -45, 0))
        weights /= weights.sum() + 1e-12
        expected = int(round(float(np.sum(indices * weights))))
        state.demo_index = max(state.demo_index, expected)
        future_idx = np.minimum(indices + 7, len(indices) - 1)
        target = np.sum(weights[:, None] * path[future_idx], axis=0)
        desired = cap_action(0.92 * (target - pos))
        # Attention is smoothed to model a coherent full-context action readout.
        desired = 0.82 * desired + 0.18 * (sc.embodiment @ state.last_action)
        command = command_from_world(desired, sc.embodiment, aware=True)
    elif method == "latent_intent":
        seq = data["latent_sequence"]
        if len(seq) == 0:
            seq = sc.true_sequence[:1]
        idx = min(state.internal_phase, len(seq) - 1)
        target = sc.deploy_objects[int(seq[idx])]
        if np.linalg.norm(pos - target) < CONTACT_RADIUS * 0.88:
            state.contact_dwell += 1
        else:
            state.contact_dwell = 0
        if state.contact_dwell >= CONTACT_DWELL and state.internal_phase < len(seq) - 1:
            state.internal_phase += 1
            state.contact_dwell = 0
            target = sc.deploy_objects[int(seq[state.internal_phase])]
        desired = cap_action(0.82 * (target - pos))
        desired = 0.82 * desired + 0.18 * (sc.embodiment @ state.last_action)
        command = command_from_world(desired, sc.embodiment, aware=True)
    else:
        raise ValueError(method)
    state.last_action = command.copy()
    return command


def run_rollout(sc: Scenario, method: str) -> Rollout:
    data = prepare_policy_data(sc)
    state = PolicyState()
    rng = np.random.default_rng(sc.seed + 100_000 + METHODS.index(method) * 10_003)
    pos = sc.deploy_start.copy()
    positions = [pos.copy()]
    commands: List[np.ndarray] = []
    phases = [0]
    true_phase = 0
    contact_count = 0
    wrong_contacts = 0
    inside_wrong: set[int] = set()
    interventions = 0
    last_progress_step = 0
    perturb_phase_thresholds = sorted(set([max(1, sc.horizon // 3), max(1, 2 * sc.horizon // 3)]))
    perturb_index = 0
    pending_recovery_deadline: int | None = None
    disturbance_events = 0
    recoveries = 0
    max_steps = 34 + sc.horizon * 24

    for step in range(max_steps):
        command = policy_action(method, pos, sc, data, state)
        command = cap_action(command, max_norm=MAX_WORLD_STEP * 1.15)
        world_delta = sc.embodiment @ command
        # Small paired execution noise prevents exact geometric coincidences without
        # changing the qualitative action-mapping test.
        world_delta += rng.normal(0.0, 0.0025, size=2)
        pos = pos + world_delta
        commands.append(command.copy())

        if true_phase < sc.horizon:
            active_id = int(sc.true_sequence[true_phase])
            active_target = sc.deploy_objects[active_id]
            if np.linalg.norm(pos - active_target) < CONTACT_RADIUS:
                contact_count += 1
            else:
                contact_count = 0
            distances = np.linalg.norm(sc.deploy_objects - pos[None, :], axis=1)
            now_inside = {int(i) for i in np.where(distances < CONTACT_RADIUS * 0.9)[0] if int(i) != active_id}
            wrong_contacts += len(now_inside - inside_wrong)
            inside_wrong = now_inside

            if contact_count >= CONTACT_DWELL:
                true_phase += 1
                contact_count = 0
                last_progress_step = step
                if pending_recovery_deadline is not None and step <= pending_recovery_deadline:
                    recoveries += 1
                    pending_recovery_deadline = None
                if perturb_index < len(perturb_phase_thresholds):
                    threshold = perturb_phase_thresholds[perturb_index]
                    if true_phase >= threshold and true_phase < sc.horizon:
                        pos = pos + sc.perturbations[min(perturb_index, 1)]
                        disturbance_events += 1
                        pending_recovery_deadline = step + 24
                        perturb_index += 1

        if pending_recovery_deadline is not None and step > pending_recovery_deadline:
            pending_recovery_deadline = None

        if true_phase < sc.horizon and step - last_progress_step > STALL_LIMIT and interventions < MAX_INTERVENTIONS:
            active_target = sc.deploy_objects[int(sc.true_sequence[true_phase])]
            direction = pos - active_target
            if np.linalg.norm(direction) < 1e-6:
                direction = np.array([1.0, 0.0])
            direction /= np.linalg.norm(direction) + 1e-9
            pos = active_target + direction * 0.16
            interventions += 1
            last_progress_step = step
            contact_count = 0
            pending_recovery_deadline = None

        positions.append(pos.copy())
        phases.append(true_phase)
        if true_phase >= sc.horizon:
            break

    p = np.asarray(positions)
    a = np.asarray(commands) if commands else np.zeros((0, 2))
    phase_arr = np.asarray(phases)
    target_distances: List[float] = []
    for k, point in enumerate(p):
        phase = min(int(phase_arr[k]), sc.horizon - 1)
        target = sc.deploy_objects[int(sc.true_sequence[phase])]
        target_distances.append(float(np.linalg.norm(point - target)))
    tracking_rmse = float(np.sqrt(np.mean(np.square(target_distances))))
    tracking_progress = float(np.mean(phase_arr / sc.horizon))
    if len(a) >= 2:
        smoothness = float(np.sqrt(np.mean(np.sum(np.diff(a, axis=0) ** 2, axis=1))))
    else:
        smoothness = 0.0
    if len(a) >= 3:
        command_jerk = float(np.sqrt(np.mean(np.sum(np.diff(a, n=2, axis=0) ** 2, axis=1))))
    else:
        command_jerk = 0.0
    assisted_completion = bool(true_phase >= sc.horizon)
    success = bool(assisted_completion and interventions == 0)
    progress = float(true_phase / sc.horizon)
    if true_phase >= sc.horizon:
        final_target_distance = 0.0
    else:
        final_target_distance = float(
            np.linalg.norm(pos - sc.deploy_objects[int(sc.true_sequence[true_phase])])
        )
    recovery_rate = float(recoveries / disturbance_events) if disturbance_events else 0.0
    return Rollout(
        scenario=sc,
        method=method,
        positions=p,
        commands=a,
        phases=phase_arr,
        success=success,
        assisted_completion=assisted_completion,
        progress=progress,
        tracking_rmse=tracking_rmse,
        tracking_progress=tracking_progress,
        interventions=interventions,
        disturbance_events=disturbance_events,
        recoveries=recoveries,
        recovery_rate=recovery_rate,
        wrong_contacts=wrong_contacts,
        smoothness=smoothness,
        command_jerk=command_jerk,
        steps=len(a),
        final_target_distance=final_target_distance,
    )


def sanity_check() -> Dict[str, object]:
    sc = make_scenario(
        seed=31415,
        shift_level=2,
        mirrored=True,
        corruption=0.28,
        horizon=6,
        scenario_id="sanity",
    )
    data = prepare_policy_data(sc)
    inferred = data["latent_sequence"][: sc.horizon]
    extraction_exact = bool(len(inferred) == sc.horizon and np.array_equal(inferred, sc.true_sequence))
    desired_world = np.array([0.08, -0.035])
    command = command_from_world(desired_world, sc.embodiment, aware=True)
    realized = sc.embodiment @ command
    mirror_error = float(np.linalg.norm(realized - desired_world))
    unaware_realized = sc.embodiment @ desired_world
    unaware_error = float(np.linalg.norm(unaware_realized - desired_world))
    latent_target_error = float(
        np.linalg.norm(sc.deploy_objects[int(inferred[0])] - sc.deploy_objects[int(sc.true_sequence[0])])
    ) if len(inferred) else float("inf")
    passed = extraction_exact and mirror_error < 1e-10 and unaware_error > 0.1 and latent_target_error < 1e-10
    result = {
        "passed": passed,
        "true_sequence": sc.true_sequence.tolist(),
        "inferred_sequence": inferred.tolist(),
        "extraction_exact": extraction_exact,
        "aware_mirror_action_error": mirror_error,
        "unaware_mirror_action_error": unaware_error,
        "first_latent_target_error": latent_target_error,
    }
    if not passed:
        raise AssertionError(f"sanity check failed: {result}")
    return result


def rollout_row(r: Rollout) -> Dict[str, object]:
    sc = r.scenario
    return {
        "scenario_id": sc.scenario_id,
        "seed": sc.seed,
        "method": r.method,
        "shift_level": sc.shift_level,
        "mirrored": int(sc.mirrored),
        "corruption": sc.corruption,
        "horizon": sc.horizon,
        "success": int(r.success),
        "assisted_completion": int(r.assisted_completion),
        "progress": r.progress,
        "tracking_rmse": r.tracking_rmse,
        "tracking_progress": r.tracking_progress,
        "interventions": r.interventions,
        "disturbance_events": r.disturbance_events,
        "recoveries": r.recoveries,
        "recovery_rate": r.recovery_rate,
        "wrong_contacts": r.wrong_contacts,
        "smoothness": r.smoothness,
        "command_jerk": r.command_jerk,
        "steps": r.steps,
        "final_target_distance": r.final_target_distance,
    }


def aggregate(rows: Sequence[Rollout]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for method in METHODS:
        rr = [r for r in rows if r.method == method]
        events = sum(r.disturbance_events for r in rr)
        recoveries = sum(r.recoveries for r in rr)
        shifted = [r for r in rr if r.scenario.shift_level == 2 or r.scenario.mirrored or r.scenario.corruption >= 0.35]
        summary[method] = {
            "episodes": len(rr),
            "success_rate": float(np.mean([r.success for r in rr])),
            "assisted_completion_rate": float(np.mean([r.assisted_completion for r in rr])),
            "mean_progress": float(np.mean([r.progress for r in rr])),
            "tracking_rmse": float(np.mean([r.tracking_rmse for r in rr])),
            "tracking_progress": float(np.mean([r.tracking_progress for r in rr])),
            "mean_interventions": float(np.mean([r.interventions for r in rr])),
            "recovery_rate": float(recoveries / events) if events else 0.0,
            "wrong_contacts": float(np.mean([r.wrong_contacts for r in rr])),
            "smoothness": float(np.mean([r.smoothness for r in rr])),
            "command_jerk": float(np.mean([r.command_jerk for r in rr])),
            "mean_steps": float(np.mean([r.steps for r in rr])),
            "robust_success_rate": float(np.mean([r.success for r in shifted])) if shifted else 0.0,
        }
    return summary


def group_summary(rows: Sequence[Rollout], factor: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in METHODS:
        grouped[method] = {}
        values = sorted({getattr(r.scenario, factor) for r in rows})
        for value in values:
            rr = [r for r in rows if r.method == method and getattr(r.scenario, factor) == value]
            events = sum(r.disturbance_events for r in rr)
            grouped[method][str(value)] = {
                "episodes": len(rr),
                "success_rate": float(np.mean([r.success for r in rr])),
                "mean_progress": float(np.mean([r.progress for r in rr])),
                "tracking_rmse": float(np.mean([r.tracking_rmse for r in rr])),
                "mean_interventions": float(np.mean([r.interventions for r in rr])),
                "recovery_rate": float(sum(r.recoveries for r in rr) / events) if events else 0.0,
                "smoothness": float(np.mean([r.smoothness for r in rr])),
            }
    return grouped


def write_outputs(
    rows: Sequence[Rollout],
    summary: Dict[str, Dict[str, float]],
    grouped: Dict[str, object],
    sanity: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(rollout_row(rows[0]).keys())
    with (OUT / "demo_prompted_policy_trials.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for rollout in rows:
            row = rollout_row(rollout)
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})
    summary_fields = [
        "method", "episodes", "success_rate", "assisted_completion_rate", "mean_progress",
        "tracking_rmse", "tracking_progress", "mean_interventions", "recovery_rate",
        "wrong_contacts", "smoothness", "command_jerk", "mean_steps", "robust_success_rate",
    ]
    with (OUT / "demo_prompted_policy_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, lineterminator="\n")
        writer.writeheader()
        for method in METHODS:
            writer.writerow({"method": method, **summary[method]})
    payload = {
        "experiment": {
            "seed": args.seed,
            "trials_per_cell": args.trials_per_cell,
            "shift_levels": [0, 1, 2],
            "mirrored": [False, True],
            "corruption_levels": [0.0, 0.18, 0.35],
            "horizons": [3, 6, 9, 12],
            "scenario_count": len({r.scenario.scenario_id for r in rows}),
            "paired_rollout_count": len(rows),
        },
        "sanity_check": sanity,
        "overall": summary,
        "by_factor": grouped,
    }
    with (OUT / "demo_prompted_policy_metrics.json").open("w") as f:
        json.dump(payload, f, indent=2)
    with (OUT / "sanity_check.json").open("w") as f:
        json.dump(sanity, f, indent=2)


def plot_overview(summary: Dict[str, Dict[str, float]]) -> None:
    metrics = ["success_rate", "mean_progress", "recovery_rate", "mean_interventions", "smoothness"]
    titles = ["autonomous success", "phase progress", "disturbance recovery", "interventions (lower)", "action change (lower)"]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.7))
    x = np.arange(len(METHODS))
    colors = [METHOD_COLORS[m] for m in METHODS]
    for ax, metric, title in zip(axes, metrics, titles):
        values = [summary[m][metric] for m in METHODS]
        ax.bar(x, values, color=colors)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(["lang", "replay", "phase", "attention", "intent"], rotation=30)
        if metric in {"success_rate", "mean_progress", "recovery_rate"}:
            ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Demonstration as an in-context task prompt: paired Monte Carlo summary")
    fig.tight_layout()
    fig.savefig(OUT / "method_overview.png", dpi=190)
    plt.close(fig)


def plot_sweeps(grouped: Dict[str, object]) -> None:
    specs = [
        ("shift_level", ["0", "1", "2"], "object/location shift"),
        ("mirrored", ["False", "True"], "mirrored action mapping"),
        ("corruption", ["0.0", "0.18", "0.35"], "demo corruption"),
        ("horizon", ["3", "6", "9", "12"], "task horizon"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (factor, values, title) in zip(axes.flat, specs):
        factor_data = grouped[factor]  # type: ignore[index]
        x = np.arange(len(values))
        for method in METHODS:
            y = [factor_data[method][value]["success_rate"] for value in values]
            ax.plot(x, y, marker="o", lw=2, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.set_xticks(x)
        ax.set_xticklabels(values)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("autonomous success")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Robustness sweeps; all other factors are marginalized")
    fig.tight_layout()
    fig.savefig(OUT / "robustness_sweeps.png", dpi=190)
    plt.close(fig)


def plot_progress(grouped: Dict[str, object]) -> None:
    horizon_data = grouped["horizon"]  # type: ignore[index]
    horizons = ["3", "6", "9", "12"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.arange(len(horizons))
    for method in METHODS:
        axes[0].plot(x, [horizon_data[method][h]["mean_progress"] for h in horizons], marker="o", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axes[1].plot(x, [horizon_data[method][h]["mean_interventions"] for h in horizons], marker="o", color=METHOD_COLORS[method])
        axes[2].plot(x, [horizon_data[method][h]["tracking_rmse"] for h in horizons], marker="o", color=METHOD_COLORS[method])
    for ax, title, ylabel in zip(axes, ["progress", "oracle interventions", "target tracking"], ["completed phase fraction", "mean count", "RMSE"]):
        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
        ax.set_xlabel("number of task phases")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Long-horizon composition and progress tracking")
    fig.tight_layout()
    fig.savefig(OUT / "long_horizon_metrics.png", dpi=190)
    plt.close(fig)


def plot_representative(sc: Scenario, rollouts: Dict[str, Rollout]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
    axes_flat = axes.flat
    object_colors = plt.cm.tab10(np.linspace(0, 1, len(sc.deploy_objects)))
    for ax, method in zip(axes_flat, METHODS):
        r = rollouts[method]
        ax.plot(sc.demo_path[:, 0], sc.demo_path[:, 1], color="0.65", ls="--", lw=1.1, label="prompt trajectory (demo scene)")
        ax.plot(r.positions[:, 0], r.positions[:, 1], color=METHOD_COLORS[method], lw=2.1, label="deployment rollout")
        for j, point in enumerate(sc.deploy_objects):
            ax.scatter(*point, s=70, color=object_colors[j], edgecolor="k", linewidth=0.5)
            ax.text(point[0] + 0.025, point[1] + 0.025, str(j), fontsize=8)
        ax.scatter(*sc.deploy_start, marker="x", color="k", s=55)
        ax.set_title(f"{METHOD_LABELS[method]}\nprogress={r.progress:.2f}, intv={r.interventions}")
        ax.axis("equal")
        ax.grid(alpha=0.16)
    ax = axes_flat[5]
    ax.axis("off")
    seq = " → ".join(str(int(x)) for x in sc.true_sequence)
    ax.text(
        0.03, 0.90,
        "Representative hard prompt\n\n"
        f"task sequence: {seq}\n"
        f"shift level: {sc.shift_level}\n"
        f"mirrored commands: {sc.mirrored}\n"
        f"demo corruption: {sc.corruption:.2f}\n"
        f"horizon: {sc.horizon}\n\n"
        "Dashed gray: raw demonstration coordinates\n"
        "Colored lines: execution in shifted scene",
        va="top", fontsize=11,
    )
    fig.suptitle("One demonstration prompt, five execution mechanisms")
    fig.tight_layout()
    fig.savefig(OUT / "representative_rollout.png", dpi=190)
    plt.close(fig)


def build_scenarios(seed: int, trials_per_cell: int, smoke: bool) -> List[Scenario]:
    shift_levels = [0, 1, 2]
    mirrors = [False, True]
    corruptions = [0.0, 0.18, 0.35]
    horizons = [3, 6, 9, 12]
    if smoke:
        shift_levels = [0, 2]
        corruptions = [0.0, 0.35]
        horizons = [3, 9]
        trials_per_cell = 1
    scenarios: List[Scenario] = []
    index = 0
    for shift in shift_levels:
        for mirror in mirrors:
            for corruption in corruptions:
                for horizon in horizons:
                    for trial in range(trials_per_cell):
                        scenario_seed = seed + index * 7919
                        scenario_id = f"s{shift}_m{int(mirror)}_c{int(round(corruption*100)):02d}_h{horizon:02d}_t{trial:02d}"
                        scenarios.append(make_scenario(scenario_seed, shift, mirror, corruption, horizon, scenario_id))
                        index += 1
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--trials-per-cell", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    sanity = sanity_check()
    scenarios = build_scenarios(args.seed, args.trials_per_cell, args.smoke)
    rows: List[Rollout] = []
    representative: Tuple[Scenario, Dict[str, Rollout]] | None = None
    for sc in scenarios:
        paired: Dict[str, Rollout] = {}
        for method in METHODS:
            rollout = run_rollout(sc, method)
            rows.append(rollout)
            paired[method] = rollout
        if representative is None and sc.shift_level == max(s.shift_level for s in scenarios) and sc.mirrored and sc.corruption == max(s.corruption for s in scenarios) and sc.horizon == max(s.horizon for s in scenarios):
            representative = (sc, paired)

    summary = aggregate(rows)
    grouped: Dict[str, object] = {
        factor: group_summary(rows, factor)
        for factor in ["shift_level", "mirrored", "corruption", "horizon"]
    }
    write_outputs(rows, summary, grouped, sanity, args)
    plot_overview(summary)
    if not args.smoke:
        plot_sweeps(grouped)
        plot_progress(grouped)
    if representative is None:
        representative = (scenarios[-1], {m: run_rollout(scenarios[-1], m) for m in METHODS})
    plot_representative(*representative)

    print(json.dumps({"sanity": sanity, "overall": summary}, indent=2))
    print(f"wrote {OUT / 'demo_prompted_policy_trials.csv'}")
    print(f"wrote {OUT / 'demo_prompted_policy_summary.csv'}")
    print(f"wrote {OUT / 'demo_prompted_policy_metrics.json'}")
    print(f"wrote plots in {OUT}")


if __name__ == "__main__":
    main()
