# Path-Consistent Safety Filtering Toy

This folder explores the PACS idea from **From Demonstrations to Safe Deployment: Path-Consistent Safety Filtering for Diffusion Policies** with a small runnable 2D experiment.

PACS addresses a failure mode of safety filters for diffusion policies / VLAs: a reactive filter can keep the next command safe while pushing the robot onto a geometric path the policy never saw during behavior cloning. PACS instead treats the action chunk as a trajectory-level object: form waypoints, generate a feasible trajectory, monitor it with reachability, and slow or stop along the same intended path when the future occupancy becomes unsafe.

## What is implemented

`run_pacs_toy.py` is intentionally narrow. It does **not** implement sara_shield, full set-based reachability, Ruckig, a diffusion model, or robot dynamics. It isolates the control distinction:

- `nominal`: raw chunk path with no safety filter.
- `reactive_cbf_like`: per-step repulsive velocity away from a moving obstacle, plus weak recovery toward the path. This is a toy CBF-like correction, not a valid CBF.
- `pacs_time_law`: keeps the demonstrated path geometry fixed and modulates only scalar progress speed `ds/dt` when a future path occupancy check predicts the moving obstacle will be too close. The jerk/acceleration-limited speed update is a toy stand-in for a trajectory generator such as Ruckig.

The toy uses a curved 2D demonstration manifold as the policy's action-chunk path. A circular human/obstacle crosses the path at randomized times and locations.

## References checked

Primary PACS sources:

- Project page: `https://tum-lsy.github.io/pacs/`
- arXiv: `https://arxiv.org/abs/2511.06385`
- Safety-filter ROS 2 repo: `https://github.com/JulianBalletshofer/pacs-ros2`
- Simulation repo: `https://github.com/JakobThumm/robomimic`

Implementation clues used for this toy:

- PACS computes a trajectory from action-chunk waypoints and performs path-consistent braking rather than reactive lateral redirection.
- The ROS 2 README says sara_shield verifies safety at 1 kHz and needs Ruckig Pro for trajectories from waypoints.
- `trajectory_parameters_panda.yaml` includes failsafe and speed/limit parameters such as `max_s_stop`, `v_safe`, joint velocity/acceleration/jerk limits, and `reachability_set_interval_size`.
- The simulation repo compares raw OSC, CBF, SSM, and PFL/PACS; SSM/PFL use `experiment_configs_selection/failsafe_waypoints/`, while CBF uses `experiment_configs_selection/cbf/`.

## Run

Use a Python environment with `numpy` and `matplotlib`:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python path_consistent_safety_filtering/run_pacs_toy.py --trials 180
```

For a quick smoke test:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python path_consistent_safety_filtering/run_pacs_toy.py --quick
```

## Outputs

Generated files live in `path_consistent_safety_filtering/outputs/`:

- `pacs_toy_metrics.csv`: per-trial metrics.
- `pacs_toy_summary.json`: aggregate metrics.
- `pacs_toy_rollout.png`: representative trajectory and speed profile.
- `pacs_toy_monte_carlo.png`: aggregate metric bars.

## Latest verified run

Command:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python path_consistent_safety_filtering/run_pacs_toy.py --trials 180
```

Results:

- `nominal`: 11.1% success, 88.9% collision, 0.0% OOD time, 4.34 s mean duration.
- `reactive_cbf_like`: 64.4% success, 29.4% collision, 15.4% OOD time, 6.46 s mean duration, 0.241 max path error.
- `pacs_time_law`: 88.9% success, 0.6% collision, 0.0% OOD time, 6.12 s mean duration, 0.00087 max path error.

The intended PACS signature shows up: the time-law filter is slower than the raw policy, but it stays on the action-chunk path and avoids almost all collisions. The reactive filter improves over raw execution, but pays for safety with lateral path deviations that become OOD under the toy demo-tube metric.

## Mapping to the real method

| Toy component | PACS analogue | Simplification |
| --- | --- | --- |
| Curved 2D path | DP/VLA action chunk converted to waypoints | No learned model; path is analytic. |
| Future clearance scan | Set-based reachability check | Uses point/circle distances, no zonotopes/capsules/uncertainty propagation. |
| Scalar `ds/dt` slowdown | Path-consistent braking / failsafe trajectory | Does not solve multi-DoF kinodynamic OTG. |
| Jerk/accel-limited speed update | Ruckig-style smooth trajectory generation | Simple Euler update instead of analytical Ruckig profiles. |
| Distance-to-path OOD score | Staying near training distribution | Uses a geometric proxy, not visual/proprioceptive density. |

## Takeaway for the slowdown question

In this toy, slowdown is implemented by lowering the target scalar speed along a fixed path, then respecting jerk/acceleration limits while updating `ds/dt`. That corresponds to the PACS intuition: the safety layer does not need to invent a new geometric command; it can preserve the action-chunk waypoints and change the timing / stopping target. In a real Ruckig-backed stack, that likely means changing the trajectory target or time parameterization and using configured velocity/acceleration/jerk limits, not issuing lateral reactive corrections.
