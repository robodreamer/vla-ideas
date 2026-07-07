# PACS toy report

## Hypothesis

For action-chunking diffusion policies, safety interventions should preserve path geometry and modify timing where possible. If the filter changes the path in configuration/task space, the next policy observation can be out of distribution even if the instantaneous safety constraint is satisfied.

## Experiment

A point robot follows a fixed curved path, representing a diffusion-policy action chunk converted to waypoints. A circular dynamic obstacle crosses the path. Three controllers are compared:

1. Raw policy path.
2. Reactive CBF-like repulsion away from the obstacle.
3. PACS-style time-law filter that slows/stops progress along the original path when future clearance is unsafe.

Metrics:

- success: reaches end without collision and without excessive OOD time;
- collision rate;
- OOD fraction: fraction of rollout outside a narrow demonstration tube;
- mean and max path error;
- duration;
- jerk proxy on scalar progress speed.

## Results from latest 180-trial run

| method | success | collision | OOD time | mean duration | max path error |
| --- | ---: | ---: | ---: | ---: | ---: |
| nominal | 0.111 | 0.889 | 0.000 | 4.34 s | 0.00075 |
| reactive_cbf_like | 0.644 | 0.294 | 0.154 | 6.46 s | 0.24085 |
| pacs_time_law | 0.889 | 0.006 | 0.000 | 6.12 s | 0.00087 |

## Interpretation

The raw policy is fast but unsafe. The reactive baseline can avoid some collisions, but it often creates a sideways detour; under the toy's demonstration-tube metric, that is exactly the kind of OOD state PACS is designed to avoid. The PACS-style controller is slower because it waits for the obstacle to clear, but it keeps path geometry nearly unchanged and resumes naturally.

The interesting part is not the exact numbers; this simulator is too simplified for that. The useful result is the qualitative decomposition:

```text
policy / VLA proposes geometry: q(s)
safety filter edits timing:      s(t)
reactive shield edits geometry:  q'(s) != q(s)
```

That decomposition gives a clean toy answer to the slowdown question: a PACS-like slowdown can be treated as a change in time parameterization or stopping target along the waypoint trajectory. Ruckig or another OTG solver can then produce a smooth feasible trajectory subject to velocity, acceleration, and jerk constraints.

## Limitations

- No real robot dynamics, joint limits, torque limits, force limits, IK, or visual observations.
- No true control barrier function or formal reachable set propagation.
- No diffusion-policy training; the demonstration distribution is represented as a handcrafted tube.
- The time-law update is only a Ruckig intuition proxy.

## Next useful extension

A better second toy would use a small 2-link or 3-link planar arm where the action chunk is joint-space waypoints. Then the path-consistent brake can be tested in joint space while workspace obstacle distances are checked against link capsules. That would map more directly to PACS's robot occupancy reasoning and to the difference between joint-space path consistency and end-effector-only consistency.
