# B-spline Action Parameterization Toy Report

This run tests the B-spline Policy intuition in a tiny 2D setting: a chunked policy emits noisy low-rate waypoints, while the plant executes at 200 Hz with velocity/acceleration limits.

The toy is intentionally not a diffusion model or robot benchmark. It isolates whether replacing a dense discrete action chunk with cubic B-spline knots/control points gives a smoother high-frequency command when the same geometric behavior is temporally scaled.

## Latest verified run

- Trials: `160`
- Low-level execution: `200 Hz`
- Policy waypoint rate: `10 Hz`
- Requested speed-up: `3.0x`
- B-spline controls: `14` per axis, degree `3`

| Method | Success | Collision/OOD | Eval duration | Max path error | Cmd jerk | Plant jerk | Predicted scalars | Full scalars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Discrete chunks, 1x | 100.0% | 0.0% | 6.75s | 0.027 | 1782.3 | 29.9 | 122 | 122 |
| Discrete chunks, 3x | 0.0% | 100.0% | 2.75s | 0.102 | 8390.3 | 114.3 | 122 | 122 |
| Discrete + curvature time-law, 3x | 100.0% | 0.0% | 4.15s | 0.042 | 4214.0 | 73.6 | 122 | 122 |
| B-spline chunks, 3x | 0.0% | 100.0% | 2.75s | 0.100 | 176.0 | 81.0 | 28 | 46 |
| B-spline + curvature time-law, 3x | 100.0% | 0.0% | 4.13s | 0.039 | 353.9 | 52.6 | 28 | 46 |

## What the toy suggests

- Naively executing the same low-rate discrete action waypoints faster is brittle: it shortens nominal duration, but the limited plant lags and leaves the demonstration tube more often.
- B-spline chunks preserve a continuous action curve. The fast B-spline variant uses fewer predicted scalars than the dense waypoint chunk and produces lower command jerk under the same speed-up.
- Both adaptive variants test retiming rather than replanning. The B-spline version computes curvature from the fitted action curve and shows why a continuous representation is convenient for local time scaling; the discrete adaptive baseline is included to avoid attributing all recovery uniquely to B-splines.

## Mapping to B-spline Policy

| Toy component | B-spline Policy analogue | Simplification |
| --- | --- | --- |
| Noisy 10 Hz waypoints | Discrete-time action chunk from a VLA/diffusion policy | The policy is analytic plus noise, not learned. |
| Cubic `BSpline(t, c, k)` | Predicted knot vector plus control points | Knots are fixed open-uniform; only control points are fit. |
| 200 Hz PD plant | Low-level robot controller | Simple point-mass limits, no manipulator dynamics. |
| 3x execution and curvature slowdown | Temporal scaling of continuous B-spline actions | No real scheduler, replanning, perception, or contact. |
| Demo-tube OOD metric | Staying near the demonstrated action manifold | Geometric proxy, not visual/proprioceptive distribution shift. |

## Generated artifacts

- `outputs/bspline_action_metrics.csv`
- `outputs/bspline_action_summary.json`
- `outputs/bspline_action_rollout.png`
- `outputs/bspline_action_monte_carlo.png`
