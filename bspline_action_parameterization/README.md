# B-spline Action Parameterization Toy

This folder explores the core idea from **B-spline Policy: Accelerating Manipulation Policies via B-spline Action Representations** with a compact runnable 2D experiment.

The paper/repo motivation is that robot policies often predict short dense action chunks at a low policy rate, while the robot controller runs much faster. B-spline Policy replaces the dense discrete chunk with a continuous B-spline action representation: a knot vector plus control points that can be decoded/resampled for higher-frequency execution and temporal speed-up.

## What is implemented

`run_bspline_action_toy.py` is intentionally narrow. It does **not** implement a diffusion model, image observations, imitation training, or real robot deployment. It isolates the action-representation question:

- `discrete_safe_1x`: noisy 10 Hz waypoint chunks executed at the original demonstration timing.
- `discrete_fast_3x`: the same waypoint chunk compressed to 3x speed, then tracked by a limited 200 Hz point-mass controller.
- `discrete_adaptive_3x`: the dense waypoint geometry with a curvature-aware scalar time law estimated from the emitted waypoint polyline.
- `bspline_fast_3x`: a compact cubic B-spline fitted to the same noisy low-rate actions and sampled at 200 Hz under the same nominal 3x timing.
- `bspline_adaptive_3x`: the same B-spline geometry, but with a curvature-aware scalar time law estimated from the fitted spline derivatives.

The toy uses a curved 2D demonstration manifold. A rollout is considered out-of-distribution / failed if the executed plant leaves a narrow tube around that manifold.

## References checked

Primary B-spline Policy sources:

- Official repo: `https://github.com/B-spline-policy/bspline-policy`
- Project page: `https://b-spline-policy.github.io/`
- arXiv: `https://arxiv.org/abs/2607.09648`

Implementation clues used for this toy:

- The repo represents actions as a dense matrix shaped like `(chunk_size + 2 * degree, 1 + action_dim)`, where column 0 stores knots and the remaining columns store control points.
- `bspline_policy.common.bspline_action.decode_bspline_action` decodes a predicted B-spline parameter matrix into regular action vectors with SciPy's `BSpline`.
- The repo's examples emphasize temporal speed-up of manipulation policy execution after fitting/predicting B-spline action chunks.

## Run

Use a Python environment with `numpy`, `scipy`, and `matplotlib`:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python bspline_action_parameterization/run_bspline_action_toy.py --trials 160
```

For a quick smoke test:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python bspline_action_parameterization/run_bspline_action_toy.py --quick
```

## Outputs

Generated files live in `bspline_action_parameterization/outputs/`:

- `bspline_action_metrics.csv`: per-trial metrics.
- `bspline_action_summary.json`: aggregate metrics and config.
- `bspline_action_rollout.png`: representative trajectory and path-error plot.
- `bspline_action_monte_carlo.png`: aggregate metric bars.

The generated writeup and LaTeX report live at:

- `bspline_action_parameterization/docs/bspline_action_toy_report.md`
- `bspline_action_parameterization/docs/bspline_action_report.tex`
- `bspline_action_parameterization/docs/bspline_action_report.pdf`

## Latest verified run

Command:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python bspline_action_parameterization/run_bspline_action_toy.py --trials 160
```

Results:

- `discrete_safe_1x`: 100.0% success, 0.0% collision/OOD, 6.75 s mean evaluation duration, 0.0268 mean max path error, 1782.3 RMS command jerk, 29.9 RMS plant jerk.
- `discrete_fast_3x`: 0.0% success, 100.0% collision/OOD, 2.75 s mean evaluation duration, 0.1020 mean max path error, 8390.3 RMS command jerk, 114.3 RMS plant jerk.
- `discrete_adaptive_3x`: 100.0% success, 0.0% collision/OOD, 4.15 s mean evaluation duration, 0.0424 mean max path error, 4214.0 RMS command jerk, 73.6 RMS plant jerk.
- `bspline_fast_3x`: 0.0% success, 100.0% collision/OOD, 2.75 s mean evaluation duration, 0.0997 mean max path error, 176.0 RMS command jerk, 81.0 RMS plant jerk.
- `bspline_adaptive_3x`: 100.0% success, 0.0% collision/OOD, 4.13 s mean evaluation duration, 0.0388 mean max path error, 353.9 RMS command jerk, 52.6 RMS plant jerk.

Evaluation duration includes the appended 0.75 s settle window, so the reported speed-up is conservative relative to command-only duration. The main signature is smoothness and timing control: the raw fast B-spline has much lower command jerk than the raw fast waypoint chunk, but still exceeds the plant/tube limits at the full 3x request. Adaptive retiming helps both dense and B-spline chunks; the B-spline version preserves similar success with many fewer predicted scalars and lower command/plant jerk.

## Mapping to the real method

| Toy component | B-spline Policy analogue | Simplification |
| --- | --- | --- |
| Noisy 10 Hz waypoints | Dense action chunk from a VLA/diffusion policy | Analytic path plus noise, not a learned model. |
| Cubic `BSpline(t, c, k)` | Knot vector plus control points | Fixed open-uniform knots; policy-predicted scalar count is control points only. |
| 200 Hz point-mass PD tracking | High-rate robot controller | No manipulator dynamics, contacts, perception, or IK. |
| 3x speed-up | Faster B-spline action rollout | No closed-loop replanning. |
| Curvature-aware time law | Safe temporal scaling of continuous action curves | Simple scalar slowdown, not a robot trajectory optimizer. |

## Takeaway

The useful part of the B-spline representation is not magic success at arbitrary speed. It gives the policy a compact continuous action object that can be decoded at high control rate and time-parameterized. In this toy, adaptive retiming is what recovers success; B-splines make that retiming smoother and more compact than dense waypoint chunks.
