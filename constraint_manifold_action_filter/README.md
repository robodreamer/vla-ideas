# Constraint-Manifold Action Filter

This PR-MPPI-inspired toy places explicit geometry between a learned action proposal and execution. A behavior-cloned controller proposes aggressive seven-step tray-motion chunks. Four execution strategies are compared:

- `penalty_only`: soft equality and obstacle penalties are baked into the proposal.
- `one_step_correction`: only the first chunk action receives sequential ambient-space correction.
- `rollout_projection`: every proposed action is projected into the equality tangent and inequality half-spaces before the toy rollout advances.
- `projection_retraction`: projected actions are followed by Gauss–Newton retraction of each finite state step onto the equality manifold.

The tray state is `q = [x, y, a, b]`. The orientation vector obeys the equality `a² + b² = 1`; sampled points along the tray obey differentiable obstacle-clearance and workspace inequalities. Projection is solved in the three-dimensional equality tangent using deterministic Dykstra half-space projection.

This is an explanatory action-filter benchmark, **not** a reproduction of PR-MPPI. It does not implement MPPI sampling or weighting; the learned BC chunk is the proposal source used to isolate the projection/retraction execution layer.

## Primary references

- Sanghyun Kim et al., [Projection-Retraction MPPI: Exact Constraint-Manifold Control for Manipulators](https://arxiv.org/abs/2608.07573), arXiv:2608.07573, August 2026.
- [PR-MPPI project page](https://rcilab.khu.ac.kr/prmppi/).

## Run

```bash
cd /home/andypark/Projects/repos/vla-ideas
PYTHONWARNINGS=error \
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  constraint_manifold_action_filter/run_constraint_manifold_action_filter.py \
  --seed 23 --trials 120 --train-samples 7000 --test-samples 1400
```

Fast verification:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  constraint_manifold_action_filter/run_constraint_manifold_action_filter.py \
  --smoke --seed 23
```

Render the report:

```bash
constraint_manifold_action_filter/docs/render_pdf.sh
```

## Latest verified result

The full run trains an Extra Trees behavior clone with held-out action RMSE `0.0889`, then evaluates 120 paired scenarios: 60 aggressive in-distribution cases and 60 more aggressive/OOD cases.

| Method | Success | Collision | p95 equality residual | Mean inequality violation | Min margin | Intervention | Deadlock |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Penalty only | 0.0% | 36.7% | 2.59e-1 | 4.91e-2 | 0.0147 | 0.000 | 0.0% |
| One-step correction | 0.0% | 33.3% | 2.56e-1 | 4.08e-2 | 0.0264 | 0.040 | 0.0% |
| Rollout projection | 0.0% | **0.0%** | 2.23e-1 | **0.0** | 0.0995 | 0.815 | 11.7% |
| Projection + retraction | **63.3%** | **0.0%** | **9.69e-11** | **0.0** | **0.1001** | 0.804 | 9.2% |

Success requires reaching the goal, avoiding collision, and satisfying the equality tolerance. Projection alone removes inequality violations but leaves finite-step equality drift, so it records zero full success despite safe progress. Retraction reduces the equality residual to numerical tolerance and converts 63.3% of all trials into full successes. In the aggressive half, projection+retraction succeeds on 95.0%; in the OOD half it succeeds on 31.7% and exposes the expected feasibility/deadlock limitation.

![Monte Carlo summary](outputs/monte_carlo_summary.png)

## Deterministic checks

`outputs/sanity_checks.json` verifies:

- equality-tangent projection;
- simultaneous equality/inequality half-space satisfaction;
- ambient correction can break equality tangency;
- finite-step equality drift scales as `O(dt²)` (measured ratio `4.0` when halving `dt`);
- Gauss–Newton retraction reaches `9.23e-13` residual in two iterations;
- contradictory half-spaces are detected as locally infeasible;
- rank-degenerate retraction fails cleanly.

## Outputs

- `outputs/trial_metrics.csv`: paired per-trial metrics.
- `outputs/summary.csv`: all/aggressive/OOD aggregates.
- `outputs/metrics.json`: configuration, BC fit, scenarios, checks, and summaries.
- `outputs/sanity_checks.json`: deterministic geometry tests.
- `outputs/monte_carlo_summary.png`: headline comparison.
- `outputs/representative_rollout.png`: paired OOD trajectory, residual, margin, and intervention traces.
- `docs/constraint_manifold_action_filter_report.tex` and `.pdf`: source-grounded report.

## Limits

- The proposal model is low-dimensional BC, not a VLA.
- The toy filters one learned chunk; it does not sample and weight MPPI trajectories.
- Equality/inequality functions and exact object geometry are known.
- Dykstra projection is a compact CPU solver, not the paper's active-set/CUDA implementation.
- Local hard constraints can be infeasible. The filter uses a stop fallback, which explains the nonzero fallback/deadlock rates under OOD proposals.
- Retraction enforces the equality but does not magically restore task progress or resolve incompatible constraints.
