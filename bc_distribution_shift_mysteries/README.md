# BC Distribution-Shift Mysteries

This self-contained experiment distills several observations from Seohong Park's August 2026 **Behavioral cloning mystery** write-up into a deterministic toy benchmark:

> Why can a policy with better held-out action prediction behave worse after it induces its own test-time state distribution?

The package contains two levels:

1. An exact one-dimensional sanity check where every policy query adds a small boundary error. Longer open-loop chunks query less often and therefore compound fewer boundary errors.
2. A learned two-dimensional obstacle-avoidance benchmark using narrow, smooth, temporally correlated demonstrations. Ridge behavioral-cloning policies sweep chunk horizon, action history, feature scaling, and linear-versus-quadratic basis capacity.

This is an explanatory toy, not a reproduction of the source benchmark, a flow-matching implementation, a VLA, or evidence that any one chunk horizon or feature scaling is universally best.

## References

- Seohong Park, [Behavioral cloning mystery](https://seohong.me/blog/behavioral-cloning-mystery/), August 2026.
- Pim de Haan, Dinesh Jayaraman, and Sergey Levine, [Causal Confusion in Imitation Learning](https://arxiv.org/abs/1905.11979), NeurIPS 2019.
- Seohong Park, Kevin Frans, Sergey Levine, and Aviral Kumar, [Is Value Learning Really the Main Bottleneck in Offline RL?](https://arxiv.org/abs/2406.09329), NeurIPS 2024.

## Run

Full deterministic sweep:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  bc_distribution_shift_mysteries/run_bc_distribution_shift_mysteries.py \
  --seed 17 --eval-episodes 96
```

Fast smoke run:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  bc_distribution_shift_mysteries/run_bc_distribution_shift_mysteries.py \
  --smoke --seed 17
```

The smoke run writes a reduced result set into the same `outputs/` directory, so rerun the full command before publishing or comparing the headline metrics.

Render the report:

```bash
bc_distribution_shift_mysteries/docs/render_pdf.sh
```

## Design

The full sweep trains 48 ridge-BC policies across:

- action horizon `H ∈ {1, 4, 8, 16}`;
- action-history conditioning `history ∈ {0, 4}`;
- input scaling `{state_focus, balanced, clock_focus}`;
- basis capacity `{linear, quadratic}`.

The two-dimensional demonstrations follow one of two smooth routes around a central obstacle. Demonstrator actions are temporally correlated. At evaluation time, every policy query adds a small seeded handoff shock and two fixed-time state perturbations push the rollout away from the expert tube.

The observation contains both corrective state features and clock features. On narrow expert trajectories those feature groups predict similar actions, but only state-relative features respond directly to rollout error. History adds previous actions, which are highly predictive in demonstrations and can therefore become a shortcut under closed-loop distribution shift.

## Metrics

- **Validation action MSE:** supervised prediction error on held-out expert states.
- **Success:** collision-free episode ending within the goal tolerance.
- **Goal distance:** final Euclidean distance from the goal.
- **Tracking RMSE:** deviation from the nominal expert route.
- **Maximum state divergence:** largest distance from the nominal route.
- **OOD fraction:** fraction of rollout states outside the configured expert-support radius.
- **On-policy oracle MSE:** action error against the scripted oracle on policy-induced states; available here only because this is a simulator.
- **Smoothness:** mean squared action difference.
- **Query count:** policy invocations per 72-step episode.

## Latest verified result

Command: `--seed 17 --eval-episodes 96`, with 90 training, 30 validation, and 96 paired evaluation episodes per model.

| Policy | Validation action MSE | Success | Goal distance | Tracking RMSE | Max divergence | Smoothness | Queries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H1, no history, state-focused, quadratic | 0.000872 | 21.9% | 0.399 | 0.237 | 0.437 | 0.01305 | 72 |
| H1, history 4, state-focused, quadratic | **0.000266** | 25.0% | 0.582 | 0.336 | 0.684 | 0.01420 | 72 |
| H8, no history, state-focused, quadratic | 0.000729 | **42.7%** | **0.217** | **0.202** | **0.322** | **0.00480** | 9 |

The history-conditioned one-step policy fits held-out expert actions substantially better, yet it has much worse goal distance, tracking error, state divergence, and on-policy oracle error than the H8 policy. The H8 policy improves success by **20.8 percentage points** over H1 and **17.7 points** over history-conditioned H1 in this setup.

Feature scaling changes rollout behavior despite nearly identical validation fit:

| H8/no-history/quadratic scaling | Validation action MSE | Success | Goal distance | Tracking RMSE | On-policy oracle MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| state focus | 0.000729 | 42.7% | 0.217 | 0.202 | 0.0912 |
| balanced | 0.000714 | **65.6%** | **0.127** | **0.185** | **0.0548** |
| clock focus | 0.000712 | 44.8% | 0.174 | 0.204 | 0.0779 |

The three validation losses differ by less than 2.5%, while success spans 22.9 percentage points. The result supports the narrow claim that supervised fit on expert states does not identify the most robust policy under induced rollout shift.

The exact sanity check also passes: H1 makes 20 queries and accumulates 0.20 final error, while H5 makes 4 queries and accumulates 0.04 final error under the scripted query-boundary mechanism.

![BC mystery summary](outputs/mystery_summary.png)

## Outputs

- `outputs/sanity_check.json`: exact query-boundary compounding check.
- `outputs/model_metrics.csv`: aggregate metrics for all 48 models.
- `outputs/rollout_trials.csv`: per-model, per-episode metrics.
- `outputs/metrics.json`: configuration, claims, sanity checks, and aggregate metrics.
- `outputs/mystery_summary.png`: selected offline-versus-rollout comparisons.
- `outputs/sweep_horizon_scaling.png`: horizon/history/scaling sweep.
- `outputs/validation_vs_rollout.png`: validation MSE versus rollout success.
- `outputs/feature_scaling_rollouts.png`: representative trajectories for the three scaling choices.
- `docs/bc_distribution_shift_mysteries_report.tex` and `.pdf`: source-grounded experiment report.

## Mapping to the source and limits

- **Open-loop versus closed-loop:** represented by chunk horizon and a per-query handoff-error mechanism, not flow-policy stochasticity or robot inference latency.
- **History can hurt:** represented by previous-action shortcuts and a larger polynomial input space, not image history or recurrent VLA state.
- **Feature scaling matters:** represented by ridge regularization selecting among correlated state and clock explanations that fit expert data similarly.
- **Large policies / beneficial overfitting:** capacity is swept with linear and quadratic bases, but this toy does not claim to reproduce billion-parameter scaling or the beneficial-overfitting result.

The benchmark is low-dimensional, demonstrations are scripted, feature groups are intentionally constructed, perturbations are synthetic, and the on-policy oracle metric is available only because the true controller is known. The result should be read as a causal-mechanism probe and a regression test for offline/online metric disagreement—not as a performance recommendation for real robots.
