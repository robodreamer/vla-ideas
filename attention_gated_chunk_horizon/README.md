# Attention-Gated Chunk Horizon Toy

This package tests a narrow mechanism inspired by **Knowing When to Stop: Adaptive Action Chunking via Internal Cross-Attention Dynamics in VLAs**.

The question is:

> Can a training-free stopping rule based on a sustained plateau in action-to-observation cross-attention entropy choose useful execution horizons more efficiently than short fixed chunks, and more reliably than state-rollout or ensemble-disagreement proxies?

This is a deterministic NumPy/Matplotlib mechanism toy. It is **not** a reproduction of the paper, a VLA evaluation, or evidence that attention entropy is calibrated uncertainty in general. The synthetic entropy signal is deliberately coupled to the toy’s future-error boundary, so the result tests stopping logic under favorable temporal alignment rather than establishing that real attention is predictive.

## Primary source and claim boundaries

Primary source checked:

- Xiaolong Shan, Shuang Dai, Yu Wang, and Jincheng Yu, [*Knowing When to Stop: Adaptive Action Chunking via Internal Cross-Attention Dynamics in VLAs*](https://arxiv.org/abs/2609.00908), arXiv:2609.00908v1, September 1, 2026.

The paper reports that action-to-VLM cross-attention entropy often rises toward a high plateau along the predicted action horizon, that higher entropy is associated with larger offline action error, and that a training-free high-and-stable plateau rule improves the evaluated average success/efficiency trade-off. Its reported rule smooths entropy, requires a `k`-step high-entropy window and a small mean entropy change, and otherwise executes the full predicted horizon. The paper evaluates real VLA policies and robot benchmarks.

This package keeps only that stopping-rule hypothesis. It does **not** use the paper's checkpoints, tasks, data, attention maps, or measured correlations. The toy attention distributions and policy-error boundary are synthetic by construction. Any result below is a test of internal consistency and failure modes of the mechanism, not a validation of the paper's empirical claims.

## Toy environment

A 2-D point robot follows a staged reference trajectory:

1. **coarse approach** — slowly changing and predictable,
2. **curved interaction** — shorter useful open-loop horizon,
3. **precision insertion** — rapid corrections and the shortest grounding horizon.

At every query, a frozen policy predicts 24 actions. Its local reference extrapolation accumulates common-mode error after a stage-dependent boundary. A synthetic cross-attention distribution becomes dispersed before that error rise, producing the rising-plateau pattern needed to test the stopping rule.

Two robustness factors are crossed:

- **physical disturbances**: unobserved impulses perturb the robot during open-loop execution;
- **visual distractors**: irrelevant clutter spreads and flickers the synthetic attention distribution without changing task dynamics.

Every method sees paired worlds, process noise, disturbances, and policy randomness.

## Compared execution rules

- `fixed_3`, `fixed_6`, `fixed_12`, `fixed_24`: fixed execution horizons.
- `state_error`: stop when the policy's predicted state rollout diverges from constant-velocity extrapolation.
- `ensemble_uncertainty`: stop when disagreement across eight perturbed action samples remains high.
- `attention_entropy`: paper-inspired moving-average, high-entropy, stable-plateau detector.
- `shuffled_attention`: permute the entropy values across action indices before applying the same detector. This preserves the entropy multiset but destroys horizon alignment.
- `oracle_stop`: future-aware control that stops when true predicted-action error is sustained above threshold. It is an unattainable reference, not a deployable method.

The state-error and ensemble methods are deliberately lightweight proxies. They are not implementations of named baselines from the paper.

## Run

Use a Python environment with NumPy and Matplotlib. The environment used for the checked artifacts was:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  attention_gated_chunk_horizon/run_attention_gated_chunk_horizon.py \
  --mode full --seed 31
```

Available modes:

```bash
# Fast mechanism check; write outside the checked output directory.
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  attention_gated_chunk_horizon/run_attention_gated_chunk_horizon.py \
  --mode sanity --seed 31 --output-dir /tmp/attention_gated_sanity

# Moderate paired evaluation.
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  attention_gated_chunk_horizon/run_attention_gated_chunk_horizon.py \
  --mode quick --seed 31
```

`--trials N` overrides the mode's main trial count. Runs are deterministic for a fixed seed and arguments.

## Metrics

- **success**: tail tracking RMSE and final error both pass fixed tolerances;
- **tracking RMSE**: full-episode position tracking error;
- **inference calls**: number of policy queries per 168-step episode;
- **mean horizon**: average actions executed per query;
- **selected action error**: future-aware diagnostic at the chosen boundary;
- **signal correlation**: per-query Pearson correlation between each internal signal and future action error;
- **robustness sweep**: success and calls as combined disturbance/distractor severity rises.

## Latest verified results

The checked full run uses 120 paired trials per condition. Exact generated values are stored in `outputs/summary.csv`, `outputs/overall_summary.csv`, and `outputs/metrics.json`.

Key result table (combined disturbance + distractor condition):

<!-- RESULTS_TABLE_START -->
| Method | Success | Tracking RMSE | Policy calls | Mean executed horizon |
| --- | ---: | ---: | ---: | ---: |
| Fixed 3 | 100.0% | 0.153 | 56.0 | 3.0 |
| Fixed 6 | 100.0% | 0.209 | 28.0 | 6.0 |
| Fixed 12 | 8.3% | 0.383 | 14.0 | 12.0 |
| Fixed 24 | 0.0% | 1.074 | 7.0 | 24.0 |
| State-error gate | 15.8% | 0.616 | 14.0 | 12.0 |
| Ensemble uncertainty | 0.0% | 0.958 | 8.2 | 20.4 |
| Attention entropy | 43.3% | 0.392 | 13.0 | 12.9 |
| Shuffled attention | 0.8% | 0.937 | 8.7 | 19.5 |
| Oracle stop | 100.0% | 0.152 | 38.5 | 4.4 |

These are deterministic point estimates from one configured simulator run, not cross-seed confidence intervals. In the combined condition, the attention gate reaches **43.3% success with 13.0 calls**, versus **8.3% with 14.0 calls** for fixed 12 and **0.8% with 8.7 calls** for the shuffled-attention control. Fixed 6 and fixed 3 remain at 100% success but require 28 and 56 calls respectively. Thus, in this configured toy, ordered attention dynamics improve the middle of the accuracy-efficiency frontier; they do not beat aggressive replanning on raw success.

Across all four conditions, attention gating averages 41.7% success. Its mean per-query correlation with future action error is about 0.55, compared with about 0.54 for ensemble disagreement and 0.13 for state error. Correlation alone is insufficient: the ensemble gate still selects overly long chunks because shared plan bias produces low disagreement at the stopping threshold.

The severity sweep also exposes a toy-specific caveat: strong distractor mixing can shorten attention-selected chunks and sometimes increase thresholded success, acting as accidental conservative regularization rather than a calibrated semantic signal. That behavior should not be read as distractor robustness.
<!-- RESULTS_TABLE_END -->

The intended reading is an accuracy-efficiency frontier, not a single universal winner. Very short fixed chunks can remain the most accurate because they replan constantly. The useful question is whether the attention gate retains much of that robustness with materially fewer policy calls, and whether destroying horizon order in the shuffled control removes the benefit.

![Headline metrics](outputs/headline_metrics.png)

![Condition robustness](outputs/condition_robustness.png)

![Mechanism example](outputs/mechanism_example.png)

## Outputs

Generated under `outputs/`:

- `trial_metrics.csv`: paired per-trial results for all methods and four conditions;
- `summary.csv`: aggregate mean/SEM by condition and method;
- `overall_summary.csv`: aggregate across all conditions;
- `signal_diagnostics.csv` and `signal_summary.csv`: internal-signal/action-error diagnostics;
- `robustness_sweep.csv`: combined-severity sweep;
- `metrics.json`: configuration, source boundary, summaries, sweep, runtime, and checks;
- `sanity_checks.json`: deterministic mechanism checks;
- `headline_metrics.png`, `condition_robustness.png`, `efficiency_frontier.png`, `robustness_sweep.png`, `mechanism_example.png`.

Report:

- `docs/attention_gated_chunk_horizon_report.tex`
- `docs/attention_gated_chunk_horizon_report.pdf`
- `docs/render_pdf.sh`

## Toy-to-real mapping

| Toy component | Intended real analogue | Important simplification |
| --- | --- | --- |
| 2-D point state | robot proprioception/end-effector state | no contacts, kinematics, cameras, or language |
| staged reference | approach / interaction / precision task phases | stages are explicit simulator labels |
| 24-action frozen plan | VLA action chunk | analytical PD-style policy, not diffusion/flow matching |
| 48-token attention distribution | action-query attention over VLM tokens | synthetic peaked-to-uniform mixture |
| entropy plateau | reduced concentration on current observation | linked to the toy error boundary by construction |
| impulses | external disturbances / execution mismatch | sparse additive acceleration |
| distractor mixing | irrelevant visual clutter | directly perturbs attention probabilities |
| inference calls | compute/control-loop cost | no wall-clock GPU timing |

## Limitations

- The attention signal is designed to become informative; the experiment tests stopping logic and controls, not whether a real VLA learns the signal.
- Entropy can be high for benign reasons. The distractor condition intentionally exposes this calibration problem.
- The ensemble shares a common biased mean, so disagreement under-detects common-mode error by design.
- The state-error proxy can fire on legitimate high-curvature motion and does not observe future disturbances.
- The oracle uses future reference and disturbance information and is only an upper-bound diagnostic.
- Success thresholds, boundary costs, and dynamics define this toy's ranking. Different choices can change the frontier.
- No image encoder, language instruction, generative action expert, contact-rich physics, learned policy, or real-time latency is modeled.
