# Prediction-Error Policy State Toy

This folder explores the state-correction mechanism in **PredVLA: A Sub-Million-Parameter Predictive-Coding Policy for Robot Manipulation** with a deterministic, low-dimensional temporal-control experiment.

Primary reference:

- H. Sawada and S. Kasahara, *PredVLA: A Sub-Million-Parameter Predictive-Coding Policy for Robot Manipulation*, arXiv:2608.26673, August 27, 2026: https://arxiv.org/abs/2608.26673

PredVLA uses hierarchical recurrent generative dynamics to predict visual features and proprioception. Observations do not directly enter those dynamics; instead, prediction errors drive online inference over latent free variables. With inference iterations set to zero, the same learned policy becomes exactly open loop.

This package isolates that distinction. It is **not a PredVLA reproduction**, does not use LIBERO, and does not test the paper's reported parameter efficiency or benchmark success rates.

## Experiment

A point robot tracks a moving 2D target. The compact latent state contains:

- fast robot position and velocity, used as a proprioceptive analogue;
- slower target position and velocity, used as a visual-feature analogue.

A shared 8-state linear recurrence is identified from synthetic demonstrations. An analytical controller acts only on the estimated latent state. Test dynamics include smooth model mismatch, process noise, and optional impulses. Sensor packets contain robot position/velocity and target position; target velocity remains latent.

All closed-loop methods use the same identified dynamics and the same scalar coefficient budget:

| Method | Observation update | Fitted coefficient budget |
| --- | --- | ---: |
| `simple_recurrent` | newest innovation through a clean-data linear recurrent observer | 136 |
| `finite_window_attention_like` | fixed recency/consistency weighting over a finite observation window | 136 |
| `learned_robust_observer` | robust-data innovation gain with clipping/gating | 136 |
| `predictive_coding` | iterative sliding-window prediction-error correction of an anchor latent | 136 |
| `predictive_coding_open_loop` | identical predictive model, but zero correction iterations | 136 available; 0 active correction |

The 136 slots are 80 identified dynamics coefficients plus 56 observer/error-lift coefficients. This is an algorithmic capacity match, **not** a claim that these estimators faithfully instantiate parameter-matched neural LSTMs, Transformers, or PredVLA.

## Perturbations and metrics

The full run uses 48 paired trials for each method and sweep value:

- visual occlusion length: `0, 10, 20, 32, 44` steps;
- observation delay: `0, 1, 3, 5, 8` steps;
- robot/target disturbance impulse: `0.00, 0.16, 0.30, 0.44, 0.60`;
- fixed episode sensor bias: `0.00, 0.02, 0.05, 0.08, 0.12`;
- predictive-coding iterations under mixed perturbations: `0, 1, 2, 4, 6, 10, 16`.

Success requires tail tracking RMSE below `0.255` and final target distance below `0.30`. The outputs also record full-episode tracking RMSE, final and tail error, latent-state RMSE, and sensory-prediction RMSE.

## Run

Use a Python environment with NumPy and Matplotlib:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  -m py_compile prediction_error_policy_state/run_prediction_error_policy_state.py

/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  prediction_error_policy_state/run_prediction_error_policy_state.py \
  --seed 23 --trials 48
```

A smoke test is available with `--quick`.

Render the report:

```bash
bash prediction_error_policy_state/docs/render_pdf.sh
```

## Latest verified results

At a five-step observation delay:

| Method | Success | Tracking RMSE |
| --- | ---: | ---: |
| Simple recurrent observer | 16.7% | 0.628 |
| Finite-window attention-like | 0.0% | 0.826 |
| Learned robust observer | 8.3% | 0.602 |
| Predictive-coding correction | **100.0%** | **0.415** |
| Predictive coding, open loop | 0.0% | 0.904 |

At eight delayed steps, predictive coding retained 66.7% success while all three closed-loop baselines and open loop were at 0%. This is the toy's clearest positive signature: timestamped errors distributed through a regenerated temporal window correct delayed state more effectively than one-pass innovation updates.

The correction-iteration sweep used a simultaneous three-step delay, 20-step visual occlusion, 0.30 impulse, and 0.025 bias:

| Iterations | Success | Tracking RMSE | State RMSE | Sensory prediction RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2.1% | 0.856 | 0.298 | 0.337 |
| 1 | 77.1% | 0.440 | 0.089 | 0.073 |
| 2 | 81.2% | 0.436 | 0.081 | 0.063 |
| 4 | 87.5% | 0.434 | 0.077 | 0.058 |
| 6 | 93.8% | 0.431 | 0.075 | 0.054 |
| 10 | **95.8%** | 0.431 | 0.073 | 0.052 |
| 16 | 93.8% | **0.430** | **0.072** | **0.052** |

Zero iterations exactly matched the dedicated open-loop method: the maximum action difference was `0.0` even when occlusion, delay, and bias were changed. This confirms that sensor values enter the predictive-coding path only through the correction loop.

The result is not uniformly favorable. Under a 44-step visual occlusion, simple recurrence achieved 47.9% success, finite-window attention-like 37.5%, predictive coding 29.2%, and the learned robust observer 0%. Repeated inference cannot recover information absent beyond the useful model horizon, and the toy's iterative method can accumulate target-model mismatch. Under high disturbance and bias, most closed-loop methods remained strong; predictive coding mainly improved tracking/state error rather than producing a large success-rate separation.

![Robustness sweeps](outputs/robustness_sweeps.png)

![Correction iterations](outputs/correction_iteration_sweep.png)

## Outputs

Generated artifacts are under `outputs/`:

- `trials.csv`: per-trial metrics for every method and sweep point;
- `summary.csv`: mean and standard error by method/sweep;
- `metrics.json`: configuration, labels, headline metrics, and claim boundary;
- `sanity_check.json`: deterministic mechanism checks;
- `parameter_budget.json`: coefficient accounting and caveat;
- `representative_rollout_metrics.json`: paired rollout metrics;
- `robustness_sweeps.png`: success and tracking error versus four perturbations;
- `correction_iteration_sweep.png`: success/tracking/state/prediction error versus inference iterations;
- `representative_rollout.png`: paired trajectories and tracking error;
- `full_run.log`: captured full-run console summary and timing.

Report files:

- `docs/prediction_error_policy_state_report.tex`;
- `docs/prediction_error_policy_state_report.pdf`;
- `docs/generated_results.md`;
- `docs/render_pdf.sh`.

## Relation to other experiments in this repository

- **`predictive_error_correction` is the nearest sibling.** Both freeze learned dynamics and make observations affect control only through prediction-error correction, with an exact zero-iteration/open-loop test. That package corrects a single current 1-D latent from current error plus a window-derived velocity cue and compares learned finite-history action policies; this package regenerates a 2-D latent trajectory from timestamped packets and compares coefficient-budgeted observer rules under separate delay, occlusion, impulse, and bias sweeps.
- **The adaptation tracks move parameters/models, not only state.** `local_residual_sim2real` updates a local transition/command model from online physical outcomes and explicitly measures source/OOD retention. `grounded_online_adaptation` updates visual attention/action parameters from reward-derived targets and measures grounding and forgetting. Here no policy/model coefficient changes during evaluation; locality is the finite inference window, and retention is not a relevant metric.
- **`anticipatory_context_chunking` predicts forward before a delayed chunk executes.** This package instead corrects backward/forward through a window after timestamped observations arrive. The two should be compared on delay, handoff/latent error, tracking, and compute, not only success. `retry_reset_recovery` uses offline bridge/reset skills plus an online router, and `conflict_aware_replay` selects retained samples during sequential training; neither is an inference-state estimator.
- **`force_embodiment_gap` and `force_feedback_demonstration_quality` are upstream offline tracks.** They study whether embodiment calibration/action coordinates and force-rich demonstrations produce better transferable models/data. Their success, damage, force-error, and morphology-shift metrics complement---but do not substitute for---state, sensory-prediction, and delay/occlusion metrics here.
- **Safety is outside the estimator comparison.** `constraint_manifold_action_filter` enforces known constraints by projection/retraction, `path_consistent_safety_filtering` preserves a learned chunk path while slowing/stopping, and `configured_failure_audit` reports monitor stops and false stops. Bounded acceleration in this toy is not a safety filter or safety guarantee.

## Mapping to PredVLA

| Toy component | PredVLA analogue | Important simplification |
| --- | --- | --- |
| 8D robot/target state | hierarchical deterministic recurrent state | one identified linear state, only implicit fast/slow partitions |
| target position packet | frozen visual feature | direct 2D coordinate, no image encoder or PCA |
| robot position/velocity packet | proprioception | point dynamics, no joints or gripper |
| model prediction `A x + B u` | generative recurrent sensory/action dynamics | linear and analytical action decoder |
| window anchor latent | posterior free variables over an inference window | one anchor rather than module/time-specific latents |
| `observation - prediction` | sensory prediction error | squared vector errors, fixed masks |
| prior-pull term | free-energy complexity term | scalar quadratic regularizer |
| repeated correction/regeneration | online error regression and recurrent-state regeneration | learned linear error lift, no gradient through a neural hierarchy |
| zero iterations | exact open-loop ablation | same construction, but in a synthetic controller |

## Sanity checks

The run asserts that:

1. identified dynamics and observer gains are finite;
2. actions remain within the configured acceleration bound;
3. changing sensor delay, bias, and occlusion cannot change open-loop actions;
4. zero predictive-coding iterations exactly match the open-loop path;
5. positive-iteration predictive coding differs from open loop;
6. sensory prediction errors remain finite.

All checks passed in the committed output set.

## Claim boundaries and limitations

- This tests a mechanism, not the PredVLA architecture or reported LIBERO results.
- There is no language encoder, frozen ResNet, action mixture density, robot-data training, contact, images, or real robot.
- The finite-window method uses fixed attention-like weights, not a Transformer.
- The simple observer is a linear recurrent baseline, not an LSTM.
- Equal scalar coefficient budgets do not establish equal expressivity, compute, optimization difficulty, or wall-clock latency.
- The predictive-coding update is a preconditioned prediction-error descent analogue, not PredVLA's exact deterministic free-energy objective and backpropagation through its hierarchy.
- Train and test dynamics are synthetically related; results can change with thresholds, calibration distribution, window length, and model mismatch.
- Long visual occlusion is a counterexample to a broad superiority claim in this setup.

The defensible conclusion is narrow: **when delayed observations remain within a useful generative-model window, iterative prediction-error correction can recover policy state that one-pass recurrent or finite-window updates fail to recover, and disabling correction gives a verifiably exact open-loop ablation.**
