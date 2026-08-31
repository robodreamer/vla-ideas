# Predictive Error Correction Toy

This package tests one narrow mechanism from **PredVLA: A Sub-Million-Parameter Predictive-Coding Policy for Robot Manipulation**:

> Can a recurrent controller recover from hidden disturbances by correcting its latent state from prediction error at inference time, even though observations are never direct recurrence inputs?

A one-dimensional robot tracks a moving target. Its nominal latent predicts robot position/velocity and target position/velocity from the previous latent, the language-like direction command, and an efference copy of the applied action. Evaluation adds hidden force pulses, unmodeled target maneuvers, observation noise, and slowly drifting proprioceptive/visual bias.

This is a compact state-space mechanism test, not a reproduction of PredVLA or a robotics benchmark.

## PredVLA details represented

The primary paper describes a hierarchical predictive-coding VLA that predicts vision, proprioception, and action rather than feeding observations directly into recurrence. At deployment, per-timestep latent free variables are optimized over a recent window. Setting the number of latent-update iterations to zero is the paper's exact open-loop ablation.

The toy preserves those structural ideas in simplified form:

- `open_loop`: recurrent latent rollout with **zero** inference iterations; the observation arrays are not read.
- `pred_error_w8`: the same prior and frozen visual-to-action bottleneck, plus online optimization of four **current-state** correction variables against current proprioceptive/visual error and a target-velocity cue estimated from a window of eight. It does not regenerate a full latent trajectory.
- The paper reports short-task module time constants `(T, V, A_top, A_bottom) = (16, 8, 5, 2)`. The toy reuses the same values in reverse, fast-to-slow order for four state coordinates; there is no module-to-coordinate correspondence.
- The toy copies the applied action into the next nominal prediction. PredVLA's lateral pathway instead uses its own previous predicted action and proprioception, so this is only an efference-copy analogue.
- A bounded visual/error bottleneck decodes the corrected latent into action.
- Observation-fed ridge baselines use fixed histories `H=1,2,4,8,16`; a larger `H=16, chunk=4` baseline predicts four actions and executes them open loop.

## Methods

- `oracle`: true-state feedback ceiling.
- `open_loop`: strict inference-off latent rollout (`iterations=0`).
- `pred_error_w8`: online prediction-error correction, 80 update iterations.
- `history_h4`: compact feed-forward finite-history policy.
- `history_h16`: larger finite-history policy.
- `history_h16_chunk4`: larger history with a four-action open-loop chunk.

The history models are trained by deterministic ridge regression on 180 oracle-controlled episodes with disturbance severity sampled in `[0,1]`. Their validation action RMSE is `0.118` for `H=4`, `0.114` for `H=16`, and `0.132` for the four-action model.

## Run

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  predictive_error_correction/run_predictive_error_correction.py \
  --seed 23 --trials 64 --train-episodes 180
```

Smoke test:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  predictive_error_correction/run_predictive_error_correction.py --smoke
```

Render the report:

```bash
predictive_error_correction/docs/render_pdf.sh
```

## Verified results

Each number below averages 64 paired schedules per method and severity.

| Severity | Method | Success | Tracking RMSE | Action RMSE vs oracle |
| ---: | --- | ---: | ---: | ---: |
| 0.0 | Open loop | 100.0% | 0.149 | 0.085 |
| 0.0 | Prediction-error correction | 100.0% | 0.144 | 0.095 |
| 1.0 | Open loop | 1.6% | 4.896 | 2.309 |
| 1.0 | Prediction-error correction | 85.9% | 0.292 | 0.310 |
| 1.0 | History H=4 | 89.1% | 0.302 | 0.341 |
| 1.0 | History H=16 | 90.6% | 0.285 | 0.323 |
| 1.5 | Open loop | 0.0% | 7.338 | 2.429 |
| 1.5 | Prediction-error correction | 50.0% | 0.511 | 0.439 |
| 1.5 | History H=4 | 64.1% | 0.408 | 0.450 |
| 1.5 | History H=16 | 65.6% | 0.384 | 0.444 |
| 2.0 | Open loop | 0.0% | 9.780 | 2.498 |
| 2.0 | Prediction-error correction | 29.7% | 0.969 | 0.524 |
| 2.0 | History H=16 | 45.3% | 0.498 | 0.590 |

At severity `1.0`, correction cuts latent RMSE from `2.737` to `0.096` and recovers `85.9%` success from the exact open-loop ablation's `1.6%`. The finite-history policies are slightly stronger on success, while correction has lower action error than either `H=4` or `H=16`.

At severity `1.5` and `2.0`, fixed history is more reliable than the current correction rule. This is an important negative result: online correction is highly beneficial relative to open loop, but the toy does **not** show it dominating a well-fit observation-fed baseline under the strongest drift.

The iteration ablation at severity `1.5` isolates inference: success rises from `0%` at zero iterations to `46.9%` at 20 or more iterations, while latent RMSE falls from `4.772` to `0.241` at 80 iterations.

![Disturbance sweep](outputs/disturbance_sweep.png)

![Inference iteration ablation](outputs/iteration_ablation.png)

## Sanity checks

`outputs/sanity_check.json` verifies eight deterministic properties, including:

- changing every observation leaves `open_loop` actions bit-identical (`max diff = 0`);
- setting correction iterations to zero produces actions bit-identical to the dedicated `open_loop` method;
- the correction path reacts to changed observations;
- correction lowers latent and tracking error on a fixed disturbed schedule;
- zero-disturbance open loop stays near oracle;
- paired methods receive identical exogenous schedules.

## Outputs

- `outputs/metrics.json`: configuration and all aggregate metrics.
- `outputs/training_metrics.json`: finite-history fit metrics and model sizes.
- `outputs/sanity_check.json`: deterministic mechanism checks.
- `outputs/disturbance_sweep_trials.csv`: per-trial metrics.
- `outputs/disturbance_sweep_summary.csv`: mean/SEM by method and severity.
- `outputs/history_sweep_summary.csv`: history-length sweep.
- `outputs/iteration_ablation_summary.csv`: exact inference-off-to-on sweep.
- `outputs/disturbance_sweep.png`
- `outputs/history_sweep.png`
- `outputs/iteration_ablation.png`
- `outputs/single_rollout.png`
- `docs/predictive_error_correction_report.tex`
- `docs/predictive_error_correction_report.pdf`

## Primary references checked

- Hiroki Sawada and Shunichi Kasahara, PredVLA paper: `https://arxiv.org/abs/2608.26673`
- Official arXiv HTML: `https://arxiv.org/html/2608.26673v1`

As checked on August 31, 2026, the paper links no official implementation. It reports a 675,732-parameter controller and simulation results on four LIBERO suites; the trainable count excludes the frozen multimodal front end and does not capture test-time optimization cost. Its ablation attributes 6.44--12.57 success-rate points to online error regression across the reported suites. Those are paper results, not results reproduced here.

## Relation to other experiments in this repository

- **`prediction_error_policy_state` is the closest sibling but is not the same implementation.** This package corrects one current 1-D latent using current sensory error plus a window-derived velocity cue and compares learned finite-history action policies under a combined severity sweep. The sibling regenerates a 2-D latent trajectory from timestamped errors over a sliding window and compares coefficient-budgeted observers under separate delay, occlusion, impulse, and bias sweeps. Their shared exact zero-iteration test is stronger evidence than either package's thresholded success alone.
- **The adaptation tracks change weights/models rather than transient state.** `local_residual_sim2real` learns local transition residuals online and measures source/OOD retention; `grounded_online_adaptation` updates a visual policy pathway from reward-derived targets and measures forgetting/grounding. This package changes no learned parameter at deployment, so its locality question is temporal-window support, not source-task retention.
- **`anticipatory_context_chunking` is feed-forward prediction before delayed execution.** It transports/corrects future robot and environment context at chunk handoff; prediction-error correction is feedback after observations arrive. `retry_reset_recovery` instead uses offline retry/reset data plus an online failure router, while `conflict_aware_replay` uses replay selection during sequential training. Their recovery/NBT metrics answer different questions from latent, tracking, action, and inference-iteration error here.
- **The force tracks are upstream offline studies.** `force_embodiment_gap` tests morphology/calibration/action-coordinate transfer, and `force_feedback_demonstration_quality` tests whether force-rich demonstrations improve imitation targets. Either could improve the prior or sensors, but neither performs inference-time latent correction.
- **Safety filters are downstream.** `constraint_manifold_action_filter` projects/retracts actions onto known constraints, `path_consistent_safety_filtering` slows/stops along the proposed chunk path, and `configured_failure_audit` monitors and stops anomalous sequences. This package only bounds actions and reports control error/jerk; it does not establish collision or constraint safety.

## Limits

The toy has no images, language encoder, learned recurrent neural dynamics, contacts, manipulation, or real-time inference stack. The correction objective is quadratic and low dimensional. Its four free variables and leaky rates are analogies to PredVLA's inference procedure, not an architectural copy. Synthetic train/test distributions overlap through severity `1.0`; severities `1.5` and `2.0` are stress tests. Claims should remain limited to the observed synthetic mechanism.
