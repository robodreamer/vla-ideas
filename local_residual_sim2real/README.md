# Local Residual Sim-to-Real: Chained Transition Toy

This package tests a narrow question motivated by **Rapid On-Robot Learning for Dynamic Manipulation Skills: Robot Juggling**:

> When a biased dynamics/policy prior encounters a local sim-to-real gap, which short-window correction rule adapts quickly without destroying the prior, and what changes when every command must remain inside a mutually reachable transition set?

The result is a deterministic 2-D throw/catch-like transition toy. **It is an adaptation and safety mechanism probe—not juggling, a VLA benchmark, an AthenaZero implementation, or a reproduction of the paper.**

## References checked

- Taeyoon Lee et al., [Rapid On-Robot Learning for Dynamic Manipulation Skills: Robot Juggling](https://arxiv.org/abs/2608.26800), arXiv:2608.26800, 2026.
- Official RAI video: [A New Benchmark in Robot Juggling](https://rai-inst.com/resources/videos/a-new-benchmark-in-robot-juggling/).
- Official platform article: [AthenaZero: A Bimanual Robot for Dynamic Manipulation](https://rai-inst.com/resources/blog/bimanual-robot-for-dynamic-manipulation/).

The paper stores transitions in memory, retrieves local neighbors with an RBF kernel, fits a local linear model regularized toward the prior, regularizes the command update, and chains skills through a precomputed mutually reachable set. The authors report five canonical three-ball patterns with less than five minutes of physical interaction; cascade reached the five-cycle target after seven resets and was consistently learned by the eighth attempt across five trials. Those are paper results. The toy's explicit uncertainty gate is an additional sparse-support fallback, not a claim that the paper uses the same gate. No paper code, robot data, or learned policy is used here.

## Toy setup

Each interaction is one synthetic release-to-catch transition:

```text
next_state = biased_prior(state, context, command)
           + localized_sim2real_residual(state, context, command)
           + small observation noise
```

The two state coordinates are normalized landing-position and landing-velocity errors. A five-skill chain succeeds only if every transition lands inside the catch ellipse. The prior is nearly correct on a source region but biased around the online adaptation region.

A learner begins with that prior, executes 240 short real-like transitions, stores observations, and replans commands from its current corrected model. Five deterministic seeds are evaluated every 20 interactions.

## Compared adaptation rules

Each rule runs **unconstrained** and with the same **mutually reachable safe-set filter**.

- `global_replacement`: ridge-fit the complete transition model from the short local window, replacing the prior prediction globally.
- `residual_finetune`: fit a global residual on top of the prior, analogous to globally fine-tuning the mismatch model.
- `nearest_memory`: kernel-average the nine closest stored residuals with no uncertainty attenuation.
- `local_residual`: kernel-local ridge-shrunk residual plus a gate based on effective neighbor count and nearest-memory distance; unsupported queries blend back toward the prior.

The safe-set filter projects a proposed command toward the nominal prior command until its predicted transition lies inside a catch ellipse, respects a conservative release-command bound, and admits a bounded one-step recovery command. A hard command clip remains as the final actuator proxy for both variants.

## Run

From the repository root, using the existing shared environment:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python -W error \
  local_residual_sim2real/run_local_residual_sim2real.py
```

Smoke test without replacing full outputs:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python -W error \
  local_residual_sim2real/run_local_residual_sim2real.py \
  --smoke --output-dir /tmp/local_residual_sim2real_smoke
```

Render the report:

```bash
local_residual_sim2real/docs/render_pdf.sh
```

## Metrics

- **Interaction/sample efficiency:** normalized chain-success AUC and first checkpoint reaching 70% chain success.
- **Prediction/control error:** one-step model error and norm of the realized landing error.
- **Reliability:** five-transition chain success and per-transition catch success.
- **Margin:** mean and 10th-percentile distance inside the catch boundary.
- **Safety:** unsafe executed commands, raw unsafe proposals, hard state violations, and filter interventions.
- **Retention/extrapolation:** source-region chain-success change from the frozen prior and out-of-distribution control error.
- **Ablations:** local memory width, ridge shrinkage, and safe set.

## Verified results

Mean over five deterministic seeds; uncertainty details are in `outputs/summary_metrics.csv`.

| Method | Safe set | Chain success | Interaction AUC | To 70% | Prediction error | Control error | Margin | Unsafe commands | Violations | Source retention | OOD control error |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Global replacement | no | 0.986 | 0.941 | 24 | 0.033 | 0.047 | 0.448 | 0.0156 | 0.0004 | -0.974 | 0.397 |
| Global replacement | yes | 0.990 | 0.955 | 20 | 0.033 | 0.059 | 0.435 | **0.0000** | 0.0012 | -0.968 | 0.345 |
| Residual fine-tune | no | 0.966 | 0.942 | 20 | 0.032 | 0.054 | 0.440 | 0.0248 | 0.0016 | -1.000 | 0.382 |
| Residual fine-tune | yes | 0.990 | 0.956 | 20 | 0.032 | 0.052 | 0.442 | **0.0000** | 0.0004 | -0.978 | 0.321 |
| Nearest-memory | no | **1.000** | **0.958** | 20 | **0.023** | **0.023** | **0.474** | 0.0000 | 0.0000 | 0.000 | 0.259 |
| Nearest-memory | yes | **1.000** | **0.958** | 20 | **0.023** | **0.023** | **0.474** | 0.0000 | 0.0000 | 0.000 | 0.259 |
| Local residual + gate | no | **1.000** | **0.958** | 20 | 0.071 | 0.071 | 0.419 | 0.0000 | 0.0000 | **0.000** | **0.180** |
| Local residual + gate | yes | **1.000** | **0.958** | 20 | 0.071 | 0.071 | 0.419 | 0.0000 | 0.0000 | **0.000** | **0.180** |

![Online learning curves](outputs/learning_curves.png)

### Bounded interpretation

- The biased prior starts at 0% five-skill chain success. Every method except unconstrained global replacement reaches at least 70% by the first 20-interaction checkpoint; that condition averages 24 interactions.
- Nearest-memory gives the smallest local prediction/control error, but it applies corrections without uncertainty attenuation and extrapolates worse than the gated local residual (`0.259` versus `0.180` OOD control error).
- Global replacement and global residual fine-tuning adapt locally but catastrophically alter the source region: source chain retention is approximately `-0.97` to `-1.00`. Both memory methods preserve source **chain success**, but not the prior exactly: source control error increases by `0.122` for nearest-memory and `0.151` for the gated local residual.
- The safe set removes executed unsafe commands for the two global methods and improves their chain-success AUC. It does **not** guarantee zero realized violations under model error: safe global replacement has a slightly higher tiny violation rate (`0.0012` versus `0.0004`).
- The safe-set variants of the two memory methods are identical because their learned commands already satisfy the modeled viable set. This is a negative but useful ablation: filtering is inactive when the adapter is both local and conservative.

![Method trade-offs](outputs/method_tradeoffs.png)

### Ablations

The local-residual ablations use only 12 adaptation transitions and a broader edge-of-memory evaluation distribution.

- Memory width `0.18` under-covers the edge distribution (`0.463` chain success), while widths `0.35–2.0` reach `1.0`; prediction error is best around `1.2` in this toy.
- Excessive ridge shrinkage toward the prior hurts: regularization `40` falls to `0.330` chain success, versus `1.0` for `0.01–1.5`.
- Safe-set on/off remains identical for the default local adapter because no default local command leaves the predicted viable set.

![Ablations](outputs/ablations.png)

## Outputs

- `outputs/metrics.json`: full configuration, method definitions, sanity result, summary rows, and ablations.
- `outputs/summary_metrics.csv`: mean, standard deviation, SEM, and valid count by method/safe-set condition.
- `outputs/condition_metrics.csv`: per-seed endpoint, safety, retention, and extrapolation metrics.
- `outputs/learning_curve.csv`: per-seed online checkpoints.
- `outputs/ablation_metrics.csv` and `ablation_summary.csv`: width, regularization, and safe-set sweeps.
- `outputs/sanity_check.json`: eight deterministic mechanism checks.
- `outputs/*.png`: learning, trade-off, and ablation plots.
- `docs/local_residual_sim2real_report.tex` and `.pdf`: source-grounded report.

## Relation to other experiments in this repository

- **The other three target tracks use different correction loci.** `grounded_online_adaptation` also updates online, but changes a small visual-policy pathway from reward-derived targets rather than a transition model and command. `predictive_error_correction` and `prediction_error_policy_state` keep weights fixed and correct inference-time latent state from sensory prediction error; this package only *measures* one-step prediction error and feeds observed transition residuals into later model updates.
- **Retention is closest to `conflict_aware_replay`, but the protocol differs.** Conflict-aware replay performs sequential/offline task training and reports negative backward transfer and continual-learning AUC. Here adaptation occurs during interaction, locality comes from kernel memory and a support gate, and retention is evaluated as source chain-success and source control-error change. Neither result establishes exact behavior preservation.
- **`retry_reset_recovery` changes data and execution routing, not the dynamics model.** Its perturbation bridges and reset-skill library are learned offline, then an online monitor selects retry/reset behavior. Its recovery, false-reset, and compounding-failure metrics complement this package's adaptation AUC, model error, and viable-transition metrics.
- **`anticipatory_context_chunking` is feed-forward compensation before delayed execution.** It predicts execution-time robot and environment context; this package corrects after physical transition outcomes arrive. `force_embodiment_gap` and `force_feedback_demonstration_quality` instead study offline interface/data causes of transfer error---calibration, morphology, and force-rich demonstrations---that could determine whether a useful prior exists before online adaptation.
- **Safety is deliberately modular.** This package's optional one-step model-based viability filter can still realize violations when its model is wrong. `constraint_manifold_action_filter` enforces known geometric constraints by projection/retraction, `path_consistent_safety_filtering` preserves chunk-path geometry by slowing/stopping, and `configured_failure_audit` uses phase/state monitors to stop anomalous sequences. Their collision, residual, path-deviation, stop, and false-stop metrics should not be conflated with this package's unsafe-command and violation rates.

## Simplifications and limitations

This is a two-dimensional analytic transition, not ball flight, a manipulator, images, language, action chunks, a VLA, or reinforcement learning. The learner observes dense transition outcomes immediately; there is no reward-credit assignment, perception, latency, occlusion, contact, actuator identification, or human intervention. The “global replacement” and “fine-tuning” baselines are linear ridge models, not neural policies. The local method uses a kernel-local zero-order ridge estimator rather than the paper’s weighted local linear regression and regularized command optimization. The viable set is an ellipse plus command/recovery bounds, not a backward-reachable set learned from robot data. Five seeds and synthetic noise provide descriptive variance only.

The main qualitative lesson is correspondingly narrow: when the mismatch is local, memory locality and uncertainty gating can preserve a useful prior outside the observed region, while a model-based safe filter can suppress unsafe commands but cannot erase model error.
