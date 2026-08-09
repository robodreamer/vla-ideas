# Anticipatory Context Chunking Toy

This folder tests the central deployment claim behind **FutureRTC: Real-Time Robot Execution with Anticipatory-Conditioned Action Chunking** with a small, falsifiable synthetic controller.

The question is narrower than a reproduction:

> Under asynchronous chunk execution, does estimating both the execution-time robot state and the environment/visual latent outperform compensating only the robot state?

A frozen analytical chunk policy tracks a maneuvering target while inference is delayed. The methods differ only in the context supplied at chunk handoff.

## Methods

- `oracle_fresh`: policy receives the true execution-time state and environment latent.
- `naive_stale`: policy receives the stale state and stale latent.
- `state_rollout`: committed actions are integrated through a nominal robot model; the latent remains stale.
- `state_correction`: a learned residual corrects the nominal state rollout; the latent remains stale.
- `obs_transport`: corrected state plus analytic latent transport (constant target velocity, frozen cue, decayed wind).
- `obs_innovation`: transport plus a learned residual for visual/environment changes that transport cannot explain.
- `policy_consistency`: the same predictor trained with an additional frozen-policy action-consistency loss.

The environment latent contains target position/velocity, a maneuver cue, and wind. It is a low-dimensional analogue of the visual latent used by FutureRTC, not an image encoder.

## References checked

Primary sources:

- Paper: `https://arxiv.org/abs/2607.24008`
- Project page: `https://jianghaiscu.github.io/FutureRTC_proj/`
- Official implementation: `https://github.com/JianghaiSCU/FutureRTC`

The official code has separate LIBERO and Kinetix branches. The local toy follows the same broad decomposition: committed-action state rollout/correction, motion-conditioned latent transport, residual innovation, and optional policy consistency. It does not use their weights, environments, or architecture.

## Run

Use the existing project Python environment:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  anticipatory_context_chunking/run_anticipatory_context_chunking.py \
  --seed 17 --trials 80 --train-samples 12000 --val-samples 2500 --train-steps 650
```

Quick smoke test:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  anticipatory_context_chunking/run_anticipatory_context_chunking.py --smoke
```

## Latest verified result

The full run evaluates 80 paired trials per method at delays `0, 2, 4, 6, 8, 10`.

At delay `d=10`:

| Method | Success | Tracking RMSE | Latent RMSE | Handoff policy error |
| --- | ---: | ---: | ---: | ---: |
| Oracle fresh context | 97.5% | 1.052 | 0.000 | 0.000 |
| Naive async (stale) | 0.0% | 3.530 | 0.454 | 2.949 |
| State rollout only | 0.0% | 1.523 | 0.454 | 1.787 |
| State correction | 0.0% | 1.508 | 0.454 | 1.588 |
| Observation transport only | 0.0% | 1.628 | 0.340 | 1.308 |
| Transport + innovation | 68.8% | 1.075 | 0.091 | 0.212 |
| + policy consistency | **81.2%** | **1.065** | 0.092 | 0.215 |

The learned state corrector reduces held-out proprioceptive RMSE from `0.161` to `0.014`, but that alone does not recover delayed control because the environment latent is still stale. Learned innovation reduces held-out latent RMSE from `0.216` (transport only) to `0.074`. The policy-consistency variant has nearly identical latent reconstruction error but improves closed-loop success at larger delays in this run.

![Delay sweep](outputs/delay_sweep.png)

## Outputs

Generated artifacts are under `outputs/`:

- `metrics.json`: configuration and aggregate results.
- `delay_sweep_summary.csv`: aggregate mean/SEM by method and delay.
- `delay_sweep_trials.csv`: per-trial metrics.
- `training_metrics.json`: held-out state/latent/policy prediction metrics.
- `sanity_check.json`: seven deterministic mechanism checks.
- `delay_sweep.png`: success, tracking, policy-error, and latent-error sweeps.
- `single_rollout.png`: representative paired trajectories and tracking error.
- `predictor_ablation.png`: held-out transport/innovation/policy-loss comparison.

Reports:

- `docs/anticipatory_context_chunking_report.md`
- `docs/anticipatory_context_chunking_report.tex`
- `docs/anticipatory_context_chunking_report.pdf`

## Interpretation

The toy supports trying the idea: as delay grows, accurate proprioceptive compensation is not enough when policy-relevant environment context changes during inference. A transport prior helps latent error but is insufficient by itself; the learned innovation is the dominant improvement. The policy-consistency term gives a smaller, closed-loop-oriented gain.

This result is deliberately bounded. It demonstrates the mechanism in a synthetic low-dimensional setting. It is not evidence that the same magnitude of gain will transfer to a VLA, image-token transport, contact-rich manipulation, or real inference hardware.
