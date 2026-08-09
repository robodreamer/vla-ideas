# Anticipatory Context Chunking: FutureRTC-Inspired Toy Report

## Bottom line

This experiment asks whether an asynchronous action-chunk policy should predict the full execution-time context rather than only rolling robot state forward. In a synthetic moving-target task, state correction is accurate but fails to recover closed-loop success at large delays because the environment latent remains stale. Adding learned latent innovation recovers most of the oracle behavior; an additional policy-consistency objective improves the delay-10 success rate from 68.8% to 81.2%.

This is a mechanism test, not a reproduction of FutureRTC.

## Primary references

- FutureRTC paper: https://arxiv.org/abs/2607.24008
- Project page: https://jianghaiscu.github.io/FutureRTC_proj/
- Official implementation: https://github.com/JianghaiSCU/FutureRTC

The authors decompose their adapter into a proprioceptive state correction module and an observation prediction module that transports stale visual features and synthesizes innovations. Their LIBERO training optionally adds a frozen-policy consistency loss. The official Kinetix branch similarly predicts an environment latent while obtaining robot state through forward simulation.

## Toy setup

A point robot tracks a maneuvering target in two dimensions. The frozen policy maps:

- robot position and velocity;
- target position and velocity;
- a rotating maneuver cue;
- a decaying stochastic wind latent;

to bounded acceleration chunks. Each chunk contains 12 actions at a 60 ms control step. At a handoff, the policy may receive context delayed by up to 10 control steps.

The true robot differs from the nominal rollout model through drag, actuator gain, wind coupling, and noise. Target acceleration follows the maneuver cue, while cue and wind dynamics make the environment latent change during inference.

Three small neural regressors are trained on synthetic on-policy-like transition samples:

1. a state residual corrector;
2. a latent innovation predictor trained with reconstruction loss;
3. the same latent predictor trained with reconstruction plus frozen-policy first-action consistency.

## Compared methods

1. **Oracle fresh:** true state and latent at handoff.
2. **Naive stale:** stale state and latent.
3. **State rollout:** nominal integration through committed actions, stale latent.
4. **State correction:** learned residual on the rollout, stale latent.
5. **Transport only:** corrected state plus analytic target/wind transport.
6. **Transport + innovation:** learned latent residual after transport.
7. **Policy consistency:** innovation predictor with an action-space consistency term.

## Metrics

- **Success:** tail tracking RMSE below 0.72 and final target error below 0.90.
- **Tracking RMSE:** root-mean-square robot-to-target distance over the episode.
- **Proprioceptive context error:** state-estimate L2 error at chunk handoffs.
- **Latent context RMSE:** environment-latent prediction error at handoffs.
- **Handoff policy error:** first-action difference from the action generated with oracle fresh context.
- **Action jump:** change between the prior executed action and the first action of the new chunk.

## Verification command

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  anticipatory_context_chunking/run_anticipatory_context_chunking.py \
  --seed 17 --trials 80 --train-samples 12000 --val-samples 2500 --train-steps 650
```

The run used 80 paired trials for each method at delays 0, 2, 4, 6, 8, and 10.

## Results

### Success rate by delay

| Method | d=0 | d=2 | d=4 | d=6 | d=8 | d=10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle fresh | 96.2% | 92.5% | 93.8% | 93.8% | 97.5% | 97.5% |
| Naive stale | 96.2% | 2.5% | 0.0% | 0.0% | 0.0% | 0.0% |
| State rollout | 96.2% | 60.0% | 3.8% | 0.0% | 0.0% | 0.0% |
| State correction | 96.2% | 57.5% | 3.8% | 0.0% | 0.0% | 0.0% |
| Transport only | 96.2% | 16.2% | 0.0% | 0.0% | 0.0% | 0.0% |
| Transport + innovation | 96.2% | 95.0% | 88.8% | 86.2% | 76.2% | 68.8% |
| + policy consistency | 96.2% | 92.5% | 90.0% | 91.2% | 83.8% | **81.2%** |

### Delay-10 detail

| Method | Success | Tracking RMSE | Proprio error | Latent RMSE | Policy error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle fresh | 97.5% | 1.052 | 0.000 | 0.000 | 0.000 |
| Naive stale | 0.0% | 3.530 | 2.864 | 0.454 | 2.949 |
| State rollout | 0.0% | 1.523 | 0.231 | 0.454 | 1.787 |
| State correction | 0.0% | 1.508 | 0.038 | 0.454 | 1.588 |
| Transport only | 0.0% | 1.628 | 0.039 | 0.340 | 1.308 |
| Transport + innovation | 68.8% | 1.075 | 0.040 | 0.091 | 0.212 |
| + policy consistency | **81.2%** | **1.065** | 0.040 | 0.092 | 0.215 |

![Delay sweep](../outputs/delay_sweep.png)

![Representative rollout](../outputs/single_rollout.png)

### Held-out predictor ablation

- Nominal state rollout RMSE: 0.161.
- Learned state-corrected RMSE: 0.014.
- Transport-only latent RMSE: 0.216; frozen-policy action RMSE: 0.546.
- Transport + innovation latent RMSE: 0.074; action RMSE: 0.074.
- Policy-consistency latent RMSE: 0.074; action RMSE: 0.073.

![Predictor ablation](../outputs/predictor_ablation.png)

## Interpretation

The intended signature appears clearly:

- stale execution context collapses quickly with delay;
- state rollout and correction materially improve state accuracy and tracking, but cannot compensate for stale task/environment information;
- simple transport reduces latent error but misses cue-driven acceleration and stochastic innovation;
- learned innovation is the main recovery mechanism;
- the policy-consistency term changes reconstruction metrics only slightly but improves closed-loop success at delays 6, 8, and 10.

One nuance is that state correction does not improve success over state rollout at delay 2, despite much better held-out state accuracy. The remaining stale latent dominates the success criterion, so extra state accuracy alone has little leverage. Transport-only also performs worse than state rollout at delay 2: a partially wrong future latent can be more harmful than consistently stale context. This is useful evidence for gating or uncertainty-aware fallback in a real adapter.

## Mapping to FutureRTC

| Toy component | FutureRTC analogue | Simplification |
| --- | --- | --- |
| Nominal integration of committed actions | rolled-forward execution-time state | 2D point dynamics, no orientation or gripper |
| Learned state residual | state correction module | small MLP, synthetic state |
| Analytic target/wind transport | motion-aware latent transport | vector features, no token grid or warping |
| Learned latent residual | gated visual innovation/synthesis | one dense vector, no spatial occlusion |
| Frozen-policy action loss | policy consistency loss | first action only, analytical policy |
| Delay sweep | asynchronous VLA inference delay | deterministic scheduler, no wall-clock service |

## Limitations and follow-ups

- No images, language, VLA backbone, flow matching, contacts, occlusion, or robot kinematics.
- Synthetic training and test distributions are closely matched.
- The environment latent is directly supervised and low-dimensional.
- Predictor inference cost is not benchmarked against a real policy.
- Success thresholds are local toy definitions.

The next useful step would be a small image/token-grid version with a moving camera/arm mask, explicit warping, occlusion/disocclusion, and uncertainty-aware fallback to stale or transported features. After that, the idea could be attached to an existing learned chunk policy in a lightweight simulator.
