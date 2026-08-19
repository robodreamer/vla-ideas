# Context versus Chunking Toy

## Narrow question

When a controller must infer a hidden persistent signed-velocity regime from noisy target positions, how does more temporal context trade against longer open-loop action chunks after unpredictable regime switches and robot disturbances?

This is a synthetic mechanism test, not a VLA-paper reproduction or evidence about a physical robot.

## Setup

The target moves with hidden drift $m_t v_0$, where $m_t \in {-1,+1}$ persists and occasionally flips. The controller observes noisy target position only. At a planning call, C1 estimates velocity from the latest observed displacement (two adjacent positions); Ck fits a bounded least-squares slope to the latest k temporal displacements. Thus C1 is noisy but meaningful, and the inferred quantity matches the declared hidden signed-velocity mode. The controller rolls an analytical PD tracker forward and commits H actions open loop.

The same mode sequence, target-velocity noise, robot impulses, and observation noise are reused across all 25 C/H configurations for each seed. The generated demonstration CSV contains real closed-loop oracle trajectories with hidden state retained only for analysis. The evaluated controller is analytical and is **not trained from the CSV**.

## Metrics

- **Success:** at least 73% of the final 45 post-transition errors are below 0.68.
- **Tracking error:** episode mean absolute target–robot separation.
- **Persistent-mode accuracy:** sign agreement between the estimated velocity and hidden mode at planning calls after a full C-transition window and at least 16 steps since the latest mode switch.
- **Restricted event-to-recovery delay:** each event is applied before its first response error. Its baseline is the median of the prior 6 errors. An event is included only if error rises by at least 0.10 within 10 steps. The threshold search begins strictly after the post-event peak, while the reported delay is measured from the event. Recovery requires 2 consecutive errors below baseline plus 25% of the observed rise. The window ends at the next event or episode end; unrecovered events contribute that censoring limit.
- **Event coverage:** impactful events divided by evaluable scheduled events. **Unrecovered** is reported explicitly among impactful events.
- **Jerk proxy:** mean squared first difference of action divided by $dt^2$.
- **Planning calls:** analytical policy invocations per episode, not wall-clock compute.

## Latest generated result

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  context_chunk_tradeoff/run_context_chunk_tradeoff.py \
  --trials 120 --seed 17
```

| Controller | Success | Error | Persistent mode accuracy | Recovery | Coverage | Unrecovered | Jerk | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 / H1 | 44.2% | 0.588 | 57.2% | 18.14 | 45.5% | 188/307 | 795.7 | 180 |
| C16 / H1 | 50.0% | 0.479 | 100.0% | 14.01 | 69.3% | 237/467 | 21.3 | 180 |
| C16 / H16 | 30.0% | 1.028 | 100.0% | 15.48 | 67.8% | 255/457 | 32.1 | 12 |

The deterministic mechanism checks passed: persistent-mode accuracy was 57.8% for C1 and 100.0% for C16, while hand-computed event traces verify impact gating, post-peak threshold search, and next-event censoring. The small fixed surprise schedule gives 13.64 event-to-recovery steps for H1 and 13.79 for H16, but is treated as descriptive rather than a pass criterion. Across the full paired C16 sweep, the per-seed H16-minus-H1 recovery gap is 2.32 steps (deterministic bootstrap 95% interval [1.29, 3.37]).

![Full metric sweep](../outputs/tradeoff_heatmaps.png)

![Reactivity–smoothness trade-off](../outputs/pareto_tradeoff.png)

![Paired rollout](../outputs/paired_rollout.png)

## Interpretation and limits

In this persistent signed-drift regime, longer context improves mode inference by averaging position noise. It need not improve immediately after a switch because old-regime samples remain in the window. Longer chunks sharply reduce planning calls and reduce the action-difference jerk proxy on average across context settings; at fixed C16, however, H16 is less smooth than H1, so smoothness is not claimed to improve monotonically. With context held at C16, H1 has lower restricted event-to-recovery delay than H16 because it can replace stale plans sooner; that reactivity claim is not generalized to every context setting. No single C/H cell is encoded as an automatic winner.

The result is limited to a one-dimensional analytical controller, synthetic matched distributions, hand-set dynamics, and a response-conditioned recovery metric. It omits images, language, learned policies, contact, inference latency, and hardware. Event coverage must be read alongside recovery because events without a demonstrable error rise are excluded rather than mislabeled as instant recoveries.
