# Conflict-Aware Replay: Memory Anchors Toy

This folder tests a narrow mechanism from **Memory Anchors for Continual Robot Learning**:

> When a new task occupies an old policy's latent region but requires a conflicting action, does deliberately replaying nearby old transitions reduce forgetting compared with generic buffer selection?

The experiment is a deterministic sequential-control toy, not a reproduction of LIBERO, diffusion-policy denoising, a VLA, or the paper's real-robot results.

## References checked

- Maximilian Du, Zhanyi Sun, Chen Xu, Paarth Shah, Masha Itkina, and Shuran Song, [Memory Anchors for Continual Robot Learning](https://arxiv.org/abs/2608.26545), arXiv:2608.26545, 2026.
- Official [Memory Anchors project page](https://robot-adaptation.github.io/MemoryAnchors/).

The implementation follows the paper's three conceptual filters: latent-state overlap, old-policy/action-label disagreement on the new task, and retrieval of old transitions near that conflict set. The local MLP is deterministic, so direct action prediction replaces diffusion action denoising.

## Toy setup

A point controller moves only in `y` while trajectory phase `u` advances from 0 to 1. Each task follows

```text
y*(u) = amplitude * sin(pi*u)^2
```

with clipped velocity feedback. The observation is

```text
[2u-1, y, sin(pi*u), cos(pi*u), task_cue]
```

Four tasks use cues `[-0.60, -0.20, 0.20, 0.60]` and alternating route amplitudes `[+0.82, -0.82, +0.60, -0.60]`. Their physical observations are identical at reset and overlap strongly during the early decision region, while required actions can be opposite. A two-layer 48-unit latent encoder and small action head are trained on the tasks in sequence.

Each seed uses 12 noisy training rollouts and 5 validation rollouts per task, 49 transitions per rollout. At the final task, buffer sizes `{12, 32, 96, 288}` are about `{0.7%, 1.8%, 5.4%, 16.3%}` of the 1,764 available old transitions.

## Compared methods

- `no_replay`: train only on the current task.
- `random`: uniform old-transition replay.
- `loss_hard`: MIR-style mining by old-sample loss increase after a 24-step current-task probe update.
- `diversity`: greedy farthest-first coreset in the current policy's latent space.
- `anchors`: reserve 20% of the buffer for conflict anchors, then fill the remaining 80% uniformly from old data.

Anchor extraction uses:

1. nearest-neighbor, 1%, 5%, and 10% old-latent distance statistics plus two-cluster K-means to isolate the closer new-task region;
2. new-task action error above old validation mean plus two standard deviations;
3. median 5-nearest-neighbor latent distance from each old transition to the filtered new conflict set.

All replay methods use the same 25% replay share in each training minibatch. Buffer size changes which distinct old transitions are available, not the replay gradient weight.

## Run

From the repository root, using the existing shared Python environment:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  conflict_aware_replay/run_conflict_aware_replay.py \
  --seed 17 --seeds 6 \
  --train-episodes 12 --val-episodes 5 --eval-rollouts 32 \
  --initial-steps 650 --train-steps 480 \
  --buffer-sizes 12 32 96 288
```

Smoke test without replacing the full outputs:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  conflict_aware_replay/run_conflict_aware_replay.py \
  --smoke --seeds 2 --buffer-sizes 12 32 \
  --output-dir /tmp/conflict_aware_replay_smoke
```

Render the report:

```bash
conflict_aware_replay/docs/render_pdf.sh
```

## Metrics

- **Closed-loop success:** tracking RMSE `< 0.23`, midpoint branch error `< 0.25`, and final error `< 0.17`.
- **Final average success:** success averaged over all four tasks after the final task.
- **Tracking RMSE:** route deviation during closed-loop policy rollouts.
- **On-policy action MSE:** policy action error against the scripted expert on policy-induced states.
- **Negative backward transfer (NBT):** for task `i`, mean drop from success immediately after learning `i` to every later stage; lower is better.
- **Continual-learning AUC:** task success averaged from its learning stage through the final stage.
- **Anchor concentration:** fraction of selected transitions in the top 10% of the locally computed anchor ranking.

The full run evaluates 6 training seeds and 32 deterministic reset perturbations per task and stage. Error bars are ±1 SEM across training seeds.

## Latest verified results

### Buffer-size sweep

| Method | Buffer | Final success | NBT ↓ | AUC ↑ | Tracking RMSE ↓ | Anchor concentration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No replay | 0 | 0.341 ± 0.037 | 0.893 ± 0.035 | 0.560 ± 0.012 | 0.480 ± 0.019 | n/a |
| Random | 12 | 0.374 ± 0.049 | 0.574 ± 0.044 | 0.508 ± 0.030 | 0.489 ± 0.030 | 0.102 |
| Loss-hard | 12 | 0.143 ± 0.042 | 0.621 ± 0.027 | 0.306 ± 0.039 | 1.557 ± 0.278 | 0.500 |
| Diversity | 12 | **0.448 ± 0.050** | **0.382 ± 0.042** | 0.504 ± 0.027 | 0.570 ± 0.108 | 0.148 |
| Anchors | 12 | 0.384 ± 0.051 | 0.524 ± 0.092 | **0.526 ± 0.026** | 0.516 ± 0.037 | 0.245 |
| Random | 32 | 0.391 ± 0.046 | 0.391 ± 0.065 | 0.447 ± 0.038 | 0.748 ± 0.160 | 0.113 |
| Diversity | 32 | 0.454 ± 0.048 | **0.377 ± 0.039** | **0.507 ± 0.033** | 0.485 ± 0.040 | 0.141 |
| Anchors | 32 | **0.471 ± 0.063** | 0.387 ± 0.053 | 0.481 ± 0.045 | **0.481 ± 0.047** | 0.264 |
| Random | 96 | 0.396 ± 0.053 | 0.367 ± 0.050 | 0.457 ± 0.047 | 0.582 ± 0.068 | 0.108 |
| Diversity | 96 | **0.470 ± 0.054** | **0.331 ± 0.065** | **0.504 ± 0.034** | **0.468 ± 0.042** | 0.149 |
| Anchors | 96 | 0.454 ± 0.043 | 0.354 ± 0.042 | 0.489 ± 0.030 | 0.520 ± 0.077 | 0.260 |
| Random | 288 | 0.396 ± 0.054 | 0.435 ± 0.038 | 0.451 ± 0.038 | 0.566 ± 0.061 | 0.099 |
| Diversity | 288 | 0.401 ± 0.035 | 0.409 ± 0.037 | 0.475 ± 0.025 | 0.531 ± 0.079 | 0.139 |
| Anchors | 288 | **0.439 ± 0.053** | **0.346 ± 0.057** | **0.492 ± 0.041** | **0.486 ± 0.038** | 0.234 |

Loss-hard rows at 32, 96, and 288 are retained in `summary_metrics.csv`; the table abbreviates them because they remain worse than the other replay selectors on NBT and/or closed-loop error.

![Buffer sweep](outputs/buffer_sweep.png)

### Bounded interpretation

- No replay catastrophically forgets old tasks: mean NBT is `0.893`, despite `0.995` success on the final current task.
- Anchor enrichment increases anchor concentration from roughly `0.10–0.11` under random replay to `0.23–0.26`.
- Mean anchor NBT is lower than random NBT at all four budgets. The reductions are `0.050`, `0.004`, `0.013`, and `0.089` absolute NBT at buffers 12, 32, 96, and 288. At buffer 288 this is a `20.5%` relative reduction, with final success improving from `0.396` to `0.439`.
- The uncertainty intervals overlap, and no hypothesis test was run; these are directional six-seed results, not evidence of statistical significance.
- Diversity is the strongest NBT baseline at the three smaller budgets. The toy therefore **does not support a claim that conflict anchors universally beat a good diversity coreset**. Anchors are best on mean NBT only at buffer 288.
- Loss-hard mining has high retrospective anchor concentration but poor control. Concentrating on probe-sensitive examples without preserving broad task support is not sufficient in this setup.
- Buffer-size trends are not monotonic. A fixed replay fraction, a finite six-seed sweep, and stability/plasticity tradeoffs all matter.

![Learning curves](outputs/learning_curves.png)

![Anchor diagnostics](outputs/anchor_diagnostics.png)

![Closed-loop routes](outputs/closed_loop_routes.png)

The closed-loop figure is a seed-17 example at buffer 32, included as a mechanism visualization rather than an aggregate claim.

## Outputs

- `outputs/metrics.json`: configuration, method definitions, headline rows, sanity checks, and full aggregate results.
- `outputs/summary_metrics.csv`: mean, standard deviation, and SEM by method and buffer size.
- `outputs/condition_metrics.csv`: per-seed continual-learning metrics.
- `outputs/stage_metrics.csv`: per-seed, per-training-stage, per-evaluated-task closed-loop metrics.
- `outputs/selection_metrics.csv`: selector diagnostics and anchor concentration.
- `outputs/sanity_check.json`: seven deterministic mechanism and buffer checks.
- `outputs/buffer_sweep.png`: success, NBT, tracking, and AUC buffer sweeps.
- `outputs/learning_curves.png`: task retention trajectories for no replay, random replay, and anchors at buffer 32.
- `outputs/anchor_diagnostics.png`: overlap/disagreement filters and representative selected buffers.
- `outputs/closed_loop_routes.png`: oldest/newest task trajectories for representative final policies.
- `docs/conflict_aware_replay_report.tex` and `.pdf`: source-grounded report.

## Simplifications and limits

The latent is a learned MLP vector, not a visual-language representation. Actions are scalar deterministic velocities, not multimodal chunks from a diffusion or flow policy. Direct prediction error replaces noise-and-denoise disagreement. The environment is scripted, all task identities are available as scalar cues, train and evaluation distributions are close, and the selection cost is not benchmarked. The experiment also uses one task ordering and does not test the paper's anchor-removal causal intervention.

A useful follow-up would sweep task orders and anchor fractions, add a joint-training upper bound, and repeat the comparison with image observations where new object-task combinations truly collapse in a visual-language encoder.
