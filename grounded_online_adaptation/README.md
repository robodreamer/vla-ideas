# Grounded Online Adaptation: Fine-Mark Toy

This package tests one narrow question motivated by **GRAFT**:

> When a frozen behavior-cloning policy misses a tiny task-critical visual cue, can training-only region supervision make a small attention pathway adapt more reliably than reward-only or globally pooled updates, while frozen-prefix replay caching reduces learner cost?

The result is a deterministic 1-D visual-alignment toy, not a reproduction of GRAFT, a VLA, actor–critic learning, multi-view robotics, or the paper's real-robot experiments.

## References checked

- Yibo Qiu, Haoliang Ye, Shu'ang Sun, Zan Huang, Ronald X. Xu, and Mingzhai Sun, [GRAFT: Grounded and Efficient Online Reinforcement Adaptation for Fine-Grained Robot Manipulation](https://arxiv.org/abs/2608.27079), arXiv:2608.27079, 2026.
- Arun Prasad, Kevin Lin, Jimmy Wu, Linqi Zhou, and Jeannette Bohg, [Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation](https://doi.org/10.15607/RSS.2024.XX.071), RSS 2024.
- Junjie Wen et al., [TinyVLA: Toward Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation](https://doi.org/10.1109/LRA.2025.3544909), IEEE RA-L 2025.

GRAFT reports four real biomedical manipulation tasks with 10 demonstrations and a matched 45-minute online budget per method. Its controlled GRAFT-Union/GRAFT comparison reports `57.5%` versus `82.5%` rolling success over the final 10 episodes per task, while prefix-KV reuse raises the reported single-step learner throughput from `2.21` to `21.96` steps/s. Those are paper results, not local measurements.

## Toy setup

A 15-patch visual strip contains:

- a broad gray object whose center is the source behavior-cloning target;
- a tiny red alignment mark, introduced only in the adaptation task and offset by 3 or 4 patches;
- two blue point distractors;
- a scalar cursor state broadcast across patches.

The source policy is trained on 2,400 examples to regress the broad-object center. Its visual prefix is a three-layer patch MLP with an explicit raw-feature skip; its BC head acts on mean-pooled frozen tokens. On the fine task, the broad object remains unchanged but the true target becomes the tiny red mark. The frozen BC policy therefore keeps choosing the coarse center.

Each online visual context receives two scalar-reward probes around the current action. The reward is

```text
r(a) = exp(-|a - target| / 2.5)
```

Because this reward shape is known, the learner inverts the two scalar rewards to obtain the unique consistent 1-D target estimate, then performs replay updates. Thus 720 interactions correspond to 360 two-probe visual contexts. This is a deliberately favorable reward-verifier proxy: it uses no mark label for the reward-only policy, but it is much easier than learning a critic from sparse robot returns.

For supervised anchors only, two training-time proposal indices identify the red mark and broad-object center. A permutation-invariant two-way assignment loss teaches two visual anchors without persistent anchor identities. Proposals are not policy inputs and are absent at evaluation.

## Compared methods

- `frozen`: source BC policy with no update.
- `full_policy`: updates the visual prefix and BC head from reward-derived targets; a full-update cost/forgetting proxy.
- `adapter`: a 13-parameter low-rank residual using only globally pooled red-mark mass and positional moment.
- `reward_attention`: two visual anchors and a residual action head trained from scalar reward probes plus source distillation.
- `supervised_anchors`: the same attention pathway plus training-only identity-free proposal supervision.
- `anchors_cached`: identical initialization, data order, losses, and outputs to `supervised_anchors`, but replay learner updates reuse cached frozen-prefix tokens.

The adapter and anchor pathways are gated by red-mark presence. Since the source retention set contains no red mark, zero forgetting for these methods is partly structural; it should not be generalized to unconstrained continual learning.

## Run

From the repository root, using the existing shared environment:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  grounded_online_adaptation/run_grounded_online_adaptation.py
```

Smoke test without replacing the full outputs:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  grounded_online_adaptation/run_grounded_online_adaptation.py \
  --smoke --output-dir /tmp/grounded_online_adaptation_smoke
```

Render the report:

```bash
grounded_online_adaptation/docs/render_pdf.sh
```

## Metrics

- **Fine success:** predicted continuous alignment lies within 0.5 patch of the red mark.
- **Interaction AUC:** fine success integrated over the 720-interaction adaptation budget.
- **Interactions to 70%:** first 48-interaction checkpoint reaching 70% exact-tolerance success; missing if never reached.
- **Forgetting:** source success before adaptation minus source success after adaptation.
- **Update cost:** measured single-thread CPU learner time and trainable parameter count.
- **Prefix compute proxy:** patch-token prefix evaluations made inside replay learner updates.
- **Grounding:** maximum anchor attention on the red mark, proposal coverage, and anchor overlap.

Main results use five seeds (`29–33`) and 900 deterministic evaluation strips per seed. Error bars are ±1 SEM across seeds.

## Verified results

| Method | Trainable params | Final fine success | Interaction AUC | Interactions to 70% | Forgetting | Learner time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen BC | 0 | 0.000 ± 0.000 | 0.000 ± 0.000 | not reached | 0.000 ± 0.000 | 0.000 s |
| Full-policy update | 42,534 | 0.637 ± 0.039 | 0.255 ± 0.027 | 576 ± 48 (`3/5` seeds) | 0.365 ± 0.105 | 4.256 s |
| Lightweight adapter | 13 | 0.287 ± 0.014 | 0.220 ± 0.008 | not reached | 0.000 ± 0.000 | 1.785 s |
| Reward-only attention | 6,371 | 0.636 ± 0.155 | 0.300 ± 0.098 | 384 ± 48 (`2/5` seeds) | 0.000 ± 0.000 | 2.195 s |
| Supervised anchors | 6,371 | **0.968 ± 0.031** | **0.506 ± 0.024** | **374 ± 18 (`5/5` seeds)** | 0.000 ± 0.000 | 2.374 s |
| Anchors + cached prefix | 6,371 | **0.968 ± 0.031** | **0.506 ± 0.024** | **374 ± 18 (`5/5` seeds)** | 0.000 ± 0.000 | **1.272 s** |

![Learning curves](outputs/learning_curves.png)

### Bounded interpretation

- The frozen source BC policy gets `0%` fine-mark success despite retaining `100%` source success: the new local cue is genuinely task-critical.
- Supervised anchors improve mean final success over reward-only attention by `33.1` percentage points and reach 70% success in all five seeds. Reward-only attention is highly variable (`0.636 ± 0.155`).
- The 13-parameter global-moment adapter preserves source behavior but plateaus near `29%`, so lightweight capacity alone is insufficient in this construction.
- The full-policy proxy reaches `63.7%` mean fine success but forgets `36.5` percentage points of source success on average and has the largest measured update cost.
- Cached and uncached supervised-anchor policies are numerically identical in the final parity check (`0.0` maximum logit difference across seeds). Caching removes all replay-side prefix evaluations in the local accounting and gives a `1.87×` measured learner-update speedup (`2.374 s` to `1.272 s`). Acting and evaluation still compute fresh prefixes.
- Anchor supervision raises red-mark attention from `0.455 ± 0.026` for reward-only attention to `0.551 ± 0.011`, but attention mass is only a mechanism diagnostic, not an explanation proof.

![Method trade-offs](outputs/method_tradeoffs.png)

![Cache runtime](outputs/cache_runtime.png)

### Sweeps and ablations

The grounding-weight sweep uses three seeds and the same 720-interaction budget. Weights `0.1` and `0.2` both reach `100%` mean final success; weight `0` averages `79.3%` with large seed variance, while excessive weight `0.8` falls to `26.4%`. The result supports moderate auxiliary grounding, not “more supervision is always better.”

The post-training contrast sweep is deliberately harsh. All methods fail at contrast `0.12`; supervised anchors reach `96.4%` at the trained contrast `0.28` but only `3.1%` at `0.40`. The non-monotonic failure exposes calibration to the training contrast and argues against a general robustness claim.

![Sweeps](outputs/sweeps.png)

![Attention diagnostics](outputs/attention_diagnostics.png)

## Outputs

- `outputs/metrics.json`: configuration, method definitions, sanity checks, aggregate rows, and bounded headline deltas.
- `outputs/summary_metrics.csv`: mean, standard deviation, SEM, and valid count by method.
- `outputs/condition_metrics.csv`: per-seed final performance, forgetting, parameter, prefix, and runtime metrics.
- `outputs/learning_curve.csv`: per-seed adaptation checkpoints.
- `outputs/contrast_sweep.csv`: post-training marker-contrast robustness sweep.
- `outputs/supervision_sweep.csv`: grounding-loss weight ablation.
- `outputs/sanity_check.json`: nine deterministic mechanism checks.
- `outputs/*.png`: learning, trade-off, grounding, cache, and sweep figures.
- `docs/grounded_online_adaptation_report.tex` and `.pdf`: source-grounded report.

## Relation to other experiments in this repository

- **The other three target tracks update elsewhere.** `local_residual_sim2real` performs online local model/command correction from observed transitions. `predictive_error_correction` and `prediction_error_policy_state` freeze policy/model weights and correct latent policy state at inference from sensory prediction error. Here online replay changes attention/action parameters from reward-derived targets, with proposal supervision used only during training.
- **`conflict_aware_replay` is the closest retention study.** It selects old transitions during sequential task training and reports negative backward transfer and continual-learning AUC. This package distills to source examples and reports source forgetting; its zero forgetting for adapter paths is aided by an explicit mark-presence gate, so it is a weaker locality test than conflict-aware overlap/disagreement selection.
- **`retry_reset_recovery` adapts behavior through an offline skill library plus an online router.** It measures retry/OOD recovery, false resets, and compounding failures rather than fine-cue success, interaction AUC, grounding, and forgetting. A grounded adapter could improve its monitor or reset-skill perception, but this toy contains no failure-state routing.
- **`anticipatory_context_chunking` addresses temporal staleness rather than spatial grounding.** It predicts execution-time robot/environment context before a delayed chunk handoff. `force_embodiment_gap` and `force_feedback_demonstration_quality` are offline data/interface tracks: they test whether matched morphology, calibration, and force-rich demonstrations reduce the gap before any online reward adaptation.
- **Safety remains external here.** `constraint_manifold_action_filter` and `path_consistent_safety_filtering` modify proposed chunks at execution to satisfy geometry or preserve a safe path; `configured_failure_audit` adds stop guards and false-stop accounting. This package reports no collision/constraint metric, so fine-mark success and attention diagnostics are not safety evidence.

## Simplifications and limits

This toy uses a 1-D synthetic strip, one visual stream, continuous scalar alignment, a known invertible reward shape, deterministic replay, and a small MLP rather than a pretrained VLM. It does not implement a critic, demonstrations mixed with online robot windows, consistency-policy action chunks, asynchronous actor–learner execution, multi-view anchors, FreeDice masks, contact dynamics, or real GPU KV-cache benchmarking. The marker-presence gate gives adapter methods an explicit task boundary and makes source retention easier. Runtime is a local CPU measurement, while prefix token evaluations are an architecture-level proxy. The five-seed intervals are descriptive; no significance test was run.

Useful follow-ups are to learn the reward model rather than invert it, remove the task-presence gate, randomize mark appearance, add multiple views and proposals, and benchmark frozen-prefix caching on an actual pretrained visual transformer.
