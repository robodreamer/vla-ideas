# Video-Prompt Shortcut-Resistance Toy

This package asks a narrow Zero-WAM-inspired question: **can a training-only future-chunk auxiliary loss make a policy rely more on a visual/video-like prompt when history and text are easier shortcuts on seen tasks?**

The corrected answer in this deterministic synthetic benchmark is **no**. Prompt-conditioned one-step BC reaches **91.7% success on unseen compositions**, while adding the future-chunk auxiliary lowers it to **69.4%**; no-prompt direct BC remains at **0%**. Both prompt-trained policies are sensitive to prompt swaps, while shuffling prompts during training removes prompt reliance and unseen success. Thus prompt conditioning helps, but this particular future-action surrogate does not provide shortcut resistance.

This is a **mechanism toy, not a reproduction of Zero-WAM**. It has no pixels, human videos, transformer, diffusion model, HumanGen data, RoboTwin environment, or robot hardware.

## Primary references

- Jiaming Zhou et al., [In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization](https://arxiv.org/abs/2608.26103), arXiv:2608.26103v2, August 27, 2026.
- [Zero-WAM project page](https://robbyant-research.github.io/Zero-WAM/).
- [Official Zero-WAM repository](https://github.com/robbyant-research/Zero-WAM).

The paper reports a causal video-action policy, 74.2K aligned human-robot pairs over 8.6K tasks, and an in-context future chunk prediction (IFP) objective intended to suppress robot-history/text shortcuts. Its IFP modules predict multiple strided future robot-video chunks from a prompt-influenced current robot representation and are removed at inference. It reports 46.95% average success over seven unseen RoboTwin 2.0 tasks, 29.50 percentage points above LingBot-VA. Those are author-reported results, not measurements from this package. As of August 31, 2026, the official repository documents a code/model/data release planned before September 15, 2026; this toy therefore does not use released Zero-WAM training code or weights.

## Toy design

Each 12-step episode composes three four-action motion primitives.

- **Seen tasks:** six dominant compositions, one per text alias.
- **Shortcut:** the text alias identifies only the first primitive, but predicts the dominant seen suffix; previous action, position, and phase make teacher-forced one-step prediction easy within each chunk.
- **Rare training probes:** 10% of training episodes change one suffix without using the held-out test composition. They make prompt use learnable while leaving the dominant shortcut attractive.
- **Unseen tasks:** the final two primitives are recomposed while reusing the same text alias. The prompt is the only reliable specification of the new order.
- **Video-like prompt:** a six-frame aligned window containing noisy motion, cumulative position, and four visual nuisance channels. It is a hand-designed feature trace, not pixels.
- **Future target:** current action plus five future actions. The auxiliary head reads the same fused history/text/prompt representation as the action head, but is used only during training and is discarded for rollout. This mirrors IFP's causal role more closely than executing the auxiliary prediction, although the target is still actions rather than strided future video chunks.
- **Training:** all prompt models use 15% language dropout. The shuffled-prompt negative control trains the same future objective against a prompt from another batch item.
- **Controlled initialization:** the three prompt-conditioned rows start from the same model initialization and minibatch order; the reported difference is not a different-seed comparison.

Compared methods:

1. `direct_bc`: history + text, no prompt.
2. `prompt_one_step_bc`: history + text + prompt, next-action BC only.
3. `ifp_future_chunk`: one-step BC plus a training-only six-action future-chunk loss.
4. `ifp_shuffled_prompt`: same losses, but prompts are shuffled during training.

## Run

Use the repository's existing `dhb-xr` Python environment (NumPy, Matplotlib, and PyTorch are already available):

```bash
cd /home/andypark/Projects/repos/vla-ideas
PY=/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python
```

Tiny sanity/smoke run:

```bash
PYTHONWARNINGS=error "$PY" \
  video_prompt_shortcut_resistance/run_video_prompt_shortcut_resistance.py \
  --mode smoke --seed 31
```

Full paired evaluation used for the checked-in outputs:

```bash
PYTHONWARNINGS=error "$PY" \
  video_prompt_shortcut_resistance/run_video_prompt_shortcut_resistance.py \
  --mode full --seed 31 --train-episodes 720 --epochs 220 --eval-trials 24
```

Render the report with the repository renderer:

```bash
video_prompt_shortcut_resistance/docs/render_pdf.sh
```

## Latest verified results

The full command produces 1,440 split/task/trial/condition scenarios, 5,760 method-condition rows, and paired correct-prompt/prompt-swap rollouts.

| Method | Seen success | Unseen success | Unseen action RMSE | Unseen chunk RMSE | Prompt-swap output shift | Hard distractor success | Hard corruption success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No-prompt direct BC | 100.0% | 0.0% | 0.862 | 3.352 | 0.000 | 0.0% | 0.0% |
| **Prompt one-step BC** | **100.0%** | **91.7%** | **0.244** | **0.931** | 0.657 | **79.9%** | 0.7% |
| Future-chunk auxiliary | 100.0% | 69.4% | 0.265 | 1.015 | 0.662 | 68.1% | **2.8%** |
| IFP + shuffled prompt | 98.6% | 0.0% | 0.868 | 3.376 | 0.002 | 0.0% | 0.0% |

On clean unseen compositions, swapping in the dominant seen-task prompt drops one-step BC success by **91.7 percentage points** and future-auxiliary success by **69.4 points**. The shuffled-prompt model's output changes by only **0.002**, showing that it learned essentially no prompt reliance. Relative to one-step prompt BC, the future loss reduces unseen success by **22.2 points**, raises action RMSE by **8.5%**, and reduces hard-distractor success by **11.8 points**. Its only headline improvement is a small hard-corruption increase from 0.7% to 2.8%, with both values near failure.

The tiny fixed sanity case uses text alias `2`: the seen shortcut says `(2, 3, 5)`, while the video-like prompt specifies unseen composition `(2, 5, 3)`. Direct BC predicts the seen sequence and fails; both correctly prompted policies predict `(2, 5, 3)` and succeed.

![Method summary](outputs/method_summary.png)

![Robustness sweeps](outputs/robustness_sweeps.png)

## Metrics

- **Task success:** exact recovery of all three primitive IDs, action RMSE below `0.38`, and final endpoint error below `1.10`.
- **Action RMSE:** root mean squared per-coordinate action error over 12 steps.
- **Chunk RMSE:** root mean squared error between four-step action sums for each primitive chunk.
- **Sequence accuracy:** exact three-primitive composition recovery.
- **Prompt-swap output shift:** RMSE between rollouts under the correct prompt and a same-text dominant seen-task prompt.
- **Prompt-swap error increase / success drop:** degradation caused by that swap.
- **Distractor robustness:** unseen success with stronger nuisance channels and drifting prompt-position distractors.
- **Prompt corruption:** unseen success under masked/noisy/reversed prompt frames.

## Outputs

- `outputs/trials.csv`: per-method paired evaluation rows.
- `outputs/summary.csv`: aggregate headline metrics.
- `outputs/metrics.json`: configuration, task split, diagnostics, summary, and sanity check.
- `outputs/sanity_check.json`: one exact seen-shortcut versus unseen-prompt case.
- `outputs/smoke_*.csv/json`: tiny deterministic smoke artifacts.
- `outputs/method_summary.png`: seen/unseen, error, and prompt-reliance comparison.
- `outputs/robustness_sweeps.png`: distractor and corruption sweep.
- `outputs/representative_rollout.png`: one paired unseen trajectory.
- `docs/video_prompt_shortcut_resistance_report.tex` and `.pdf`: report source and rendered PDF.

## Interpretation

The local result supports only the prompt-conditioning mechanism: the correct-prompt versus swapped-prompt intervention shows that the one-step policy genuinely uses the prompt rather than merely succeeding on recomposed tasks. Prompt shuffling removes that dependence while retaining 98.6% seen-task success through the text/history shortcut.

It does **not** support the proposed future-loss mechanism. Once the auxiliary head is kept training-only and all prompt variants share initialization, future-action prediction substantially lowers clean success and modestly raises error, while also hurting distractor performance. This may reflect objective interference, an overly easy/aligned prompt, or the mismatch between contiguous action regression here and Zero-WAM's strided future-video denoising. The negative result is more informative than attributing gains to an auxiliary head that is also executed at inference.

## Relation to other experiments in this repository

- `demo_prompted_policy` is the closest semantic companion: both use a demonstration as a runtime task specification and perform no target-time weight update. That track compares hand-built replay, phase alignment, full-trajectory attention, and event-level intent under scene, embodiment, corruption, and horizon shifts, scoring progress/recovery/interventions. This track gives away correspondence and instead trains matched BC architectures to isolate text/history shortcut pressure, using held-out recomposition and prompt-swap interventions.
- `context_chunk_tradeoff`, `anticipatory_context_chunking`, `async_chunking_compare`, and `streaming_action_denoising` study execution-time context: observation history, stale state/latent prediction, chunk handoff, configured latency, and continuity. Here the future window is a **training target**, not an open-loop execution schedule; there is no asynchronous inference, so tracking delay and boundary-jump metrics are intentionally absent.
- `grounded_online_adaptation`, `local_residual_sim2real`, and `conflict_aware_replay` change parameters or memory using target interactions and report adaptation AUC, extrapolation, retention, or forgetting. This package keeps weights fixed at deployment and asks whether the training objective alone changes in-context prompt use.
- `cross_embodiment_world_model` and `force_embodiment_gap` isolate transfer across action or force/morphology interfaces. This toy assumes the video-like motion is already aligned to robot action coordinates, so it cannot answer their embodiment questions; its unique variable is whether the learned policy follows the correct prompt rather than a seen text/history shortcut.

## Limitations and next steps

- Prompt motion is aligned to robot action coordinates; real human-to-robot correspondence is the hard problem.
- Inputs are low-dimensional traces, not RGB video embeddings.
- The action head directly reads a prompt-fused latent instead of acting through a predicted robot-video representation.
- The task has fixed phase alignment and a fixed 12-step horizon.
- Training uses only six primitive families and synthetic Gaussian perturbations.
- The auxiliary predicts one contiguous action window rather than multiple strided future video chunks.
- Robustness to hard prompt corruption remains poor (2.8% for the best method).
- No statistical uncertainty across independent training seeds is reported; evaluation trials are paired, but the model seed is fixed.
- A stronger follow-up would sweep the auxiliary weight/horizon, remove phase alignment, render image observations, predict strided latent future frames, and average over several training seeds.
