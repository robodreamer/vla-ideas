# VLA Ideas

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-managed-6A5ACD?logo=uv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)
![Git LFS](https://img.shields.io/badge/assets-Git%20LFS-0A97B0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/robodreamer/vla-ideas?style=social)](https://github.com/robodreamer/vla-ideas/stargazers)

Toy research prototypes for exploring visual-language-action timing, policy conditioning, and execution under latency. The repository currently contains twenty-six compact experiment tracks plus a lightweight research-notes index:

- `recap_pi`: RECAP-style advantage-conditioned navigation demos in 2D and 3D.
- `async_chunking_compare`: a lightweight simulator for comparing synchronous and asynchronous chunked-control strategies under inference delay.
- `prefix_rl_chunking`: a toy PPO + action-prefix-loss demo inspired by RL-for-chunked-VLA prefix-stability discussions.
- `path_consistent_safety_filtering`: a PACS-inspired toy comparing path-consistent braking against reactive CBF-like correction for action chunks.
- `bspline_action_parameterization`: a B-spline Policy-inspired toy comparing dense waypoint chunks against compact continuous B-spline action chunks under speed-up.
- `turbo_vla_direct_control`: a TurboVLA-inspired direct V+L→A chunk-policy toy comparing 32 Hz direct fusion against lower-rate LLM-bottleneck-style execution.
- `openvla_oft_systems_toy`: an OpenVLA-OFT-inspired systems toy comparing serial-like action availability with parallel continuous chunks and closed-loop refresh under delayed observations.
- `explorative_policy_chunks`: an Explorative Modeling/XM-inspired toy showing that best-of-K action chunks, when seeded with candidate diversity, can avoid multimodal BC averaging in one forward pass.
- `anticipatory_context_chunking`: a FutureRTC-inspired toy comparing stale context, state-only compensation, latent transport, learned innovation, and policy-consistency training under asynchronous handoff delay.
- `context_chunk_tradeoff`: a hidden-temporal-mode toy that sweeps observation-history context against open-loop chunk horizon under paired disturbances and mode switches.
- `bc_distribution_shift_mysteries`: a BC diagnostic sweeping chunk horizon, previous-action history, feature scaling, and basis capacity while comparing expert-state validation loss against closed-loop success and state divergence.
- `demo_prompted_policy`: an S1-inspired toy comparing task-label, replay, phase, full-demo alignment, and latent-intent mechanisms when one demonstration prompts execution under scene and embodiment shift.
- `constraint_manifold_action_filter`: a PR-MPPI-inspired toy filtering aggressive learned action chunks with equality-tangent/inequality-half-space projection and finite-step retraction.
- `conflict_aware_replay`: a Memory Anchors-inspired sequential-learning toy comparing random, hard, diversity, and latent-overlap/action-disagreement replay under limited buffers.
- `retry_reset_recovery`: a FLARE-inspired recovery-data toy separating robot-pose retry robustness from monitor-gated object-state reset skills.
- `streaming_action_denoising`: a FlashVLA-inspired systems toy comparing isolated, few-step, and staggered chunk refinement under delay and disturbances.
- `force_feedback_demonstration_quality`: a PHABS-inspired matched-data toy testing whether cleaner haptic force demonstrations improve fragile-object imitation under shift.
- `prediction_error_policy_state`: a PredVLA-inspired temporal-control toy comparing iterative prediction-error state correction with one-pass observers and exact open loop.
- `configured_failure_audit`: a defense-only TrapVLA-inspired synthetic audit for conditional phase-local failures, mitigation, and execution monitoring.
- `video_prompt_shortcut_resistance`: a Zero-WAM-inspired composition toy testing whether future-chunk supervision increases reliance on the correct video-like prompt instead of text/history shortcuts.
- `instruction_conditioned_async_control`: an Instruct-to-Act-inspired toy comparing direct blocking planner actions with synchronous and asynchronous sparse-instruction control.
- `local_residual_sim2real`: a robot-juggling-inspired adaptation toy comparing global and memory-local residual updates with mutually reachable safe-set filtering.
- `cross_embodiment_world_model`: a CLAP-inspired action-harmonization toy comparing raw, canonical end-effector, and learned transition-latent interfaces across embodiments.
- `force_embodiment_gap`: a Koala-inspired contact toy separating visual aliasing from force calibration, morphology, and action-coordinate transfer.
- `grounded_online_adaptation`: a GRAFT-inspired fine-mark toy comparing reward-only and supervised visual anchors with exact cached/uncached policy parity.
- `predictive_error_correction`: a PredVLA-inspired current-state correction toy comparing exact open loop, inference-time prediction-error updates, and finite-history policies.
- `notes`: source-grounded research notes and a reference index for ideas that may become experiments.

## Preview

| RECAP-conditioned navigation | Async chunking comparison |
| --- | --- |
| ![RECAP 3D preview](recap_pi/outputs/game_3d_recap.gif) | ![Async chunking comparison](async_chunking_compare/outputs/async_chunking_dynamic_monte_carlo.png) |

## Overview

This repo is organized as a small ideas lab rather than a polished framework. The focus is on quickly testing control and conditioning hypotheses, then exporting visual artifacts and short writeups that make the behavior easy to inspect.

### `recap_pi`

`recap_pi` contains toy navigation tasks that compare plain imitation-style rollouts against RECAP-style conditioning. It includes:

- 2D obstacle-navigation demos and comparison figures.
- 3D drone-heist style rollouts with animated outputs.
- extra variants for RL-token, counterfactual, and latent-adaptation experiments.
- a compact concept note and experiment writeups under `recap_pi/docs/`.

### `async_chunking_compare`

`async_chunking_compare` studies delay compensation for chunked action execution. It compares synchronous planning, stale-state async planning, future-state rollouts, and simple prefix/history-conditioned surrogates, then exports figures and trial CSVs for inspection.

### `prefix_rl_chunking`

`prefix_rl_chunking` turns the PPO + prefix-CFM stability idea into chunked-control toys, including a compact 1D reacher and a richer 2D pick/place environment. It compares a BC reference, PPO-only improvement, and PPO with an explicit prefix-copy loss, then exports metrics and summary plots.

### `bspline_action_parameterization`

`bspline_action_parameterization` explores B-spline action chunks as compact continuous action representations for faster high-rate execution. It compares dense discrete waypoints against fitted cubic B-splines and a simple curvature-aware time law.

### `turbo_vla_direct_control`

`turbo_vla_direct_control` explores TurboVLA's practical claim that execution-level manipulation benefits from a compact direct vision+language-to-action path. It trains a tiny bidirectional cross-attention chunk policy and a heavier transformer-core bottleneck proxy, then evaluates how receding-horizon refresh rate changes closed-loop behavior.

### `openvla_oft_systems_toy`

`openvla_oft_systems_toy` explores the output-interface/timing side of OpenVLA-OFT: parallel continuous chunks make actions available differently from a serial-like generator, while delayed feedback makes chunk horizon and refresh cadence consequential. It is a deterministic analytical toy, not a reproduction of OpenVLA-OFT or its reported results.

### `explorative_policy_chunks`

`explorative_policy_chunks` distills Explorative Modeling/XM into a VLA action-chunk setting. It compares ordinary K=1 behavior cloning against best-of-K candidate chunks on an ambiguous over/under obstacle-routing toy, demonstrating how seeded candidate diversity plus best-of-K credit assignment can preserve committed multimodal futures without iterative inference.

### `anticipatory_context_chunking`

`anticipatory_context_chunking` tests the FutureRTC claim that asynchronous chunk handoffs need execution-time observation/environment context, not only a rolled-forward robot state. A frozen synthetic chunk policy is evaluated with stale context, state rollout/correction, analytic latent transport, learned innovation, and a frozen-policy consistency loss across inference delays.

### `context_chunk_tradeoff`

`context_chunk_tradeoff` isolates a complementary execution trade-off: history helps a controller infer a hidden persistent target-motion mode from noisy observations, while action-chunk horizon controls policy refresh after surprises. In the supported C16 comparison, H1 recovers faster than H16; this is not generalized across every context setting. The toy emits a deterministic sanity check plus a paired 2D context/horizon Monte Carlo map and is not a VLA-paper reproduction.

### `bc_distribution_shift_mysteries`

`bc_distribution_shift_mysteries` distills the Behavioral Cloning Mystery write-up into an exact query-boundary sanity check and a trained 2D obstacle-avoidance benchmark. A 48-policy ridge-BC sweep shows how open-loop chunk horizon, previous-action shortcuts, correlated feature scaling, and basis capacity can leave expert-state validation MSE misaligned with closed-loop success, tracking error, and policy-induced state divergence.

### `demo_prompted_policy`

`demo_prompted_policy` treats one trajectory as an inference-time task prompt. It compares a fixed language/task prior, mapped replay, phase retrieval, soft whole-demonstration alignment, and a hand-engineered latent interaction sequence while sweeping scene transforms, mirrored action mappings, prompt corruption, perturbations, and long horizons.

### `constraint_manifold_action_filter`

`constraint_manifold_action_filter` wraps aggressive behavior-cloned tray-motion chunks in explicit execution geometry. It compares soft penalties, one-step correction, rollout-time equality/inequality projection, and projection plus nonlinear retraction, exposing both finite-step equality drift and locally infeasible hard constraints.

### `conflict_aware_replay`

`conflict_aware_replay` studies sequential policies whose physical observations overlap while required actions conflict. It compares no replay, random replay, loss-hard mining, latent diversity, and Memory Anchors-style overlap/disagreement selection across replay budgets, reporting both closed-loop success and normalized backward transfer.

### `retry_reset_recovery`

`retry_reset_recovery` separates recoverable robot-pose drift from state-breaking object failures. Success-only, generic recovery, perturb+bridge, monolithic reset, and monitor-gated object-skill policies are evaluated on clean, retry, reset, mixed, and hard failure scenarios.

### `streaming_action_denoising`

`streaming_action_denoising` models FlashVLA's staggered action-buffer scheduling and cleaner-to-noisier causal coupling. It compares configured throughput, cold start, tracking, chunk-boundary continuity, handoff staleness, and success against isolated, few-step, and future-state-conditioned baselines.

### `force_feedback_demonstration_quality`

`force_feedback_demonstration_quality` uses matched synthetic haptics-on/off demonstrations for a bimanual fragile-object handoff and insertion. Identical ridge imitation policies test whether force-profile quality, rather than force annotation alone, changes damage, slip/drop, retry, and shifted-object robustness.

### `prediction_error_policy_state`

`prediction_error_policy_state` isolates observation-driven latent correction in a compact generative recurrent controller. It compares simple recurrence, finite-window attention-like updates, a learned observer, iterative predictive-coding correction, and an exact open-loop ablation under occlusion, delay, disturbance, and sensor bias.

### `configured_failure_audit`

`configured_failure_audit` is a deliberately constrained defensive evaluation. A fixed synthetic token and hard-coded phase-local residuals demonstrate why clean validation can miss conditional failures, then test data filtering, token-channel shrinkage, trigger-invariance regularization, and execution-time safety stops. It has no external-model, dataset, or robot interface.

### `video_prompt_shortcut_resistance`

`video_prompt_shortcut_resistance` creates seen tasks where text and action history predict an easy dominant suffix, then evaluates held-out compositions that require the aligned video-like prompt. It compares direct and prompt-conditioned BC with a future-chunk objective, prompt shuffling, corruption, and distractor interventions.

### `instruction_conditioned_async_control`

`instruction_conditioned_async_control` separates a slow sparse route planner from a high-frequency learned controller in a partially observed navigation-like environment. It measures the reliability, blocking, staleness, smoothness, and safety trade-off between direct planner actions, synchronous instructions, and asynchronous online planning.

### `local_residual_sim2real`

`local_residual_sim2real` begins with a biased transition prior and adapts from short real-like rollouts. Global replacement, global residual fitting, nearest-memory correction, and uncertainty-gated local residuals are compared with and without a mutually reachable safe-set filter for chained throw/catch-like transitions.

### `cross_embodiment_world_model`

`cross_embodiment_world_model` tests which action coordinates let source-embodiment experience predict target object dynamics and rerank candidate actions with few target demonstrations. It separates padded raw controls, exact canonical end-effector actions, transition latents, and a latent-to-EE curriculum interface.

### `force_embodiment_gap`

`force_embodiment_gap` studies contact states that look identical in vision but differ in load or jam direction. Vision-only, temporal, raw-force, mismatched-gripper, and matched multimodal policies expose how force observability differs from calibrated morphology/action-coordinate transfer.

### `grounded_online_adaptation`

`grounded_online_adaptation` adapts a frozen coarse-alignment policy to a tiny visual mark using reward-derived replay. Global updates, a low-rank adapter, reward-only attention, supervised visual anchors, and cached anchors are compared for sample efficiency, grounding, forgetting, and learner cost.

### `predictive_error_correction`

`predictive_error_correction` keeps recurrent weights fixed and corrects the current latent state from sensory prediction errors at inference time. Exact open loop, current-state correction, finite-history policies, and open-loop chunks are evaluated under hidden force, target maneuvers, observation noise, and drift.

## How the experiments relate

The tracks intentionally isolate different intervention points rather than treating every improvement as the same kind of adaptation:

| Family | Tracks | Main distinction |
| --- | --- | --- |
| Task prompting, transfer, and data interfaces | `demo_prompted_policy`, `video_prompt_shortcut_resistance`, `cross_embodiment_world_model`, `force_embodiment_gap`, `force_feedback_demonstration_quality` | Runtime task prompts, cross-embodiment action semantics, sensor/morphology interfaces, and collection-time target quality are evaluated separately. |
| Timing, planning, and action delivery | `async_chunking_compare`, `prefix_rl_chunking`, `bspline_action_parameterization`, `turbo_vla_direct_control`, `openvla_oft_systems_toy`, `anticipatory_context_chunking`, `context_chunk_tradeoff`, `streaming_action_denoising`, `instruction_conditioned_async_control` | Planner latency, decoder scheduling, handoff-state compensation, representation, context, horizon, and prefix stability are distinct mechanisms. |
| Adaptation, prediction, and recovery | `recap_pi`, `explorative_policy_chunks`, `conflict_aware_replay`, `retry_reset_recovery`, `prediction_error_policy_state`, `predictive_error_correction`, `grounded_online_adaptation`, `local_residual_sim2real` | The update locus ranges from conditioning and candidate selection to replay, recovery skills, transient latent inference, policy updates, and local transition memory. |
| Safety and diagnostics | `path_consistent_safety_filtering`, `constraint_manifold_action_filter`, `bc_distribution_shift_mysteries`, `configured_failure_audit` | Execution filtering, geometric projection, benign closed-loop distribution shift, and hidden conditional-behavior auditing answer different failure questions. |

Each new experiment README and PDF includes a more detailed comparison against its nearest repository neighbors, including negative results and confounds found during review.

## Research Notes

[`notes/`](notes/README.md) is the repository knowledge base for VLA and embodied-AI references. Notes preserve source links, evidence level, key claims, limitations, and concrete experiment hooks without turning every useful article into a prototype.

- [X Square Robot embodied-AI stack note](notes/2026-08-05-x-square-embodied-ai-stack.md)
- [OpenVLA-OFT primary-source note](notes/2026-08-06-openvla-oft.md)

## Quick Start

Set up the shared Python environment:

```bash
cd recap_pi
uv sync
```

Run the core RECAP demos:

```bash
cd recap_pi
uv run python recap_demo_complex_2d.py
uv run python recap_demo_game_3d.py
```

Run the async chunking comparison:

```bash
cd /home/andypark/Projects/playground/vla-ideas
recap_pi/.venv/bin/python async_chunking_compare/run_async_chunking_compare.py
```

Run the prefix-RL chunking toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
python prefix_rl_chunking/run_prefix_rl_chunking.py
python prefix_rl_chunking/run_prefix_rl_pickplace_2d.py
```

Run the PACS path-consistent safety toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python path_consistent_safety_filtering/run_pacs_toy.py --trials 180
```

Run the B-spline action parameterization toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python bspline_action_parameterization/run_bspline_action_toy.py --trials 160
```

Run the TurboVLA direct-control toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python turbo_vla_direct_control/run_turbo_vla_toy.py --train-steps 480 --eval-episodes 200
```

Run the OFT-inspired systems toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python openvla_oft_systems_toy/run_openvla_oft_systems_toy.py --seed 17 --trials 96
```

Run the Explorative Policy chunks toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python explorative_policy_chunks/run_explorative_policy_toy.py --steps 900
```

Run the anticipatory-context chunking toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python anticipatory_context_chunking/run_anticipatory_context_chunking.py --seed 17 --trials 80 --train-samples 12000 --val-samples 2500 --train-steps 650
```

Run the context-versus-chunking toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python context_chunk_tradeoff/run_context_chunk_tradeoff.py --trials 120 --seed 17
```

Run the BC distribution-shift mysteries toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python bc_distribution_shift_mysteries/run_bc_distribution_shift_mysteries.py --seed 17 --eval-episodes 96
```

Run the S1-inspired demonstration-prompted policy toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python demo_prompted_policy/run_demo_prompted_policy.py --seed 23 --trials-per-cell 6
```

Run the PR-MPPI-inspired constraint-manifold filter toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
PYTHONWARNINGS=error /home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python constraint_manifold_action_filter/run_constraint_manifold_action_filter.py --seed 23 --trials 120 --train-samples 7000 --test-samples 1400
```

Run the Memory Anchors-inspired conflict-aware replay toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python conflict_aware_replay/run_conflict_aware_replay.py --seed 17 --seeds 6 --train-episodes 12 --val-episodes 5 --eval-rollouts 32 --initial-steps 650 --train-steps 480 --buffer-sizes 12 32 96 288
```

Run the FLARE-inspired retry/reset recovery toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
PYTHONWARNINGS=error /home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python retry_reset_recovery/run_retry_reset_recovery.py --seed 29 --trials 120 --train-demos 48 --max-steps 230
```

Run the FlashVLA-inspired streaming action-denoising toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python streaming_action_denoising/run_streaming_action_denoising.py --seed 23 --trials 48 --episode-steps 240
```

Run the PHABS-inspired force-feedback demonstration toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python force_feedback_demonstration_quality/run_force_feedback_demonstration_quality.py --seed 37 --demo-scenarios 72 --demo-repeats 2 --eval-trials-per-shift 180
```

Run the PredVLA-inspired predictive-state correction toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python prediction_error_policy_state/run_prediction_error_policy_state.py --seed 23 --trials 48
```

Run the defense-only configured-failure audit:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python configured_failure_audit/run_configured_failure_audit.py --seed 31
```

Run the Zero-WAM-inspired prompt shortcut-resistance toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
PYTHONWARNINGS=error /home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python video_prompt_shortcut_resistance/run_video_prompt_shortcut_resistance.py --mode full --seed 31 --train-episodes 720 --epochs 220 --eval-trials 24
```

Run the Instruct-to-Act-inspired asynchronous-control toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
PYTHONWARNINGS=error /home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python instruction_conditioned_async_control/run_instruction_conditioned_async_control.py --seed 41 --trials 64 --episode-steps 260
```

Run the local-residual sim-to-real adaptation toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
PYTHONWARNINGS=error /home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python local_residual_sim2real/run_local_residual_sim2real.py
```

Run the CLAP-inspired cross-embodiment world-model toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python cross_embodiment_world_model/run_cross_embodiment_world_model.py --seed 23 --seeds 8 --source-episodes 150 --target-pool-episodes 48 --test-transitions 900 --planning-queries 320 --shots 0 1 2 4 8 16 32 --source-ee-label-fraction 0.01
```

Run the force-embodiment-gap toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
PYTHONWARNINGS=error /home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python force_embodiment_gap/run_force_embodiment_gap.py --seed 31 --train-episodes 180 --validation-episodes 45 --eval-episodes 120 --sweep-episodes 32
```

Run the grounded online-adaptation toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
PYTHONWARNINGS=error /home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python grounded_online_adaptation/run_grounded_online_adaptation.py
```

Run the predictive-error-correction toy:

```bash
cd /home/andypark/Projects/repos/vla-ideas
PYTHONWARNINGS=error /home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python predictive_error_correction/run_predictive_error_correction.py --seed 23 --trials 64 --train-episodes 180
```

## What Gets Generated

The experiment scripts write visual outputs and reports directly into the repo so results stay easy to compare across iterations.

### `recap_pi/outputs`

- rollout GIFs for plain and RECAP-conditioned policies.
- side-by-side comparison plots for 2D and 3D tasks.
- additional assets for RL-token, counterfactual, and latent-adaptation variants.

### `async_chunking_compare/outputs`

- single-run and Monte Carlo comparison plots.
- delay-sweep plots.
- per-trial CSV exports for static and dynamic settings.

### `prefix_rl_chunking/outputs`

- PPO/BC comparison metrics and training curves for the 1D and 2D examples.
- summary plots showing success, safety stops, prefix-copy error, and representative rollouts.

### `path_consistent_safety_filtering/outputs`

- Monte Carlo metrics for raw, reactive CBF-like, and PACS-style time-law controllers.
- representative trajectory/speed plots showing path-consistent slowdown versus lateral deviation.

### `bspline_action_parameterization/outputs`

- Monte Carlo metrics comparing discrete waypoint chunks, fast B-spline chunks, and B-spline chunks with a curvature-aware time law.
- representative rollout and path-error plots showing jerk reduction and local slowdown under speed-up.

### `turbo_vla_direct_control/outputs`

- BC training/latency metrics for a direct V+L→A chunk policy and an LLM-bottleneck proxy.
- receding-horizon success, distance, jerk, and refresh-rate stress plots.
- representative rollouts around a central keep-out zone.

### `openvla_oft_systems_toy/outputs`

- aggregate CSV for configured latency, action-output rate, tracking, success, and jerk proxy;
- a matched-rollout and trade-off plot;
- a generated experiment report with metric definitions, results, interpretation, and scope boundary.

### `explorative_policy_chunks/outputs`

- best-of-K metrics comparing K=1 BC against K=2/4/8 Explorative Policy-style chunks with explicit diversity seeding.
- representative over/under obstacle trajectories showing mode averaging versus committed candidate chunks.
- training curves and K-sweep plots for success, collision, and oracle reconstruction error.

### `anticipatory_context_chunking/outputs`

- paired delay-sweep trial and summary metrics for stale, state-only, transport, innovation, policy-consistency, and oracle methods.
- held-out state/latent predictor ablations and deterministic mechanism sanity checks.
- delay-sweep, representative-rollout, and predictor-ablation plots.

### `context_chunk_tradeoff/outputs`

- deterministic mechanism checks and paired per-trial/aggregate CSV and JSON metrics for the full context-by-horizon sweep;
- heatmaps for success, tracking error, recovery delay, and planning-call proxy;
- a reliability/error trade-off plot and a matched reactive-versus-open-loop rollout.

### `bc_distribution_shift_mysteries/outputs`

- exact query-boundary sanity-check metrics plus aggregate and per-trial CSV/JSON for 48 ridge-BC policies;
- chunk/history/scaling sweep plots and validation-loss-versus-rollout comparisons;
- representative rollouts showing how correlated feature scalings extrapolate after perturbations;
- a generated LaTeX experiment report and rendered PDF.

### `demo_prompted_policy/outputs`

- paired per-rollout and aggregate metrics for five demonstration-use mechanisms;
- robustness sweeps over scene shift, mirrored action mappings, prompt corruption, and horizon;
- a deterministic intent-extraction/motor-mapping sanity check and representative rollout;
- a generated LaTeX experiment report and rendered PDF.

### `constraint_manifold_action_filter/outputs`

- paired trial and aggregate metrics for penalty, correction, projection, and projection+retraction filters;
- deterministic tangent, half-space, drift, retraction, infeasibility, and degeneracy checks;
- summary and representative residual/margin/intervention plots;
- a generated LaTeX experiment report and rendered PDF.

### `conflict_aware_replay/outputs`

- 102 replay-method/buffer conditions with summary, stage, selection, and closed-loop metrics;
- buffer-sweep and learning-curve plots plus anchor diagnostics and representative routes;
- deterministic overlap, action-conflict, selection, and buffer-size checks;
- a generated LaTeX experiment report and rendered PDF.

### `retry_reset_recovery/outputs`

- paired clean, retry, reset, mixed, and hard-failure rollout metrics;
- pose-magnitude, monitor-quality, and missing-reset-skill ablations;
- data-support, robustness, recovery, false-reset, and representative-trajectory plots;
- a generated LaTeX experiment report and rendered PDF.

### `streaming_action_denoising/outputs`

- 2,592 paired trial rows separating staggered scheduling, causal continuation, and handoff-state compensation;
- latency/disturbance sweeps, throughput/cold-start proxies, and representative rollouts;
- deterministic scheduler, refinement, staleness, continuity, and throughput checks;
- a generated LaTeX experiment report and rendered PDF.

### `force_feedback_demonstration_quality/outputs`

- matched haptics-on/off demonstration metrics and 3,600 downstream policy trials;
- condition summaries and object-property shift sweeps for success, damage, slip/drop, force error, and retries;
- deterministic matching and force-quality sanity checks plus four figures;
- a generated LaTeX experiment report and rendered PDF.

### `prediction_error_policy_state/outputs`

- 5,136 paired temporal-control trials and 107 method/perturbation aggregates;
- occlusion, delay, disturbance, bias, and correction-iteration sweeps;
- coefficient-budget accounting, exact open-loop checks, and representative trajectories;
- a generated LaTeX experiment report and rendered PDF.

### `configured_failure_audit/outputs`

- a fixed-rate/mode/mitigation sweep for a synthetic conditional-failure policy;
- paired clean/triggered metrics, phase-local data and activation probes, and monitor-stop diagnostics;
- defensive sanity checks and five figures with no real-model integration path;
- a generated LaTeX experiment report and rendered PDF.

### `video_prompt_shortcut_resistance/outputs`

- paired seen/unseen composition trials for direct, prompt-conditioned, future-chunk, and prompt-shuffled policies;
- prompt-swap reliance diagnostics plus distractor and frame-corruption sweeps;
- deterministic sanity/smoke checks and three figures;
- a generated LaTeX experiment report and rendered PDF.

### `instruction_conditioned_async_control/outputs`

- 320 default paired trial rows plus planner-latency, cadence, staleness, controller, and planner-swap sweeps;
- post-hoc instruction-relabeling data and transparent controller training metrics;
- scheduling sanity checks and three figures;
- a generated LaTeX experiment report and rendered PDF.

### `local_residual_sim2real/outputs`

- online adaptation curves for four residual/model-update rules with safe-set on/off;
- paired safety, source-retention, extrapolation, memory-width, and regularization metrics;
- eight deterministic mechanism checks and three figures;
- a generated LaTeX experiment report and rendered PDF.

### `cross_embodiment_world_model/outputs`

- few-shot target prediction and candidate-reranking metrics across raw, canonical, latent, and curriculum action interfaces;
- per-seed representation diagnostics and target-shot sweeps;
- seven deterministic action-semantics checks and three figures;
- a generated LaTeX experiment report and rendered PDF.

### `force_embodiment_gap/outputs`

- paired nominal policy trials plus calibration and morphology stress sweeps;
- success, damage, progress, validation-error, and hardware-interface metrics;
- deterministic visual-alias/action-frame checks and four figures;
- a generated LaTeX experiment report and rendered PDF.

### `grounded_online_adaptation/outputs`

- online learning curves, marker-contrast and supervision sweeps, grounding diagnostics, and source-forgetting metrics;
- cached/uncached policy parity, learner-time, and prefix-evaluation accounting;
- nine deterministic mechanism checks and five figures;
- a generated LaTeX experiment report and rendered PDF.

### `predictive_error_correction/outputs`

- paired disturbance-severity trials plus history-length and correction-iteration sweeps;
- exact open-loop, current-state correction, finite-history, and chunked-policy comparisons;
- eight deterministic mechanism checks and four figures;
- a generated LaTeX experiment report and rendered PDF.

## Current Snapshot

The existing `recap_pi` docs report these latest verified headline numbers:

- 2D plain imitation: 190 average effective steps, 17% success, 83% collisions.
- 2D RECAP-conditioned: 41 average effective steps, 100% success, 0% collisions.
- 3D plain imitation: 222 average effective steps, 19% success, 55% artifact pickup, 24% uplink activation, 79% collisions.
- 3D RECAP-conditioned: 111 average effective steps, 74% success, 100% artifact pickup, 74% uplink activation, 19% collisions.

## Repository Layout

```text
vla-ideas/
├── README.md
├── notes/                         # indexed VLA / embodied-AI research notes
│   ├── README.md
│   └── YYYY-MM-DD-*.md
├── recap_pi/
│   ├── pyproject.toml
│   ├── recap_demo_complex_2d.py
│   ├── recap_demo_game_3d.py
│   ├── recap_demo_*.py
│   ├── outputs/
│   └── docs/
├── async_chunking_compare/
│   ├── run_async_chunking_compare.py
│   ├── outputs/
│   └── docs/
├── prefix_rl_chunking/
│   ├── run_prefix_rl_chunking.py
│   ├── run_prefix_rl_pickplace_2d.py
│   ├── outputs/
│   └── docs/
├── path_consistent_safety_filtering/
│   ├── run_pacs_toy.py
│   ├── outputs/
│   └── docs/
├── bspline_action_parameterization/
│   ├── run_bspline_action_toy.py
│   ├── outputs/
│   └── docs/
├── turbo_vla_direct_control/
│   ├── run_turbo_vla_toy.py
│   ├── outputs/
│   └── docs/
├── openvla_oft_systems_toy/
│   ├── run_openvla_oft_systems_toy.py
│   ├── outputs/
│   └── docs/
├── explorative_policy_chunks/
│   ├── run_explorative_policy_toy.py
│   ├── outputs/
│   └── docs/
├── anticipatory_context_chunking/
│   ├── run_anticipatory_context_chunking.py
│   ├── outputs/
│   └── docs/
├── context_chunk_tradeoff/
│   ├── run_context_chunk_tradeoff.py
│   ├── outputs/
│   └── docs/
├── demo_prompted_policy/
│   ├── run_demo_prompted_policy.py
│   ├── outputs/
│   └── docs/
├── constraint_manifold_action_filter/
│   ├── run_constraint_manifold_action_filter.py
│   ├── outputs/
│   └── docs/
├── conflict_aware_replay/
│   ├── run_conflict_aware_replay.py
│   ├── outputs/
│   └── docs/
├── retry_reset_recovery/
│   ├── run_retry_reset_recovery.py
│   ├── outputs/
│   └── docs/
├── streaming_action_denoising/
│   ├── run_streaming_action_denoising.py
│   ├── outputs/
│   └── docs/
├── force_feedback_demonstration_quality/
│   ├── run_force_feedback_demonstration_quality.py
│   ├── outputs/
│   └── docs/
├── prediction_error_policy_state/
│   ├── run_prediction_error_policy_state.py
│   ├── outputs/
│   └── docs/
├── configured_failure_audit/
│   ├── run_configured_failure_audit.py
│   ├── outputs/
│   └── docs/
├── video_prompt_shortcut_resistance/
│   ├── run_video_prompt_shortcut_resistance.py
│   ├── outputs/
│   └── docs/
├── instruction_conditioned_async_control/
│   ├── run_instruction_conditioned_async_control.py
│   ├── outputs/
│   └── docs/
├── local_residual_sim2real/
│   ├── run_local_residual_sim2real.py
│   ├── outputs/
│   └── docs/
├── cross_embodiment_world_model/
│   ├── run_cross_embodiment_world_model.py
│   ├── outputs/
│   └── docs/
├── force_embodiment_gap/
│   ├── run_force_embodiment_gap.py
│   ├── outputs/
│   └── docs/
├── grounded_online_adaptation/
│   ├── run_grounded_online_adaptation.py
│   ├── outputs/
│   └── docs/
└── predictive_error_correction/
    ├── run_predictive_error_correction.py
    ├── outputs/
    └── docs/
```

## Reports

LaTeX reports share one renderer and Docker setup under [`tools/`](tools/). Each idea folder keeps a thin `docs/render_pdf.sh` wrapper, or you can run:

```bash
./tools/render_latex_pdf.sh path/to/report.tex
```

- [`bspline_action_parameterization/README.md`](bspline_action_parameterization/README.md)
- [`bspline_action_parameterization/docs/bspline_action_toy_report.md`](bspline_action_parameterization/docs/bspline_action_toy_report.md)
- [`bspline_action_parameterization/docs/bspline_action_report.pdf`](bspline_action_parameterization/docs/bspline_action_report.pdf)
- [`path_consistent_safety_filtering/README.md`](path_consistent_safety_filtering/README.md)
- [`path_consistent_safety_filtering/docs/pacs_toy_report.md`](path_consistent_safety_filtering/docs/pacs_toy_report.md)
- [`turbo_vla_direct_control/README.md`](turbo_vla_direct_control/README.md)
- [`turbo_vla_direct_control/docs/turbo_vla_toy_report.md`](turbo_vla_direct_control/docs/turbo_vla_toy_report.md)
- [`turbo_vla_direct_control/docs/turbo_vla_direct_control_report.pdf`](turbo_vla_direct_control/docs/turbo_vla_direct_control_report.pdf)
- [`openvla_oft_systems_toy/README.md`](openvla_oft_systems_toy/README.md)
- [`openvla_oft_systems_toy/outputs/openvla_oft_systems_report.md`](openvla_oft_systems_toy/outputs/openvla_oft_systems_report.md)
- [`openvla_oft_systems_toy/docs/openvla_oft_systems_toy_report.tex`](openvla_oft_systems_toy/docs/openvla_oft_systems_toy_report.tex)
- [`openvla_oft_systems_toy/docs/openvla_oft_systems_toy_report.pdf`](openvla_oft_systems_toy/docs/openvla_oft_systems_toy_report.pdf)
- [`explorative_policy_chunks/README.md`](explorative_policy_chunks/README.md)
- [`explorative_policy_chunks/docs/explorative_policy_toy_report.md`](explorative_policy_chunks/docs/explorative_policy_toy_report.md)
- [`explorative_policy_chunks/docs/explorative_policy_toy_report.pdf`](explorative_policy_chunks/docs/explorative_policy_toy_report.pdf)
- [`anticipatory_context_chunking/README.md`](anticipatory_context_chunking/README.md)
- [`anticipatory_context_chunking/docs/anticipatory_context_chunking_report.md`](anticipatory_context_chunking/docs/anticipatory_context_chunking_report.md)
- [`anticipatory_context_chunking/docs/anticipatory_context_chunking_report.pdf`](anticipatory_context_chunking/docs/anticipatory_context_chunking_report.pdf)
- [`context_chunk_tradeoff/README.md`](context_chunk_tradeoff/README.md)
- [`context_chunk_tradeoff/docs/context_chunk_tradeoff_report.md`](context_chunk_tradeoff/docs/context_chunk_tradeoff_report.md)
- [`context_chunk_tradeoff/docs/context_chunk_tradeoff_report.tex`](context_chunk_tradeoff/docs/context_chunk_tradeoff_report.tex)
- [`context_chunk_tradeoff/docs/context_chunk_tradeoff_report.pdf`](context_chunk_tradeoff/docs/context_chunk_tradeoff_report.pdf)
- [`demo_prompted_policy/README.md`](demo_prompted_policy/README.md)
- [`demo_prompted_policy/docs/demo_prompted_policy_report.pdf`](demo_prompted_policy/docs/demo_prompted_policy_report.pdf)
- [`constraint_manifold_action_filter/README.md`](constraint_manifold_action_filter/README.md)
- [`constraint_manifold_action_filter/docs/constraint_manifold_action_filter_report.pdf`](constraint_manifold_action_filter/docs/constraint_manifold_action_filter_report.pdf)
- [`conflict_aware_replay/README.md`](conflict_aware_replay/README.md)
- [`conflict_aware_replay/docs/conflict_aware_replay_report.pdf`](conflict_aware_replay/docs/conflict_aware_replay_report.pdf)
- [`retry_reset_recovery/README.md`](retry_reset_recovery/README.md)
- [`retry_reset_recovery/docs/retry_reset_recovery_report.pdf`](retry_reset_recovery/docs/retry_reset_recovery_report.pdf)
- [`streaming_action_denoising/README.md`](streaming_action_denoising/README.md)
- [`streaming_action_denoising/docs/streaming_action_denoising_report.pdf`](streaming_action_denoising/docs/streaming_action_denoising_report.pdf)
- [`force_feedback_demonstration_quality/README.md`](force_feedback_demonstration_quality/README.md)
- [`force_feedback_demonstration_quality/docs/force_feedback_demonstration_quality_report.pdf`](force_feedback_demonstration_quality/docs/force_feedback_demonstration_quality_report.pdf)
- [`prediction_error_policy_state/README.md`](prediction_error_policy_state/README.md)
- [`prediction_error_policy_state/docs/prediction_error_policy_state_report.pdf`](prediction_error_policy_state/docs/prediction_error_policy_state_report.pdf)
- [`configured_failure_audit/README.md`](configured_failure_audit/README.md)
- [`configured_failure_audit/docs/configured_failure_audit_report.pdf`](configured_failure_audit/docs/configured_failure_audit_report.pdf)
- [`video_prompt_shortcut_resistance/README.md`](video_prompt_shortcut_resistance/README.md)
- [`video_prompt_shortcut_resistance/docs/video_prompt_shortcut_resistance_report.pdf`](video_prompt_shortcut_resistance/docs/video_prompt_shortcut_resistance_report.pdf)
- [`instruction_conditioned_async_control/README.md`](instruction_conditioned_async_control/README.md)
- [`instruction_conditioned_async_control/docs/instruction_conditioned_async_control_report.pdf`](instruction_conditioned_async_control/docs/instruction_conditioned_async_control_report.pdf)
- [`local_residual_sim2real/README.md`](local_residual_sim2real/README.md)
- [`local_residual_sim2real/docs/local_residual_sim2real_report.pdf`](local_residual_sim2real/docs/local_residual_sim2real_report.pdf)
- [`cross_embodiment_world_model/README.md`](cross_embodiment_world_model/README.md)
- [`cross_embodiment_world_model/docs/cross_embodiment_world_model_report.pdf`](cross_embodiment_world_model/docs/cross_embodiment_world_model_report.pdf)
- [`force_embodiment_gap/README.md`](force_embodiment_gap/README.md)
- [`force_embodiment_gap/docs/force_embodiment_gap_report.pdf`](force_embodiment_gap/docs/force_embodiment_gap_report.pdf)
- [`grounded_online_adaptation/README.md`](grounded_online_adaptation/README.md)
- [`grounded_online_adaptation/docs/grounded_online_adaptation_report.pdf`](grounded_online_adaptation/docs/grounded_online_adaptation_report.pdf)
- [`predictive_error_correction/README.md`](predictive_error_correction/README.md)
- [`predictive_error_correction/docs/predictive_error_correction_report.pdf`](predictive_error_correction/docs/predictive_error_correction_report.pdf)
- [`recap_pi/README.md`](recap_pi/README.md)
- [`recap_pi/docs/rl_tokens_experiment_report.md`](recap_pi/docs/rl_tokens_experiment_report.md)
- [`recap_pi/docs/recap_concept_writeup.pdf`](recap_pi/docs/recap_concept_writeup.pdf)
- [`async_chunking_compare/README.md`](async_chunking_compare/README.md)
- [`async_chunking_compare/docs/async_chunking_experiment_report.md`](async_chunking_compare/docs/async_chunking_experiment_report.md)
- [`async_chunking_compare/docs/async_chunking_report.pdf`](async_chunking_compare/docs/async_chunking_report.pdf)
- [`prefix_rl_chunking/README.md`](prefix_rl_chunking/README.md)
- [`prefix_rl_chunking/docs/prefix_rl_chunking_report.md`](prefix_rl_chunking/docs/prefix_rl_chunking_report.md)
- [`prefix_rl_chunking/docs/blog_outcome_mapping.md`](prefix_rl_chunking/docs/blog_outcome_mapping.md)
- [`prefix_rl_chunking/docs/prefix_rl_chunking_report.pdf`](prefix_rl_chunking/docs/prefix_rl_chunking_report.pdf)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=robodreamer/vla-ideas&type=Date)](https://www.star-history.com/#robodreamer/vla-ideas&Date)

## Notes

- Large media artifacts are tracked with Git LFS.
- The code here is intentionally lightweight and experiment-oriented, not a general-purpose library.
