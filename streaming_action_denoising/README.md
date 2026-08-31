# Streaming Action Denoising Toy

This track tests a narrow systems/control interpretation of **FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference**.

FlashVLA's central source claim is that a buffer of action chunks at staggered denoising levels can emit one executable chunk per steady-state inference step, while chunk-wise causal attention gives noisier future chunks access to cleaner near-execution chunks. This toy asks what those two mechanisms look like in a deterministic delayed-control model.

It is **not** a FlashVLA reproduction. There is no VLA, transformer, learned flow field, image/language input, training, GPU timing, LIBERO, RoboTwin, or real robot. The causal mechanism is an analytical endpoint/action-continuation surrogate, and all decoder timing is configured rather than measured.

## Compared methods

- `isolated_n_step`: independently refines one chunk for six sequential passes before it is available.
- `few_step_distilled`: a two-refinement degraded quality/capacity proxy with lower latency, residual chunk error, and hand-coded smoothing/attenuation. It is not a trained distilled model or a fair estimate of what distillation can achieve.
- `streaming_no_causal`: maintains staggered chunk ages and emits one mature chunk per pass after warm-up, but refines slots independently.
- `streaming_causal`: uses the same staggered scheduler plus cleaner-to-noisier endpoint and boundary-action continuation. It deliberately does not roll the latest observation to the handoff state.
- `streaming_causal_compensated`: adds a separately labeled handoff-state surrogate that blends the causal predecessor endpoint with a simple rollout from the latest observation. This diagnostic is not claimed as part of FlashVLA's decoder mechanism.
- `future_state_conditioned`: keeps six-step isolated decoding but plans from a nominal rollout of the committed action suffix.

A disturbed 2-D point mass tracks a maneuvering target with eight-action chunks. Paired trials sweep decoder pass latency and disturbance magnitude.

## References checked

Primary sources:

- Paper: <https://arxiv.org/abs/2608.27384>
- Official repository: <https://github.com/z-lab/flashvla> (inspected at `5227b039ebd4f6b5cad0c27d2d6098932f0f7ed3`)

The source reports real learned-policy and hardware results. Its paper uses staggered action buffers, chunk-wise causal attention, a cold-start fill, and one executable chunk per steady streaming pass. The official repository also reports LIBERO, RoboTwin, cross-architecture, and latency results. Those numbers are source context only; none are local measurements.

## Relation to other experiments in this repository

- `instruction_conditioned_async_control` decouples a slow **semantic planner** from a fast language-conditioned controller. FlashVLA instead pipelines iterative refinement **inside the action decoder**. This package has no sparse instruction handoff or planner/controller hierarchy; the mechanisms are complementary rather than competing definitions of “asynchronous.”
- `async_chunking_compare` and `anticipatory_context_chunking` focus on prediction-to-execution observation delay: roll committed actions forward, correct robot state, and optionally predict the changing visual/environment latent. The FlashVLA toy focuses on decoder scheduling and inter-chunk continuation. Its `future_state_conditioned` row is included specifically as the separate observation-delay-compensation control.
- `context_chunk_tradeoff` varies temporal observation context and open-loop action horizon, while `openvla_oft_systems_toy` varies parallel chunk availability and refresh cadence. Here chunk size and observation format are fixed; the manipulated variables are denoising depth, staggered slot scheduling, and decoder latency.
- `turbo_vla_direct_control` studies a compact direct vision+language-to-action architecture and the benefit of high refresh rate. FlashVLA is compatible with a direct or larger VLA backbone; it changes how an iterative action head is decoded rather than whether the model uses a direct V+L→A path.
- `prefix_rl_chunking` studies training-time stability of copied committed prefixes under RL. The causal FlashVLA surrogate instead passes cleaner predicted endpoint/action context to noisier future slots at inference; it has no prefix-copy loss or policy optimization.
- The prompt tracks (`demo_prompted_policy`, `video_prompt_shortcut_resistance`) test task specification and shortcut-resistant prompt use. This package has no language/prompt input, so it makes no grounding or prompt-reliance claim.

## Baseline fairness and claim boundary

The cleanest decoder-only ablation is `streaming_no_causal` versus `streaming_causal`: both use the same staggered scheduler, slot count, per-slot refinement rule, decoder-noise tensors, launch policy, and environment traces. `streaming_causal_compensated` then adds observation/handoff correction as a separate factor. The isolated, few-step, and future-state rows are diagnostic controls, not architecture- or training-compute-matched learned baselines. In particular, the few-step row is deliberately degraded and must not be read as evidence against real distillation. Reported `chunks/s` is configured back-to-back decoder service capacity under the equal-cost joint-pass assumption, not the simulator's consumed chunk rate or measured GPU throughput.

## Run

Full deterministic sweep used for the checked outputs:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  streaming_action_denoising/run_streaming_action_denoising.py \
  --seed 23 --trials 48 --episode-steps 240
```

Smoke test without overwriting the checked report:

```bash
rm -rf /tmp/streaming-action-denoising-smoke
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  streaming_action_denoising/run_streaming_action_denoising.py \
  --smoke --trials 3 --episode-steps 96 \
  --output-dir /tmp/streaming-action-denoising-smoke --no-report
```

Render the generated TeX report:

```bash
./streaming_action_denoising/docs/render_pdf.sh
```

## Verified results

The full run evaluates 48 paired trials per method at each point of a `3 latency scales × 3 disturbance scales` sweep.

Default condition (`latency_scale=1.0`, `disturbance_scale=1.0`):

| Method | Success | Tracking RMSE | Boundary jump | Handoff error | Decoder capacity | Cold start |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Isolated 6-step | 77.1% | 0.459 | 0.235 | 0.125 | 8.33 | 120 ms |
| Few-step degraded proxy | 72.9% | **0.444** | 0.318 | 0.049 | 25.00 | **40 ms** |
| Staggered, no coupling | 31.2% | 0.464 | 0.283 | 0.026 | **50.00** | 120 ms |
| Staggered + causal only | 4.2% | 0.777 | **0.040** | 0.539 | **50.00** | 120 ms |
| Staggered + causal + handoff compensation | 64.6% | 0.480 | 0.043 | **0.002** | **50.00** | 120 ms |
| Future-state conditioned | 75.0% | 0.474 | 0.163 | 0.002 | 8.33 | 120 ms |

Stress condition (`latency_scale=2.0`, `disturbance_scale=1.8`):

- isolated six-step decoding drops to `25.0%` success and incurs `46` mean deadline-miss ticks because its configured 240 ms decode exceeds the 200 ms execution horizon;
- the few-step proxy reaches `64.6%` success, but has a larger `0.393` boundary jump;
- uncoupled streaming reaches `22.9%` success with a `0.374` boundary jump;
- causal-only streaming reaches `0.0%` success: its boundary jump is only `0.044`, but handoff-state error grows to `0.524`;
- causal streaming plus the separately labeled handoff compensation reaches `52.1%` success with a `0.066` boundary jump and no steady deadline misses;
- future-state conditioning reaches `50.0%` success but shares isolated decoding's deadline misses.

![Control quality](outputs/control_quality.png)

![Robustness sweeps](outputs/robustness_sweeps.png)

![Systems proxies](outputs/systems_proxies.png)

## What the local result supports

- **Scheduling:** under the explicit equal-cost joint-pass assumption, staggering changes the steady sequential cadence from six passes per emitted chunk to one. It does not remove the six chunk-slot refinements or causal interaction work.
- **Cold start:** streaming does not get a free first action. It pays the same six-pass `120 ms` configured warm-up as full isolated decoding in this setup.
- **Continuity is not state alignment:** causal-only continuation cuts the default boundary jump from `0.283` to `0.040`, but its handoff error rises from `0.026` to `0.539` and success falls to `4.2%`. A smooth action boundary can still be conditioned on the wrong execution state.
- **Compensation is separate:** adding the explicitly labeled handoff-state surrogate reduces handoff error to `0.002` and raises success to `64.6%`. This is a diagnostic combination, not attribution of observation-delay compensation to FlashVLA's decoder mechanism.
- **Trade-off:** the compensated streaming row remains worse than isolated/future-state baselines on default success and tracking RMSE. The toy supports distinct scheduling, continuity, and state-alignment mechanisms, not a blanket superiority claim.

## Outputs

- `outputs/trial_metrics.csv`: all 2,592 per-trial rows.
- `outputs/summary_metrics.csv`: mean/SEM aggregates for 54 method/condition cells.
- `outputs/systems_metrics.csv`: throughput, cold-start, pass-count, slot-update, and causal-interaction proxies.
- `outputs/metrics.json`: config, exact command, default/stress summaries, and claim boundaries.
- `outputs/sanity_checks.json`: eight deterministic mechanism checks.
- `outputs/control_quality.png`: default success, tracking, continuity, and handoff metrics.
- `outputs/robustness_sweeps.png`: latency and disturbance sweeps.
- `outputs/systems_proxies.png`: configured throughput/cold-start/work accounting.
- `outputs/representative_rollout.png`: paired representative tracking and actions.
- `docs/streaming_action_denoising_report.tex`: report generated from the full run.
- `docs/streaming_action_denoising_report.pdf`: TeX-rendered report.

## Important limitations

- Throughput and latency are fixed-cost scheduling proxies, not wall-clock or GPU measurements.
- A streaming joint pass is assumed to cost one isolated pass; memory, attention, compilation, and kernel behavior are omitted.
- The few-step baseline is a deterministic degradation model, not trained distillation.
- The causal surrogate chains analytical planned endpoints and actions; it is not chunk-wise transformer attention.
- The compensated streaming row adds a hand-built state rollout/blend that is not claimed as a FlashVLA component.
- Future-state conditioning uses a nominal model with plant mismatch, but no learned predictor error.
- Success thresholds and dynamics are toy-specific. The full CSV should be preferred over a single ranking.
