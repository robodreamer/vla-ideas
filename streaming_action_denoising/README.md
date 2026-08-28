# Streaming Action Denoising Toy

This track tests a narrow systems/control interpretation of **FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference**.

FlashVLA's central source claim is that a buffer of action chunks at staggered denoising levels can emit one executable chunk per steady-state inference step, while chunk-wise causal attention gives noisier future chunks access to cleaner near-execution chunks. This toy asks what those two mechanisms look like in a deterministic delayed-control model.

It is **not** a FlashVLA reproduction. There is no VLA, transformer, learned flow field, image/language input, training, GPU timing, LIBERO, RoboTwin, or real robot. The causal mechanism is an analytical endpoint/action-continuation surrogate, and all decoder timing is configured rather than measured.

## Compared methods

- `isolated_n_step`: independently refines one chunk for six sequential passes before it is available.
- `few_step_distilled`: a two-refinement quality/capacity proxy with lower latency and residual chunk error. It is not a trained distilled model.
- `streaming_no_causal`: maintains staggered chunk ages and emits one mature chunk per pass after warm-up, but refines slots independently.
- `streaming_causal`: uses the same staggered scheduler plus cleaner-to-noisier endpoint and boundary-action continuation.
- `future_state_conditioned`: keeps six-step isolated decoding but plans from a nominal rollout of the committed action suffix.

A disturbed 2-D point mass tracks a maneuvering target with eight-action chunks. Paired trials sweep decoder pass latency and disturbance magnitude.

## References checked

Primary sources:

- Paper: <https://arxiv.org/abs/2608.27384>
- Official repository: <https://github.com/z-lab/flashvla>

The source reports real learned-policy and hardware results. Its paper uses staggered action buffers, chunk-wise causal attention, a cold-start fill, and one executable chunk per steady streaming pass. The official repository also reports LIBERO, RoboTwin, cross-architecture, and latency results. Those numbers are source context only; none are local measurements.

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

| Method | Success | Tracking RMSE | Boundary jump | Handoff error | Configured chunks/s | Cold start |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Isolated 6-step | 77.1% | 0.459 | 0.235 | 0.125 | 8.33 | 120 ms |
| Few-step distilled proxy | 72.9% | **0.444** | 0.318 | 0.049 | 25.00 | **40 ms** |
| Staggered, no coupling | 31.2% | 0.464 | 0.283 | 0.026 | **50.00** | 120 ms |
| Staggered + causal coupling | 64.6% | 0.480 | **0.043** | **0.002** | **50.00** | 120 ms |
| Future-state conditioned | 75.0% | 0.474 | 0.163 | 0.002 | 8.33 | 120 ms |

Stress condition (`latency_scale=2.0`, `disturbance_scale=1.8`):

- isolated six-step decoding drops to `25.0%` success and incurs `46` mean deadline-miss ticks because its configured 240 ms decode exceeds the 200 ms execution horizon;
- the few-step proxy reaches `64.6%` success, but has a larger `0.393` boundary jump;
- uncoupled streaming reaches `22.9%` success with a `0.374` boundary jump;
- causal streaming reaches `52.1%` success with a `0.066` boundary jump and no steady deadline misses;
- future-state conditioning reaches `50.0%` success but shares isolated decoding's deadline misses.

![Control quality](outputs/control_quality.png)

![Robustness sweeps](outputs/robustness_sweeps.png)

![Systems proxies](outputs/systems_proxies.png)

## What the local result supports

- **Scheduling:** under the explicit equal-cost joint-pass assumption, staggering changes the steady sequential cadence from six passes per emitted chunk to one. It does not remove the six chunk-slot refinements or causal interaction work.
- **Cold start:** streaming does not get a free first action. It pays the same six-pass `120 ms` configured warm-up as full isolated decoding in this setup.
- **Continuity:** causal coupling is the active improvement inside the toy buffer: it reduces default boundary jump from `0.283` to `0.043` and raises success from `31.2%` to `64.6%` relative to the uncoupled scheduler.
- **Trade-off:** causal streaming is not best on every metric. At the default condition it is smoother and less stale, but its tracking RMSE and success remain worse than the isolated and future-state baselines. The toy therefore supports the mechanism, not a blanket superiority claim.

## Outputs

- `outputs/trial_metrics.csv`: all 2,160 per-trial rows.
- `outputs/summary_metrics.csv`: mean/SEM aggregates for 45 method/condition cells.
- `outputs/systems_metrics.csv`: throughput, cold-start, pass-count, slot-update, and causal-interaction proxies.
- `outputs/metrics.json`: config, exact command, default/stress summaries, and claim boundaries.
- `outputs/sanity_checks.json`: seven deterministic mechanism checks.
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
- Future-state conditioning uses a nominal model with plant mismatch, but no learned predictor error.
- Success thresholds and dynamics are toy-specific. The full CSV should be preferred over a single ranking.
