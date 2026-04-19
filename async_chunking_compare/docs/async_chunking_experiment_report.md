# Async Chunking Toy Experiment Report

This note records the current comparison status for toy delayed-action VLA deployment ideas in `async_chunking_compare/`.

## Bottom line

This artifact should still be read as a delay-compensation toy, not a finished benchmark. In the static-goal setting, `VLASH` is the strongest method, the learned `Train-Time Prefix` baseline closes part of the gap without explicit rollout, and the crude `RTC Hard Stitch` fails badly because hard prefix freezing without inpainting breaks trajectory consistency. In the moving-goal setting, the ranking is more mixed.

The main qualitative pattern matches the motivating intuition from the papers:

- `Synchronous` is stable but slow because the robot pauses while replanning.
- `Naive Async` improves utilization but suffers stale-state oscillation.
- `RTC Hard Stitch` is a useful negative control, not a faithful RTC implementation.
- `Train-Time Prefix` learns to continue committed motion from stale observations plus the committed prefix.
- `History + Prefix` tests whether short action history can recover more of the missing state without explicit future rollout.
- `VLASH` is strongest on the static-goal task because it aligns policy input to the future execution-time state directly.

## Implemented experiments

Script:

- `async_chunking_compare/run_async_chunking_compare.py`

Outputs:

- `outputs/async_chunking_static_single_run.png`
- `outputs/async_chunking_static_monte_carlo.png`
- `outputs/async_chunking_dynamic_single_run.png`
- `outputs/async_chunking_dynamic_monte_carlo.png`
- `outputs/async_chunking_delay_sweep.png`
- `outputs/async_chunking_static_metrics.csv`
- `outputs/async_chunking_static_trials.csv`
- `outputs/async_chunking_dynamic_metrics.csv`
- `outputs/async_chunking_dynamic_trials.csv`
- `outputs/async_chunking_delay_sweep.csv`

### Static-goal scenario

Setup summary:

- Goal at `x=10.0`
- Chunk horizon `H=18`
- Delay `d=8`
- Control step `dt=0.02s`
- Velocity disturbance std `0.035`

Current results (means; aggregate plots include standard-error bars):

- Synchronous: `100%` success, `11.09s` mean settle, `0.164` final error, `99.4` RMS jerk
- Naive Async: `0%` success, `nans` mean settle, `6.469` final error, `64.3` RMS jerk
- RTC Hard Stitch: `0%` success, `nans` mean settle, `22.022` final error, `55.0` RMS jerk
- Train-Time Prefix: `81%` success, `12.24s` mean settle, `0.649` final error, `24.8` RMS jerk
- History + Prefix: `41%` success, `15.38s` mean settle, `2.080` final error, `38.5` RMS jerk
- VLASH: `100%` success, `9.15s` mean settle, `0.054` final error, `101.5` RMS jerk

![Static single-run comparison](../outputs/async_chunking_static_single_run.png)

![Static Monte Carlo metrics](../outputs/async_chunking_static_monte_carlo.png)

Interpretation:

- `Synchronous` eventually reaches the target but wastes wall-clock time in every planning gap.
- `Naive Async` keeps moving, but its commands were planned for the wrong physical state and it oscillates.
- `Train-Time Prefix` clearly improves over naive async by using the committed prefix as an implicit delay cue.
- `History + Prefix` checks whether short control history can close more of the gap without future-state rollout; in the current toy it helps less than prefix-only conditioning.
- `VLASH` is still better because it uses the exact rolled-forward future state instead of having to infer it.
- `RTC Hard Stitch` demonstrates why hard handoff alone is not enough; the missing inpainting step matters.

### Moving-goal scenario

Setup summary:

- Goal oscillation amplitude `1.20`
- Goal oscillation period `2.60s`
- Same `H=18`, `d=8`, `dt=0.02s`

Current results (means; aggregate plots include standard-error bars):

- Synchronous: `3%` within tolerance, `1.781` tail mean error, `1.617` final error, `106.4` RMS jerk
- Naive Async: `1%` within tolerance, `7.925` tail mean error, `8.059` final error, `61.2` RMS jerk
- RTC Hard Stitch: `0%` within tolerance, `18.645` tail mean error, `29.850` final error, `53.2` RMS jerk
- Train-Time Prefix: `2%` within tolerance, `2.602` tail mean error, `1.065` final error, `26.0` RMS jerk
- History + Prefix: `2%` within tolerance, `2.589` tail mean error, `1.596` final error, `33.7` RMS jerk
- VLASH: `3%` within tolerance, `1.798` tail mean error, `1.340` final error, `83.3` RMS jerk

![Dynamic single-run comparison](../outputs/async_chunking_dynamic_single_run.png)

![Dynamic Monte Carlo metrics](../outputs/async_chunking_dynamic_monte_carlo.png)

Interpretation:

- The moving target increases the penalty for stale state because the reference itself shifts during the delay window.
- `Train-Time Prefix` still helps by damping the large stale-state oscillations, but it lags the moving reference.
- `History + Prefix` tests whether recent action context helps tracking beyond prefix-only conditioning.
- `VLASH` stays competitive on moving-goal tracking because it plans against the execution-time state instead of the stale one, but the current metrics do not show clean dominance over every baseline.
- This is the scenario where the biological internal-model analogy is most visible in the toy.

### Delay sweep

![Delay sweep](../outputs/async_chunking_delay_sweep.png)

Interpretation:

- Small delays are tolerable for most methods.
- As `d/H` grows, `Naive Async` degrades first, then the learned prefix model, while `VLASH` remains usable longer.
- The hard-stitch negative control becomes unstable quickly, reinforcing that RTC's actual guidance step is doing real work.

## Current conclusion

The toy now exports per-trial metrics for uncertainty-aware analysis (`32` trials per method on the static task and `32` on the moving-goal task). Even with that improvement, the artifact should still be read as a delay-compensation toy rather than a finished benchmark.
