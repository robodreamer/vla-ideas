# Async Chunking Compare

This folder contains a lightweight point-mass simulator for comparing asynchronous VLA execution ideas under chunked control and inference delay.

Implemented methods:

- `synchronous`: compute a new chunk only after the old chunk finishes, with the robot idling during the delay window.
- `naive_async`: overlap inference with execution, but plan from the stale current state.
- `rtc_hard`: a deliberately crude RTC-style hard prefix stitch that freezes committed actions without diffusion inpainting.
- `vlash`: roll committed actions forward and plan from the estimated future execution state.
- `prefix_conditioned`: a lightweight train-time surrogate that predicts a postfix action from stale state plus committed prefix.
- `history_conditioned`: the same idea plus a short recent-action history window, as a proxy for a recurrent/history-aware controller.
- `naive_reflex`: naive async plus a small per-step residual controller, as a toy A2C2-style reflex loop.
- `vlash_reflex`: VLASH plus the same reflex loop.

Notes:

- `rtc_hard` is intentionally a toy failure mode, not a faithful implementation of RTC's gradient-guided inpainting.
- `prefix_conditioned` and `history_conditioned` are lightweight toy surrogates. They are not learned VLAs.
- This folder is currently best read as a delay-compensation intuition demo, not a finished benchmark.

Run:

```bash
cd /home/andypark/Projects/playground/vla-ideas
recap_pi/.venv/bin/python async_chunking_compare/run_async_chunking_compare.py
```

Outputs are written to `async_chunking_compare/outputs/`.

The script also exports per-trial CSVs for uncertainty-aware analysis:

- `async_chunking_compare/outputs/async_chunking_static_trials.csv`
- `async_chunking_compare/outputs/async_chunking_dynamic_trials.csv`

The generated writeup lives at:

- `async_chunking_compare/docs/async_chunking_experiment_report.md`
