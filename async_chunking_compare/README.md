# Async Chunking Compare

This folder contains a lightweight point-mass simulator for comparing asynchronous VLA execution ideas under chunked control and inference delay.

Implemented methods:

- `synchronous`: compute a new chunk only after the old chunk finishes, with the robot idling during the delay window.
- `naive_async`: overlap inference with execution, but plan from the stale current state.
- `rtc_hard`: a deliberately crude RTC-style hard prefix stitch that freezes committed actions without diffusion inpainting.
- `vlash`: roll committed actions forward and plan from the estimated future execution state.
- `naive_reflex`: naive async plus a small per-step residual controller, as a toy A2C2-style reflex loop.
- `vlash_reflex`: VLASH plus the same reflex loop.

Notes:

- `rtc_hard` is intentionally a toy failure mode, not a faithful implementation of RTC's gradient-guided inpainting.
- `prefix_conditioned` is a lightweight toy train-time surrogate: a small regression model predicts a constant postfix action from stale state plus committed prefix. It is not a learned VLA.
- This folder is currently best read as a delay-compensation intuition demo, not a finished benchmark.

Run:

```bash
cd /home/andypark/Projects/playground/vla-ideas
recap_pi/.venv/bin/python async_chunking_compare/run_async_chunking_compare.py
```

Outputs are written to `async_chunking_compare/outputs/`.

The generated writeup lives at:

- `async_chunking_compare/docs/async_chunking_experiment_report.md`
