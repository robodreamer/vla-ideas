# VLA Ideas

This snapshot contains a self-contained RECAP-style conditioning experiment for a 2D double-integrator navigation task with two obstacles.

## Setup

Install the project environment with:

```bash
uv sync
```

Run the experiment with:

```bash
uv run python recap_demo_complex_2d.py
```

The script writes artifacts into `outputs/`:

- `complex_comparison.png`
- `plain_rollouts.gif`
- `recap_rollouts.gif`

## Current Snapshot Metrics

Latest verified run:

- Plain imitation: 190 average effective steps, 17% success, 83% collisions
- RECAP advantage-conditioned: 41 average effective steps, 100% success, 0% collisions
- Duration improvement: 4.67x faster
- Success improvement: 6.00x higher
