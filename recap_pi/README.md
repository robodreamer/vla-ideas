# VLA Ideas

This directory contains toy RECAP-style examples that illustrate the ideas discussed in the RECAP paper.

## Setup

Install the project environment with:

```bash
cd recap_pi
uv sync
```

Run the 2D obstacle-navigation example with:

```bash
cd recap_pi
uv run python recap_demo_complex_2d.py
```

Run the 3D key-and-portal game example with:

```bash
cd recap_pi
uv run python recap_demo_game_3d.py
```

The scripts write artifacts into `outputs/`:

- `complex_comparison.png`
- `plain_rollouts.gif`
- `recap_rollouts.gif`
- `game_3d_comparison.png`
- `game_3d_plain.gif`
- `game_3d_recap.gif`

## Current Snapshot Metrics

Latest verified 2D run:

- Plain imitation: 190 average effective steps, 17% success, 83% collisions
- RECAP advantage-conditioned: 41 average effective steps, 100% success, 0% collisions
- Duration improvement: 4.67x faster
- Success improvement: 6.00x higher

Latest verified 3D run:

- Plain imitation: 221 average effective steps, 10% success, 40% key pickup, 90% collisions
- RECAP advantage-conditioned: 124 average effective steps, 62% success, 78% key pickup, 38% collisions
- Duration improvement: 1.79x faster
- Success improvement: 6.25x higher
