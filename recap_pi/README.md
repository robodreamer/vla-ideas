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

Run the 3D drone-heist game example with:

```bash
cd recap_pi
uv run python recap_demo_game_3d.py
```

Render the TeX concept note to PDF with the cached Docker image:

```bash
cd recap_pi/docs
./render_pdf.sh
```

The scripts write artifacts into `outputs/`:

- `complex_comparison.png`
- `plain_rollouts.gif`
- `recap_rollouts.gif`
- `game_3d_comparison.png`
- `game_3d_plain.gif`
- `game_3d_recap.gif`
- `game_3d_plain_3d.gif`
- `game_3d_recap_3d.gif`

The writeup lives in `docs/`:

- `recap_concept_writeup.tex`
- `recap_concept_writeup.pdf`

## Current Snapshot Metrics

Latest verified 2D run:

- Plain imitation: 190 average effective steps, 17% success, 83% collisions
- RECAP advantage-conditioned: 41 average effective steps, 100% success, 0% collisions
- Duration improvement: 4.67x faster
- Success improvement: 6.00x higher

Latest verified 3D run:

- Plain imitation: 222 average effective steps, 19% success, 55% artifact pickup, 24% uplink activation, 79% collisions
- RECAP advantage-conditioned: 111 average effective steps, 74% success, 100% artifact pickup, 74% uplink activation, 19% collisions
- Duration improvement: 1.99x faster
- Success improvement: 3.88x higher
