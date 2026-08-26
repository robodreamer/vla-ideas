# VLA Ideas

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-managed-6A5ACD?logo=uv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)
![Git LFS](https://img.shields.io/badge/assets-Git%20LFS-0A97B0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/robodreamer/vla-ideas?style=social)](https://github.com/robodreamer/vla-ideas/stargazers)

Toy research prototypes for exploring visual-language-action timing, policy conditioning, and execution under latency. The repository currently contains eleven compact experiment tracks plus a lightweight research-notes index:

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
└── context_chunk_tradeoff/
    ├── run_context_chunk_tradeoff.py
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
