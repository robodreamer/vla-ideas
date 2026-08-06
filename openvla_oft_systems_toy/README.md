# OpenVLA-OFT Systems Toy

This is a compact, deterministic **OFT-inspired systems toy**, not an OpenVLA-OFT reproduction. It makes the recipe's output-side implications tangible in a delayed-observation 2D tracking task: a serial/autoregressive-like continuous-action generator is compared with a parallel continuous chunk regressor, both open-loop and with closed-loop chunk refresh.

The source recipe combines parallel decoding, action chunking, continuous actions, and an L1 regression objective. Here the controller emits clipped continuous velocity commands; its analytical target is a small L1-regression proxy rather than a learned OpenVLA action head. The simulated latency values are explicit experiment parameters, not hardware measurements.

## What it measures

- simulated per-decision latency and action-output throughput (including discarded predicted suffixes on early refresh);
- mean and final tracking error plus a terminal success threshold;
- a finite-difference jerk proxy for command smoothness;
- the open-loop chunk versus refresh-cadence trade-off with 160 ms delayed observations and small disturbances.

`autoregressive_serial` serializes one action every 55 ms. The three parallel methods decode an 8-step continuous chunk in 60 ms; they execute the chunk open-loop, or replan after 4 or 2 steps. All trial seeds are deterministic.

## Run

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python openvla_oft_systems_toy/run_openvla_oft_systems_toy.py --seed 17 --trials 48
```

Outputs (ignored by Git) are written to `openvla_oft_systems_toy/outputs/`:

- `openvla_oft_systems_metrics.csv`
- `openvla_oft_systems_summary.png`
- `openvla_oft_systems_report.md`

## Reports

The 96-trial deterministic run is documented in three linked forms:

- [generated Markdown result](outputs/openvla_oft_systems_report.md)
- [source-grounded LaTeX report](docs/openvla_oft_systems_toy_report.tex)
- [rendered PDF](docs/openvla_oft_systems_toy_report.pdf)

Render the PDF with the shared repository renderer:

```bash
./openvla_oft_systems_toy/docs/render_pdf.sh
```

## Scope boundary

The toy does not use OpenVLA weights, images, language grounding, LoRA, LIBERO, ALOHA, real inference timing, or an actual learned L1 action head. It tests only the systems-level hypothesis that parallel action chunks can improve action availability, but their benefit depends on chunk horizon and closed-loop refresh under stale feedback. The LaTeX report labels primary-source claims, local toy design, and locally generated results separately. See [`notes/2026-08-06-openvla-oft.md`](../notes/2026-08-06-openvla-oft.md) for primary sources and reported paper results.
