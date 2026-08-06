# OpenVLA-OFT Systems Toy

This folder explores one systems implication of **OpenVLA-OFT: Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success** with a compact deterministic 2D experiment.

The source recipe combines parallel decoding, action chunking, continuous actions, and L1 regression. This toy asks a narrower control question: when observations are delayed, how do serial action availability, parallel chunks, and chunk-refresh cadence affect a tracking controller? It is not an OpenVLA-OFT reproduction; the controller is analytical, latency is configured, and no learned OpenVLA action head is used.

## What is implemented

`run_openvla_oft_systems_toy.py` is intentionally narrow. It does **not** implement OpenVLA weights, image or language inputs, LoRA, L1 training, LIBERO, ALOHA, robot dynamics, or hardware timing. It isolates the output-interface/timing distinction:

- `autoregressive_serial`: one continuous action becomes available after a configured 55 ms serial-like decision delay; the previous command is held while the synthetic decoder is busy.
- `parallel_open_loop`: an 8-step continuous action chunk becomes available after a configured 60 ms decision delay and is consumed open loop.
- `parallel_refresh_4`: the same chunk generator is refreshed after four executed actions.
- `parallel_refresh_2`: the same chunk generator is refreshed after two executed actions.

The plant tracks a moving 2D target with 160 ms delayed observations and seeded velocity gusts. Early refresh discards an unused chunk suffix, so action-output throughput is reported as generated actions rather than useful plant-rate actions.

## References checked

Primary OpenVLA-OFT sources:

- Research paper: <https://arxiv.org/abs/2502.19645>
- Project page: <https://openvla-oft.github.io/>
- Official repository: <https://github.com/moojink/openvla-oft>

The source-grounded recipe and the boundary between reported paper results and this local toy are documented in [`notes/2026-08-06-openvla-oft.md`](../notes/2026-08-06-openvla-oft.md).

## Run

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python openvla_oft_systems_toy/run_openvla_oft_systems_toy.py --seed 17 --trials 96
```

Generated files live in `openvla_oft_systems_toy/outputs/`:

- `openvla_oft_systems_metrics.csv`
- `openvla_oft_systems_summary.png`
- `openvla_oft_systems_report.md`

## Latest generated headline

The deterministic 96-trial run has its highest terminal-window success with `parallel_refresh_4` (33.3%); `parallel_refresh_2` has the lowest mean tracking error (0.311) but returns to 28.1% success. The toy's point is the trade-off, not an OpenVLA-OFT performance claim: early refresh reduces stale-chunk exposure while increasing decisions from a four-tick-old observation and introducing more command boundaries.

See the generated report for the full table and interpretation.

## Reports

- [generated experiment report](outputs/openvla_oft_systems_report.md)
- [LaTeX report source](docs/openvla_oft_systems_toy_report.tex)
- [generated PDF](docs/openvla_oft_systems_toy_report.pdf)

`docs/render_pdf.sh` remains a wrapper for the repository LaTeX renderer when it is available.

## Mapping to the source recipe

| Toy component | OpenVLA-OFT relation | Simplification |
| --- | --- | --- |
| Parallel 8-step velocity chunk | Parallel continuous action chunk | Analytical controller, not a learned MLP action head. |
| Continuous clipped velocity commands | Continuous action representation | No robot action normalization or dataset actions. |
| Serial-like one-action timing model | Autoregressive output contrast | Configured delay, not measured VLA decoding. |
| Refresh after 8/4/2 actions | Receding-horizon chunk execution | No inference queue, model latency jitter, or deadline policy. |
| L1-target proxy | OFT's L1 regression objective | No optimization or training occurs. |

## Takeaway

Parallel chunks in this configured toy make more actions available than the serial-like condition, but they do not make a fixed open-loop suffix robust to stale feedback. Refreshing midway improves terminal-window success here; refreshing more often slightly improves mean error but worsens the terminal metric and jerk proxy. Those local observations motivate a refresh/latency sweep, not a claim about OFT's reported benchmark or hardware results.
