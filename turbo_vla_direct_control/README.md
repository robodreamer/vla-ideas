# TurboVLA Direct Control Toy

This folder explores the practical control implication behind **TurboVLA: Efficient Vision-Language-Action Model**.

Primary reference:

- TurboVLA paper page: https://arxiv.org/abs/2607.27205
- Paper PDF used for notes: https://arxiv.org/pdf/2607.27205

## Core idea

TurboVLA argues that execution-level robot control does not need to route every observation and instruction through a large LLM core. Instead, a compact model can encode vision and language separately, fuse them with lightweight bidirectional cross-attention, and decode a continuous action chunk directly. The paper reports a 0.2B-parameter model, about 31 ms latency, around 32 Hz action updates, and under 1 GB inference VRAM on an RTX 4090.

This toy asks whether that matters for control, not just for a model-size table.

## What the toy does

`run_turbo_vla_toy.py` trains two behavior-cloned chunk policies on a synthetic language-conditioned reach/drag task:

- `direct_32hz`: compact direct V+L→A policy with bidirectional cross-attention and an ACT-style chunk decoder.
- `llm_bottleneck_11hz`: heavier transformer-core proxy for an LLM-centric V→L→A execution path.
- `direct_throttled_11hz`: same direct policy, but forced to refresh chunks only every three 32 Hz ticks. This isolates the latency/control-rate mechanism.
- `direct_shuffled_language`: same direct policy with the wrong instruction token, to verify that the instruction is actually used.

The rendered observation contains the cursor/object, a central keep-out obstacle, and four fixed possible goal markers. The goal markers are separate input channels rather than natural images, so this is an instruction-use sanity check rather than a full visual-grounding benchmark. There is no selected-goal halo; the instruction token selects which marker matters. The expert target is a future velocity chunk that bends around the obstacle. Evaluation uses paired small disturbances plus occasional gusts so stale chunks have visible consequences.

## Relation to other repo ideas

- `async_chunking_compare`: same stale-chunk / refresh-delay question, now tied to a TurboVLA-style direct V+L→A architecture.
- `path_consistent_safety_filtering`: obstacle entry is treated as a path/safety failure rather than only final-goal error.
- `bspline_action_parameterization`: keeps attention on continuous action chunks and jerk/smoothness.
- `prefix_rl_chunking`: complements chunk-stability work by stressing receding-horizon refresh behavior.

## Run

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python turbo_vla_direct_control/run_turbo_vla_toy.py --train-steps 480 --eval-episodes 200
```

Outputs are written to `turbo_vla_direct_control/outputs/`, including:

- `turbo_vla_direct_control/outputs/turbo_vla_latency_metrics.png`
- `turbo_vla_direct_control/outputs/turbo_vla_paired_deltas.png`
- `turbo_vla_direct_control/outputs/turbo_vla_example_rollouts.png`

Reports are generated at:

- `turbo_vla_direct_control/docs/turbo_vla_toy_report.md`
- `turbo_vla_direct_control/docs/turbo_vla_direct_control_report.tex`
- `turbo_vla_direct_control/docs/turbo_vla_direct_control_report.pdf`

## Latest generated headline

Best policy in the latest run: `direct_32hz` with 93.0% success, mean final distance 0.037, and refresh period 1 control step(s).

See the generated report for the full table and interpretation.

## Simplifications

- Tiny learned encoders replace DINOv3/BERT.
- The “LLM bottleneck” baseline is a larger transformer-core proxy, not an actual LLM.
- The task is synthetic 2D control, not LIBERO or a real robot.
- Absolute hardware latency claims are from the paper; this script only measures local toy forward time and uses refresh-rate ablations to expose the control consequence.
- Goal locations are fixed and channelized; the shuffled-language ablation checks instruction use, but randomized visual-language grounding is left as future work.
