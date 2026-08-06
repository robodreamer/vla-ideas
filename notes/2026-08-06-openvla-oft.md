# OpenVLA-OFT: optimized fine-tuning for speed and success

- **Captured:** 2026-08-06
- **Evidence label:** primary paper, project page, and official code repository
- **Tags:** `openvla`, `oft`, `parallel-decoding`, `action-chunking`, `continuous-actions`, `l1-regression`

## Source-grounded recipe

Kim, Finn, and Liang's OFT recipe changes OpenVLA's fine-tuning action interface: predict a future action chunk in parallel, represent the actions continuously, and optimize an L1 regression loss. The official repository describes an MLP action head that generates a continuous action chunk; its example config enables L1 regression and disables diffusion. The project page presents these alongside action chunking as the recipe.

## Reported results (not reproduced here)

The paper reports that OpenVLA-OFT increases OpenVLA's average LIBERO success from 76.5% to 97.1% over four suites and increases action-generation throughput by 26x. The project page summarizes 25--50x faster inference and 20%+ success-rate improvement, and reports 97.1% average LIBERO success. Those are authors' benchmark/hardware results, not outcomes from this repository.

## Why it matters here

[`openvla_oft_systems_toy/`](../openvla_oft_systems_toy/README.md) tests only a systems-level slice of the recipe. It compares serial/autoregressive-like action availability with parallel continuous chunks in a delayed-observation 2D dynamic task, then varies chunk refresh. This cleanly complements `async_chunking_compare`'s delay-compensation alternatives and `turbo_vla_direct_control`'s direct-policy/refresh-rate framing.

## Caveats

- This toy is not a reproduction: it has no OpenVLA backbone, images, language, LoRA, LIBERO/ALOHA, action-head training, or real robot.
- Its controller is analytical and uses simulated latency parameters; no local timing should be read as an OFT throughput claim.
- Continuous outputs and an L1-style target make the output interface concrete, but do not establish that L1 optimization or parallel decoding alone causes the paper's task results.
- The task is intentionally partial observability plus disturbances, so results explain latency/chunk/refresh trade-offs rather than VLA quality or generalization.

## Primary references

1. **Kim, Finn, Liang — “Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success”** (arXiv:2502.19645, 2025). Paper and reported LIBERO/throughput findings. <https://arxiv.org/abs/2502.19645>
2. **OpenVLA-OFT project page.** Recipe summary, reported results, and implementation FAQ. <https://openvla-oft.github.io/>
3. **moojink/openvla-oft.** Official code, configurations, continuous action-head example, and checkpoints. <https://github.com/moojink/openvla-oft>
