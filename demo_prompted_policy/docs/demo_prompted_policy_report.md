# Demonstration-Prompted Policy: An S1-Inspired Mechanism Probe

## Question

When one demonstration is supplied at inference time, which representation is useful for execution after the scene, motor convention, prompt quality, and horizon change?

## Source boundary

Skild AI's August 2026 S1 post presents video demonstrations as in-context robot task specifications and discusses correspondence, progress tracking, robustness, recovery, and execution without task-specific post-training. The model is closed and its training details are deferred. This package does not reproduce S1.

## Toy mapping

- A video prompt becomes a 2D trajectory through an ordered set of objects.
- Scene/viewpoint mismatch becomes a similarity transform plus object jitter.
- Embodiment mismatch becomes a mirrored command map.
- Prompt errors become trajectory bumps, distractor approaches, and noisy dwell segments.
- Long-horizon composition becomes 3–12 ordered interactions.
- Recovery is measured after two paired state perturbations.

The five methods isolate task-label priors, literal path replay, local phase retrieval, soft full-trajectory alignment, and a semantic sustained-contact event sequence.

## Result

The full seed-23 sweep contains 432 scenarios per method. The hand-engineered latent-intent controller completes all scenarios without intervention, while trajectory and phase baselines remain brittle. This is a deliberately sharp sufficient-statistic result: the successful method is given object correspondence and an event parser closely matched to the simulator.

The main conclusion is not that a simple parser solves video imitation. It is that runtime demonstration prompting requires separating **what happened and what remains** from the exact demonstrator path and motor coordinates.

## Follow-ups

1. Remove object IDs and learn cross-scene correspondence from visual features.
2. Learn event boundaries and progress under ambiguous/repeated interactions.
3. Replace scripted controllers with a chunk policy trained across prompt/deployment pairs.
4. Evaluate recovery without oracle unsticks.
5. Introduce demonstrations whose errors change intent rather than only trajectory quality.
