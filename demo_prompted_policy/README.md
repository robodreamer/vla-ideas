# Demonstration-Prompted Policy Toy

This S1-inspired experiment asks a narrow question: **what information must a controller recover when one video demonstration is the runtime task prompt?**

A synthetic demonstration specifies an ordered sequence of object interactions. The deployment scene then changes object geometry, action convention, prompt quality, and task length. Five hand-built controllers compare progressively richer ways of using the demonstration:

- `language_prior`: follows a fixed task-label prior without using the demonstrated sequence.
- `nearest_demo_replay`: maps and replays the demonstration path, but assumes the original motor convention.
- `phase_retrieval`: retrieves a demonstration phase from object-relative distance features.
- `full_demo_attention`: softly aligns the current scene to the whole demonstration trajectory.
- `latent_intent`: extracts sustained object-contact events and executes that semantic sequence in the deployment scene.

This is an explanatory mechanism probe, **not** an implementation or reproduction of Skild S1. The strongest method receives deliberately favorable hand-engineered object correspondences and contact-event structure; its result should be read as evidence for the value of intent/progress abstraction, not as a learned-policy benchmark.

## Primary reference

- Skild AI, [Introducing S1: In-Context Learning for Robotics](https://skild.ai/blogs/s1), August 2026.

The post frames a video demonstration as an in-context task specification and emphasizes scene/viewpoint/embodiment correspondence, long-horizon progress tracking, robustness, and recovery without task-specific weight updates. The underlying model and training details are not public in this release.

## Run

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  demo_prompted_policy/run_demo_prompted_policy.py \
  --seed 23 --trials-per-cell 6
```

The full run evaluates 432 paired scenarios per method across:

- scene/object transform levels `0, 1, 2`;
- normal and mirrored action mappings;
- demonstration corruption `0.00, 0.18, 0.35`;
- horizons of `3, 6, 9, 12` object interactions;
- six seeded trials per factor cell.

Render the report:

```bash
demo_prompted_policy/docs/render_pdf.sh
```

## Latest verified result

Command: `--seed 23 --trials-per-cell 6` (432 scenarios and 2,160 rollouts).

| Method | Autonomous success | Mean progress | Recovery | Interventions | Robust success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Language prior | 0.0% | 27.3% | 32.2% | 2.00 | 0.0% |
| Nearest replay | 2.5% | 10.5% | 43.9% | 1.95 | 0.9% |
| Phase retrieval | 4.2% | 19.7% | 38.6% | 1.91 | 3.3% |
| Full-demo attention | 0.2% | 4.4% | 8.0% | 2.00 | 0.3% |
| Latent intent | **100.0%** | **100.0%** | **100.0%** | **0.00** | **100.0%** |

The deterministic sanity check confirms that the latent parser recovers a six-step sequence despite a shifted scene, mirrored commands, and a corrupted prompt; embodiment-aware command translation has zero error while unaware replay has `0.16` error in the check.

The separation is intentionally sharp. Path-level alignment is brittle under repeated objects, prompt mistakes, perturbations, and long horizons, while the event-level method is built around the task's true semantic sufficient statistic. A useful learned follow-up would remove object IDs and learn correspondence, event segmentation, progress, and recovery from pixels.

![Method summary](outputs/method_overview.png)

## Metrics

- **Autonomous success:** all prompted interactions completed with no oracle intervention.
- **Progress:** fraction of the demonstrated interaction sequence completed.
- **Recovery rate:** fraction of injected disturbances followed by timely progress.
- **Tracking RMSE:** distance to the currently intended object.
- **Interventions:** oracle unsticks after prolonged no-progress periods.
- **Robust success:** autonomous success on the hardest transform, mirror, or corruption cases.
- **Smoothness / command jerk:** first- and second-difference action magnitudes.

## Outputs

- `outputs/demo_prompted_policy_trials.csv`: per-method paired rollout metrics.
- `outputs/demo_prompted_policy_summary.csv`: aggregate metrics.
- `outputs/demo_prompted_policy_metrics.json`: configuration, factor sweeps, sanity checks, and summaries.
- `outputs/sanity_check.json`: exact intent-extraction and mirrored-action checks.
- `outputs/method_overview.png`: headline comparison.
- `outputs/robustness_sweeps.png`: transform, embodiment, corruption, and horizon sweeps.
- `outputs/long_horizon_metrics.png`: progress, interventions, and tracking versus horizon.
- `outputs/representative_rollout.png`: a hard paired scenario.
- `docs/demo_prompted_policy_report.tex` and `.pdf`: source-grounded report.

## Limits

- States are 2D coordinates, not images or video embeddings.
- Object identity/correspondence is given to the policies.
- The latent-intent method uses a hand-designed sustained-contact parser that matches the simulator's event structure.
- There is no meta-learning, pretrained foundation model, language model, action chunk learner, or real robot.
- Oracle interventions are a diagnostic accounting mechanism, not autonomous recovery.
- The local numbers do not validate S1's company-reported results.
