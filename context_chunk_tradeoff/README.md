# Context versus Chunking Toy

This self-contained experiment asks a deliberately narrow question:

> When noisy observations conceal a persistent temporal mode, when is more observation history valuable, and when do short receding-horizon chunks beat longer open-loop execution after surprises?

It generates synthetic closed-loop oracle demonstrations for a one-dimensional tracker. The target has a hidden signed-velocity mode; only noisy target-position observations are exposed to the evaluated controller. The sweep deliberately does not train a learner from the demonstration CSV: a transparent least-squares estimator isolates how `C` recent observations affect target-velocity inference before the controller emits an `H`-step action chunk. Unannounced target-mode switches and robot velocity impulses are injected during execution.

This is an explanatory, falsifiable control toy. It is not a reproduction of a paper, a trained VLA, or evidence that the same values transfer to image histories, language intent, contact-rich manipulation, model inference latency, or real robots.

## Run

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  context_chunk_tradeoff/run_context_chunk_tradeoff.py --trials 120 --seed 17
```

Fast deterministic smoke run:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  context_chunk_tradeoff/run_context_chunk_tradeoff.py --smoke --seed 17
```

Render the generated LaTeX report through the shared helper:

```bash
context_chunk_tradeoff/docs/render_pdf.sh
```

## Design and comparisons

The full paired Monte Carlo sweep crosses context `C ∈ {1, 2, 4, 8, 16}` and chunk horizon `H ∈ {1, 2, 4, 8, 16}`. Every cell receives the same seeded mode switches, target noise, disturbances, and observation noise.

- `C1/H1`: short-history reactive baseline.
- `C16/H1`: long-history reactive baseline.
- `C16/H16`: long-context, long open-loop baseline.
- The other cells form the 2D trade-off map rather than assuming a global winner.

The deterministic sanity check establishes the estimator and metric mechanics: C16 improves persistent-mode inference over C1, events are applied before response measurement, threshold search begins after a demonstrated peak, and the next event censors the active recovery window. The short fixed surprise schedule is descriptive only; the paired C16/H1 versus C16/H16 comparison supplies the supported chunk-reactivity evidence.

## Metrics

- **Success/reliability:** fraction of episodes whose final tracking window stays within tolerance.
- **Tracking/task error:** mean absolute target–robot separation.
- **Restricted event-to-recovery delay:** an event must produce a measurable error rise; threshold search begins after that peak, but delay is measured from the event and censored at the next event or episode end. Coverage and unrecovered counts are reported alongside it.
- **Smoothness:** squared action-difference jerk proxy.
- **Compute proxy:** planning calls per episode, not measured wall-clock latency.
- **Mode accuracy:** sign agreement between inferred and hidden velocity mode.

## Latest verified full result

The generated artifacts used `--trials 120 --seed 17` (3,000 paired episodes):

| Controller | Success | Tracking error | Persistent mode accuracy | Recovery | Coverage | Unrecovered | Jerk | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 / H1 short-history reactive | 44.2% | 0.588 | 57.2% | 18.14 | 45.5% | 188/307 | 795.7 | 180 |
| C16 / H1 long-history reactive | **50.0%** | **0.479** | **100.0%** | **14.01** | 69.3% | 237/467 | **21.3** | 180 |
| C16 / H16 long open loop | 30.0% | 1.028 | **100.0%** | 15.48 | 67.8% | 255/457 | 32.1 | **12** |

The deterministic mechanism checks passed: persistent-mode accuracy was `57.8%` for C1 and `100.0%` for C16, and hand-computed traces verify impact gating, post-peak threshold search, and next-event censoring. The small fixed surprise schedule produced `13.64` recovery steps for H1 and `13.79` for H16, but is not a pass criterion. With context held at C16, the mean per-seed H16-minus-H1 recovery gap was `2.32` steps with a deterministic bootstrap 95% interval of `[1.29, 3.37]`.

![Context/chunk sweep](outputs/tradeoff_heatmaps.png)

## Outputs

- `outputs/sanity_check.json`: fixed-seed mechanism checks.
- `outputs/expert_demonstrations.csv`: 12 generated closed-loop oracle demonstrations with hidden state retained for analysis.
- `outputs/sweep_trials.csv`, `sweep_summary.csv`, and `metrics.json`: deterministic per-trial and aggregate results.
- `outputs/tradeoff_heatmaps.png`, `pareto_tradeoff.png`, and `paired_rollout.png`: visual comparisons.
- `docs/context_chunk_tradeoff_report.md` and `.tex`: generated reports.
- `docs/context_chunk_tradeoff_report.pdf`: rendered report when a local LaTeX engine or permitted Docker daemon is available.

## Interpretation and limits

The observed evidence is restricted to the controlled mechanism. It supports the inference that history can disambiguate a persistent hidden mode, and that committing many actions can leave a controller stale between observations. Longer chunks reduce planning calls and reduce the jerk proxy on average across context settings, but smoothness is not monotonic at fixed C16. The faster short-chunk recovery result is specific to the paired C16 comparison and does not imply a universal preference for large context or short chunks, nor a performance claim for a real VLA system. Next tests should separate sensing and planning latency, use non-hand-designed visual observations, measure actual inference cost, and test physical contact disturbances.
