# Causal Interaction Memory Toy

This package builds a **deterministic, self-contained memory benchmark** seeded by **ZeVA**. It does **not** reproduce the paper or claim that the toy recovers identified causal models. Instead, it asks a narrower question:

> If related tasks share hidden mass, friction, and joint-stiffness factors, does a phase-aware dual-timescale interaction memory help retries and transfer more than raw history, flat transition retrieval, or lightweight online fine-tuning?

## References checked

- [Zeva: In-Context Causal Learning for Self-Evolving Vision-Language-Action Models](https://arxiv.org/abs/2608.30880), arXiv:2608.30880, 2026.
- [ZeVA project page](https://air-embodied-brain.github.io/Zeva/)

The source motivation is long-horizon interaction memory for VLA systems. This toy keeps only the memory question and replaces vision, robot control, and world modeling with explicit low-dimensional hidden dynamics.

## Toy setup

Each trial samples a family of related tasks that share hidden physical properties:

- **mass** → mostly affects the `lift` phase,
- **friction** → mostly affects the `slide` phase,
- **joint stiffness** → mostly affects the `turn` phase.

A target task is attempted up to **4 times**. Each attempt predicts three phase-specific force corrections. If a phase misses its tolerance band, the attempt stops, a biased but informative transition estimate is stored, and the method retries from the start.

Related support tasks from the same family provide prior interaction memory. Hidden properties are never given directly to the compared methods.

## Compared methods

- `raw_history`: retrieve whole support-task traces with weak phase structure.
- `transition_retrieval`: phase-specific kNN retrieval over related-task transitions.
- `phase_dual_memory`: slow family consolidation + fast retry cache + cross-phase retrieval.
- `online_finetune`: ridge-style parametric update from related tasks and retry observations.

## Run

Full verified run:

```bash
PYTHONWARNINGS=error \
python causal_interaction_memory/run_causal_interaction_memory.py \
  --seed 29 --trials 240 --support-tasks 4 --max-attempts 4
```

Smoke test without replacing checked outputs:

```bash
PYTHONWARNINGS=error \
python causal_interaction_memory/run_causal_interaction_memory.py \
  --smoke --seed 29 --output-dir /tmp/causal_interaction_memory_smoke
```

Render the report:

```bash
./causal_interaction_memory/docs/render_pdf.sh
```

## Latest verified results

The main run evaluates **240 paired target tasks** with **4 related support tasks** per family.

| Method | First-attempt transfer | Retry success | Retry gain | Mean attempts | Robust split success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw history | 30.8% | 53.3% | 22.5 pts | 3.17 | 38.3% |
| Transition retrieval | 44.6% | 83.3% | 38.8 pts | 2.40 | 63.3% |
| Phase-aware dual memory | **48.3%** | **95.4%** | **47.1 pts** | **1.85** | **85.0%** |
| Online fine-tune | 39.6% | 69.6% | 30.0 pts | 2.73 | 43.3% |

`phase_dual_memory` solves `79.2%` of tasks by the **second** attempt and `92.1%` by the **third**, versus `57.5%` and `74.6%` for flat transition retrieval.

At robustness severity `1.50`, final success is:

- raw history: `27.5%`
- transition retrieval: `57.5%`
- phase-aware dual memory: **`80.6%`**
- online fine-tune: `36.3%`

![Headline metrics](outputs/headline_metrics.png)

![Retry curves](outputs/retry_curves.png)

## Sample efficiency and transfer

Across support-memory sweep sizes `{0, 2, 4, 8, 12}`:

- With **2** related tasks, dual memory already reaches **98.8%** final success.
- With **0** related tasks, online fine-tuning is the strongest fallback (`53.8%`) because it can still exploit retry updates, while raw history nearly collapses (`5.6%`).
- With **12** related tasks, online fine-tuning matches the best first-attempt transfer (`54.4%`) but still trails dual memory on final retry success (`75.0%` vs. `98.1%`).

![Sample efficiency](outputs/sample_efficiency.png)

## Memory ablations

Dual-memory ablations on 180 held-out trials:

| Setting | Success | Transfer | Solved by attempt 2 |
| --- | ---: | ---: | ---: |
| Full dual memory | **96.7%** | **41.7%** | **76.1%** |
| No fast retry memory | 41.7% | 41.7% | 41.7% |
| No slow consolidation | 34.4% | 0.6% | 8.3% |
| No cross-phase retrieval | 96.1% | 41.7% | 76.1% |
| No phase-aware prior | 88.3% | 15.6% | 51.1% |

The strongest losses come from removing the **fast retry cache** or **slow consolidation**. Removing phase-aware retrieval is a smaller but still meaningful degradation (`96.7%` to `88.3%`), while cross-phase covariance mainly trims the tail (`96.7%` to `96.1%`).

![Ablations](outputs/ablations.png)

## Outputs

- `outputs/trial_metrics.csv`: per-trial method metrics.
- `outputs/summary.csv`: aggregate rows by method and split.
- `outputs/retry_curves.csv`: cumulative solve rate by retry budget.
- `outputs/sample_efficiency.csv`: support-memory sweep.
- `outputs/robustness.csv`: hidden-shift severity sweep.
- `outputs/ablations.csv`: dual-memory ablation table.
- `outputs/metrics.json`: config, summaries, claims, and runtime.
- `outputs/sanity_checks.json`: deterministic mechanism checks.
- `outputs/*.png`: generated figures.
- `docs/causal_interaction_memory_report.tex` and `.pdf`: maintained report source and rendered PDF.

## Sanity checks

The checked run passes deterministic mechanism checks for:

- repeatability with the same seed,
- support memory improving dual-memory success,
- transition retrieval improving the seen failed phase,
- dual memory improving cross-phase prediction after an observed failure,
- dual memory outperforming raw history on overall retry success,
- dual memory retaining the best hard-split robustness.

## Important limitations

- This is a **mechanism toy**, not a VLA, world model, or robot benchmark.
- Hidden mass/friction/joint factors are simulator latents; memory quality is measured by control outcomes, not by causal identification.
- Retry observations are low-dimensional force-correction summaries, not images, language, or dense trajectories.
- Online fine-tuning is a small ridge-style proxy, not gradient-based adaptation of a pretrained VLA.
- The dual-memory advantage depends on shared family structure across related tasks; changing that structure could alter the ranking.
