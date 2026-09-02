# Contact-Consequence Prediction Toy

This package builds a **deterministic, self-contained insertion benchmark** seeded by **Facet-0**. It does **not** reproduce Facet-0, train a VLA, or establish robot safety. It asks a narrower mechanism question:

> When the same visual state can hide opposite misalignment and jam outcomes, what changes when behavior cloning reads force, predicts the next wrench, uses that consequence in candidate valuation, and adapts a bounded local correction to shifted part dynamics?

## Primary sources and claim boundary

Primary references checked on September 2, 2026:

- [Facet-0: A Robotic Foundation Model for Contact-Rich Precise Manipulation](https://arxiv.org/abs/2609.01596), arXiv:2609.01596, September 1, 2026.
- [Official Facet-0 project page](https://pine-lab-ntu.github.io/facet-0/), PINE Lab, Nanyang Technological University.

The paper describes a model that conditions on a causal wrist-wrench history, jointly predicts action chunks and their future wrench profiles, values action–wrench proposals with a deployment-trained critic, and uses a bounded local actor for part-specific adaptation. The paper reports **82% mean success** over five sub-millimeter computer-assembly tasks versus **15%** for its strongest matched baseline, with controlled variants at **16%** after semantic–contact alignment, **38%** after value-guided refinement, and **82%** with bounded local adaptation. Those are **authors’ reported real-system results**, not measurements from this repository.

This package tests only an abstracted causal chain—observe contact, predict consequence, value candidates, adapt locally—in a synthetic 2-D simulator. It makes no claim about Facet-0’s dataset, model scale, flow matching, VLM semantics, real-robot latency, placement accuracy, or generalization. The near-perfect adapted result is a deterministic point estimate favored by a directly observable one-parameter stiffness structure; it is not evidence of general adaptation robustness.

## Toy setup

A part starts above a slot with hidden signed lateral error. The visual observation exposes insertion depth and a **quantized error magnitude but not its sign**, so `+x` and `-x` misalignment are exactly aliased. Once contact begins, a two-axis wrist wrench reveals the sign and severity of the interaction.

Each simulated part changes:

- lateral stiffness,
- jam amplification,
- insertion friction,
- motion/force transmission,
- effective clearance.

An expert generates nominal demonstrations. Ridge regressors fit behavior-cloning policies and a next-wrench model. Deployment-style perturbed transitions fit a scalar critic. Evaluation expands part-dynamics shift beyond the nominal training range.

A successful episode reaches full insertion before 58 control steps without crossing the configured hard wrench threshold. Metrics also retain jams, recovery, intervention-threshold crossings, peak force, action variation, final alignment, and wrench-prediction error.

## Compared methods

1. **Action-only BC** — ridge behavior cloning from sign-aliased visual features.
2. **Force-conditioned BC** — the same model family with current axial/lateral wrench inputs.
3. **Future-wrench auxiliary** — force-conditioned BC plus a learned next-wrench vector, lateral magnitude, and wrench norm; predicted contact intensity gates a shallow axial probe.
4. **Action–wrench critic reranking** — a deployment-trained critic reranks bounded axial candidates using each candidate’s predicted wrench consequence.
5. **Bounded local adaptation** — critic reranking plus a causal local stiffness estimate and a residual lateral correction clipped to `0.03`, below the global action limit of `0.075`.

The auxiliary wrench is never commanded. The simulator executes only Cartesian-like lateral/axial actions.

## Toy-to-real mapping

| Toy component | Facet-0 motivation | Major simplification |
| --- | --- | --- |
| Sign-aliased slot state | Visually similar progress can hide clean insertion versus jam | Six visual scalars, no images or language |
| Current two-axis wrench | Causal wrist-wrench history conditions contact behavior | One instantaneous 2-D wrench, not a 10-frame 6-D history |
| Next-wrench regressor | Joint action–future-wrench proposal | Separate ridge consequence model, not joint flow matching |
| Candidate critic | Action–Wrench Critic values contact consequences | One-step scalar ridge value, not a distributional return critic |
| Local stiffness estimate | Bounded part-specific adaptation on frozen representations | Hand-designed online estimator and clipped residual, not TD3+BC |
| External action limits | Executable action remains bounded before control | Kinematic clipping only; no compliant controller or certified force bound |

## Run

Verified full run from the repository root:

```bash
PYTHONWARNINGS=error \
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  contact_consequence_prediction/run_contact_consequence_prediction.py \
  --mode full
```

Quick run without replacing checked outputs:

```bash
PYTHONWARNINGS=error \
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  contact_consequence_prediction/run_contact_consequence_prediction.py \
  --mode quick --output-dir /tmp/contact_consequence_prediction_quick
```

Mechanism sanity mode:

```bash
PYTHONWARNINGS=error \
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  contact_consequence_prediction/run_contact_consequence_prediction.py \
  --mode sanity --output-dir /tmp/contact_consequence_prediction_sanity
```

Render the report:

```bash
contact_consequence_prediction/docs/render_pdf.sh
```

## Verified full results

The main split uses **320 paired episodes per method** at shift severity `1.35`.

| Method | Success | Jam rate | Recovery given jam | Mean steps | Intervention | Predicted next-wrench MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Action-only BC | 0.0% | 0.0%* | 0.0% | 58.0 | 16.6% | 0.352 |
| Force-conditioned BC | 62.2% | 45.3% | 89.7% | 39.1 | 3.4% | 0.145 |
| Future-wrench auxiliary | 62.8% | 35.9% | **100.0%** | 38.9 | 3.4% | 0.144 |
| Action–wrench critic reranking | 72.5% | 5.6% | 72.2% | 33.1 | 3.4% | 0.149 |
| Bounded local adaptation | **100.0%** | **2.8%** | **100.0%** | **23.3** | 3.4% | **0.121** |

\*The action-only policy usually stalls before satisfying the toy’s jam-event predicate; its 0% completion and 0.180 final alignment error show that this is not benign behavior. These are deterministic point estimates from one configured simulator run, not confidence intervals. Conditional recovery uses 145, 115, 18, and 9 observed jams for force-conditioned BC, the future-wrench auxiliary, critic reranking, and bounded adaptation respectively.

Bounded reading of the result:

- Force conditioning adds **62.2 percentage points** over action-only BC because lateral wrench sign breaks the exact visual alias.
- Future-wrench gating adds only **0.6 points** of completion, but lowers jam incidence by **9.4 points** and makes all observed jams recoverable. This supports consequence prediction as a useful control signal in this construction, not a general claim that an auxiliary loss must improve BC.
- Critic reranking adds **9.7 points** over the auxiliary policy and cuts jam incidence from 35.9% to 5.6%.
- Bounded adaptation adds **27.5 points** over critic reranking and halves mean episode length relative to force-conditioned BC.
- The four contact-aware methods have the same mean main-split peak wrench (`1.477`) because they share the initial contact probe. This run therefore **does not support a peak-force reduction claim** for bounded adaptation.

![Headline metrics](outputs/headline_metrics.png)

![Paired trajectories](outputs/paired_trajectories.png)

## Robustness to shifted part dynamics

Success at severity `0.70 / 1.00 / 1.35 / 1.70 / 2.05`:

| Method | 0.70 | 1.00 | 1.35 | 1.70 | 2.05 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Action-only BC | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Force-conditioned BC | 100.0% | 93.3% | 60.8% | 53.3% | 54.2% |
| Future-wrench auxiliary | 100.0% | 92.5% | 62.5% | 53.3% | 54.2% |
| Critic reranking | 100.0% | 96.7% | 70.8% | 55.8% | 54.2% |
| Bounded adaptation | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **99.2%** |

The frozen consequence model and critic lose their edge at the hardest shift, while the local estimator retains performance. That is evidence for this toy’s adaptation mechanism, but also a warning: the nearly perfect adapted curve follows from a favorable one-parameter stiffness structure and direct force observability.

![Robustness](outputs/robustness.png)

## Contact-role ablation

The ablation withholds anticipated wrench from candidate scoring or removes reranking entirely. It is intended to diagnose the toy critic, not correspond one-for-one with Facet-0’s reported ablations. See `outputs/contact_role_ablation.csv` for checked values.

![Wrench prediction](outputs/wrench_prediction.png)

## Outputs

- `outputs/trial_metrics.csv` — 1,600 paired main-split episode rows.
- `outputs/summary.csv` — aggregate main metrics.
- `outputs/robustness.csv` — five shift severities × five methods.
- `outputs/contact_role_ablation.csv` — action/wrench scoring ablation.
- `outputs/metrics.json` — configuration, method definitions, diagnostics, summaries, claims, and runtime.
- `outputs/sanity_checks.json` — exact-alias, force-sign, bounds, and determinism checks.
- `outputs/headline_metrics.png` — completion, recovery, and contact load.
- `outputs/paired_trajectories.png` — paired hard-shift depth and lateral-wrench traces.
- `outputs/robustness.png` — success and load versus shift severity.
- `outputs/wrench_prediction.png` — nominal next-wrench calibration plot.
- `docs/contact_consequence_prediction_report.tex` and `.pdf` — maintained report source and rendered artifact.

## Important limitations

- This is a synthetic 2-D insertion simulator with manually chosen dynamics and thresholds.
- The exact visual alias is constructed; real images may expose weak cues that this observation deletes.
- Ridge BC, ridge consequence prediction, and a one-step ridge critic are much smaller and less expressive than a VLA and distributional critic.
- The auxiliary head predicts two wrench components plus sign-invariant magnitude targets; it is not a joint action–wrench flow-matching model.
- Local adaptation estimates one effective stiffness online. It does not learn a neural actor, use interventions, or protect a semantic backbone from forgetting.
- Bounds limit commanded displacement, not physical energy or force. No passivity, collision, or certified safety guarantee is implemented.
- The critic is trained on synthetic one-step utility and can lose its advantage under severe model shift.
- Completion, jam, recovery, and intervention thresholds are simulator definitions, not manufacturing acceptance criteria.
- No statistical significance test is reported; episodes are paired and deterministic for mechanism comparison.
