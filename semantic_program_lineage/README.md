# Semantic Program Lineage

This deterministic toy isolates one idea from [SUN: Persistent Programs For Language-Grounded Control-to-Learning-to-Real Policies](https://arxiv.org/abs/2608.31167): define task semantics once, then compile stage guards, reward targets, demonstration labels, and success checks from the same typed specification.

The package is a mechanism probe, not a SUN/Kuafu reproduction. It has no language parser, MPC, RL training, visual policy, simulator, or robot interface.

## Question

How much can small hand-copied inconsistencies distort training, execution, optimization, and evaluation even when every interface appears individually reasonable?

A five-stage 2-D pick/carry/release task compares:

- `compiled_spec`: all interfaces share one `StageSpec` source;
- `copied_demo_labels`: stage boundaries advance 1–2 frames;
- `copied_stage_guards`: looser, partially misaligned transition conditions;
- `copied_rewards`: offset targets and grip preferences;
- `copied_success_check`: a looser terminal evaluator;
- `copied_all`: every copied inconsistency together.

A stagewise ridge policy is trained from scripted demonstrations. A bounded reward-controller blend is selected on validation returns, then held fixed on deterministic held-out geometries.

## Run

```bash
PYTHONWARNINGS=error \
python semantic_program_lineage/run_semantic_program_lineage.py --seed 13
```

Smoke test without replacing checked outputs:

```bash
PYTHONWARNINGS=error \
python semantic_program_lineage/run_semantic_program_lineage.py \
  --smoke --output-dir /tmp/semantic_program_lineage_smoke
```

Render the report:

```bash
./semantic_program_lineage/docs/render_pdf.sh
```

## Verified results

The checked run uses 48 training, 20 validation, and 36 test geometries.

| Lineage | True success | Stage mismatch | Interface drift exposed |
| --- | ---: | ---: | --- |
| Compiled spec | **100.0%** | 9.1% | none |
| Copied demo labels | 0.0% | 1.1% | 30.1% label error |
| Copied stage guards | 100.0% | 88.3% | 0.89-step mean boundary error |
| Copied rewards | 0.0% | 1.0% | 0.079 mean target drift |
| Copied success check | 100.0% | 9.1% | 33.3% false positives on three crafted near-miss states |
| Copied all | 0.0% | 66.8% | all four drift diagnostics |

Interpretation is deliberately narrow:

- shared semantics preserved successful execution in this toy;
- copied labels and rewards changed what the policy learned or what validation selected;
- loose guards can preserve terminal success while making internal stage traces misleading;
- nominal rollout agreement does not prove evaluator agreement—the copied success check accepts a crafted near miss rejected by the compiled predicate.

The 100%/0% gaps are properties of this transparent stress test, not estimates of real-system effect sizes.

![Headline metrics](outputs/headline_metrics.png)

![Interface drift](outputs/interface_drift.png)

![Reward selection](outputs/reward_selection.png)

![Representative trajectories](outputs/trajectory_examples.png)

## Outputs

- `outputs/summary_metrics.csv`: held-out aggregate results;
- `outputs/episode_metrics.csv`: per-rollout metrics;
- `outputs/interface_consistency.csv`: direct label/guard/reward/evaluator drift;
- `outputs/selection_metrics.csv`: validation return and success by reward blend;
- `outputs/metrics.json`: configuration, source mapping, sanity checks, and summaries;
- `outputs/sanity_check.json`: expert, label-shift, and evaluator-probe assertions;
- `outputs/*.png`: summary, interface, selection, and trajectory figures;
- `docs/semantic_program_lineage_report.tex` and `.pdf`: generated report.

## Mapping to SUN

SUN defines geometric/contact relations once and compiles aligned MPC costs, satisfaction predicates, RL rewards, transition guards, and diagnostics. This toy retains only the *single semantic lineage versus duplicated interfaces* distinction. Its explicit stage table stands in for a typed executable; scripted demonstrations and linear controllers stand in for the much richer control-to-learning pipeline.

## Limitations

- synthetic 2-D set-point dynamics and privileged state;
- hand-designed, fully visible inconsistencies;
- scripted demonstrations and linear stage policies;
- reward blending is validation-time controller selection, not RL;
- crafted evaluator probes are tiny and do not measure real false-positive rates;
- no perception, language grounding, contact physics, embodiment shift, or hardware uncertainty.
