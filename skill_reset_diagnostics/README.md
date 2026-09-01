# Skill Reset Diagnostics

This deterministic toy is inspired by [A Fine-Grained Benchmark for Evaluating Vision-Language-Action Policies in Long-Horizon Tasks](https://arxiv.org/abs/2608.30536) and the official [Behavior-Skill repository](https://github.com/nubot-nudt/Behavior-Skill).

It asks a narrow diagnostic question: how different does a long-horizon policy look when every constituent skill is evaluated from a valid restored precondition instead of only when earlier skills happen to reach it?

This is not a Behavior-Skill reproduction. It uses an explicit reliability model rather than BEHAVIOR assets, BDDL predicates, learned VLAs, or simulator state snapshots.

## Toy benchmark

Five household chains are split into semantic skills such as:

- `MoveTo`
- `PickUpFrom`
- `OpenDoor` / `CloseDoor`
- `PlaceIn` / `PlaceOn`
- `Pour`
- `SweepSurface`
- `Attach`
- `Release`

Each skill instance has a natural-language instruction, named preconditions, local success conditions, a saved latent reset signature, and a horizon equal to twice its nominal demonstration length.

The experiment compares:

- ordinary full rollouts that stop at the first failed skill;
- exact independent resets;
- pose-jitter resets;
- realistic pose/grasp/joint/tool/precision mismatch;
- off-nominal resets;
- equal-cost nominal-demo and recovery-data allocation sweeps.

## Run

```bash
PYTHONWARNINGS=error \
python skill_reset_diagnostics/run_skill_reset_diagnostics.py \
  --seed 41 --trajectories-per-task 18
```

Smoke test without replacing checked outputs:

```bash
PYTHONWARNINGS=error \
python skill_reset_diagnostics/run_skill_reset_diagnostics.py \
  --smoke --seed 41 --output-dir /tmp/skill_reset_diagnostics_smoke
```

Render the report:

```bash
./skill_reset_diagnostics/docs/render_pdf.sh
```

## Verified results

The checked run contains 90 full trajectories across five task families.

| Evaluation | Completion metric | Valid reset fraction |
| --- | ---: | ---: |
| Full rollout | 0.0% task success; 25.9% successful-skill fraction | n/a |
| Exact reset | **56.9% TSCR** | 100.0% |
| Pose jitter | 47.0% TSCR | 100.0% |
| Realistic reset | 44.2% TSCR | 86.4% |
| Off-nominal reset | 41.1% TSCR | 49.3% |

The gap is reachability, not a claim that reset evaluation replaces full rollouts. `PlaceIn`, `Pour`, and `Attach` are attempted in only `0.0%`, `11.1%`, and `5.6%` of full-rollout occurrences, respectively, so aggregate task failure alone cannot distinguish their local capability from earlier blockers.

![Headline metrics](outputs/headline_metrics.png)

![Reachability bias](outputs/reachability_bias.png)

![Skill-type metrics](outputs/stsr_heatmap.png)

![Reset ablations](outputs/reset_ablations.png)

## Allocation ranking

One synthetic data pack is allocated to each skill type in turn. Value is measured as lift in the fraction of successfully completed skills across full chains.

- Best nominal-demo target: `PickUpFrom`, `+13.4` percentage points.
- Best recovery-data target: `PickUpFrom`, `+5.0` percentage points.
- Tie-aware demo ranking correlation with measured value: full-rollout heuristic `0.75`, reset-only heuristic `-0.30`.
- Tie-aware recovery ranking correlation: full-rollout heuristic `0.75`, reset-only heuristic `0.59`.

This negative result is useful: reset suites reveal unreached bottlenecks, but allocation value still depends on where a skill sits in the chain. The strongest prioritization combines local reset diagnostics with reachability/propagation information rather than treating either view as sufficient.

![Allocation value](outputs/allocation_value.png)

## Outputs

- `outputs/full_rollouts.csv` and `full_skill_rollouts.csv`;
- `outputs/skill_eval.csv` and `trajectory_reset_summary.csv`;
- `outputs/full_summary.csv`, `reset_summary.csv`, `stsr_summary.csv`, and `family_summary.csv`;
- `outputs/allocation_ranking.csv` and `skill_catalog.csv`;
- `outputs/metrics.json` and `sanity_checks.json`;
- five generated plots;
- `docs/skill_reset_diagnostics_report.tex` and `.pdf`.

## Sanity checks

The run verifies:

- deterministic repeated evaluation;
- exact resets satisfy every precondition;
- TSCR and reset validity decrease monotonically as reset mismatch grows;
- exact-reset TSCR exceeds full-task success;
- navigation is easier than contact-rich, articulation, and tool-use families;
- both allocation sweeps produce a nonzero best target;
- all ranking metrics are finite and stable across independent processes.

## Limitations

- deterministic latent reliability model, not images or robot dynamics;
- abstract reset mismatch rather than simulator snapshots;
- no VLA training or action generation;
- equal-cost synthetic data packs instead of retraining on demonstrations;
- full-task success is intentionally harsh and remains zero in this configuration;
- reset results expose local capability but do not model whether a real robot can restore the same preconditions safely or precisely.
