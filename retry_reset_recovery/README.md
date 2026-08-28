# Retry / Reset Recovery Toy

This FLARE-inspired track asks whether recovery should be split into two data and execution problems:

- **Retry:** recover from in-distribution robot-pose errors while the environment remains task-valid.
- **Reset:** invoke an object-centric skill after a state-breaking failure such as a drop, topple, or wedge.

The experiment is a deterministic 2-D pick/place mechanism probe, **not** a reproduction of FLARE or a VLA benchmark. It uses locally weighted behavior cloning over scripted demonstrations so the support of each training set and the consequences of gating remain inspectable.

## Primary reference

- Ganlong Zhao, Zijia Tang, Xingping Chen, Zhanghui Kuang, Ye Tian, and Guanbin Li. [FLARE: A Failure-Aware Framework for Autonomous Correction and Recovery in Visual-Language Robotic Manipulation](https://arxiv.org/abs/2608.26645), arXiv:2608.26645, accepted to CVPR 2026.

The source method augments success trajectories with perturbation-to-bridge continuations for retry robustness, mines object-centric reset skills for state-breaking failures, and uses an online MLLM monitor to switch between task execution and reset skills. This toy preserves that retry/reset separation but replaces the VLA, MLLM, vision, and real robot with low-dimensional policies and a configurable noisy state monitor.

## Compared methods

- `success_only_bc`: narrow, monotonic success demonstrations only.
- `generic_recovery_bc`: mixes task data with task-agnostic returns to a global home pose.
- `perturb_bridge_bc`: discards perturbation actions and trains only bridge-to-task continuations.
- `monolithic_resets`: mixes task, retry, and all reset actions into one policy without an execution gate.
- `monitor_gated_reset_skills`: perturb+bridge task policy plus object-centric reset policies selected by a noisy monitor.

The task places three objects in sequence. Paired scenarios include clean execution, robot-pose displacement, toppled/dropped/wedged objects, mixed retry/reset failures, and a three-failure stress case.

## Run

Full checked run:

```bash
cd /home/andypark/Projects/repos/vla-ideas
PYTHONWARNINGS=error \
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  retry_reset_recovery/run_retry_reset_recovery.py \
  --seed 29 --trials 120 --train-demos 48 --max-steps 230
```

Smoke test without overwriting the checked outputs:

```bash
PYTHONWARNINGS=error \
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  retry_reset_recovery/run_retry_reset_recovery.py \
  --smoke --seed 29 --output-dir /tmp/retry-reset-smoke
```

Render the TeX report:

```bash
./retry_reset_recovery/docs/render_pdf.sh
```

## Latest verified results

The full run evaluates 120 paired scenarios per method.

| Method | Success | Completion | ID recovery | OOD recovery | False resets / ep. | Compounding failures / ep. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Success-only BC | 15.0% | 30.6% | 25.0% | 0.0% | 0.000 | 0.217 |
| Generic recovery BC | 0.0% | 0.0% | 0.0% | n/a | 0.000 | **0.000** |
| Perturb + bridge BC | 23.3% | 47.5% | 91.7% | 0.0% | 0.000 | 0.275 |
| Monolithic resets | 55.0% | 64.7% | 90.2% | **67.1%** | 5.025 | 0.167 |
| Monitor-gated reset skills | **56.7%** | **68.9%** | **100.0%** | 53.0% | 0.575 | 0.100 |

At a `0.50` robot-pose perturbation, perturb+bridge succeeds on `93.8%` of paired episodes versus `12.5%` for success-only BC. This isolates the retry-data benefit: broader pose support improves recovery without teaching the random perturbation itself.

For state-breaking failures, the gated library attains the best overall success and completion while cutting false resets from `5.025` to `0.575` per episode relative to monolithic mixing. It does **not** dominate every failure type: monolithic reset training recovers toppled objects more reliably in this implementation, while the gated policies are much stronger on dropped objects and tie on wedged objects.

![Method summary](outputs/method_summary.png)

![Robustness](outputs/robustness_sweeps.png)

## Monitor and library ablations

The monitor is part of the result, not a free oracle. On failure-rich episodes, the default monitor reaches `47.8%` success; a weak detector falls to `26.7%`, and an over-triggering monitor falls to `18.9%` while producing `6.26` false resets per episode.

The reset-library ablation is intentionally reported even where it is awkward. Removing dropped or wedged recovery sharply reduces success. Removing the toppled skill increases aggregate success in this toy, revealing that the learned toppled reset is harmful often enough to outweigh its local benefit. A real system would need per-skill validation, confidence thresholds, and a safe fallback rather than assuming every collected reset skill should always be callable.

## Outputs

- `outputs/trial_metrics.csv`: per-method paired rollout metrics.
- `outputs/summary.csv`: all/split aggregates.
- `outputs/robustness.csv`: pose-magnitude and failure-type sweeps.
- `outputs/ablations.csv`: monitor-quality and missing-skill tests.
- `outputs/metrics.json`: configuration, data coverage, summaries, claims, and ablations.
- `outputs/sanity_checks.json`: deterministic mechanism checks.
- `outputs/data_coverage.png`: robot-pose support by task stage.
- `outputs/method_summary.png`: headline success/recovery/failure comparison.
- `outputs/robustness_sweeps.png`: retry magnitude and OOD failure results.
- `outputs/ablations.png`: monitor and reset-library ablations.
- `outputs/representative_rollout.png`: paired hard-case trajectories.
- `docs/retry_reset_recovery_report.tex` and `.pdf`: source-grounded report.

## Important limitations

- Low-dimensional k-nearest-neighbor BC replaces a visual-language-action model.
- Failure identity and object state are explicit simulator variables; the monitor is a configured stochastic classifier, not an MLLM.
- Scripted reset demonstrations, task geometry, and grasp/release rules are simplified.
- Generic recovery is deliberately task-agnostic and fails badly; this is a warning about data semantics, not a claim that all broad recovery datasets fail.
- The default monitor parameters are chosen to represent a strong but imperfect classifier. The ablations show the result is sensitive to that assumption.
- Success rankings depend on this toy's failure mix. Per-failure rows and the complete CSV should be preferred over the overall average.
