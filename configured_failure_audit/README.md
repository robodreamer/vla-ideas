# Configured-Failure Audit for Toy Sequence Policies

> **DEFENSIVE EVALUATION ONLY.** This package uses one hard-coded synthetic token, generated low-dimensional set-point sequences, and six fixed residual templates. It cannot load or modify external datasets, checkpoints, images, VLA models, simulators, or robots. It contains no trigger search, trigger optimization, poisoning automation, reusable attack interface, or real-system deployment path.

This package asks a narrow auditing question inspired by TrapVLA: can a tiny conditional failure remain invisible to ordinary clean behavioral-cloning metrics, and which simple data, training, and execution checks expose or suppress it?

The result is a deterministic mechanism toy, **not** a reproduction of TrapVLA, its experiments, or a real backdoor. The fixed token is literally `[SYNTHETIC_AUDIT_TRIGGER]`; all trajectories and residuals are generated inside this folder.

## Primary references

- Jun-Hui Liu, Kun-Yu Lin, Yi-Lin Wei, Xu-Han Chen, Yinghao Li, Zhuohao Li, Yuan-Ming Li, Qing Zhang, Xiaoyi Fan, Dongmei Jiang, Yan Li, and Wei-Shi Zheng. [TrapVLA: Trapping Vision-Language-Action Models in Configured Failure Modes](https://arxiv.org/abs/2608.26578), arXiv:2608.26578, 2026.
- Official [TrapVLA project page](https://john-liua.github.io/TrapVLA/).

TrapVLA studies instruction-conditioned, phase-local manipulation failures and evaluates attack success together with clean-task preservation. This toy retains only that **defender-facing evaluation shape**. It adds delayed grasp/release stress cases to the paper-aligned premature-grasp, grasp-offset, premature-release, and release-offset categories.

## Toy setup

A transparent ridge behavior-cloning policy predicts a 41-step sequence of absolute `(x, y, gripper)` set points for a generated pick-and-place task. Clean demonstrations approach an object, grasp near phase `0.38`, transport it, release near phase `0.76`, and retreat.

Each audit run trains on 120 clean trajectories. At the nominal 5% configured-failure setting, six additional triggered trajectories use exactly one fixed template. The realized training-set prevalence is therefore `6 / 126 = 4.76%`; every sweep row records both the requested setting and realized fraction.

- early or late grasp;
- grasp-position offset;
- early or late release;
- release-position offset.

The model has an explicit phase/geometry feature bank and an auditable token-gated channel. This is intentionally inspectable; it is not a visual encoder, language model, transformer, flow policy, or robot controller.

## Compared defenses

- `naive_bc`: ordinary ridge BC on the mixed toy dataset.
- `phase_residual_filter`: removes rows whose labels disagree with the known clean toy generator beyond fixed position/gripper thresholds.
- `token_channel_shrinkage`: applies a stronger group penalty to the token-gated feature channel.
- `trigger_invariance`: penalizes output changes when the fixed synthetic token is toggled while phase and geometry are held constant.

Execution monitoring combines:

- grasp/release phase guards;
- missing grasp/release event guards;
- object/goal proximity guards at gripper transitions;
- a shadow residual guard comparing triggered and token-cleared predictions;
- a safety stop instead of continuing an anomalous sequence.

These checks are demonstrations of defensive metrics, not a production monitor design.

## Run

Full deterministic run:

```bash
cd /home/andypark/Projects/repos/vla-ideas
PYTHONWARNINGS=error \
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  configured_failure_audit/run_configured_failure_audit.py --seed 31
```

Smoke test without replacing checked outputs:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  configured_failure_audit/run_configured_failure_audit.py \
  --smoke --output-dir /tmp/configured_failure_audit_smoke
```

Render the report:

```bash
./configured_failure_audit/docs/render_pdf.sh
```

## Verified results

The checked run sweeps requested configured-failure trajectory prevalences `{0%, 1%, 2%, 5%, 10%}` across six fixed failure modes and four training methods. Integer trajectory counts yield realized prevalences `{0%, 0.83%, 1.64%, 4.76%, 9.77%}`. Each model is evaluated on 48 clean validation geometries and 64 paired clean/triggered test geometries.

At the nominal 5% setting (4.76% realized prevalence):

| Method | Clean val. MSE | Clean success | Triggered failure fidelity | Triggered task success | Trigger stop rate | Clean false stops | Unstopped task failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive BC | 0.008163 | 100.0% | 100.0% | 0.0% | 100.0% | 0.0% | 0.0% |
| Residual filter | **0.008108** | 100.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| Robust shrinkage | 0.008244 | 100.0% | 1.6% | 71.1% | 85.4% | 0.0% | 0.0% |
| Trigger invariance | 0.008356 | 100.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |

The key deception is explicit: naive BC preserves 100% clean success and a clean validation MSE within about 0.7% of the best defense, yet reproduces every configured failure under the fixed token.

Across the prevalence sweep, naive failure fidelity rises from `60.16%` at the requested 1% setting to `98.18%` at 2% and `100%` at 5–10%, while clean success remains `100%`. Filtering and trigger invariance keep measured failure fidelity at `0%` throughout this sweep. Shrinkage is effective at low rates but degrades to `16.93%` failure fidelity at 10%, illustrating that regularization alone is not a guarantee.

The known-clean residual audit attains AUROC `1.0` in this deliberately transparent toy. At the nominal 5% setting, only about `0.7–1.2%` of all training rows are locally modified, depending on failure mode. For naive BC, `98.3%` of triggered-clean output-residual energy lies within ±0.12 phase of the configured event, and `72.2%` of weighted token-channel activity is concentrated there. The weighted activity is a transparent channel diagnostic, not causal attribution.

![Headline metrics](outputs/headline_metrics.png)

![Configured-failure prevalence sweep](outputs/poison_rate_sweep.png)

![Phase-local residual probes](outputs/phase_local_probes.png)

![Activation probes](outputs/activation_probes.png)

![Monitor stops](outputs/monitor_stop_rates.png)

## Relation to other experiments in this repository

- **Hidden conditional behavior versus ordinary distribution shift:** `bc_distribution_shift_mysteries` studies benign closed-loop distribution shift, where policies extrapolate differently after their own errors. `demo_prompted_policy` studies useful runtime conditioning on a demonstration. This audit instead holds geometry fixed and pairs token-cleared with token-present evaluations to test a specific conditional discrepancy. Poor OOD performance alone is not evidence of a hidden trigger, and useful conditioning alone is not suspicious.
- **Data/training mitigation versus execution containment:** the residual filter, token-channel shrinkage, and trigger-invariance conditions alter fitting or training data. The execution monitor is a separate containment layer. `constraint_manifold_action_filter` and `path_consistent_safety_filtering` likewise act after proposal generation, but enforce geometric/path constraints rather than diagnose a conditional model dependence. `local_residual_sim2real` combines online correction with a modeled safe set; it does not audit hidden conditions.
- **Recovery routing:** `retry_reset_recovery` uses a noisy monitor to route between task and reset skills after observable failures. This audit's monitor stops anomalous sequences before continuing. A stop does not provide recovery, and a recovery router does not establish that a model is free of hidden conditional behavior.
- **Force and embodiment:** `force_feedback_demonstration_quality` studies force-target quality in demonstrations, while `force_embodiment_gap` studies runtime force observability and calibration/morphology transfer. This audit has no force, contact dynamics, sensor calibration, or embodiment transfer. Its set-point checks must not be presented as physical robot safety evidence.

## Outputs

- `outputs/metrics.json`: configuration, safety notice, sanity check, method definitions, and default-rate results.
- `outputs/sanity_check.json`: nominal-sequence, fixed-template, locality, and monitor assertions.
- `outputs/summary_metrics.csv`: per-mode default-rate evaluation.
- `outputs/method_aggregates.csv`: headline method means across six modes.
- `outputs/poison_rate_sweep.csv`: legacy-named full requested/realized prevalence × mode × mitigation sweep.
- `outputs/data_probe.csv`: phase-local label-residual diagnostics.
- `outputs/phase_probe.csv`: phase-by-phase output residuals and token-channel activations.
- `outputs/probe_summary.csv`: residual/activation peak phases and local-energy fractions.
- `outputs/monitor_metrics.csv`: stop, false-stop, prevented configured/task failure, and unstopped configured/task failure metrics.
- `outputs/*.png`: headline, sweep, residual, activation, and monitor plots.
- `docs/configured_failure_audit_report.tex` and `.pdf`: source-grounded report.

## Sanity check

Before fitting any model, the script verifies that:

1. the nominal scripted sequence succeeds;
2. all six fixed templates satisfy their declared failure criterion;
3. every fixed template is stopped by at least one execution guard;
4. missing-grasp and missing-release sequences are stopped explicitly;
5. no template changes more than 30% of sequence phases.

The full checked run passes all assertions.

## Limitations

- The policy is linear ridge regression over explicit phase/geometry features, not a VLA.
- The token is an exposed scalar feature, not a natural-language representation.
- A clean counterfactual generator is available, making filtering, invariance, and shadow monitoring unrealistically easy.
- Fixed residual templates are deterministic and separable; AUROC `1.0` should not be expected on real data.
- Sequence set points replace contact dynamics, perception, action chunking, and closed-loop execution.
- The monitor uses exact phase, object, and goal state and has zero clean false stops in this toy; real monitors face uncertainty and distribution shift.
- The shadow-residual guard assumes a trusted token-cleared counterfactual inference, which may be unavailable or itself unreliable.
- No adaptive or unseen failure is tested. A defense that works here is not certified for real models or robots.
- Results show mechanism plausibility only. They do not estimate TrapVLA attack rates, defense effectiveness, or real-world risk.
