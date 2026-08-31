# Force Embodiment Gap

This deterministic toy asks a narrow question inspired by RAI Institute's Koala gripper work:

> When contact-rich demonstrations must transfer from a handheld teaching tool to a robot gripper, what is lost by visual aliasing, and what extra benefit comes from matching morphology, force calibration, and action coordinates?

The package compares five behavior-cloning conditions on a stuck-fastener task and a tight insertion task. During contact, multiple physically different states render the same visual progress. The learner must decide whether to keep loading, release, or correct a signed lateral jam.

This is an explanatory low-dimensional benchmark. It is **not** a reproduction of RAI Institute's Large Behavior Models, Koala hardware, data scale, neural architecture, or robot results.

## Source boundary

Primary sources checked:

- RAI Institute, [Getting a Grip on Robotic Data Collection](https://rai-inst.com/resources/blog/handheld-robotic-data-collection/), May 12, 2026.
- Amar Hajj-Ahmad et al., [Koala Gripper: Co-designing Robotic Grippers and Data-Capture Devices for Scaling Dexterous Manipulation Learning](https://arxiv.org/abs/2608.20546), 2026.
- [Official Koala project page](https://koalagripper.rai-inst.com/).

The source's central contribution is **embodiment parity**: the handheld and robot-side grippers share geometry and action coordinates so the demonstrated hand action maps directly to execution. The paper's approximately 24--27 N pinch-force measurements over 10--40 mm test objects are hardware capability/consistency checks. It does not report a quantitative policy-success comparison that isolates force as a demonstration input.

This toy therefore separates two ideas:

1. **Source-backed:** matched morphology and action coordinates reduce an embodiment gap.
2. **Hypothetical extension tested here:** if synchronized force is available, force can disambiguate visually identical contact states, but only when its calibration and mechanical frame transfer correctly.

## Conditions

| Method | Observation / action interface |
| --- | --- |
| Vision only | Current quantized visual state; raw motor-command BC |
| Temporal history | Current vision plus four previous visual/action pairs |
| Vision + force | Current vision plus raw two-axis force from nominal matched demonstrations |
| Mismatched-gripper force | Force/proprio BC trained on a different handheld gripper and transferred without recalibration |
| Matched multimodal BC | Vision, calibrated force, and prior physical action in a shared matched action frame |

The matched model predicts physical effort and maps it through the current gripper gains. This deliberately idealizes a known co-calibration/action-coordinate mapping; it is not learned hardware identification.

## Tasks

- **Stuck fastener:** visual progress is exactly flat while axial contact force ramps toward an unknown breakaway threshold. Too little effort stays stuck; excessive overshoot damages the part.
- **Tight insertion:** the same axial ambiguity is combined with a signed lateral jam. Vision exposes only quantized alignment magnitude, while lateral force exposes correction direction.

Expert demonstrations use feedback rules, then deterministic Extra Trees regressors clone each observation/action interface. Evaluation scenarios are paired across methods.

## Run

Use the repository's existing shared Python environment:

```bash
cd /home/andypark/Projects/repos/vla-ideas

# Fast end-to-end check (overwrites outputs with reduced metrics)
PYTHONWARNINGS=error \
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  force_embodiment_gap/run_force_embodiment_gap.py --smoke --seed 31

# Full deterministic run used for the checked-in artifacts
PYTHONWARNINGS=error \
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  force_embodiment_gap/run_force_embodiment_gap.py \
  --seed 31 --train-episodes 180 --validation-episodes 45 \
  --eval-episodes 120 --sweep-episodes 32

# Render the LaTeX report
force_embodiment_gap/docs/render_pdf.sh
```

## Latest verified result

The full seed-31 run uses 120 paired episodes per task and method at nominal hardware, plus 32 paired episodes per task/method/point in each sweep.

<!-- METRICS_START -->
| Method | Overall success | Damage | Mean final progress | Validation action RMSE |
| --- | ---: | ---: | ---: | ---: |
| Vision only | 0.0% | 0.0% | 0.000 | 0.1517 |
| Temporal history | 40.0% | 0.0% | 0.402 | 0.0688 |
| Vision + force | **80.0%** | 0.0% | **0.803** | 0.0283 |
| Mismatched-gripper force | 73.3% | 0.4% | 0.733 | 0.0383 |
| Matched multimodal BC | 77.9% | 0.0% | 0.780 | 0.0284 |

At nominal hardware, raw vision + force is narrowly best overall; matched multimodal BC does **not** dominate that in-distribution condition. The contact-alias effect is clearer by task: temporal history reaches 67.5% on fasteners but only 12.5% on signed-jam insertion, while force and matched multimodal BC both reach 87.5% insertion success.

Validation RMSE is only a within-interface diagnostic: raw motor targets, mismatched motor targets, and physical-effort targets have different coordinates and scales, so the values should not be ranked across all rows.

The matched interface's advantage appears under transfer. Across the tested calibration scales its worst success is 71.9%, versus 0.0% for raw vision + force. Across morphology gains its worst success is 73.4%, versus 0.0% for raw vision + force and 17.2% for mismatched-gripper force. The mismatched policy is not uniformly worse: it reaches 92.5% nominal fastener success and an accidental 100% at calibration scale 1.30, but falls to 54.2% on nominal insertion and incurs 23.4% damage at morphology gain 1.30. Those reversals are why this toy supports a calibration/coordinate claim rather than a universal ranking.
<!-- METRICS_END -->

![Method summary](outputs/method_summary.png)

### Calibration and morphology stress

The calibration sweep changes the force-sensor scale, cross-axis coupling, and bias. Raw-force methods receive the shifted readings. Matched multimodal BC is given the exact co-calibration inverse; this is an explicit idealization.

The morphology sweep changes axial gain by 0.70--1.30 and lateral gain inversely. Raw motor-command policies transfer unchanged. Matched multimodal BC retains shared physical action coordinates and maps them through the known deployment gains.

![Calibration sweep](outputs/calibration_sweep.png)

![Morphology sweep](outputs/morphology_sweep.png)

## Metrics and outputs

- `outputs/summary_metrics.csv`: per-task and overall nominal aggregates.
- `outputs/trial_metrics.csv`: per-episode nominal results.
- `outputs/calibration_sweep.csv`: aggregate calibration stress results.
- `outputs/morphology_sweep.csv`: aggregate morphology stress results.
- `outputs/metrics.json`: configuration, model definitions, validation RMSE, sweeps, and summaries.
- `outputs/sanity_checks.json`: visual-alias, calibration-inverse, action-frame, and deterministic-replay checks.
- `outputs/method_summary.png`: nominal success, damage, and progress.
- `outputs/calibration_sweep.png`, `outputs/morphology_sweep.png`: robustness curves.
- `outputs/representative_rollouts.png`: paired fastener/insertion traces.
- `docs/force_embodiment_gap_report.tex` and `.pdf`: bounded report and rendered artifact.

## Sanity checks

The script fails the run unless all checks pass:

- two contact states with identical vision require substantially different expert actions;
- the known affine force calibration is exactly inverted;
- shared physical action coordinates execute the same effort across morphology gains;
- a seeded matched-policy rollout replays identically.

## Interpretation and limits

The toy is constructed so contact force is causally informative. It can show that vision history does not fully resolve a contact alias and that uncalibrated force/motor coordinates may transfer poorly. It cannot establish that real robot policies need force, that force is sufficient, or that a specific Koala/RAI policy uses force as input.

## Relation to other experiments in this repository

- `force_feedback_demonstration_quality` changes the **quality of the collected demonstrations** by simulating haptic feedback on/off, then holds the ridge policy family fixed and measures force-profile error, damage, retries, and unseen-object shift. This track instead holds scripted expertise fixed and changes the learner's observation/action hardware interface; it uniquely targets visual contact aliasing plus calibration/morphology transfer.
- `cross_embodiment_world_model` also compares raw, canonical, and latent action semantics, but for smooth full-state world-model prediction with 0--32 target demonstrations and candidate reranking. This package trains BC policies once, then stresses force sensing and motor-to-physical execution in contact; it has no learned world model or few-shot refit.
- `demo_prompted_policy` and `video_prompt_shortcut_resistance` use a demonstration video as a runtime semantic prompt. No such prompt exists here: the relevant hidden variables are contact load and signed jam direction, and the intervention is a force/action coordinate change rather than prompt corruption or task recomposition.
- `context_chunk_tradeoff` and the asynchronous chunking tracks study observation history, stale context, and chunk scheduling. `temporal_history` here is only a fixed four-step BC baseline, with no inference delay or chunk commitment. Conversely, the online-adaptation tracks update from reward or observed target transitions; this package performs no deployment-time learning and gives the matched method exact calibration/morphology transforms.

Dynamics, thresholds, sensor transforms, calibration metadata, and the expert are synthetic and low-dimensional. Images are hand-designed features, Extra Trees replace a VLA, demonstrations are independent rather than human, and the matched model receives exact hardware transforms. There is no language input, whole-body motion, learned representation, force-sensor noise estimation, latency, wear, slip, or unmodeled compliance. A real test should pair the same demonstrations and tasks across matched and deliberately mismatched physical grippers, report force/action calibration error, and include policy success, damage, and recovery metrics.

The matched condition also bundles exact calibration/action-coordinate transforms with one-step physical-action history, while the raw-force baseline omits that history. The sweeps therefore identify the bundled matched interface, not the separate causal contribution of calibration, morphology mapping, and temporal input. A clean follow-up should add calibrated-force/physical-action ablations with identical history features.
