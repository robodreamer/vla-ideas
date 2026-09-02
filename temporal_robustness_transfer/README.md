# Temporal Robustness Transfer Toy

This package asks:

> If an imitation policy is trained on demonstrations at multiple execution speeds, does it inherit the expert's operating envelope—or only fit the demonstrated trajectories?

The deterministic toy compares five temporal/control choices on matched insertion scenarios:

1. **Raw elapsed-time scaling:** one policy uses whole-cycle normalized clock time. Fixed and speed-scaled phases therefore move relative to its clock.
2. **Phase normalization:** the policy receives normalized task phase but not execution rate.
3. **Speed conditioning:** phase features are explicitly conditioned on the requested speedup factor.
4. **Force feedback:** the speed-conditioned policy receives a synthetic contact residual that adjusts grip, lateral/yaw correction, and insertion speed near contact.
5. **Dynamics augmentation:** a dynamics-aware feature basis is trained with extra synthetic rates and randomized compliance/servo parameters.

A scripted expert is retained as an upper bound. Every learned method uses the same standardized multi-output ridge regressor; representation, runtime feedback, or synthetic training support is what changes. The force-feedback condition receives a privileged simulator-aligned correction, while dynamics augmentation adds richer features and additional demonstrations including out-of-range rates; neither is a like-for-like same-data policy comparison.

## Primary sources and claim boundary

- Enwerem, Baras, and Belta, **“Does Imitation Learning Preserve Temporal Robustness in Dexterous Manipulation? An Expert-Learner Comparison Across Task Execution Speeds.”** arXiv: `https://arxiv.org/abs/2609.01453`
- Official code, records, task specifications, and diagnostics: `https://github.com/coenwerem/parcelstow`

ParcelStow evaluates expert and learner under matched initial conditions. Its parcel-insertion demonstrations use speedup factors in `r ∈ [0.5, 2]`; evaluation includes `r = {0.5, 1, 1.5, 2, 2.25, 2.5, 3}`. The reported primary comparison has expert and ACT-A both at 100/100 nominal successes, but at `r = 2` the expert succeeds in 84/100 episodes and ACT-A in 53/100. At `r = 2`, 45 of ACT-A's 47 failures are classified as insertion misalignment or insertion jam. The release also reports that no acquired episode lacking force closure completes the task in its evaluated records.

Those are **source results**. The numbers below are **local synthetic results**. This package does not run Isaac Lab, load ParcelStow records/checkpoints, implement ACT or Diffusion Policy, reproduce contact mechanics, or estimate real-robot performance. It tests plausible interventions that the source paper does not evaluate. In particular, the paper does not establish that phase normalization, force feedback, or dynamics augmentation closes the expert–learner gap.

## Synthetic task

The toy preserves a few structural ideas from ParcelStow:

- acquisition and final task structure include fixed-duration components;
- post-acquisition manipulation is sped up by `r`, giving total duration `6.3 + 7.8/r` seconds (`14.1 s` at `r=1`, `10.2 s` at `r=2`);
- task stages are ordered: acquisition, lift, reorientation, pre-insertion, insertion, release, settling;
- the insertion outcome depends on pre-insertion lateral/yaw error, insertion speed, grip margin, compliance, and a receptacle-force proxy;
- force closure is a sign test at acquisition, while continuous force margin is not treated as a calibrated success probability.

Each scenario varies initial lateral/yaw error, friction, compliance, servo gain, and force-sensor bias. The expert target includes speed-dependent dynamic lead, insertion speed, and grip demand. Policies predict lateral correction, yaw correction, insertion-speed command, and grip command over 81 normalized phase samples.

Base training uses 48 demonstrations at each of `r = {0.5, 1, 1.5, 2}` (192 demonstrations). Evaluation uses 180 matched scenarios at each of:

- demonstrated anchors: `0.5, 1, 1.5, 2`;
- held-out interpolation: `0.75, 1.25, 1.75`;
- extrapolation: `2.25, 2.5, 3`.

Dynamics augmentation adds 144 simulator-generated demonstrations at broader rates and dynamics. It therefore tests whether an explicit model-based coverage intervention helps; it is not a same-data comparison. The force-feedback condition uses a favorable synthetic contact residual; it is a feedback-controller hypothesis, not evidence that a particular real sensor/controller will work.

## Run

Dependencies are NumPy and Matplotlib.

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  temporal_robustness_transfer/run_temporal_robustness_transfer.py --mode full
```

Quick sweep:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  temporal_robustness_transfer/run_temporal_robustness_transfer.py \
  --mode quick --output-dir /tmp/temporal_robustness_quick
```

Sanity mode:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  temporal_robustness_transfer/run_temporal_robustness_transfer.py \
  --mode sanity --output-dir /tmp/temporal_robustness_sanity
```

Aliases `--quick` and `--sanity-only` are also supported. Render the report with:

```bash
temporal_robustness_transfer/docs/render_pdf.sh
```

## Verified full-sweep results

Each method is evaluated on 1,800 episodes: 180 matched scenarios at each of 10 speeds.

| Method | Nominal (`r=1`) | `r=2` | Held-out interpolation | Extrapolation | Drop (`r=1→2`) | Extrapolation AUC (`r∈[2,3]`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scripted expert | 100.0% | 100.0% | 100.0% | 75.4% | 0.0 pp | 81.5% |
| Raw elapsed time | 99.4% | 84.4% | 66.7% | 2.6% | 15.0 pp | 12.5% |
| Normalized phase | 100.0% | 90.0% | 99.8% | 1.3% | 10.0 pp | 12.2% |
| Speed-conditioned | 100.0% | 98.9% | 100.0% | 50.4% | 1.1 pp | 57.8% |
| **Speed + force feedback** | **100.0%** | **100.0%** | **99.6%** | **67.2%** | **0.0 pp** | **75.1%** |
| Dynamics augmentation | 100.0% | 100.0% | 100.0% | 64.6% | 0.0 pp | 72.3% |

These are deterministic point estimates from one configured simulator run, not cross-seed confidence intervals.

Within the toy:

- nominal success does not identify temporal robustness: raw time, phase, and speed-conditioned policies are all at least 99.4% at `r=1`, but extrapolation success ranges from 1.3% to 50.4%;
- phase normalization nearly eliminates held-out interpolation failures but does not encode the quadratic speed/contact demand, so it collapses beyond the training range;
- explicit speed conditioning raises extrapolation success by 49.1 percentage points over phase normalization;
- force feedback adds 16.9 points over speed conditioning in extrapolation and lowers mean pre-insertion misalignment from `1.72 mm` to `0.96 mm` and peak force from `4.72 N` to `4.25 N`;
- dynamics augmentation adds 14.3 extrapolation points over speed conditioning, but it uses 144 additional synthetic demonstrations and richer dynamics features;
- all methods encounter a physical-envelope cliff at `r=3`; even the scripted expert succeeds in only 26.1% of those episodes.

These results are produced by the toy equations and thresholds. They support an experiment-design conclusion—not a real-system performance claim: **speed should be treated as an intervention with a measured operating envelope, and temporal representation, contact feedback, and dynamics coverage should be ablated separately.**

![Operating envelope](outputs/operating_envelope.png)

![Stage diagnostics](outputs/stage_diagnostics.png)

![Insertion proxies](outputs/insertion_proxies.png)

![Representative extrapolation trajectory](outputs/representative_trajectory.png)

## Metrics and diagnostics

- **Success / operating envelope:** final settling success versus requested `r`.
- **Temporal drop:** success at `r=1` minus success at `r=2`.
- **Interpolation and extrapolation success:** averages over the held-out rate groups above.
- **Range-normalized AUC:** trapezoidal area under success versus speed inside `[0.5,2]` or `[2,3]`.
- **Ordered stage pass rate:** an episode cannot pass a later stage after failing an earlier one.
- **Force-closure proxy:** sign of acquisition grip reserve after a lateral-offset penalty. As in the source diagnostic, absence is treated as a necessary-condition failure, not a complete success score.
- **Insertion misalignment:** pre-insertion lateral/yaw residual exceeds the toy clearance tolerance.
- **Insertion jam:** peak receptacle-force proxy is too high or achieved depth is below `50 mm`.
- **Contact/geometry proxies:** mean lateral misalignment, yaw error, peak receptacle force, insertion depth, and action RMSE relative to the scripted target.

## Toy-to-real mapping

| Toy component | Intended real counterpart | Important mismatch |
| --- | --- | --- |
| Normalized phase | skill progress, phase estimator, event-based state machine | phase is given perfectly here |
| Speedup `r` | requested task-rate intervention | only one scalar changes the schedule |
| Ridge action predictor | compact imitation policy | no images, language, history, chunks, or multimodality |
| Grip reserve / force closure | grasp wrench feasibility | scalar proxy replaces contact geometry and wrench space |
| Force residual | tactile/wrist-force closed-loop correction | uses privileged, low-noise simulator structure |
| Dynamics augmentation | randomized simulation or model-generated corrective data | labels share the toy's equations and include out-of-range rates |
| Misalignment / jam | receptacle geometry and contact failure | threshold proxy replaces rigid-contact simulation |

A real follow-up should use the same expert/learner initial-condition draws, pre-register demonstrated and extrapolation speeds, separate phase estimation from speed conditioning, preserve raw force/contact streams, and report paired confidence intervals for success, stage progression, insertion pose, contact force, insertion depth, and force-closure sign. Chunk horizon, temporal ensembling, and replanning frequency should also be varied because the source paper identifies ACT's temporally extended prediction as an unresolved possible mechanism.

## Outputs

- `outputs/sanity_check.json`: deterministic schedule, expert, and feedback assertions.
- `outputs/episode_metrics.csv`: one row per method/speed/scenario rollout.
- `outputs/speed_sweep.csv`: aggregate success, stage, force, geometry, and failure metrics by method and speed.
- `outputs/stage_diagnostics.csv`: ordered pass rates by method and regime.
- `outputs/summary_metrics.csv`: headline method comparison.
- `outputs/metrics.json`: configuration, training-set sizes, summaries, and pairwise comparisons.
- `outputs/operating_envelope.png`
- `outputs/stage_diagnostics.png`
- `outputs/insertion_proxies.png`
- `outputs/representative_trajectory.png`
- `docs/temporal_robustness_transfer_report.tex` and rendered PDF.

## Limits

- No Isaac Lab, robot, hand, object mesh, collision/contact solver, camera, VLA, ACT, Diffusion Policy, or DAgger.
- Perfect task phase is available to all methods except the deliberately raw-clock baseline.
- Force feedback is represented as direct residual correction toward the simulator target; real sensing delay, calibration, saturation, and controller stability are absent.
- Dynamics augmentation is generated by the same equations used for evaluation and includes rates beyond the base demonstration range.
- Stage predicates and thresholds are inspired by the source structure but are not a reproduction of ParcelStow's simulator predicates.
- One deterministic seed/configuration is reported; there is no training-seed confidence interval.
- The expert's extrapolation cliff and all policy rankings depend on hand-designed equations.
- The toy cannot establish which mechanism caused the paper's reported ACT-A gap or whether any intervention transfers to hardware.
