# Force-Feedback Demonstration Quality Toy

This package asks a prospective PHABS-inspired question:

> If the same demonstrators perform the same bimanual fragile-object handoff/insertion with haptic force feedback on versus off, do the haptic demonstrations have cleaner force profiles, and do identical imitation policies trained from those data transfer better to harder objects?

The answer in this **synthetic mechanism test** is yes. That is a local simulator result, not a result reported by PHABS. The source paper presents an early handheld haptic teleoperation prototype and a pilot with three participants; the pilot reports better operator task success and confidence with force feedback, but it does not report force-profile-quality statistics or a downstream haptic-on/off learned-policy comparison. This package deliberately does not upgrade the pilot into those claims.

## Primary reference

- Mosier et al., **“PHABS: A Handheld Haptic Device for Force-Annotated Bimanual Teleoperation.”** OpenReview: `https://openreview.net/pdf?id=Lmbnt2VNDM`

PHABS motivates collecting synchronized bimanual pose and force demonstrations while returning fingertip force cues to the operator. This toy isolates one possible downstream mechanism: feedback changes the *demonstration target distribution*, not only the annotation format.

## What is simulated

A fragile object is:

1. approached and grasped by the left hand;
2. transferred through a bimanual handoff;
3. carried by the right hand;
4. inserted into tight packaging with both hands;
5. released.

Matched scenarios use identical mass, friction, fragility, insertion tightness, alignment, and visual estimates under haptics on/off. Haptic demonstrators regulate toward a feasible no-slip/no-damage force envelope with lower lag and noise. Visual-only demonstrators use uncertain visual property estimates, a larger generic margin, slower corrections, and more oscillation.

Every downstream policy is the same four-output ridge imitation model. Only its training targets differ:

- `pose_only_visual`: visual-only pose traces; force is replaced by a generic pose-gated squeeze.
- `force_annotated_visual`: the force actually executed without haptic feedback is retained.
- `haptic_force_visual_motion`: visual-only speed/balance targets are retained, but force is replaced by the exactly matched haptic rollout. This crossed ablation isolates force-target quality while holding the non-force targets fixed.
- `passive_noisy_force`: visual-only force is logged with lag, noise, bias, and dropout.
- `haptic_force_rich`: haptic-collected pose and force targets are both retained.

The crossed condition makes the claim boundary sharper. Comparing it with `force_annotated_visual` changes only the force targets. Comparing it with `haptic_force_rich` then measures the remaining effect of the generated speed/balance targets. This is still not a causal human-subject result: the simulator constructs both kinds of demonstrations.

## Run

Dependencies are only NumPy and Matplotlib.

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  force_feedback_demonstration_quality/run_force_feedback_demonstration_quality.py \
  --seed 37 --demo-scenarios 72 --demo-repeats 2 \
  --eval-trials-per-shift 180
```

Sanity check only:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  force_feedback_demonstration_quality/run_force_feedback_demonstration_quality.py \
  --sanity-only
```

Small smoke sweep:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  force_feedback_demonstration_quality/run_force_feedback_demonstration_quality.py \
  --quick --output-dir /tmp/force_feedback_quick
```

Render the report:

```bash
force_feedback_demonstration_quality/docs/render_pdf.sh
```

## Latest verified full sweep

The full deterministic run evaluates 720 matched trials per method: 180 trials at each of four shift severities. Distribution shift raises mass and insertion tightness, lowers friction and fragility, and increases visual property-estimation error.

| Training condition | Success | Damage | Slip/drop | Force RMSE | Retries | Hard-shift success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pose-only / visual | 48.6% | 29.0% | 33.8% | 2.256 | 0.729 | 19.4% |
| Force-annotated / visual | 72.9% | 16.0% | 14.6% | 1.024 | 0.314 | 46.1% |
| Haptic force, visual motion | 78.1% | 10.4% | 13.5% | 0.505 | 0.188 | 52.2% |
| Noisy passive force | 53.6% | 10.7% | 41.0% | 0.848 | 0.263 | 35.6% |
| **Haptic force-rich** | **78.3%** | **10.4%** | **13.1%** | **0.505** | **0.189** | **52.2%** |

At fixed visual-only speed, balance, observations, and model class, substituting the matched haptic force targets improves average success by 5.1 percentage points, reduces force-profile RMSE by 50.7%, reduces retries by 40.3%, lowers damage by 5.6 points, and improves hard-shift success by 6.1 points. Moving from that crossed condition to the full haptic targets adds only 0.3 success points in this run. Thus the local separation is driven almost entirely by the constructed force-target quality, not by different non-force targets. These are simulator outcomes caused by the assumptions below, not estimates of real PHABS performance.

The matched demonstration generator also shows the intended mechanism before policy training:

- visual-only demo force RMSE: `0.991 N`; haptic: `0.265 N`;
- visual-only force-error variance: `0.339`; haptic: `0.010`;
- visual-only retries: `0.292`; haptic: `0.194`;
- visual-only rollout success: `82.6%`; haptic: `87.5%`.

![Matched demonstrations](outputs/matched_demo_force_profiles.png)

![Policy summary](outputs/policy_summary.png)

![Robustness sweep](outputs/robustness_sweep.png)

## Metrics

- **Success:** no sampled drop, damage, or terminal insertion failure within the retry budget.
- **Object damage / excess force:** binary damage event and mean force beyond the phase-dependent fragility limit.
- **Slip/drop:** sampled from sustained force deficit and handoff imbalance.
- **Force-profile RMSE / variance:** error relative to the simulator’s feasible phase- and object-conditioned force target.
- **Retries:** insertion reattempts induced by speed, alignment, force, and handoff errors.
- **Robustness:** success at the hardest shift and area under the success-versus-shift curve.

## Relation to other experiments in this repository

- **Data quality versus policy conditioning:** `bc_distribution_shift_mysteries` changes chunk horizon, history, feature scaling, and model basis while holding the expert data source conceptually fixed; `demo_prompted_policy` uses a demonstration as runtime task conditioning. This experiment instead changes the supervised action targets at collection time. Its crossed ablation holds observation and non-force targets fixed, so it tests target quality rather than a richer policy input or prompt.
- **Recovery data and online correction:** `retry_reset_recovery` broadens training support with bridge/reset examples, while `local_residual_sim2real` corrects a deployed prior from recent transition data. This package has neither recovery demonstrations nor online adaptation: all differences are inherited from the original matched demonstration set.
- **Execution-time safety filters:** `constraint_manifold_action_filter` and `path_consistent_safety_filtering` modify or stop proposed actions during execution using known constraints or predicted occupancy. Such filters can contain bad commands but cannot reconstruct missing contact targets in the training data. The present toy is a training/data mitigation and provides no deployment-time safety guarantee.
- **Force and embodiment:** `force_embodiment_gap` asks whether force resolves runtime visual aliasing and whether calibration/morphology preserve force semantics across hardware. Here the policy receives no runtime force measurement and there is no handheld-to-robot morphology transfer; force matters only as demonstration supervision. The two experiments are complementary and should not be merged into a claim that force-rich data automatically transfer across embodiments.

## Outputs

- `outputs/sanity_check.json`: matched-pair, determinism, and force-quality assertions.
- `outputs/demonstration_metrics.csv`: one row per haptic-on/off demonstration rollout.
- `outputs/policy_trials.csv`: one row per policy evaluation rollout.
- `outputs/shift_sweep.csv`: aggregate metrics by method and shift.
- `outputs/summary_metrics.csv`: headline method comparison.
- `outputs/metrics.json`: configuration, sanity, demonstration, training, and evaluation summaries.
- `outputs/matched_demo_force_profiles.png`
- `outputs/policy_summary.png`
- `outputs/robustness_sweep.png`
- `outputs/representative_policy_rollout.png`
- `docs/force_feedback_demonstration_quality_report.tex` and rendered `.pdf`

## Simplifications and limits

- No PHABS hardware, torque sensing, kinematics, bilateral controller, robot, camera, language, or VLA is implemented.
- Object properties and task phase are compact numeric features rather than pixels or learned representations.
- Policies do not observe runtime force; this toy tests force as an action target, not force-conditioned feedback control.
- The same hand-designed equations define demonstration generation and evaluation; this creates favorable structure for a small regressor.
- Haptic feedback is modeled as lower lag/noise plus correction toward a known force envelope. Real operators may adapt differently.
- Damage, slip, and insertion failure are calibrated stochastic proxies, not contact mechanics.
- The shift sweep changes several factors together and uses one random seed/configuration.
- The ridge policy predicts one action per phase, not chunks or a multimodal action distribution.
- The toy supports a **plausibility argument and experiment design**, not a causal or empirical claim about PHABS.

A useful real study would use matched participants and objects, randomize haptics on/off order, retain synchronized raw force/pose streams, train the same policy family and data budget per condition, and report paired confidence intervals for task success, peak/excess force, drops, retries, and unseen-object robustness.
