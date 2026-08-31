# Cross-Embodiment World Model: CLAP Action-Harmonization Toy

This package tests a narrow mechanism motivated by **CLAP: Cross-Embodiment Video World Models are Zero-Shot Physical Simulators**:

> If different embodiments interact with objects under shared physics, which action representation makes source experience useful for a target embodiment with very little data?

Two source embodiments drive the same synthetic 2-D object dynamics through different raw control spaces. A third, novel embodiment supplies 0–32 demonstration episodes. The experiment compares padded raw controls, canonical end-effector (EE) controls, learned transition latents, and latent-to-EE curriculum grounding for one-step target prediction and candidate-action reranking.

This is a deterministic vector-state mechanism test. It is **not** a reproduction of CLAP's video diffusion models, Open X-Embodiment/EgoDex training, learned 32-D video LAM, perceptual metrics, or robot results.

## References checked

- Kechen Liu and Ola Shorinwa, [CLAP: Cross-Embodiment Video World Models are Zero-Shot Physical Simulators](https://arxiv.org/abs/2608.27406), arXiv:2608.27406v1, August 27, 2026.
- Official [CLAP project page](https://omni-clap.github.io/).
- Official [omni-CLAP/clap repository](https://github.com/omni-CLAP/clap).

The paper and project describe the following relevant ideas represented in this toy:

1. raw joint/action spaces differ sharply across embodiments;
2. CLAP's canonical robot interface is a normalized absolute 7-D EE pose/gripper action; the paper reports that absolute EE conditioning outperformed relative EE conditioning in its cross-embodiment comparison;
3. CLAP's latent-action model extracts a learned 32-D proxy action from frame pairs, allowing shared video dynamics pretraining to include action-unlabeled data such as human video;
4. CLAP-CURR replaces the latent action head with a 7-D EE MLP and jointly fine-tunes that head and the retained video-model backbone on action-labeled robot video;
5. latent-only real-world planning still requires a separately trained EE-to-latent adapter, while CLAP also evaluates few-shot novel-embodiment adaptation and candidate planning.

The local PCA latent, ridge predictors, and 2-D EE commands are deliberately much simpler than those methods. References to zero-shot transfer below mean transfer within this synthetic setting with the target's exact raw-to-EE map available; they do not reproduce CLAP's real-world zero-shot claim.

## Toy setup

The object state is

```text
[x, y, vx, vy]
```

and the shared transition function depends only on this state and a canonical 2-D EE command. Embodiment identity never enters the object dynamics. Instead, each embodiment maps its raw control to EE motion differently:

- source A: 2-D raw control;
- source B: 3-D redundant raw control;
- target: novel 4-D raw control.

The maps use different linear, quadratic, and cross-coordinate terms. The same padded raw vector therefore has different physical meaning across source robots, while equal EE commands induce exactly equal object transitions.

Each full run uses:

- 3,600 source transitions (150 episodes × 12 steps × 2 embodiments);
- only 36 source transitions with visible EE labels (`1%`), while all 3,600 source transitions are available as unlabeled state-transition pairs;
- target sweeps of `{0, 1, 2, 4, 8, 16, 32}` episodes, or 0–384 transitions;
- 900 held-out target transitions per seed;
- 320 held-out planning queries with 10 target-raw-action candidates each;
- 8 deterministic data/model seeds.

## Compared methods

- `raw_joint`: pad raw controls to four coordinates, add embodiment-specific branches, and jointly fit source plus upweighted target demonstrations. A novel target branch is unconstrained without target examples.
- `canonical_ee`: directly condition a shared world model on exact 2-D EE actions. It is trained only on the 36 EE-labeled source transitions plus available target demonstrations.
- `learned_latent`: regress passive state change, extract a 2-D PCA action latent from source transition residuals, train latent-conditioned shared dynamics on all source transitions, and learn the target raw-to-latent alignment from few demonstrations.
- `latent_to_ee_curriculum`: use the same all-source latent dynamics pretraining, then learn an EE-to-latent grounding bridge from labeled source/target data and fit a conservative residual correction.

The learned-latent deployment adapter intentionally reflects the action-alignment limitation discussed by CLAP, whose latent-only planning path uses a separately trained EE-to-latent adapter. Canonical EE and curriculum methods receive the target embodiment's exact kinematic raw-to-EE conversion, so they can be evaluated at zero target demonstrations; raw and latent target branches cannot. This is a controlled interface assumption, not evidence of real-robot zero-shot deployment.

## Metrics

- **State RMSE:** root mean squared error over the four next-state coordinates.
- **Position RMSE:** next-object-position RMSE.
- **Candidate top-1:** fraction of planning queries where predicted and true costs select the same candidate.
- **Candidate regret:** true goal-distance cost of the selected candidate minus the oracle candidate cost.
- **Planning success:** fraction with candidate regret at most `0.003`.

Candidate reranking is a one-step analogue of using a world model to discriminate policy proposals. It is not closed-loop robot planning.

## Run

Full deterministic sweep from the repository root:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  cross_embodiment_world_model/run_cross_embodiment_world_model.py \
  --seed 23 --seeds 8 \
  --source-episodes 150 --target-pool-episodes 48 \
  --test-transitions 900 --planning-queries 320 \
  --shots 0 1 2 4 8 16 32 \
  --source-ee-label-fraction 0.01
```

Smoke test without replacing the checked outputs:

```bash
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  cross_embodiment_world_model/run_cross_embodiment_world_model.py \
  --smoke --output-dir /tmp/cross_embodiment_world_model_smoke
```

Render the report:

```bash
cross_embodiment_world_model/docs/render_pdf.sh
```

## Latest verified results

### Four target demonstrations

Four episodes contain 48 target transitions.

| Method | State RMSE ↓ | Candidate top-1 ↑ | Regret ↓ | Planning success ↑ |
| --- | ---: | ---: | ---: | ---: |
| Raw / padded joints | 0.0247 ± 0.0009 | 0.757 ± 0.009 | 0.00124 | 0.855 |
| Canonical EE | **0.0109 ± 0.0011** | **0.930 ± 0.006** | **0.00013** | **0.986** |
| Learned latent | 0.0247 ± 0.0009 | 0.749 ± 0.012 | 0.00132 | 0.848 |
| Latent→EE curriculum | 0.0109 ± 0.0011 | 0.925 ± 0.004 | 0.00015 | 0.983 |

Values are mean ± SEM across eight seeds where shown.

![Few-shot sweep](outputs/few_shot_sweep.png)

### Bounded interpretation

- At zero target demonstrations, canonical EE and curriculum grounding already obtain state RMSE `0.0132` and `0.0122`, while ungrounded raw and latent target interfaces are near `0.106–0.109`. In this construction, a known geometric action interface matters more than the specific grounded model variant.
- At four demonstrations, canonical EE and curriculum reduce state RMSE by about `56%` relative to the raw baseline and raise candidate top-1 from `0.757` to `0.930` and `0.925`.
- The learned latent explains `99.99%` of source action-residual variance, but its target performance closely tracks raw controls after alignment. This intentionally shows that a compact transition latent does not by itself remove the deployment-time alignment problem.
- Canonical EE is slightly better than curriculum on most reranking rows, although prediction RMSE is nearly tied. The toy therefore **does not establish a curriculum advantage over direct EE training**; it only shows that latent pretraining can be grounded without losing the shared-physics benefit under extreme EE-label sparsity.
- One target episode is underdetermined for the target-specific raw/latent branches and can be worse than zero-shot passive/source predictions. The sweep exposes this small-sample instability rather than hiding it.
- Error bars overlap for several EE-versus-curriculum comparisons, and no significance tests were run.

![Representation diagnostics](outputs/representation_diagnostics.png)

![Candidate reranking example](outputs/candidate_reranking_example.png)

## Relation to other experiments in this repository

- `demo_prompted_policy` and `video_prompt_shortcut_resistance` study a demonstration as a **runtime task specification**. They use ordered prompt content and score task progress, recomposition, recovery, or prompt-swap sensitivity without target-time weight updates. This track has no task prompt or long-horizon sequence; it retrains closed-form target interfaces on 0--32 target episodes and isolates how raw, exact EE, and transition-latent action coordinates affect one-step prediction and candidate regret.
- `force_embodiment_gap` is the closest embodiment-transfer companion, but its mismatch is in force sensing, morphology gains, and motor-versus-physical BC coordinates during contact. Here observations are full object state, contacts are smooth, and the embodiment question is confined to the **action conditioning of a world model**. Its unique outputs are target state RMSE and proposal reranking rather than policy success under calibration stress.
- `grounded_online_adaptation`, `local_residual_sim2real`, and `conflict_aware_replay` update a policy or residual from sequential target interactions and explicitly measure interaction efficiency, extrapolation, retention, or forgetting. This track instead uses a fixed few-shot demonstration pool, has no reward or continual task sequence, and does not claim online sample efficiency; the sweep is an action-representation comparison under matched data budgets.
- The timing/context tracks (`context_chunk_tradeoff`, `anticipatory_context_chunking`, and `streaming_action_denoising`) vary history, stale context, chunk handoff, or configured inference delay. This experiment has none of those mechanisms: candidate scoring is single-step and offline, so it uniquely avoids conflating action harmonization with execution scheduling.

## Sanity checks

`outputs/sanity_check.json` records seven deterministic checks. They verify:

- equal state + equal EE action gives exactly equal next state across embodiments;
- the same raw command gives substantially different source EE effects (`L2 = 1.187`);
- repeated data generation is bitwise deterministic;
- states remain finite;
- target action dimensionality differs;
- shot counts are sorted/unique;
- candidate pools are nontrivial.

All checks pass in the verified run.

## Outputs

- `outputs/metrics.json`: configuration, method definitions, headline rows, sanity checks, and all aggregate metrics.
- `outputs/summary_metrics.csv`: mean, standard deviation, and SEM by method and target-shot count.
- `outputs/per_seed_metrics.csv`: held-out prediction and reranking metrics for every seed/condition.
- `outputs/latent_diagnostics.csv`: per-seed latent explained variance and source-label accounting.
- `outputs/sanity_check.json`: deterministic mechanism checks.
- `outputs/few_shot_sweep.png`: target prediction and reranking sweep.
- `outputs/representation_diagnostics.png`: raw-control mismatch, shared latent coordinates, and target object motion.
- `outputs/candidate_reranking_example.png`: one held-out candidate-selection example.
- `docs/cross_embodiment_world_model_report.tex` and `.pdf`: source-grounded report.

## Simplifications and limitations

This toy predicts a four-number state, not pixels or videos. PCA replaces CLAP's learned variational latent-action model; ridge regression replaces a diffusion video backbone; exact simulator EE values replace noisy forward kinematics; and one-step reranking replaces receding-horizon policy planning. The source/target distributions are synthetic and close, observations expose full state, no contacts are discontinuous, and there are only three embodiments. The extreme 1% EE-label split is an experimental stress test, not a claim about CLAP's data mix.

Useful follow-ups are to vary EE-label fraction, corrupt target kinematics, use multi-step rollout loss, add image observations/occlusion, and compare uncertainty-aware reranking when the model is wrong.
