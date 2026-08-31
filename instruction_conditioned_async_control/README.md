# Instruction-Conditioned Asynchronous Control Toy

This package isolates the planner/controller scheduling mechanism behind **Instruct-to-Act**: a sparse, slow high-level instruction planner drives a high-frequency language-conditioned controller in a partially observed sequential environment.

It is a **mechanism toy, not a Dreamer, VLM, VLA, or paper reproduction**. There are no pixels, pretrained language/vision model, RSSM, latent imagination, actor–critic training, GPU timing measurements, or real robot. Planner latency and error are configured; the controller is transparent ridge regression.

## Sources checked

Primary sources:

- Paper: <https://arxiv.org/abs/2608.26788>
- Project page: <https://zinengtang.github.io/instruct-to-act/>
- Official code: <https://github.com/zinengtang/instruct-to-act-code> (inspected at `f46a12b2cfaece7f76e789a11c2eef58af33b31d`)

The source method separates a pretrained VLM planner from an environment-specific, language-conditioned world-model controller; relabels segments of controller rollouts with instructions for behavior cloning; and supports asynchronous online planning while control continues. The official code exposes the same concepts in `inference/algorithms.py` and `training/annotator.py`.

Source-reported task scores, model sizes, and throughput are context only. None are local measurements.

## Toy environment and learned controller

A continuous 2-D point agent must visit a hidden ordered route of three or four named stations (`amber`, `cobalt`, `mint`, `depot`). The controller receives:

- proprioception (position and velocity),
- station-relative vectors,
- nearby moving-hazard information only within a sensing radius,
- one sparse text instruction.

It does **not** observe the hidden route, route stage, or future hazard motion. Actions are 2-D accelerations at every 80 ms environment tick.

The script first generates autonomous analytical-expert rollouts without language labels, segments completed behavior at station boundaries, and only then attaches one of four instruction dialects to every step in each completed segment. It then fits three closed-form ridge controllers:

| Capacity | Labeled steps | Features | Held-out action MSE |
| --- | ---: | --- | ---: |
| Small | 900 | target delta + velocity | 0.1441 |
| Medium | 4,200 | + local hazard avoidance and distance terms | 0.0892 |
| Large | 9,234 | + velocity/distance and hazard-proximity terms | 0.0213 |

This is an inspectable behavior-cloning analogue of post-hoc instruction supervision, not world-model RL.

## Compared methods

- `controller_only`: self-operating learned policy with a modal route prior and no planner calls.
- `direct_blocking`: an intentionally stressed foil in which the slow planner directly emits open-loop low-level action chunks; each call blocks control. It includes an explicit deterministic action-token/precision perturbation (`0.85` acceleration units) and an extra target-error probability (`+0.18`) to model the burden of using a high-level planner as a high-rate actuator. This is not an Instruct-to-Act baseline and is not a capacity-matched comparison.
- `sync_instruction`: the learned controller acts at control frequency, but waits at every instruction boundary for the planner.
- `async_online`: after one cold-start instruction, the controller keeps acting while a sparse background request prepares the next instruction. Late instructions cause measured stale execution rather than blocking.
- `oracle`: zero-latency correct stage instructions plus the analytical expert controller.

The central scheduling comparison is `sync_instruction` versus `async_online`: they use the same learned controller, planner profile, route/disturbance trace, planner-error random draws, and per-tick controller-noise draws. They differ in whether instruction planning blocks control. The asynchronous default additionally includes the disclosed four-tick transport delay, so it is not given a latency advantage. The direct-action foil and analytical oracle answer different diagnostic questions and should not be ranked as matched learned baselines.

## Relation to other experiments in this repository

- `streaming_action_denoising` (FlashVLA) decouples work **inside the low-level action decoder**: chunks at staggered denoising levels advance together and a cleaner chunk conditions noisier future chunks. This package decouples a **sparse semantic planner from a high-rate controller**. It has no denoising buffer, flow time, or decoder-level causal attention; the two mechanisms could be stacked.
- `async_chunking_compare` and `anticipatory_context_chunking` compensate prediction-to-execution delay by estimating the robot state, and in the latter case the changing observation/environment latent, at chunk handoff. Here the controller observes the current toy state every tick; the delayed object is the next semantic instruction, not the sensor context used by an action chunk.
- `context_chunk_tradeoff` and `openvla_oft_systems_toy` vary observation-history length, action-chunk horizon, parallel action availability, or refresh cadence. This experiment holds the controller at one action per tick and varies sparse instruction scheduling; an instruction's dwell time is not an open-loop action horizon.
- `turbo_vla_direct_control` studies a compact direct vision+language-to-action path that removes a large execution-time language-model bottleneck. This toy intentionally retains a slow high-level language planner and asks whether a separate compact controller can absorb that planner's latency.
- `prefix_rl_chunking` regularizes a policy to copy already committed action prefixes during asynchronous chunk replacement. This package has no committed action-prefix input, diffusion/flow policy, or RL update; its continuity failure is stale semantic intent rather than prefix drift.
- `demo_prompted_policy` and `video_prompt_shortcut_resistance` study whether a demonstration/video prompt specifies task intent and whether training prevents history/text shortcuts. The present parser receives explicit station words and tests only a narrow shared semantic interface, not learned prompt grounding or correspondence.

## Metrics

- route success;
- completion steps/time, with failures charged the full horizon;
- idle planning steps/fraction;
- stale-instruction ticks: active instructions intended for an already completed stage;
- wrong-target ticks: stale instructions, planner mistakes, or parse/interface failures;
- control smoothness: accumulated action change per active tick;
- safety violations: moving-hazard contacts and wall hits;
- planner calls and path length.

## Run and verify

Full deterministic paired evaluation used for the checked outputs:

```bash
cd /home/andypark/Projects/repos/vla-ideas
PYTHONWARNINGS=error MPLBACKEND=Agg \
  /home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  instruction_conditioned_async_control/run_instruction_conditioned_async_control.py \
  --seed 41 --trials 64 --episode-steps 260
```

Smoke run without overwriting checked outputs:

```bash
cd /home/andypark/Projects/repos/vla-ideas
rm -rf /tmp/instruction-conditioned-async-control-smoke
PYTHONWARNINGS=error MPLBACKEND=Agg \
  /home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  instruction_conditioned_async_control/run_instruction_conditioned_async_control.py \
  --smoke --trials 4 --episode-steps 120 \
  --output-dir /tmp/instruction-conditioned-async-control-smoke --no-report
```

Render and verify the real PDF/artifacts:

```bash
cd /home/andypark/Projects/repos/vla-ideas
./instruction_conditioned_async_control/docs/render_pdf.sh
PYTHONWARNINGS=error \
  /home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python \
  instruction_conditioned_async_control/verify_package.py
```

## Verified default results

The full evaluation uses 64 paired trials per method with identical routes, starts, moving hazards, and disturbance traces.

| Method | Success | Steps | Idle | Stale ticks | Wrong-target ticks | Smoothness | Safety | Planner calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Controller only | 15.6% | 230.5 | 0.0 | 0.0 | 207.1 | 0.384 | 21.70 | 0.0 |
| Direct/blocking planner actions | 56.2% | 225.1 | 95.5 | 17.3 | 43.2 | 1.236 | 27.62 | 9.6 |
| Synchronous instructions | **84.4%** | 139.3 | 33.4 | 0.0 | 32.0 | 0.375 | 15.41 | 3.4 |
| Asynchronous online | **84.4%** | **121.6** | **9.0** | 7.7 | 41.1 | **0.331** | **8.31** | 3.5 |
| Oracle | 100.0% | 78.8 | 0.0 | 0.0 | 0.0 | 0.408 | 7.78 | 3.6 |

![Method comparison](outputs/method_comparison.png)

The default toy does **not** make asynchronous planning universally best. With common planner-error and controller-noise draws, synchronous and asynchronous instructions tie at `84.4%` success. Asynchronous planning has `17.8` fewer horizon-capped completion steps, cuts idle time by `24.4` ticks, and reduces the toy safety-violation count from `15.41` to `8.31`, but incurs `7.7` stale-instruction ticks and more wrong-target execution. The point is the mechanism/trade-off: asynchrony converts blocking into bounded stale execution rather than removing planner error or latency.

## Sweeps

The fuller paired evaluation includes:

- planner latency: `0, 6, 12, 24` ticks for direct, synchronous, and asynchronous methods;
- asynchronous planning cadence: `4, 8, 16, 32` ticks;
- extra instruction transport/staleness delay: `0, 4, 8, 16` ticks;
- controller capacity × action noise: `small/medium/large × 0.00/0.07/0.16`;
- planner swaps: balanced, strong/slow, fast/terse, held-out semantic synonyms, and an opaque cipher negative control.

Selected findings:

- At 24 planner-latency ticks, direct generation falls to `12.5%` success with `165.8` idle ticks; asynchronous control remains at `90.6%` in the 32-trial paired sweep with `23.0` cold-start idle ticks, though stale ticks rise to `25.8`.
- Increasing asynchronous cadence from 4 to 32 ticks leaves planner calls sparse (`3.47`) but increases horizon-capped completion from `102.0` to `128.7` steps and stale execution from `0.0` to `27.0` ticks.
- Increasing extra transport delay from 0 to 16 ticks raises stale execution from `2.8` to `27.1` ticks and safety violations from `7.50` to `9.28`.
- The feature/data-capacity sweep is most visible in safety: at zero added action noise, the small controller averages `11.22` violations versus `7.50` for the large controller. This is not a clean capacity-only ablation because feature set and sample count change together.
- With shared planner-error draws, the held-out-synonym and balanced planners both reach `90.6%` success in the 32-trial swap sweep, while the opaque cipher interface gets `0%`. This checks parser/interface compatibility, not open-ended language generalization.

![Mechanism sweeps](outputs/mechanism_sweeps.png)

![Representative rollout](outputs/representative_rollout.png)

## Outputs

- `outputs/trial_metrics.csv`: 320 default per-trial rows.
- `outputs/summary_metrics.csv`: method mean/SEM aggregates.
- `outputs/sweep_trial_metrics.csv`: all fuller paired sweep trials.
- `outputs/sweep_summary.csv`: sweep mean/SEM aggregates.
- `outputs/metrics.json`: config, exact command, profiles, summaries, training data, and claim boundary.
- `outputs/training_metrics.json`: ridge features, weights, sample counts, and train/test MSE.
- `outputs/instruction_relabeling_segments.csv`: post-hoc segment labels and action summaries.
- `outputs/sanity_checks.json`: deterministic mechanism checks.
- `outputs/method_comparison.png`, `mechanism_sweeps.png`, `representative_rollout.png`.
- `docs/instruction_conditioned_async_control_report.tex` and rendered `.pdf`.

## Important limitations

- The task and parser make language grounding far easier than real vision-language control.
- Planner latency/error are configured knobs, not measured model/API performance.
- The direct baseline's action and target errors are explicit toy assumptions; different values change its ranking.
- Controller ``capacity'' sweeps jointly change feature set and training-sample count, so they are not a clean parameter-count ablation.
- The learned controller is feed-forward ridge regression, not recurrent state estimation or a learned world model.
- Safety, staleness, and smoothness are local proxies in a navigation-like single-agent task.
- There is no multi-agent communication, long-horizon crafting, contact-rich manipulation, or real-time operating-system concurrency.
