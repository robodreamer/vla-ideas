# Blog Outcome Mapping

This note maps the qualitative and quantitative claims in the motivating KinetIQ Ascend article onto the local toy examples. The toy is not meant to match the article's industrial scale, robot hardware, or absolute reliability targets; it checks whether the same *directions* of improvement appear in a controlled chunked-policy setting.

## Outcomes from the article

The article emphasizes four observable improvements from RL over BC:

1. **Speed / throughput:** RL turns higher execution speed into useful task throughput instead of failed grasps or retries.
2. **Reliability:** RL improves task success rate and reduces failures/timeouts.
3. **Robustness / long-tail behavior:** RL reduces slow or bad outliers; prefix regularization prevents action-prefix drift under async chunking.
4. **Safety-aware exploration:** safety-triggered terminations become part of the learning signal rather than behaviors to imitate.

## Toy metrics used here

The toy examples use analogous metrics:

- **Speed:** mean chunks to completion; lower is faster. For the 2D task, a BC timeout at the horizon is treated as the baseline cycle limit.
- **Reliability:** success rate and object-goal error.
- **Robustness:** success under reset jitter, safety-stop rate, and prefix-copy MSE.
- **Prefix stability:** prefix-copy MSE with PPO-only vs PPO + prefix loss.

## 1D reacher outcome

| method | success | safety stops | mean chunks | speed gain vs BC | prefix MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| BC reference | 1.000 | 0.000 | 11.33 | 1.00x | 0.1201 |
| PPO only | 1.000 | 0.000 | 7.00 | 1.62x | 0.1195 |
| PPO + prefix loss | 1.000 | 0.000 | 7.68 | 1.48x | 0.0306 |

Observed:

- RL keeps reliability at 100% while finishing faster.
- PPO-only gives the largest speed gain: 11.33 → 7.00 chunks, or **1.62x faster**.
- PPO + prefix loss is still **1.48x faster** than BC while reducing prefix-copy error by roughly **74% vs PPO-only**.

## 2D pick/place outcome

Latest verified full run uses deterministic dynamics with reset jitter, so the evaluation measures robustness across small pose/object variations.

| method | success | grasp | safety stops | mean chunks | object-goal error | prefix MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BC reference | 0.000 | 0.981 | 0.000 | 22.00 | 0.196 | 0.1048 |
| PPO only | 1.000 | 1.000 | 0.000 | 8.59 | 0.096 | 0.1324 |
| PPO + prefix loss | 0.988 | 0.988 | 0.000 | 8.68 | 0.078 | 0.0347 |

Observed:

- BC usually reaches the object and grasps it, but times out before reliable placement.
- PPO converts that partial skill into successful completion: **0% → 100% success** for PPO-only and **0% → 98.8%** for PPO + prefix loss.
- Mean episode length falls from the 22-chunk timeout horizon to about 8.6 chunks, a roughly **2.5x shorter cycle**.
- Object placement error falls by **51%** for PPO-only and **60%** for PPO + prefix loss vs BC.
- PPO + prefix loss reduces prefix-copy error by roughly **74% vs PPO-only**, while preserving nearly the same speed and success.
- Safety stops remain at 0% in the final policies, so the faster policy did not buy speed by violating the toy safety monitor.

## Interpretation

The toy now exhibits the same qualitative pattern as the article:

- **RL vs BC:** RL improves task completion and speed beyond the behavior-cloned demonstrator.
- **Speed and reliability together:** the richer 2D task improves both success and completion time, rather than trading one away.
- **Robustness:** evaluation with reset jitter shows the RL policies finishing across small pose/object variations where BC times out.
- **Prefix-loss benefit:** the explicit prefix regularizer gives a measurable stability gain: much lower prefix-copy error with only a small speed cost.

The main limitation is that the toy's BC baseline is intentionally weak on final placement, so the 2D success gain is more dramatic than the production numbers in the article. The useful comparison is directional: RL turns a partial BC skill into faster, more reliable task completion, and prefix regularization preserves the async chunking contract during RL.
