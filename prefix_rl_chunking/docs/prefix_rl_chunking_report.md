# Prefix RL Chunking Toy Report

## Setup

This experiment is a deliberately small analogy for RL-over-chunked-VLA setups that combine PPO, sparse rewards, flow-style stochastic policy sampling, async action prefixes, and an explicit prefix-copy regularizer.

- State: 1D end-effector position/velocity, one inactive joint, and goal delta.
- Action chunk: 14 steps, 2 action dimensions.
- Async prefix: 5 future actions copied from the previous chunk.
- Execution: first 4 actions of each chunk are executed before replanning.
- Policy: small MLP producing a Gaussian chunk distribution and a value estimate.
- BC pretraining: supervised imitation of a conservative PD chunker that copies the prefix.
- RL: clipped PPO objective with GAE.
- Prefix stabilization: optional MSE term on the prefix slots of the predicted chunk.
- Safety: episode terminates when a simple force proxy crosses a threshold.

## Result snapshot

Latest verified full run:

| method | success | safety stops | mean chunks | prefix MSE |
| --- | ---: | ---: | ---: | ---: |
| BC reference | 1.000 | 0.000 | 11.33 | 0.1201 |
| PPO only | 1.000 | 0.000 | 7.00 | 0.1195 |
| PPO + prefix loss | 1.000 | 0.000 | 7.68 | 0.0306 |

## Takeaway

The toy reproduces the qualitative tradeoff described in the motivating reference article:

- PPO can improve speed beyond the behavior-cloned reference policy.
- If RL is allowed to optimize only executed actions, copied-prefix fidelity is not protected.
- Adding a prefix-copy term keeps the async conditioning contract intact, at a small speed cost in this simple setup.

The toy does not implement true CFM velocity fields or real robot force/torque monitoring. The Gaussian chunk sampler is a compact proxy for the ODE-to-SDE/log-probability machinery used by flow-model RL implementations.
