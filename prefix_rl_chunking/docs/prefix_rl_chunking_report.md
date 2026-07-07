# Prefix RL Chunking Toy Report

## Setup

This folder contains two deliberately small analogies for RL-over-chunked-VLA setups that combine PPO, sparse rewards, flow-style stochastic policy sampling, async action prefixes, and an explicit prefix-copy regularizer.

Shared ingredients:

- Chunked action policies initialized by behavior cloning.
- Gaussian chunk sampling as a compact stand-in for flow-policy ODE-to-SDE exploration/log-probability machinery.
- Clipped PPO with GAE and a learned value baseline.
- Async action prefixes copied from the previous chunk.
- Optional prefix-copy loss on the prefix slots of the predicted chunk.
- Sparse success/safety rewards plus small shaping terms for stable toy training.

The original 1D script uses a reacher plus inactive-joint drift probe. The 2D extension below adds object interaction, obstacle avoidance, and gripper actions.

## 2D pick/place extension

`run_prefix_rl_pickplace_2d.py` adds a more fully fledged toy than the 1D reacher:

- 2D end-effector position/velocity dynamics.
- A graspable object and target placement region.
- A rectangular obstacle and force/safety termination proxy.
- Three-dimensional action chunks: x/y acceleration plus gripper command.
- BC initialization from a conservative staged pick/place demonstrator.
- PPO fine-tuning with optional action-prefix regularization.
- Trajectory plot showing end-effector and object paths.

Latest verified 2D run:

| method | success | grasp | safety stops | mean chunks | object-goal error | prefix MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BC reference | 0.000 | 1.000 | 0.000 | 22.00 | 0.183 | 0.0995 |
| PPO only | 1.000 | 1.000 | 0.000 | 8.00 | 0.041 | 0.1020 |
| PPO + prefix loss | 1.000 | 1.000 | 0.000 | 9.00 | 0.068 | 0.0308 |

## 1D result snapshot

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
