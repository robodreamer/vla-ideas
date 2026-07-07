# Prefix RL Chunking Toy

This folder contains a compact toy experiment for an RL + prefix-stabilized chunking idea: use RL to push a chunked manipulation policy past a conservative behavior-cloned demonstrator, while adding an explicit action-prefix loss so asynchronous chunk transitions stay smooth.

The demo is intentionally small. It is not a faithful implementation of Humanoid's stack, CFM training, real-robot safety systems, or PPO-at-scale. It isolates the part that is easiest to test in this repo style:

- a chunked action policy initialized by behavior cloning;
- PPO-style online improvement against sparse/generic rewards;
- a copied action prefix that represents the already-committed actions during async inference;
- an optional prefix-copy loss added to the PPO objective;
- a safety stop that terminates episodes when a crude force proxy crosses a threshold;
- CSV/plot artifacts for comparing BC, PPO-only, and PPO + prefix-loss variants.

## Why this idea

The motivating reference article argues that BC alone caps speed and quality at the demonstrator, while RL can optimize directly against the robot's dynamics and sparse task outcomes. It also highlights a specific stability issue for async action-prefix conditioning: under RL, rewards only apply to executed actions, so the model can stop copying the prefix slots; those off-distribution prefixes then feed back into future chunks and can destabilize training. The proposed fix is to add a masked prefix-CFM regression term to the PPO objective.

This toy mirrors that logic with a 1D reaching task plus an inactive joint:

```text
loss = PPO clipped policy loss + value loss + optional prefix-copy MSE
reward = sparse success + sparse recoverable-error/safety terms
```

The stochastic chunk sampler is a Gaussian stand-in for the article's CFM/ODE-to-SDE policy. It keeps the same RL bookkeeping shape: sampled chunks, old log probabilities, clipped probability ratios, a value baseline, and prefix regularization.

## References checked

Primary motivation:

- Humanoid, "KinetIQ Ascend: Towards 100% Manipulation Reliability and Superhuman Speed": `https://thehumanoid.ai/technology/kinetiq-ascend/`

Methods and papers referenced by that article:

- Schulman et al., "Proximal Policy Optimization Algorithms": `https://arxiv.org/abs/1707.06347`
- Lipman et al., "Flow Matching for Generative Modeling": `https://arxiv.org/abs/2210.02747`
- Liu et al., "Flow-GRPO: Training Flow Matching Models via Online RL": `https://arxiv.org/abs/2505.05470`
- Chen et al., "π RL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models": `https://arxiv.org/abs/2510.25889`
- McAllister et al., "Flow Matching Policy Gradients": `https://arxiv.org/abs/2507.21053`
- Bergmeister et al., "Reinforce Adjoint Matching": `https://arxiv.org/abs/2605.10759`
- Zhang et al., "ReinFlow": `https://arxiv.org/abs/2505.22094`
- Black et al., "Training-Time Action Conditioning for Efficient Real-Time Chunking": `https://arxiv.org/abs/2512.05964`
- de Haan et al., "Causal Confusion in Imitation Learning": `https://arxiv.org/abs/1905.11979`
- Physical Intelligence, "π0.5": `https://arxiv.org/abs/2504.16054`
- NVIDIA, "GR00T N1": `https://arxiv.org/abs/2503.14734`

Codebase references checked:

- Flow-GRPO codebase: `https://github.com/yifan123/flow_grpo`
  - `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py` shows an SDE step with tractable per-step log probability.
  - `scripts/train_sd3.py` contains the clipped ratio policy-loss pattern used for flow-model RL.
- Physical Intelligence OpenPI: `https://github.com/Physical-Intelligence/openpi`
  - `packages/openpi-client/src/openpi_client/action_chunk_broker.py` is a minimal action-chunk execution wrapper.
- NVIDIA Isaac-GR00T: `https://github.com/NVIDIA/Isaac-GR00T`
  - `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` describes an action head for a flow-matching diffusion policy and includes RTC/action-overlap handling.
  - `gr00t/data/state_action/action_chunking.py` contains action chunk data utilities.

## Run

Use any Python environment with `torch`, `numpy`, and `matplotlib`:

```bash
cd /home/andypark/Projects/repos/vla-ideas
python prefix_rl_chunking/run_prefix_rl_chunking.py
```

For a fast smoke test:

```bash
python prefix_rl_chunking/run_prefix_rl_chunking.py --quick
```

Outputs are written to `prefix_rl_chunking/outputs/`:

- `prefix_rl_chunking_metrics.csv`
- `prefix_rl_chunking_training_curves.csv`
- `prefix_rl_chunking_summary.png`

## Latest verified run

Using the local Python environment at `/home/andypark/Projects/hmnd-repos/hmnd/hmnd_robot/.pixi/envs/default/bin/python3.11`:

- BC reference: 100% success, 0% safety stops, 11.33 mean chunks, prefix MSE 0.1201.
- PPO only: 100% success, 0% safety stops, 7.00 mean chunks, prefix MSE 0.1195.
- PPO + prefix loss: 100% success, 0% safety stops, 7.68 mean chunks, prefix MSE 0.0306.

Interpretation: PPO pushes the conservative BC policy to finish faster. The prefix-loss variant gives up a little speed in this toy run but keeps the prefix-copy error about 4x lower, matching the blog's stability intuition.
