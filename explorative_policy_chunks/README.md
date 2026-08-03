# Explorative Policy Chunks

This folder distills the Explorative Modeling / XM idea into a VLA action-chunk toy.

The core question: if robot demonstrations contain multiple valid futures, can a one-step action policy avoid the MSE “average trajectory” failure without paying diffusion-style iterative inference? The toy trains:

- `BC K=1`: a standard one-shot behavior-cloning chunk policy.
- `XM K>1`: a multi-candidate chunk policy trained with a best-of-K reconstruction loss.

A point robot must route around a circular keep-out zone. Demonstrations go either above or below the obstacle, but the policy context does not reveal which route the demonstrator used. The K=1 policy tends to blur the two modes through the obstacle. The best-of-K policy, with small head-specific route biases and a weak anti-collapse regularizer, can keep committed chunks alive in one network forward pass; a tiny feasibility critic then selects a safe candidate.

The limitation is explicit: this is **not** evidence that best-of-K alone discovers modes from a perfectly symmetric initialization. It demonstrates the core credit-assignment mechanism once candidate diversity is seeded, and shows why a single averaged BC chunk is the wrong abstraction for ambiguous futures.

Run:

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python explorative_policy_chunks/run_explorative_policy_toy.py --steps 900
```

Outputs are written to `explorative_policy_chunks/outputs/`:

- `metrics.json` / `metrics.csv`
- `exploration_k_sweep.png`
- `representative_trajectories.png`
- `training_curves.png`

The generated writeup lives at:

- `explorative_policy_chunks/docs/explorative_policy_toy_report.md`

## Notes

This is not a faithful reproduction of the full XM codebase or paper results. It is a small VLA-focused analogy for the repo: best-of-K credit assignment plus seeded candidate asymmetry can raise action-chunk expressivity, reduce mode averaging, and preserve one-forward-pass deployment. Extra K heads are a candidate budget, not a guarantee that every head becomes a meaningful route mode.
