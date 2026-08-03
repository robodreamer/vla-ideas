# Explorative Policy Chunks Toy Report

## Why this belongs in `vla-ideas`

Explorative Modeling argues that multimodality can be handled by factoring *training* instead of repeatedly factoring *generation*. The VLA-relevant version is an action-chunk policy: train K candidate chunks with a best-of-K reconstruction loss, then execute with one network forward pass instead of averaging all valid futures into an unsafe mean chunk.

This toy keeps the robotics part intentionally small. A point robot must move from left to right around a circular keep-out zone. Demonstrations contain two equally good modes, above and below the obstacle, while the policy input does not reveal which mode the demonstrator picked. A standard one-shot behavior-cloning policy trained with MSE therefore predicts the mean of the two modes, which cuts through the obstacle. An Explorative Policy-style multi-head chunker trains only the closest candidate head for each demo, so different heads can specialize to different route modes. The implementation deliberately includes small head-specific route biases and a weak anti-collapse term; the claim here is not that best-of-K alone always discovers modes from a symmetric initialization, but that best-of-K credit assignment can preserve seeded candidate futures instead of averaging them away.

## Mapping to the XM idea

- **K=1 / BC**: ordinary reconstructive behavior cloning; one generated chunk is compared to the expert chunk.
- **K>1 / XM**: the policy emits K candidate chunks; the loss backpropagates only through the candidate closest to the expert chunk. This mirrors Forward XM's best-of-K credit assignment, amortized here as K action heads rather than K separate latent samples. The tiny route-bias initialization is a toy stand-in for whatever stochasticity, latent conditioning, or architectural asymmetry seeds distinct futures in a larger model.
- **Single forward inference**: all candidate chunks come from one model call; a cheap geometric critic selects a feasible committed chunk for this toy. The critic is deliberately simple and only proves that the candidates include safe committed modes, not that the hidden demonstrator choice can be recovered from unobserved information.
- **Simplification**: the real paper studies images, video, language, robot manipulation, and world models. This script isolates only the multimodal action-chunk mechanism.

## Source notes

- Project page: https://explorative-modeling.github.io/
- Code repository: https://github.com/alexiglad/XM
- The repo README describes Forward XM as exploring K candidate outputs and backpropagating only through the closest candidate; it also notes that `K = 1` is the no-exploration baseline.

## Latest metrics

| method | success | collision | any safe cand. | oracle best MSE | chosen MSE | both-side coverage | side sep. | forwards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BC K=1 | 0.0% | 100.0% | 0.0% | 0.02730 | 0.02730 | 0.0% | 0.000 | 1 |
| XM K=2 | 100.0% | 0.0% | 100.0% | 0.00009 | 0.05445 | 100.0% | 0.677 | 1 |
| XM K=4 | 100.0% | 0.0% | 100.0% | 0.00008 | 0.05280 | 100.0% | 2.323 | 1 |
| XM K=8 | 100.0% | 0.0% | 100.0% | 0.00008 | 0.05597 | 100.0% | 2.313 | 1 |


## Result sanity checks

- The held-out demo route is intentionally hidden, so the selected safe chunk is not expected to match the demonstrator's arbitrary over/under choice. That is why `chosen MSE` can be worse than the K=1 average even when task success is perfect.
- `oracle best MSE` measures whether one of the emitted candidates matches the hidden route. It drops from K=1's averaged path to near-zero for K≥2.
- `both-side coverage` checks that the multi-head model is actually representing over/under alternatives, not just outputting duplicate safe curves. `side sep.` is only a rough diagnostic because extra heads for K=4/8 can drift into non-useful outlier routes.
- `any safe cand.` separates candidate expressivity from the toy critic, but it is not the strongest evidence here; `oracle best MSE` is the main mode-preservation check.
- Limitation: K is a candidate budget, not a guarantee that every head becomes a meaningful mode. In this two-mode toy, K=4/8 usually still reduce to two useful over/under candidates plus redundant or outlier heads. Without the route-bias initialization, this small deterministic toy often collapses back toward the averaged solution.

## Generated artifacts

- `outputs/metrics.json` and `outputs/metrics.csv`
- `outputs/exploration_k_sweep.png`
- `outputs/representative_trajectories.png`
- `outputs/training_curves.png`

## Takeaway

The useful distilled idea is not “use XM as-is for everything,” nor that best-of-K removes the need to seed diversity. It is: for VLA action chunks, best-of-K credit assignment plus a source of candidate asymmetry is a compact way to keep several plausible futures alive without paying diffusion-style iterative inference at deployment time. In this toy, K=1 blurs the two route modes; K≥2 with seeded heads can represent both over/under chunks and lets a light safety or task critic pick a committed trajectory.

## Run

```bash
cd /home/andypark/Projects/repos/vla-ideas
/home/andypark/Projects/repos/dhb-xr/.pixi/envs/default/bin/python explorative_policy_chunks/run_explorative_policy_toy.py --steps 900
```
