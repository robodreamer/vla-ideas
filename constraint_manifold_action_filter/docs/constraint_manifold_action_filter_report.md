# Constraint-Manifold Action Filtering: A PR-MPPI-Inspired Toy

## Question

Can explicit equality-tangent and inequality-half-space geometry make aggressive learned action chunks safe, and why is finite-step retraction still needed?

## Source boundary

PR-MPPI projects each sampled rollout velocity into the equality tangent and active inequality half-spaces, filters the nominal horizon again, and retracts the executed finite step onto the equality manifold. This toy isolates that execution-layer geometry around a BC chunk. It does not reproduce MPPI sampling, weighting, robot kinematics, or the paper's CUDA implementation.

## Setup

A 2D tray has center `(x,y)` and orientation vector `(a,b)` constrained by `a²+b²=1`. Sampled points along the tray must keep positive differentiable clearance from circular obstacles and remain inside the workspace. An Extra Trees behavior clone supplies seven-step chunks, then seeded gains, centerline shortcuts, noise, and radial orientation bias create aggressive and OOD proposals.

## Result

Across 120 paired scenarios, penalty and one-step methods reach the goal but violate the equality and collide in roughly one third of trials. Rollout projection removes measured inequality violations but leaves second-order finite-step equality drift. Projection plus retraction reduces p95 equality residual to `9.69e-11`, keeps zero measured inequality violation, and achieves 63.3% full success. Its OOD success falls to 31.7%, with local infeasibility and deadlock remaining visible.

## Follow-ups

1. Add actual sampled MPPI candidate weighting and nominal-horizon refiltering.
2. Compare active-set, Dykstra, QP, and GPU-batched projection solvers.
3. Add relaxed/slack constraints and a hierarchy for infeasible states.
4. Learn constraint functions or uncertainty-aware safety margins.
5. Replace coordinate BC with image-conditioned action chunks.
