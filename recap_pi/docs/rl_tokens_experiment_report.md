# RECAP + RL Tokens Toy Experiment Report

This note records the current experiment status for the proposed "`RECAP + RL Tokens`" idea built on top of the existing `recap_pi` toy environments.

## Bottom line

The current toy implementations do **not** outperform offline RECAP. The experiments were still useful because they isolate where the proposed online correction module is failing:

- Offline RECAP is already strong on these toy tasks.
- The online recovery module helps some disturbed states but harms enough nominal or borderline states to lose overall.
- The recovery editor is not yet preserving the same long-horizon structure as the offline RECAP policy.

This means we do **not** currently have evidence for the claim that the newer idea is better than RECAP in these toy settings.

## Implemented experiments

### 2D disturbed navigation

Script:

- `recap_pi/recap_demo_rl_tokens_2d.py`

Outputs:

- `recap_pi/outputs/upgraded_2d_comparison.png`
- `recap_pi/outputs/upgraded_plain.gif`
- `recap_pi/outputs/upgraded_recap.gif`
- `recap_pi/outputs/upgraded_rl_tokens.gif`

Setup summary:

- Start from the 2D obstacle-navigation RECAP toy.
- Add shared velocity disturbances during evaluation.
- Compare:
  - plain behavior cloning
  - offline RECAP
  - RECAP plus an online recovery intervention during a post-disturbance window

Current verified results:

- Plain BC: `199` effective steps, `33%` success, `67%` collisions
- RECAP offline: `93` effective steps, `78%` success, `22%` collisions
- RECAP + RL Tokens: `116` effective steps, `67%` success, `33%` collisions

Delta vs RECAP:

- Success: `-11.1` points
- Effective steps: `-23.9` worse
- Collision rate: `-11.1` points worse

Interpretation:

- RECAP already recovers from many 2D disturbances on its own.
- The online recovery window is too blunt and tends to over-intervene.
- In a short-horizon 2D domain, handoff between a strong offline policy and an online correction module is fragile.

### 3D latent strategy adaptation

Script:

- `recap_pi/recap_demo_latent_adapt_3d.py`

Outputs:

- `recap_pi/outputs/latent_adapt_3d_comparison.png`
- `recap_pi/outputs/latent_plain_3d.gif`
- `recap_pi/outputs/latent_recap_3d.gif`
- `recap_pi/outputs/latent_adapt_3d.gif`

Setup summary:

- Keep the same disturbed 3D drone-heist task.
- Replace direct action residuals with a strategy-level latent style token.
- Train the policy on two successful modes:
  - a faster route
  - a safer higher-clearance route
- Let a learned gate decide whether to switch from the default fast style to the safer style after disturbance.

Current verified results:

- Plain BC: `270` effective steps, `12%` success, `62%` artifact completion, `38%` uplink completion, `88%` collisions
- RECAP fixed-style: `211` effective steps, `38%` success, `100%` artifact completion, `69%` uplink completion, `50%` collisions
- RECAP + latent adapt: `226` effective steps, `31%` success, `100%` artifact completion, `69%` uplink completion, `50%` collisions

Delta vs RECAP:

- Success: `-6.2` points
- Artifact completion: `+0.0` points
- Uplink completion: `+0.0` points
- Effective steps: `-15.0` worse
- Collision rate: `+0.0` points

Additional diagnostic:

- Gate dataset: `1920` disturbed states
- Positive style-switch labels: `0.1%`

Interpretation:

- Moving from action residuals to a strategy-level latent was the right structural test, but it still did not add useful information over RECAP in this toy.
- The gate rarely finds states where switching to the safe style beats simply continuing with the RECAP default style.
- This is stronger evidence than the earlier RL-token result: once RECAP already internalizes the useful long-horizon correction signal, even a cleaner high-level adaptation mechanism may have very little left to improve unless it brings new predictive information.

### 3D counterfactual planner

Script:

- `recap_pi/recap_demo_counterfactual_3d.py`

Outputs:

- `recap_pi/outputs/counterfactual_3d_comparison.png`
- `recap_pi/outputs/counterfactual_plain_3d.gif`
- `recap_pi/outputs/counterfactual_recap_3d.gif`
- `recap_pi/outputs/counterfactual_planner_3d.gif`

Setup summary:

- Keep the same disturbed 3D drone-heist task and latent-conditioned RECAP base policy.
- Train a small value model on the offline return labels.
- At disturbance time, run short imagined rollouts under candidate strategies and choose the better one using:
  - exact toy dynamics as a world model
  - a learned value bootstrap at the horizon
  - a state-aware bias that prefers the safer mode before uplink and the faster mode after uplink

Current verified results:

- Plain BC: `288` effective steps, `5%` success, `35%` artifact completion, `10%` uplink completion, `95%` collisions
- RECAP fixed-style: `192` effective steps, `45%` success, `100%` artifact completion, `75%` uplink completion, `45%` collisions
- RECAP + planner: `216` effective steps, `35%` success, `100%` artifact completion, `85%` uplink completion, `45%` collisions

Delta vs RECAP:

- Success: `-10.0` points
- Artifact completion: `+0.0` points
- Uplink completion: `+10.0` points
- Effective steps: `-23.5` worse
- Collision rate: `+0.0` points

Additional diagnostic:

- Safe-style planner interventions: `9`

Interpretation:

- This is the first variant that clearly adds some *new* useful information beyond local risk. The counterfactual planner improves the disturbed policy's ability to recover to the uplink stage.
- However, it still does not beat RECAP end to end. The planner is helping mid-horizon subgoal completion without converting that gain into full extract success.
- In other words, world-model-style lookahead appears directionally correct, but the current planner still lacks the right objective or option set for full task completion.

### 3D disturbed drone-heist

Script:

- `recap_pi/recap_demo_rl_tokens_3d.py`

Outputs:

- `recap_pi/outputs/upgraded_3d_comparison.png`
- `recap_pi/outputs/upgraded_3d_plain.gif`
- `recap_pi/outputs/upgraded_3d_recap.gif`
- `recap_pi/outputs/upgraded_3d_rl_tokens.gif`

Setup summary:

- Start from the 3D drone-heist task: artifact -> uplink -> extract.
- Add shared velocity disturbances during evaluation.
- Compare the same three modes as in 2D.

Current verified results:

- Plain BC: `285` effective steps, `6%` success, `44%` artifact completion, `6%` uplink completion, `94%` collisions
- RECAP offline: `180` effective steps, `50%` success, `100%` artifact completion, `69%` uplink completion, `38%` collisions
- RECAP + RL Tokens: `195` effective steps, `44%` success, `94%` artifact completion, `62%` uplink completion, `50%` collisions

Delta vs RECAP:

- Success: `-6.2` points
- Artifact completion: `-6.2` points
- Uplink completion: `-6.2` points
- Effective steps: `-15.1` worse
- Collision rate: `-12.5` points worse

Interpretation:

- The 3D task gives the online module more room to matter, but the current recovery controller still underperforms RECAP.
- The main failure mode is subgoal disruption: once the online controller activates, it can repair local motion but still weakens the global sequence that RECAP had learned.
- This is visible in the lower uplink completion rate. The online module is not preserving the artifact -> uplink -> extract structure as reliably as the offline conditioned policy.

## Why the newer idea is not better yet

The right conclusion from the current experiments is not that online correction is useless. It is that the **current toy implementation** is not yet the right formulation.

The likely reasons are:

1. RECAP already solves much of the disturbance response by itself.
   The baseline is strong, so the online module needs to be highly selective to help rather than interfere.

2. The action-level editor was too local, and the latent-style switch still did not add enough new information.
   The residual controller reacts to near-term disturbance states, while the latent switch changes mode more cleanly, but neither one improves enough long-horizon outcomes to beat RECAP.

3. Counterfactual lookahead helps, but the current planner is only partially aligned with the real objective.
   The planner improves uplink completion in 3D, which suggests imagined futures are the right direction, but it still does not optimize the full artifact -> uplink -> extract sequence well enough to surpass RECAP.

4. Recovery supervision is misaligned with deployment.
   The hand-designed recovery targets and even the safer latent mode make sense intuitively, but they are not necessarily the same choices a globally optimal RECAP policy would make from disturbed states.

5. There may be almost no beneficial intervention mass under the current intervention space.
   The latent experiment's `0.1%` positive gate-label rate is the clearest sign: the problem may not just be optimization, but that the available intervention options rarely improve on RECAP at all.

## What would need to change to beat RECAP credibly

The next versions should focus on:

1. A gated editor trained with explicit "intervene / do not intervene" supervision.
   The current recovery window is too coarse.

2. A long-horizon objective for the online module.
   Especially in 3D, the editor should optimize downstream subgoal completion, not just immediate stabilization.

3. A stronger source of new information.
   Counterfactual world-model rollouts are the most promising direction so far, but they likely need richer options than a fast-vs-safe style switch and a terminal objective that explicitly rewards successful extraction rather than just local recovery.

4. Perturbation-centric training data.
   If the online method is supposed to help under disturbances, the training distribution should emphasize exactly those off-trajectory states.

## Current conclusion

For the present toy codebase, the evidence is:

- RECAP clearly beats plain imitation.
- The attempted "`RECAP + RL Tokens`" extension does **not** yet beat RECAP.

That is still a useful result. It tells us the bar for improving over offline RECAP is higher than simply attaching a local online recovery module.
