# R2R verification and campaign

All generated outputs live below `experiments/r2r/results/` and are ignored by
Git. JSONL metrics, checkpoints, cache mirrors, and provenance files are the
source of record. Jobs use the official R2I Apptainer image at
`.containers/r2i.sif`, L40S GPUs for integration/ToyMemory/BSuite, and H100 GPUs
for Memory Maze.

## Local correctness suite

```bash
apptainer exec --cleanenv \
  --bind "$PWD:$PWD" --pwd "$PWD" .containers/r2i.sif \
  python -m unittest discover -s tests -v
```

The suite covers exact window/batch mappings, update cadence, BF16 complex
round trips and memory accounting, cache boundaries/generations/duplicates,
wraparound and mirrored persistence, fixed-seed sampling parity, a full-BPTT
gradient oracle, reset masking, and one-primal/one-reverse behavior.

## GPU integration smokes

```bash
sbatch experiments/r2r/slurm_integration.sh
```

The two array tasks exercise the production replay/cache/learner path at
`64x64` and `1024x4`, including a 4096-transition prefill and two learner
updates.

## Staged ToyMemory gate

Run this only from the pushed commit:

```bash
experiments/r2r/submit_toy_gate.sh
```

The controller launches the 256-transition job only if the 128-transition job
created `SUCCESS`. Each uses window 64, batch 64, seed 0, a one-million-step
uniform replay, 4096 random-prefill transitions, train ratio 1024, and balanced
128-episode greedy evaluation every 1000 environment steps. Success requires
exactly 100% actor accuracy and exactly 100% model reward-choice accuracy before
25,000 steps.

## Cue/query distance diagnostic

Submit independent seed-0 jobs at literal cue-to-query distances 8, 16, 32,
and 64:

```bash
experiments/r2r/submit_toy_distances.sh
```

All four jobs use window 64, batch 64, the production uniform replay and BF16
state-adjoint cache, 4096 random-prefill transitions, train ratio 1024, and the
same balanced 128-episode evaluation and strict early-stop gate as the staged
ToyMemory runs. ToyMemory's terminal transition follows the query, so these
distances correspond to episode sizes 10, 18, 34, and 66 respectively. The
first three dependencies fit inside the learner window; distance 64 does not.

## Matched distance-8 mechanism comparison

After changing learner/reset semantics, run all three controls for the full
25,000-step budget from one pushed commit:

```bash
experiments/r2r/submit_toy_compare.sh
```

The array contains a random-policy world-model-only arm, a full-agent direct
BPTT arm, and the full-agent BF16 state-adjoint R2R arm. All use seed 0,
distance 8, `T=64`, `B=64`, 4096 random-prefill transitions, uniform replay,
and train ratio 1024. Evaluation reports deterministic-prior reward choices as
well as the mean predicted reward margin between the correct and incorrect
answer, so partial model learning remains visible below the 100% gate. These
comparison arms do not stop early.

## Reward-acquisition and R2I-window controls

Run the balanced terminal-reward falsifier and cache-disabled R2I `1024x4`
window control together:

```bash
experiments/r2r/submit_toy_controls.sh
```

Both arms use distance 8, seed 0, 4096 random-prefill transitions, uniform
replay, train ratio 1024, 25,000 interactions, and exactly 5226 learner
updates. The falsifier uses direct BPTT at `64x64`, collects random actions,
and optimizes an equal-class mean over terminal +1/-1 reward rows only. It
reports teacher-forced posterior reward accuracy separately from imagined
one-step model choice. The R2I arm uses the native objective and full agent at
`1024x4` with state-gradient caching disabled. The configured one-million-row
replay differs from R2I's historical capacity but cannot bind in a 25k run.

Continue the cache-disabled `1024x4` control from its 25k checkpoint to 100k
total interactions with:

```bash
experiments/r2r/submit_toy_r2i_long.sh
```

The continuation reuses the checkpoint and persistent replay sidecar in place,
keeps the original JSONL history, records separate continuation provenance,
and reports the cumulative expected 23,976 learner updates. Override
`R2R_SOURCE_CAMPAIGN` or `R2R_TARGET_STEPS` when resuming another campaign.

If native reward acquisition remains at chance, run the first minimal
acquisition probe with:

```bash
experiments/r2r/submit_toy_reward10_wm.sh
```

This keeps the native world-model loss, uniform replay, `64x64` batch shape,
random behavior, optimizers, and train ratio fixed. Its sole scientific change
is `loss_scales.reward: 10`; it runs distance 8 for 50,000 interactions and
11,476 learner updates. It is a diagnostic gate before testing the same scale
with the actor or state-gradient cache.

If reward scale 10 remains at chance through 50k, run the predeclared next
single-variable value with:

```bash
R2R_REWARD_SCALE=32 experiments/r2r/submit_toy_reward10_wm.sh
```

If both global reward-scale probes remain at chance, run the targeted
rare-terminal-row probe with:

```bash
experiments/r2r/submit_toy_terminal_weight_wm.sh
```

This launches normalized terminal weights 10 and 100 as separate seed-0
world-model-only arms. Both retain uniform replay and every native R2I
world-model term. Only the relative weighting of terminal versus nonterminal
rows inside the reward loss changes; normalization keeps the mean reward-loss
weight at one. These remain diagnostic acquisition arms, not production R2R
defaults.

If normalized row weighting still misses the positive control's acquisition
transition, test whether native auxiliary-gradient competition is the remaining
bottleneck with:

```bash
experiments/r2r/submit_toy_weighted_reward_wm.sh
```

This crosses terminal weights 10 and 100 with `loss_scales.reward: 10`. The
earlier scale-only and weight-only arms identify the interaction: all native
losses and uniform replay remain present, while terminal reward gradients now
dominate both within the reward term and relative to auxiliary model terms.

If that interaction still fails, isolate auxiliary-gradient interference with:

```bash
experiments/r2r/submit_toy_auxiliary_ablation.sh
```

This is a 25k, seed-0, world-model-only 2x2 factorial. Every arm uses the exact
balanced terminal reward objective; KL (`dyn+rep`) and
reconstruction/continuation losses are independently enabled or disabled. The
reward-only cell must reproduce the earlier positive control, while the other
three cells identify the smallest native auxiliary group that prevents cue
acquisition. Replay sampling and all optimizer settings remain fixed.

After the reward-only oracle passes and every auxiliary cell fails, promote the
proven acquisition objective through the full agent with:

```bash
experiments/r2r/submit_toy_reward_only_promotion.sh
```

This runs matched 50k direct-BPTT and full-R2R arms at distance 8. Both train
actor and critic normally; the world-model scalar is the balanced terminal
reward objective that passed the acquisition audit. The full-R2R arm enables
the dense BF16 state/adjoint cache, while the direct arm is cache-disabled.
This is an explicit ToyMemory diagnostic profile, not the unchanged native R2I
objective.

If the reward-only world model learns but the actor exploits repeated imagined
post-query rewards, audit continuation training with:

```bash
experiments/r2r/submit_toy_continuation_audit.sh
```

Both 25k world-model-only cells retain the proven balanced reward objective and
restore only the continuation loss. `cont_shared` uses the native shared-core
gradient; `cont_detached` trains the same continuation head on stop-gradient
features so it cannot erase cue memory. This distinguishes whether continuation
can be restored faithfully or requires gradient isolation before another
full-agent promotion.

If either 25k continuation cell is still pre-transition, extend both existing
checkpoints to 50k with:

```bash
experiments/r2r/submit_toy_continuation_extend.sh
```

The continuation preserves model/optimizer counters and the fixed evaluation
grid. For these T=64 arms, the durable mirror contains completed 64-row chunks:
a 25k source therefore restores through 24,960 and loses up to 40 partial-tail
replay rows. The launcher records this qualification explicitly; replay remains
uniform.

After shared continuation retains model acquisition, promote it through matched
full-agent runs with:

```bash
experiments/r2r/submit_toy_cont_promotion.sh
```

These 50k distance-8 arms retain the exact balanced terminal reward plus the
native continuation loss. KL and observation reconstruction remain disabled by
the preceding factorial. Direct BPTT is cache-off; full R2R uses the dense BF16
state/adjoint cache. Actor-plus-model retention determines whether learned
termination removes the repeated-reward imagination exploit.

The native-scale sparse-prior pair failed, the dynamics-scale-zero direct-BPTT
oracle passed, and the `0.05` full-R2R distance-8 arm subsequently retained
exact actor and model accuracy through 60k. The matched direct-BPTT `0.05` arm
remained at chance, while a second `0.005` full-R2R run independently retained
the same distance-8 solution by 50k. Promote a passing scale to distances 16
and 32 with:

```bash
experiments/r2r/submit_toy_cont_distances.sh
```

Both jobs remain `T=64`, `B=64`, seed 0, use the equal-class sparse-reward
objective, and run 60k interactions by default. The submission gate rechecks
the retained distance-8 source. It also creates a detached Git worktree at the
exact submitted commit so queued array tasks cannot silently pick up later
checkout edits. Each distance writes `ROBUST_SUCCESS` only after five
consecutive balanced 128-episode panels have actor accuracy 1.0, model
reward-choice accuracy 1.0, and finite model margin of at least 0.1, with the
exact gate still passing at the requested endpoint. No replay, cache,
optimizer, sampling, imagination, batch, or window setting changes.
`R2R_TARGET_STEPS`, `R2R_DYN_SCALE`, and `R2R_SEED` are explicit overrides for
continuations or replication campaigns.

The seed-0 `0.05` promotion solved distance 32 within 60k. At distance 16 its
world model solved by 23,096, but its actor remained at chance through 60k; an
additional-time-only continuation first solved the actor at 71k and retained
the joint solution through 100k. Because `0.005` solved distances 8, 16, and 32
within one common 60k budget, that 100-times-reduced dynamics scale is the
selected fast replication profile. This is a ToyMemory acquisition diagnostic,
not a native-R2I-objective result: reward-class balancing, zero representation
and reconstruction scales, and the reduced dynamics scale must be reported as
scientific deviations.

After all three seed-0 runs pass that retained gate, launch the exact same
60k profile for seeds 1 and 2 at distances 8, 16, and 32:

```bash
experiments/r2r/submit_toy_sparse_seed_audit.sh
```

The submission validates every seed-0 summary, config, clean provenance, and
final retention streak before allocating GPUs. Its defaults select dynamics
scale `0.005`, the retained 50k distance-8 source, the retained 60k
distance-16/32 campaign, and immutable training snapshot `421666e`. Override
`R2R_DYN_SCALE`, `R2R_DISTANCE8_SOURCE`, `R2R_DISTANCE8_FINAL_STEP`,
`R2R_SEED0_CAMPAIGN`, or `R2R_TRAINING_COMMIT` only when promoting a different
fully qualified profile.

If the cached distance-8 world model solves but its actor remains at chance at
50k, continue that exact checkpoint and objective without changing any
hyperparameter:

```bash
R2R_SOURCE_CAMPAIGN=CAMPAIGN \
  experiments/r2r/submit_toy_cont_full_extend.sh
```

The default target is 100k cumulative interactions. Model, actor, critic,
optimizer variables, counters, uniform replay, and dense cache are restored;
agent, replay-sampler, and environment RNG streams restart, so this is not a
trajectory-identical continuation. The durable T=64 mirror can replace at most
one incomplete 64-row tail after a crash. The continuation records retention
separately rather than treating more training as a hidden hyperparameter
change.
The continuation also logs scalar summaries of imagined continuation and
trajectory weights; these are read-only diagnostics for detecting returns that
continue past the learned terminal transition.

For a phase-conditioned snapshot from any saved ToyMemory checkpoint, run
`probe_toy_imagination.py` on a Slurm GPU allocation. It loads the checkpoint's
exact adjacent `config.yaml` and reports actor actions, one-step reward and
continuation, critic returns, and imagination weights for each real episode
phase without updating the checkpoint. The default eight stochastic draws per
state expose sampled-prior failures that a deterministic reward-choice metric
can hide.

If that probe shows unconstrained nonterminal rewards and an untrained sampled
prior, run the matched sparse-prior repair:

```bash
experiments/r2r/submit_toy_sparse_prior_promotion.sh
```

This keeps uniform replay, `T=64`, `B=64`, continuation, optimizers, actor,
critic, and imagination settings fixed. Its reward objective gives equal weight
to the three real ToyMemory reward classes (nonterminal 0, terminal +1, and
terminal -1), and restores only the native dynamics KL at scale 0.5 so the
sampled prior used by actor imagination is trained. Representation KL and
reconstruction stay disabled because the earlier acquisition factorial found
that those gradients suppress the cue. The two 50k arms are direct BPTT and
full BF16 state-adjoint R2R from the same reset-corrected commit.

The native-scale pair at campaign `20260825T233154Z` completed all 50k steps,
but even direct BPTT remained at chance. This rules out attributing that result
to the cache and makes dynamics scale 0.5 the next acquisition confound to
isolate. Run the bounded scale falsifier with:

```bash
experiments/r2r/submit_toy_sparse_dyn_falsifier.sh
```

It runs direct BPTT at dynamics scales 0 and 0.05 plus full R2R at 0.05. The
zero-scale arm tests whether the three-class reward objective itself still
acquires the cue; the matched 0.05 arms test the smallest native-mechanism
repair in both direct and cached training. Model-only and joint five-panel
retention reports are written separately.

If the zero-scale oracle acquires but 0.05 blocks the matched direct control,
run the next tenfold scale reduction with:

```bash
R2R_DYN_SCALE=0.005 experiments/r2r/submit_toy_sparse_dyn_promotion.sh
```

This keeps the same dynamics KL mechanism. Adam's per-parameter moments largely
preserve learning in parameters used only by the stochastic prior, while the
KL contribution to shared recurrent parameters is reduced tenfold relative to
the proven reward objective.

## Gated downstream campaign

After both ToyMemory markers exist, submit the seed-0 BSuite grid:

```bash
python experiments/r2r/submit_campaign_stage.py \
  --campaign CAMPAIGN --stage bsuite-seed0
```

Build the seed-1/2 promotion manifest after seed 0, then submit it:

```bash
python experiments/r2r/promote_bsuite.py --campaign CAMPAIGN --stage seed0
python experiments/r2r/submit_campaign_stage.py \
  --campaign CAMPAIGN --stage bsuite-promoted
```

After promoted jobs finish, run the three-seed `>=2T` gate and, only if it
unlocks Memory Maze, submit the two Memory Maze stages:

```bash
python experiments/r2r/promote_bsuite.py --campaign CAMPAIGN --stage qualify
python experiments/r2r/submit_campaign_stage.py \
  --campaign CAMPAIGN --stage mmaze-seed0
python experiments/r2r/submit_campaign_stage.py \
  --campaign CAMPAIGN --stage mmaze-promoted
```

All training launchers forward the preemption signal, flush checkpoint/replay/
cache state, and requeue exit codes 75 or 138. Long BSuite and Memory Maze runs
use one-day b3 allocations and continue across those durable requeue slices.
