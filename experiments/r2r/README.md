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
