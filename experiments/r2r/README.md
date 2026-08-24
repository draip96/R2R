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
