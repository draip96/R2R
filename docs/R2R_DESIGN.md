# R2R design

R2R extends Recall2Imagine (R2I) with dense replay-aligned MIMO state and
boundary-adjoint caching. It deliberately does not add an EMA model, burn-in,
age gates, version rejection, damping, consistency losses, prioritization, or
any other freshness policy.

## Scientific invariants

R2R keeps R2I's uniform replay distribution and the exact sampler/offset random
number calls. Cache validity never filters, rejects, prioritizes, or resamples a
learner sequence. The world-model objective, MIMO architecture, optimizer and
gradient clipping, train ratios, imagination settings, and benchmark budgets
are unchanged. The four learner shapes all contain 4096 transitions:

| Window | Batch | Transitions per update |
| ---: | ---: | ---: |
| 64 | 64 | 4096 |
| 128 | 32 | 4096 |
| 256 | 16 | 4096 |
| 1024 | 4 | 4096 |

The upstream configuration already paired a 1024-step window with batch 4 and
its BSuite profile paired a 256-step window with batch 16. No learning-rate,
optimizer, train-ratio, or loss-normalization change is coupled to the R2R
window presets; only `batch_length` and `batch_size` differ among them.

That source snapshot is not evidence of a historical window-only ablation: the
1024x4 default and 256x16 BSuite profiles target different tasks, and the latter
also changed its replay capacity, interaction budget, reset mode, architecture,
and train ratio. The repository history introduces those profiles together and
does not expose an earlier/later controlled window-size edit. R2R therefore
makes the causal control explicit: within each campaign grid, every setting is
fixed and only the matched `T x B` preset changes.

## Cache timeline

For a sampled physical replay window `s:e`, the learner gathers the cached
boundary `(x, z, action)_(s-1)` and future adjoint `G_e`. Missing or overwritten
entries use the learned/zero initial state and zero future adjoint. The learner
runs the native parallel MIMO scan once and minimizes the native mean model loss
plus the real inner product between `G_e` and `x_e`.

Zero-valued state taps are inputs to that same scan. A single reverse pass
therefore produces normal parameter gradients and `G_(s-1:e-1)` together; no
second model forward or backward is performed. After the optimizer update, R2R
writes `x_(s:e)` and those adjoints, both computed under the same pre-update
parameters, directly to their replay slots. Thus an adjoint is saved from the
reverse pass associated with the forward pass that produced its state. Relative
to a freshly overwritten state boundary, the future information represented by
an adjoint is naturally one learner update older; R2R performs no extra work to
refresh it.

Real episode resets zero recurrent boundaries and prevent cached credit from
crossing episodes. A sampled-window boundary is not fabricated as a reset.
Native parameter-gradient clipping remains enabled, while cached adjoints are
not clipped. Non-finite state or adjoint writes abort the learner.

## Physical alignment and persistence

Replay samples expose stable physical slots, per-chunk overwrite generations,
and predecessor slots without changing selection probabilities. The sidecar is
aligned to the replay's final chunk boundary while the configured replay size
remains exactly 1,000,000 transitions. It stores complex MIMO state and adjoint
as raw BF16 real/imaginary bit patterns, categorical state and preceding action
in compact integer arrays, and validity/generation metadata.

Writes validate the sampled generation and discard stale writes. For duplicate
slots, the last batch row wins deterministically. Dirty ranges are mirrored with
a validity-last publication protocol so an interrupted copy cannot expose a
partially published entry. Replay chunk generations remain authoritative after
wraparound and resume.

For Memory Maze's `5 x 512` complex MIMO state, state plus adjoint require
20,480 bytes per transition, or approximately 19.1 GiB for one million
transitions, before the ordinary replay and compact metadata.
