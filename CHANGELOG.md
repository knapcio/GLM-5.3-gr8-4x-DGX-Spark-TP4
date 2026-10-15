# Changelog

## v0.2 — 2026-10-15

v0.1 plus one patch, fourteen in all. Upstream base (`6448a16`) and the other thirteen patches are unchanged.

- DFlash2 after prefix reuse and alone on MTP (0014), behind `G6_DFL_REUSE=1|2`, default off; serving uses mode 2.
  Upstream runs a request on MTP only for its whole life when its prompt hit the prefix cache inside the drafter's
  2,048-position window or it was not alone at its prompt tail (upstream KNOWN-ISSUES #1). Mode 1 lets prefix-hit turns
  keep DFlash2, continuing the drafter's context ring from the previous turn of the same conversation or warming it up
  over a few MTP rounds; mode 2 also promotes a session that is alone on MTP. Drafting.
- `G6ENV_X` in `docker/env.example` gains `G6_DFL_REUSE=2`; image tag `gr8-glm53-recipe:v0.2`.

Multi-turn script (code-heavy, prefix-cache hits, T=0, c1, one run): prefix-hit turns ×1.20 decode, fresh turns ×1.09
(driven by one turn that was not alone at its prompt tail), replayed turns ×1.01; see the README's Multi-turn section.
RigMark full run c1 prose / code / structured 35.5 / 54.8 / 66.6 tok/s, C1/C2/C4 aggregate
38.8 / 54.3 / 67.6, prefill within 3 % of the baseline.
T=0 / KL mean KL 2.7e-3 vs the previous configuration (batching noise 5.0e-3); sampled qeval 578/600 (unchanged).

## v0.1 — 2026-10-15

First release: upstream gr8 `6448a16` plus thirteen patches. The fabric MTU fix is always on; every other patch is
behind a flag that defaults to off.

- Fabric: negotiate the QP MTU from the active ports (0001, always on).
- Replicated-GEMV splits across the four ranks with a per-layer all-gather: layer input (0002), indexer wq_b and the
  MTP draft layer (0003). Bit-exact.
- INT4 g32 draft heads for MTP (0004) and DFlash2 (0005). Drafting.
- DSel: three experiments, off (0006–0008), and drafting-cost overrides (0009); serving uses probe 32 and an MTP
  cost of 2.9. Drafting.
- MoE shared expert on a forked stream during the router (0010). Bit-exact.
- Decode attention round 1, scores and context kernels (0011), and round 2, eager transpose reduction, 32-bit context
  kernel, rank-by-count selector, grouped indexer (0012). Bit-exact with the serving flags; `G6_ATTN_CTX2=9` is an
  experimental mode that is not exact and is off.
- MTP confidence stop for solo MTP rounds (0013). Drafting.
- Dockerfile: the pinned upstream image with the patched TP=4 engine; `docker/env.example` with the serving flags.

RigMark c1 decode screens prose / code / structured 35.8 / 52.5 / 67.4 tok/s (baseline `d9188e3` + MTU fix, full run:
29.0 / 48.1 / 60.6); full run 35.4 / 54.9 / 66.7, C1/C2/C4 aggregate 39.4 / 55.7 / 67.5 (baseline 34.7 / 50.0 /
63.7), prefill within 1 %. T=0 logprobs on 75 prompts identical to the previous configuration; sampled qeval 578/600.
