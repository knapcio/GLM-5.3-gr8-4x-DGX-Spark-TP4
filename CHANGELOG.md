# Changelog

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
