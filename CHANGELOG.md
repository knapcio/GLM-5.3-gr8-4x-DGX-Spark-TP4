# Changelog

## v0.2.1 — 2026-10-15

Documentation only; no patch or flag changed. README shows the current release only, with a decode table by prompt
type and concurrency (`bench/conc.py`) and a decode-vs-context script (`bench/ctxlen.py`); comparisons with earlier
releases moved to this file and [docs/history.md](docs/history.md).

## v0.2 — 2026-10-15

v0.1 plus one patch, fourteen in all. Upstream base (`6448a16`) and the other thirteen patches are unchanged.

- DFlash2 after prefix reuse and alone on MTP (0014), behind `G6_DFL_REUSE=1|2`, default off; serving uses mode 2.
  Upstream runs a request on MTP only for its whole life when its prompt hit the prefix cache inside the drafter's
  2,048-position window or it was not alone at its prompt tail (upstream KNOWN-ISSUES #1). Mode 1 lets prefix-hit turns
  keep DFlash2, continuing the drafter's context ring from the previous turn of the same conversation or warming it up
  over a few MTP rounds; mode 2 also promotes a session that is alone on MTP. Drafting.
- `G6ENV_X` in `docker/env.example` gains `G6_DFL_REUSE=2`; image tag `gr8-glm53-recipe:v0.2`.

Measured against v0.1 on the same machines (detail in [docs/history.md](docs/history.md)):

- Multi-turn script (code-heavy, prefix-cache hits, T=0, c1, one run): prefix-hit turns ×1.20 decode, fresh turns
  ×1.09 (driven by one turn that was not alone at its prompt tail), replayed turns ×1.01.
- RigMark full run (single-turn, fresh prompts), c1 prose / code / structured 35.5 / 54.8 / 66.6 tok/s vs 35.4 / 54.9 /
  66.7 (+0.3 / −0.2 / −0.1 %); C1 / C2 / C4 aggregate 38.8 / 54.3 / 67.6 vs 39.4 / 55.7 / 67.5 (−1.5 / −2.5 / +0.1 %),
  one run each, variance not established; cold prefill 8K / 32K / 64K / 128K 1369 / 1370 / 1312 / 1157 vs 1367 / 1373 /
  1311 / 1181 tok/s (128K −2.0 %, within 3 % of the baseline).
- T=0 top-20 logprobs vs an earlier configuration (GEMV split + INT4 draft heads + DSel probe 32), prefix-cached pass: mean KL 2.7e-3 (batching noise, c1 vs c4: 5.0e-3); sampled qeval
  578/600, identical per seed; T=0 qeval 71/75, the same four misses.

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

Measured against the baseline, upstream gr8 `d9188e3` plus the MTU fix only, on the same machines (complete stacks:
the recipe pins `6448a16`; detail in [docs/history.md](docs/history.md)):

- RigMark full run, c1 prose / code / structured 35.4 / 54.9 / 66.7 tok/s vs 29.0 / 48.1 / 60.6 (+22.1 / +14.1 /
  +10.1 %); decode screens 35.8 / 52.5 / 67.4.
- RigMark C1 / C2 / C4 aggregate 39.4 / 55.7 / 67.5 vs 34.7 / 50.0 / 63.7 (+13.5 / +11.4 / +6.0 %).
- Cold prefill 8K / 32K / 64K / 128K 1367 / 1373 / 1311 / 1181 vs 1361 / 1375 / 1307 / 1188 tok/s (within 1 %); 96K
  needle answer unchanged.
- T=0 top-20 logprobs on the 75 qeval prompts identical to the previous development configuration; sampled qeval
  578/600; T=0 qeval 71/75 (baseline 73/75, borderline tasks).
