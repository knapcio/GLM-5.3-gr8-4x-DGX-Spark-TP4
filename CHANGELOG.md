# Changelog

## v0.4 — 2026-10-16

v0.3 plus one patch, twenty-two in all. Upstream base (`6448a16`) unchanged; patches 0001–0021 differ only in the series count of their subject lines.

- DFlash2 drafter body on INT4 (0022), `G6_DFL_BODY_I4=1`: the drafter's body GEMVs read INT4 g32 copies of their MXFP8
  records, made at boot, with the same splits: 382 MB of records instead of 701 MB. Drafts only; the target, its
  weights and the verify steps are unchanged. Also in the patch, off: the MTP draft's eh_proj on INT4
  (`G6_MTP_EHPROJ_I4`). Drafting.
- The DSel cost table stays v0.3's. A table re-measured with 0022 prices a DFlash2 round lower (`d:5.726` instead of
  `d:7.285`) and was level on prose and code and slower on structured in our screens.
- `G6ENV_X` in `docker/env.example` gains `G6_DFL_BODY_I4=1`; image tag `gr8-glm53-recipe:v0.4`.

Measured against v0.3 on the same machines:

v0.4 vs v0.3: RigMark c1 decode screens (three runs each, interleaved) prose +0.3 %, code +2.6 %, structured +5.2 %;
RigMark full run prose 36.4 / code 55.9 / structured 71.1 (v0.3 36.2 / 56.2 / 65.8), C1/C2/C4 aggregate 43.6 / 56.1 / 68.6
(v0.3 43.1 / 55.6 / 68.6). `bench/decset.py` (24 prompts, c1, two runs each, interleaved): prose 41.64 vs 40.68 (+2.4 %,
faster on 8 of 12 prompts), code 52.54 vs 54.30 (−3.2 %, mostly one prompt: 53.0 vs 63.8; median per prompt 52.6 vs
53.1), structured 57.72 vs 56.25 (+2.6 %). Decode table (`bench/conc.py`, aggregate, median of three boots each,
two of them interleaved in one session): prose 35.3 / 48.9 / 63.3 (v0.3 36.2 / 49.3 / 63.3), code 46.7 / 59.1 / 75.1 (v0.3 46.3 / 59.2 / 75.5),
structured 72.1 / 65.6 / 84.9 (v0.3 70.6 / 65.7 / 85.0) for c1 / c2 / c4; its single prose prompt at c1 is slower on
v0.4 in all three boots (35.1 / 35.7 / 35.3 vs 36.2 / 36.2 / 36.3), a drafting change moves individual prompts both
ways. Cold prefill 1375 / 1378 / 1313 / 1160 tok/s (8K-128K, unchanged). T=0
logprobs: mean KL 2.7e-3 vs the same earlier configuration as v0.3 (noise scale 5.0e-3); sampled qeval 578/600 (unchanged);
T=0 qeval 72/75; 96K needle correct.

## v0.3 — 2026-10-16

v0.2 plus seven patches, twenty-one in all (0021 only corrects the table tool's usage text). Upstream base (`6448a16`) unchanged. Of patches 0001–0014 only 0011 and 0012
changed, in one comment line of the attnk bench; their code is unchanged.

- Decode attention round 3 (0015): the scores kernel reads a head's query lane-transposed with 5 vector loads instead
  of 18 (`G6_ATTN_QT=1`), and the step's front kernels (KV append, index append and preparation) run in one launch
  (`G6_ATTN_FRONT=1`). Also in the patch and off: `G6_ATTN_ABS2`, `G6_ATTN_VP2`, `G6_ATTN_CTX4`. Bit-exact.
- Parallel DFlash2 drafter step (0016), `G6_DRAFT_PAR=1`: in a solo DFlash2 round the drafter step runs on its own
  stream under MTP's first form. Bit-exact.
- Decode step timeline (0017), `G6_S2_TIMELINE=N`: GPU span, host time and GPU idle per step kind. Diagnostics only,
  off in serving.
- DSel cost model (0018, 0020), `G6_S2_DSEL_MODEL=1` with the frozen cost table `G6_S2_DSEL_TABLE`: the drafter choice
  prices both drafters from measured milliseconds, prices MTP rounds as the confidence stop runs them, and (table key
  `r`, 0020) prices DFlash2 from its rounds as they ran. `tools/s2_dsel_table.py` (0018, 0019) writes the table from a
  timeline boot. The table was measured on these four machines; other hardware should measure its own. Drafting.
- `G6ENV_X` in `docker/env.example` gains `G6_ATTN_QT=1 G6_ATTN_FRONT=1 G6_DRAFT_PAR=1 G6_S2_DSEL_MODEL=1
  G6_S2_DSEL_TABLE=…` and drops `G6_S2_DSEL_MCOST=2.9` (unused with the model on); image tag `gr8-glm53-recipe:v0.3`.

Measured contributions, RigMark c1 decode (detail in [docs/history.md](docs/history.md#per-lever-contributions)):

- Attention round 3 (`QT` + `FRONT`): prose +0.6 %, code +0.7 %, structured +1.2 %.
- `G6_DRAFT_PAR=1`: prose +0.4 %, code +1.1 %, structured +0.8 %.
- DSel model with the table (`r:0.95`) vs `G6_S2_DSEL_MCOST=2.9`: prose +1.8 %, code +3.0 %, structured −3.6 % (two
  runs each).

Measured against v0.2 on the same machines:

v0.3 vs v0.2: RigMark c1 decode screens (two runs each) prose +3.5 %, code +4.6 %, structured -2.5 %; RigMark full run
prose 36.2 / code 56.2 / structured 65.8 (v0.2 35.5 / 54.8 / 66.6), C1/C2/C4 aggregate 43.1 / 55.6 / 68.6 (v0.2 38.8 / 54.3 /
67.6). Decode table (bench/conc.py, aggregate): prose 36.2 / 49.2 / 63.3 (v0.2 34.9 / 48.9 / 62.8), code 46.3 / 59.2 / 75.5
(v0.2 43.4 / 58.8 / 74.8), structured 70.6 / 65.7 / 85.0 (v0.2 69.5 / 65.2 / 84.3) for c1 / c2 / c4. Cold prefill
1367 / 1380 / 1311 / 1174 tok/s (8K-128K, unchanged). T=0 logprobs: mean KL 2.7e-3 vs an earlier configuration (noise scale
5.0e-3); sampled qeval 578/600 (unchanged); T=0 qeval 72/75; 96K needle correct.

## v0.2.2 — 2026-10-15

Documentation only: the README keeps one decode table (by prompt type and concurrency) and one prefill table; the
RigMark decode tables moved to [docs/history.md](docs/history.md).

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
