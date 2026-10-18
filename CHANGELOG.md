# Changelog

## v0.8 — 2026-10-19

v0.7 plus six patches, thirty-nine in all, and one step in the serving procedure. Upstream base (`6448a16`) unchanged;
patches 0001–0033 differ only in the series count of their subject lines.

- kv_b as the release's block FP8 (0034–0036, `G6_KVB=fp8`): upstream's open PR
  [#98](https://github.com/kindlingai/gr8/pull/98) by spectra-kindling, its three commits as published. The attention's
  UK / UV weights are read as E4M3 with one f32 scale per 128 × 128 block instead of the BF16 copy the loader makes
  (half the bytes every decode step); each weight is rounded to the same BF16 value the BF16 loader stores, so the
  products are the same. One conflict with 0015 resolved here: 0015's ABS2 / VP2 widths read FP8 too.
- 0023's L2 prefetch reads kv_b at its stored width under `G6_KVB=fp8` (0037; E4M3 bytes plus the block scales;
  queue ranges 16 → 20) and reports a range dropped by a full queue instead of dropping it silently (0038).
- `kvb8_test` covers 0015's widths, alone and under QT (0039).
- Serving procedure: compact the host memory on every box before each start
  (`sudo sh -c 'sync; echo 1 > /proc/sys/vm/compact_memory'`). No setting changes; GB10's GPU allocates from host
  memory, and on a fragmented box it gets small physical pages.
- `docker/env.example`: `G6ENV_X` gains `G6_KVB=fp8`; image tag `gr8-glm53-recipe:v0.8`.

Exactness: `kvb8_test` 155/155 byte-identical (decode and prefill, every absorb / value-projection variant, under QT);
identical T=0 output on seven fresh boots of the cluster; KL vs the dh32f configuration 2.734e-03, the same as v0.7;
sampled qeval 573/600 and T=0 qeval 72/75 on each of the three full runs.

Where the gain comes from (RigMark prose c1, all from fresh, compacted boots on 2026-10-18/19): v0.7 38.3 as released
(no compaction) → v0.7 39.4 with compaction (+2.9 %) → v0.8 40.6 (+3.0 % from 0034–0038). The interleaved A/B of
the FP8 kv_b alone (`bench/decset.py`, three fresh-boot pairs, same image) gave prose +2.3 %, code +1.8 %, structured
+1.9 %, 3 of 3 pairs in every class.

Measured (medians of three full runs, 2026-10-18/19): RigMark prose 40.6 / code 61.2 / structured 73.4 (v0.7 38.3 /
58.4 / 70.2); concurrency C1 48.2 / C2 61.7 / C4 76.2 aggregate (v0.7 46.4 / 58.8 / 73.5); prefill after a warm-up
8K 1341 / 32K 1362 / 64K 1308 / 128K 1205 (v0.7 1371 / 1368 / 1317 / 1202; within the boot-to-boot spread).
`bench/conc.py` medians of three boots in [docs/history.md](docs/history.md).

## v0.7 — 2026-10-18

v0.6 plus three patches, thirty-three in all. Upstream base (`6448a16`) unchanged; patches 0001–0030 differ only in the
series count of their subject lines.

- S2 per-session MTP confidence stop for shared rounds (0031, KStopB): the shared chain still drafts K drafts for the
  round; 0031 lets each session verify fewer of them than the others (the most j with p_1 * .. * p_j >= tau for every
  i <= j). Thresholds are set per active session count (`G6_MTP_KSTOPB_TAU_S<n>`, frozen values measured 2026-10-17)
  and `G6_MTP_LPLOG=1` logs every shared round's verification and timing. Supplied entries replace the defaults in
  order (the defaults are 0.6 / 0.88 / 0.95 for positions 1–3, so two supplied entries leave position 3 at 0.95); past
  position 3 there is no stop (tau 0), and the runtime signature carries all seven positions so every rank agrees.
  Enabling it captures additional main widths at boot (~115 MiB per extra width per rank) — the memory cost rides with
  the throughput gain. An ABBA A/B (6 blocks, 07:54–19:21) against v0.6 measured c2 +1.50 % and c4 +3.70 % (geometric
  mean over paired blocks; the C/B isolation runs the same pattern), with no short-context change. Shared rounds
  verify at most `--mtp` drafts per session (the recipe serves `--mtp 2`), within the seven threshold positions.
- S2 drafter-shape fixes for block-12 drafters (0032, always on): boot now captures every DFlash2 first-form row count
  up to the verify width, the per-step row budget includes the verify width, and the acceptance histogram ceiling is
  widened to 16 entries, past anything a 12-block round can produce. Built and verified with the AlayaNeW/GLM-5.3-DFlash2
  drafter (MIT): with the recipe's
  B=8 incoai drafter the shapes were already covered and nothing observable changes. The Alaya drafter itself lost the
  A/B (decode code −21 %, prose −48 %, C1 −31 % at c1/c4) and is not shipped; the fixes stay because they are what
  makes a wider drafter servable at all. Note the scope: actually serving a B=12 drafter also took experimental engine
  changes (the drafter binary, the v2 pack format) that this recipe deliberately does not ship — 0032 only removes the
  S2-side ceilings so the engine can host such a drafter.
- SELX prefill selection on by default (0033): `G6_PF_SELX_CTX` now defaults to 53248 (was env-only, default off).
  Below 53248 tokens the selection path is identical with the flag off, so short contexts are unaffected; the ABBA
  measured against v0.6 (4 runs) ran its prefill screens below the threshold, so its C1/C4 spread (−2.3 %) is
  boot-to-boot variance, not a SELX cost. The final-image prefill table shows the select effect: 64K +0.8 %, 128K
  +5.1 % vs v0.6 (an earlier full run on the same owner decision showed 64K +1.4 %, 128K +3.9 % — boot-to-boot spread
  on the same lever). Accepted as the release default by owner decision, with the numbers in
  [docs/history.md](docs/history.md).
- `docker/env.example`: `G6ENV_X` gains the KStopB flags; image tag `gr8-glm53-recipe:v0.7`.

Measured against v0.6 on the same machines (medians of the release's three fullq runs, 2026-10-18): RigMark full run
prose 38.3 / code 58.4 / structured 70.2 (v0.6 38.5 / 58.3 / 70.2), concurrency C1 46.4 / C2 58.8 / C4 73.5 aggregate
(v0.6 46.3 / 56.7 / 73.3), prefill 8K 1371 / 32K 1368 / 64K 1317 / 128K 1202 (v0.6 1375 / 1381 / 1306 / 1157; across
the three runs prefill spans 1323–1377 / 1366–1378 / 1316–1324 / 1194–1216). The decode table (3 × `conctabx.sh`
medians) put the c2/c4 lifts where KStopB predicts: code c2 +5.7 %, c4 +4.8 %; prose c2 +6.4 %, c4 +9.1 %; c1 and
structured level with v0.6 (across the release's full runs structured c1 spans 70.0–74.2). Quality gate: T=0 KL vs the
dh32f reference 2.734e-03, the same value v0.6 measured the same
way; sampled qeval 573/600 vs the reference 578/600 (~1 standard error apart, identical in all three full runs); T=0
qeval 72/75; 96K needle correct.

## v0.6 — 2026-10-16

v0.5 plus four patches, thirty in all. Upstream base (`6448a16`) unchanged; patches 0001–0026 differ only in the series
count of their subject lines.

- CP decode in one chain (0029), `G6_CP_ONECHAIN=1`: past the replication boundary, the other ranks' head groups run as
  one 48-head attention call (one `k_vget`, one call whose merge writes each group's partial, the three puts in the S1
  order) instead of three separate chains — 8 launches a layer instead of 18. Bit-exact: `engine/attn/tests/cpgroups.cu`
  byte-compares the merged partials against the three calls, and the long-prompt decode output is identical (same SHA as
  flags-off at 70K / 80K / 90K-token prompts). TP=4 only, and the boot refuses `G6_ATTN_FLIP` with it.
- The launcher's `--share-limit` takes any multiple of 64 from 4096 (0027), not only powers of two, so a context-share
  limit such as 98304 (the cap is 212224 at TP=4) is accepted. The Dockerfile also copies the patched launcher options
  into the runtime image, so the limit works in the served container.
- Decode buckets: nine slots instead of eight (0028), so a limit off the doublings no longer merges the 2048 bucket away
  and short-context steps keep their own graph. With no limit, or a power-of-two limit, the buckets, graphs and outputs
  are those of before.
- S2 K-stop (0030): a session that `G6_DFL_REUSE` promoted keeps the shared chain's K drafts; at most `kmax` of them are
  verified (the confidence stop's `tau` has `kmax` + 1 entries). No new flag: it tightens 0013 wherever the K-stop is on.
- `docker/env.example`: `G6ENV_X` gains `G6_CP_ONECHAIN=1`; image tag `gr8-glm53-recipe:v0.6`. `SHARE_LIMIT` stays off:
  it costs maximum context (310K → 212K); it is a documented option ([README](README.md#reproducing),
  [docs/history.md](docs/history.md#long-context-one-chain-and-the-ninth-bucket-v06)).

Measured against v0.5 on the same machines: RigMark full run prose 38.5 / code 58.3 / structured 70.2 (v0.5 37.8 / 57.9 /
70.2), C1/C2/C4 aggregate 46.3 / 56.7 / 73.3 (v0.5 46.1 / 57.4 / 72.5); the decode table (median of three boots, details
in [docs/history.md](docs/history.md#decode-by-prompt-type-and-concurrency-releases)) is level with v0.5 within about
1 tok/s per cell; cold prefill 1375 / 1381 / 1306 / 1157 tok/s (8K–128K). The release's gains are in long-context
decode ([docs/history.md](docs/history.md#long-context-one-chain-and-the-ninth-bucket-v06)): `G6_CP_ONECHAIN=1`
+1.7…+3.6 % at 70K–90K and +9.5 % at a ~120K-token prompt; `SHARE_LIMIT=98304` (without the one-chain call)
+8.3…+13.6 % at 70K–90K. Short prompts: level. T=0 top-20 logprobs on the 75 qeval prompts vs v0.5, prefix-cached pass:
34/75 identical, mean KL 2.7e-3 (batching-noise scale 5.0e-3). Sampled qeval 577/600 (T=1.0 with rejection sampling;
v0.5 577/600, the dh32f reference 578/600); T=0 qeval 71/75; 96K needle correct.

## v0.5 — 2026-10-16

v0.4 plus four patches, twenty-six in all, and one upstream launcher option. Upstream base (`6448a16`) unchanged;
patches 0001–0022 differ only in the series count of their subject lines.

- Paced L2 prefetch (0023), `G6_L2PF=15`: in the windows where DRAM is idle (the two all-reduce waits, the column
  gathers, the latency-bound attention) one warp on a side stream prefetches the layer's next weights into L2 in small
  chunks at a set rate, the pacing of [dsgx10](https://github.com/blaesus/dsgx10). Bit-exact.
- MoE decode kernel (0026): a split table per step row count, `G6_K1_SPLIT=1:/4,3:2x1,4-32:/4` (fc1 CTA shape and fc2
  n-split with the same summation order), and a translation prefetch at the start of each expert's fc1,
  `G6_K1_TLBW=2048`: a step's routed experts touch about 5,300 2 MB pages per rank, more than the GPU's translation
  cache covers (about 4 GB), and the kernel ran at 85 % of a plain read of the same bytes. Also in the patch, off:
  `G6_IMG_VMM` and the `k1_audit` bench. Bit-exact.
- Two experiments in the series, off: the all-reduce wait folded into the norm kernel (0024, measured level) and the
  MTP first form without its head on DFlash2 rounds (0025, measured slower on prose).
- `SPEC_ACCEPT=rejection` (upstream launcher option): drafts at temperature > 0 are accepted by rejection sampling
  instead of coupled sampling. The output distribution is the target's in both; a seed's text now depends on the
  drafting configuration. Temperature 0 is unchanged.
- `docker/env.example`: `G6ENV_X` gains `G6_L2PF=15 G6_K1_SPLIT=1:/4,3:2x1,4-32:/4 G6_K1_TLBW=2048`, plus
  `SPEC_ACCEPT=rejection`; image tag `gr8-glm53-recipe:v0.5`.
- README: the results are RigMark's (decode by workload, concurrency, prefill); the `bench/conc.py` table moved to
  [docs/history.md](docs/history.md).

Measured against v0.4 on the same machines:

v0.5 vs v0.4: RigMark full run prose 37.8 / code 57.9 / structured 70.2 (v0.4 36.4 / 55.9 / 71.1), C1/C2/C4 aggregate
46.1 / 57.4 / 72.5 (v0.4 43.6 / 56.1 / 68.6); a second run after a warm-up request, without the concurrency section:
prose 38.0 / code 59.1 / structured 70.2. `bench/decset.py` (24 prompts, c1, T=0; three runs each of v0.5's
levers, interleaved): prose 43.35 vs 41.38 (+4.8 %), code 55.30 vs 53.14 (+4.1 %), structured 60.02 vs 57.57
(+4.3 %); of that, `G6_L2PF=15` alone +2.7 % prose and the K1 split table plus translation prefetch +2.0 % on top (detail in
[docs/history.md](docs/history.md#decode-over-24-prompts-v05-levers)). At
T=1.0, top_p 0.95 (two runs each): `SPEC_ACCEPT=rejection` vs coupled prose +0.2 %, code +2.4 %, structured +6.2 %.
Decode table (`bench/conc.py`, aggregate, median of three boots, c1 / c2 / c4): prose 37.4 / 51.5 / 66.3 (v0.4 35.3 / 48.9 / 63.3), code
48.9 / 61.7 / 78.9 (v0.4 46.7 / 59.1 / 75.1), structured 74.9 / 68.3 / 88.7 (v0.4 72.1 / 65.6 / 84.9). Cold prefill
(warm-up run) 1359 / 1383 / 1316 / 1154 tok/s (8K-128K; v0.4 1375 / 1378 / 1313 / 1160); the full run's first prefill
after boot read 1299 at 8K, and cold TTFT at 8K / 32K on our own prompts was level (two boots each). T=0 top-20 logprobs on the 75 qeval prompts, each configuration from a fresh boot: `G6_L2PF=15` vs v0.4, and the K1 split table and translation prefetch on top of it, 75/75 prompts identical, KL 0. Sampled qeval 577/600 (T=1.0 with rejection sampling; v0.4 578/600 with coupled
sampling, standard error 0.8 pp); T=0 qeval 72/75; 96K needle correct.

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
