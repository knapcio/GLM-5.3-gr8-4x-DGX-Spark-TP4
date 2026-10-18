# Measurement history

A/B measurements behind the releases: each release against the previous one and against the upstream baseline, the
per-lever contributions, and the multi-turn test for patch 0014. The README shows only the current release; per-release
summaries are in [CHANGELOG.md](../CHANGELOG.md).

## Baseline

The baseline is upstream gr8 `d9188e3` plus the MTU fix only (the engine without this recipe's speed patches),
measured on the same four machines as the recipe. The recipe pins upstream `6448a16`. The upstream changes in between
are engine-side (fabric relay back-pressure, giving host heap back after load, S2 prefill and notice scheduling); the
front end, launcher and loader are identical. The comparisons below are of complete stacks, not of the patches alone.

v0.1 was measured with the v0.2 `G6ENV_X` line without `G6_DFL_REUSE=2`. v0.2's line is v0.3's with
`G6_S2_DSEL_MCOST=2.9` in place of `G6_ATTN_QT=1 G6_ATTN_FRONT=1 G6_DRAFT_PAR=1 G6_S2_DSEL_MODEL=1 G6_S2_DSEL_TABLE=…`.

## Decode by prompt type and concurrency: releases

[`bench/conc.py`](../bench/conc.py): aggregate tok/s **[per-stream tok/s]**, RigMark's prose, code and structured
prompts, 256 output tokens, temperature 0, `reasoning_effort: low`, medians of three rounds per cell, c identical
requests sent at once. v0.7, v0.6, v0.5 and v0.4: median per cell over three boots (v0.7's three boots are within
0.5 tok/s of each other in every cell; v0.6's three boots are within 2 tok/s of each other, most cells within 1.3;
v0.5's within 0.3). v0.4 was measured against three v0.3 boots, two of each interleaved in one session (v0.3's three-boot medians:
prose 36.2 / 49.3 / 63.3, code 46.3 / 59.2 / 75.5, structured 70.6 / 65.7 / 85.0). Earlier rows: one boot per release.

| Release | Prompt type | c1 | c2 | c4 |
|---|---|---:|---:|---:|
| v0.7 (median of three boots) | prose | 37.4 [38.5] | 54.8 [29.6] | 72.2 [20.1] |
| v0.7 (median of three boots) | code | 48.9 [50.1] | 65.3 [35.2] | 82.7 [23.3] |
| v0.7 (median of three boots) | structured | 74.8 [77.7] | 68.2 [37.1] | 88.8 [26.0] |
| v0.6 (median of three boots) | prose | 37.3 [38.4] | 51.3 [27.4] | 66.2 [18.5] |
| v0.6 (median of three boots) | code | 48.8 [50.0] | 61.6 [32.9] | 78.9 [22.4] |
| v0.6 (median of three boots) | structured | 75.0 [77.9] | 68.3 [37.1] | 88.7 [26.1] |
| v0.5 (median of three boots) | prose | 37.4 [38.5] | 51.5 [27.4] | 66.3 [18.5] |
| v0.5 (median of three boots) | code | 48.9 [50.1] | 61.7 [32.8] | 78.9 [22.4] |
| v0.5 (median of three boots) | structured | 74.9 [77.7] | 68.3 [37.2] | 88.7 [26.0] |
| v0.4 (median of three boots) | prose | 35.3 [36.2] | 48.9 [26.1] | 63.3 [17.7] |
| v0.4 (median of three boots) | code | 46.7 [47.8] | 59.1 [31.5] | 75.1 [21.4] |
| v0.4 (median of three boots) | structured | 72.1 [74.8] | 65.6 [35.7] | 84.9 [24.9] |
| v0.3 | prose | 36.2 [37.3] | 49.2 [26.1] | 63.3 [17.7] |
| v0.3 | code | 46.3 [47.4] | 59.2 [31.5] | 75.5 [21.4] |
| v0.3 | structured | 70.6 [73.2] | 65.7 [35.7] | 85.0 [25.0] |
| v0.2 (2026-10-15) | prose | 34.9 [35.8] | 48.9 [25.9] | 62.8 [17.6] |
| v0.2 (2026-10-15) | code | 43.4 [44.3] | 58.8 [31.2] | 74.8 [21.3] |
| v0.2 (2026-10-15) | structured | 69.5 [72.0] | 65.2 [35.4] | 84.3 [24.8] |

## Decode over 24 prompts: v0.4 vs v0.3

[`bench/decset.py`](../bench/decset.py): 12 prose, 6 code and 6 structured prompts, c1, temperature 0,
`reasoning_effort: low`, 384 output tokens; per kind, all completion tokens over all decode time (first token to end).
Two runs per release, alternating, fresh boot each.

| Release | prose | code | structured |
|---|---:|---:|---:|
| v0.4 | 41.57 / 41.71 | 52.26 / 52.82 | 57.72 / 57.71 |
| v0.3 | 40.86 / 40.50 | 54.32 / 54.28 | 56.23 / 56.26 |

v0.4 is faster on 8 of the 12 prose prompts. Most of the code difference is one prompt (63.8 → 53.0 tok/s); the median
over the six code prompts is 52.6 vs 53.1.

## Decode over 24 prompts: v0.5 levers

`bench/decset.py`, c1, T=0, fresh boot per run, one session (2026-10-16). v0.4 controls: prose 40.97 / 40.94 / 41.82 /
41.79, code 53.24 / 53.25 / 53.04 / 53.02, structured 57.21 / 57.33 / 57.87 / 57.87 (two of them on the v0.5 image with
its flags off).

| Configuration | prose | code | structured |
|---|---:|---:|---:|
| v0.4 (mean of four) | 41.38 | 53.14 | 57.57 |
| + `G6_L2PF=15` (mean of three) | 42.48 | 54.38 | 59.12 |
| + `G6_K1_SPLIT` + `G6_K1_TLBW` = v0.5 (mean of three) | 43.35 | 55.30 | 60.02 |

## RigMark: releases and baseline

[RigMark](https://github.com/alexellis/rigmark) 1.0.0, thinking on, `reasoning_effort: low`, decode tok/s; settings in
the README's Reproducing section. RigMark sends single-turn requests.

| c1 decode | prose | code | structured |
|---|---:|---:|---:|
| v0.7, full run (median of three) | 38.3 | 58.4 | 70.2 |
| v0.6, full run | 38.5 | 58.3 | 70.2 |
| v0.5, full run | 37.8 | 57.9 | 70.2 |
| v0.5, run after a warm-up request (no concurrency section) | 38.0 | 59.1 | 70.2 |
| v0.4, full run | 36.4 | 55.9 | 71.1 |
| v0.2, full run | 35.5 | 54.8 | 66.6 |
| v0.1, full run | 35.4 | 54.9 | 66.7 |
| v0.1, decode screens (two runs) | 35.76 / 35.74 | 52.51 / 52.56 | 67.51 / 67.36 |
| baseline (`d9188e3` + MTU fix), full run | 29.0 | 48.1 | 60.6 |

| Concurrency, aggregate end-to-end tok/s | C1 | C2 | C4 |
|---|---:|---:|---:|
| v0.6 | 46.3 | 56.7 | 73.3 |
| v0.5 | 46.1 | 57.4 | 72.5 |
| v0.4 | 43.6 | 56.1 | 68.6 |
| v0.2 | 38.8 | 54.3 | 67.6 |
| v0.1 | 39.4 | 55.7 | 67.5 |
| baseline (`d9188e3` + MTU fix) | 34.7 | 50.0 | 63.7 |

| Prefill, cold, tok/s | 8K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|
| v0.7 (median of three full runs) | 1371 | 1368 | 1317 | 1202 |
| v0.6 (full run) | 1375 | 1381 | 1306 | 1157 |
| v0.5 (after a warm-up request) | 1359 | 1383 | 1316 | 1154 |
| v0.5 (full run, first prefill after boot) | 1299 | 1374 | 1306 | 1160 |
| v0.4 | 1375 | 1378 | 1313 | 1160 |
| v0.3 | 1367 | 1380 | 1311 | 1174 |
| v0.2 | 1369 | 1370 | 1312 | 1157 |
| v0.1 | 1367 | 1373 | 1311 | 1181 |
| baseline (`d9188e3` + MTU fix) | 1361 | 1375 | 1307 | 1188 |

Prefill (v0.1) is within 1 % of the baseline: the levers act on decode. v0.2: prefill within 3 % of the baseline
(128K: 1157 vs 1188). A 96K-token needle prompt returns the same answer as the baseline in v0.1 and v0.2.

## Per-lever contributions

Measured contribution is RigMark c1 decode on short screens, each step on top of the levers above it, so the steps use
different control runs and do not add up exactly to the release totals. The DSel model (0018–0020) is compared with
v0.2's `G6_S2_DSEL_MCOST=2.9`, which it replaces.

| Patch | Flags (serving) | Measured |
|---|---|---|
| 0001 | always on | needed to boot on a switched fabric; no speed effect |
| 0002, 0003 | `G6_QA_SPLIT=1` `G6_IDXQ_SPLIT=1` `G6_MTP_SPLIT=1` | prose +6.5 % |
| 0004, 0005 | `G6_DRAFT_HEAD_I4=1` `G6_DFL_HEAD_I4=1` | prose +0.6 %, code +1.5 % |
| — (DSel probe interval) | `G6_S2_DSEL_PROBE=32` | prose +3 % |
| 0006–0008 | off | not used |
| 0009 | `G6_S2_DSEL_MCOST=2.9` (v0.1–v0.2; replaced by 0018 in v0.3) | prose +2.2 %, code −1.9 % |
| 0010 | `G6_ROUTE_OVL=1` | prose +1.7 % |
| 0011 | `G6_ATTN_SC2=4` `G6_ATTN_CTX2=4` | prose +3.4–4.7 %, code / structured +4–7 %; −8.6 ms/step at c4 (12 rows) |
| 0012 | `G6_ATTN_SC2_TR=2` `G6_ATTN_CTX3=2` `G6_ATTN_SEL2=16` `G6_IDX_GRP=16` | prose +0.7 % at 1K context; at 32K context up to −3.1 ms/step at c1, −10.3 ms/step at c4 |
| 0013 | `G6_MTP_KSTOP=1` `G6_MTP_KSTOP_KMAX=3` `G6_MTP_KSTOP_TAU=0,0.5,0.75` | prose +1.2 %, code −2.1 % |
| 0014 | `G6_DFL_REUSE=2` | multi-turn script: prefix-hit turns ×1.20, fresh ×1.09 (see [below](#multi-turn-prefix-cache-hits)); RigMark single-turn rows similar to v0.1 in these single full runs (it sends fresh prompts); variance not established |
| 0015 | `G6_ATTN_QT=1` `G6_ATTN_FRONT=1` | prose +0.6 %, code +0.7 %, structured +1.2 %; attnk bench −0.8 ms/step at 2K context and 3 rows, −1.5 to −1.9 ms/step at 12 rows |
| 0016 | `G6_DRAFT_PAR=1` | prose +0.4 %, code +1.1 %, structured +0.8 % |
| 0017 | off (`G6_S2_TIMELINE=N` when measuring the DSel table) | diagnostics only |
| 0018–0020 | `G6_S2_DSEL_MODEL=1` `G6_S2_DSEL_TABLE=…` (with `r:0.95`) | vs `G6_S2_DSEL_MCOST=2.9`: prose +1.8 %, code +3.0 %, structured −3.6 % (two runs each). Without key `r` (0018 alone) a structured request that a dip moved to MTP stayed there: structured −21 % on the cluster; `r` (0020) brings it back to DFlash2 |
| 0022 | `G6_DFL_BODY_I4=1` (with v0.3's DSel table) | prose +0.3 %, code +2.6 %, structured +5.2 % (three runs each, interleaved with v0.3). With a DSel table re-measured on it (`d:5.726`): prose +0.4 %, code +3.9 %, structured −1.0 % (two runs each); v0.3's table is kept |
| 0023 | `G6_L2PF=15` | `bench/decset.py` (24 prompts, c1, three runs vs four controls): prose +2.7 %, code +2.3 %, structured +2.7 %; one-GPU bench of 78 decode layers: -10.1 us per layer at 3 rows, -38.6 at 12 rows |
| 0024 | off (`G6_AR_WAITFOLD=1` measured) | decset: level (+0.3 % prose, within noise); one-GPU bench -0.18 ms per step |
| 0025 | off (`G6_MTP_HEAD_SKIP=4` measured) | decset: prose -1.0 %, code +4.2 % from one prompt whose text changed, structured -1 % |
| 0026 | `G6_K1_SPLIT=1:/4,3:2x1,4-32:/4` `G6_K1_TLBW=2048` (on top of 0023) | decset: prose +2.0 %, code +1.7 %, structured +1.5 % (three runs each); `k1_audit` at served routes: split table -11 us per MoE call at 3 rows (-2.9 %), translation prefetch -8.5 us per call with the route overlap; the MoE kernel reaches 85 % of a plain read of the same bytes with all 75 layers resident, 91 % with 8 |
| 0027 | off (launcher rule for `--share-limit`) | no speed effect; accepts limits such as 98304 that the power-of-two rule kept out |
| 0028 | off (`SHARE_LIMIT=98304` measured) | long-context decode below (one chain and the ninth bucket); with no limit the buckets, graphs and outputs are unchanged |
| 0029 | `G6_CP_ONECHAIN=1` | long-context decode (below): 70K / 80K / 90K-token prompts 35.96 / 37.23 / 36.92 vs 34.71 / 36.32 / 36.29 tok/s flags-off (one run each, T=0, same output SHA); a 120K-prompt decode screen 37.0 vs 33.8 tok/s. Short prompts: level |
| 0030 | (with `G6_MTP_KSTOP=1`) | correctness: a `G6_DFL_REUSE`-promoted session verifies at most `kmax` of the shared chain's drafts; no speed effect measured beyond the K-stop's own |
| 0031 | `G6_MTP_KSTOPB=1` `G6_MTP_LPLOG=1` `G6_MTP_KSTOPB_TAU_S1..S4` (with 0013's solo stop on) | ABBA, six blocks (2026-10-17, 07:54–19:21, control interleaved with candidate): c2 +1.50 %, c4 +3.70 % (geometric mean over paired blocks; per-block c4 +3.5 to +3.8 %); a held-out combined workload set +2.44 %; short-context decode (c1) unchanged. The shared chain still drafts K drafts; each session verifies its own fewer count, the most j with the running product of draft probabilities past the per-position thresholds. Thresholds frozen per active session count (S1 0.7/0.7, S2 0.7/0.8, S3 0.7/0.88, S4 0.7/0.75); `G6_MTP_LPLOG=1` logs every shared round's verification and timing. Extra boot cost: additional main widths captured (~115 MiB per extra width per rank) |
| 0032 | always on | correctness only for the recipe's drafter: with the B=8 incoai drafter every captured shape was already covered, outputs unchanged. Built and verified with the AlayaNeW DFlash2 B=12 drafter (MIT), which then lost the A/B (below) and is not shipped; the fixes stay because they make a wider drafter servable at all |
| 0033 | default on (`G6_PF_SELX_CTX` overrides) | ABBA, four runs (2026-10-17, 19:51–20:37, v0.6 flags-off control interleaved). Caveat: the ABBA's prefill screens ran 8K/32K prompts, below the 53248 threshold, so SELX never activated in them — those runs measure boot-to-boot variance, not a SELX cost (C1 −2.3 %, C4 −2.3 % against a same-code control). The prefill effect shows in the prefill table (median of the release's three full runs): 64K +0.8 %, 128K +3.9 % vs v0.6 (1371/1368/1317/1202 vs 1375/1381/1306/1157 tok/s at 8K–128K; the three runs span 1323–1377 / 1366–1378 / 1316–1324 / 1194–1216, so the 128K gain is the stable one and the short columns sit within boot-to-boot spread). Accepted as the release default by owner decision (long-context prefill-select gain, short contexts unaffected: below 53248 the selection path is identical with the flag off) |
| — | `SPEC_ACCEPT=rejection` (upstream option) | decset at T=1.0, top_p 0.95 (two runs each): prose +0.2 %, code +2.4 %, structured +6.2 %; T=0 unchanged |

## Long context: one chain and the ninth bucket (v0.6)

Past the replication boundary R, decode attention runs the other ranks' head groups on each rank's own keys. Upstream
does that as one chain per remote head group (three at TP=4): 18 launches a layer. Patch 0029, `G6_CP_ONECHAIN=1`,
folds them into one 48-head call: 8 launches a layer, byte-identical partials (`engine/attn/tests/cpgroups.cu`).
Patch 0028 adds a ninth decode bucket, so the launcher's `--share-limit` can take a limit off the power-of-two
doublings (98304) without merging the 2048 bucket away. Both were measured on one engine build (ident `8f52350c`),
T=0, one run each, fresh boot per arm; decode tok/s on long prompts:

| Configuration | 70K | 80K | 90K | ~120K (ctx table) |
|---|---:|---:|---:|---:|
| flags off | 34.71 | 36.32 | 36.29 | 33.8 |
| `G6_CP_ONECHAIN=1` | 35.96 | 37.23 | 36.92 | 37.0 |
| `SHARE_LIMIT=98304` (no one-chain) | 39.43 | 39.32 | 41.14 | 36.7 |

- The one-chain arm's output SHA is identical to flags-off at all three sizes (bit-exact); the ninth bucket regroups
  verify rows, so the `SHARE_LIMIT` arm's text differs (drafting class).
- TTFT is level across the arms (80K: 12.9–13.1 s, 90K: 11.7–11.8 s; the 70K prompt includes a cold start,
  55–56 s).
- At short prompts the arms are level: `bench/decset.py` prose 42.69 (`SHARE_LIMIT`) vs 42.36 (off); the serving
  configuration's own numbers (RigMark, conc table above) are level with v0.5.
- `SHARE_LIMIT` costs maximum context: 310K → 212K. It stays off in the serving configuration; the one-chain call is
  on.

## Multi-turn (prefix-cache hits)

Upstream runs a request on MTP only for its whole life when its prompt hit the prefix cache inside the DFlash2
drafter's 2,048-position window, or when it was not alone at its prompt tail (upstream KNOWN-ISSUES #1). A later turn
of a conversation whose new tokens are fewer than 2,048 is in the first case. Patch 0014 lets those turns draft with
DFlash2. The test is a small multi-turn script: three conversations (A, B, C; A3 is conversation A's third turn),
code-heavy, with turns sized to hit the prefix cache; T=0, c1, decode tok/s, one run each, v0.1 vs v0.2 on the same
machines.

| Turn | Case | v0.1 (MTP only) | v0.2 (DFlash2) | v0.2 / v0.1 |
|---|---|---:|---:|---:|
| A2 | prefix hit inside the window | 45.51 | 57.69 | ×1.27 |
| A3 | prefix hit | 45.34 | 50.49 | ×1.11 |
| B2 | prefix hit | 47.30 | 68.34 | ×1.44 |
| A4 | prefix hit; another conversation had taken over the drafter's context ring | 43.88 | 45.06 | ×1.03 |
| C1 | fresh, not alone at its prompt tail at the start | 44.92 | 57.40 | ×1.28 |

Mean of the per-turn ratios (`bench/multiturn.py compare`): prefix-hit turns (A2, A3, B2, A4) ×1.20 decode, fresh turns
×1.09 (driven by C1; A1 and B1 are ×1.01), replayed turn ×1.01. Single run: no variance is established.

- This is one run of a small script ([`bench/multiturn.py`](../bench/multiturn.py), stdlib Python against the local
  API), not a benchmark; all rows come from the same pair of runs.
- The turns are code-heavy. We expect prose turns to gain less, because DSel picks MTP for prose anyway; this was not
  measured.
- A turn whose conversation no longer owns the drafter's context ring (A4) warms the drafter up over a few MTP rounds
  instead of continuing it, and gains little.
- 0014 is a drafting lever: T=0 text on these turns differs between v0.1 and v0.2 (see the exactness classes in the
  README).

## Wider drafters: the AlayaNeW DFlash2 A/B (v0.7)

The recipe's drafter is incoai's DFlash2, which drafts up to B=8 tokens per round. AlayaNeW's
[GLM-5.3-DFlash2](https://huggingface.co/AlayaNeW/GLM-5.3-DFlash2) (MIT) drafts B=12 with a different draft head
layout. Patch 0032 makes B=12 drafters servable at all; this A/B measured whether the wider drafter is worth serving.

RigMark c1 decode (medians; the control ran twice, 22:0x and 22:2x, with matching results) and `bench/conc.py` at c4:

| Configuration | prose | code | structured | drafts | draft tok/round | accepted/round | C1 | C2 | C4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v0.6 drafter (B=8 incoai) | 35.17 | 56.72 | 70.46 | 5496 | 3.34 | 2.46 | 40.8 | 58.5 | 72.6 |
| AlayaNeW DFlash2 (B=12) | 18.36 | 44.66 | 71.69 | 4880 | 11.00 | 3.08 | 28.3 | 55.9 | 70.2 |

The Alaya drafter accepts more per round (+25 %) but proposes 3.3× more draft tokens per round (11.0 vs 3.34): the
verify cost of the wide blocks outweighs the acceptance gain everywhere except structured, where it is level. Per the
release acceptance rule (win per workload) it loses and is not shipped; the B=8 incoai drafter stays. The S2 fixes
(0032) remain because they are what makes any wider drafter servable. Measured 2026-10-18 00:16–00:37.

## Quality checks against earlier configurations

| Check | Result |
|---|---|
| T=0 identity / KL, v0.7 | top-20 logprobs on the 75 qeval prompts vs the dh32f reference configuration, prefix-cached pass: 34/75 identical, mean KL 2.734e-03 on the 3,590 common positions (noise scale 5.0e-3, as below; the c1-vs-c4 scale run gave 30/75, KL 4.994e-03). The same value v0.6 measured the same way |
| Sampled qeval / T=0 qeval / needle, v0.7 | 573/600 sampled with `SPEC_ACCEPT=rejection` (the dh32f reference 578/600, standard error 0.85 pp, ~1 se apart; identical in both full runs); 72/75 T=0 (74/75 on the first full run; 71–74 across the release's boots); 96K needle correct |
| T=0 identity / KL, v0.6 | top-20 logprobs on the 75 qeval prompts vs the v0.5 configuration, prefix-cached pass: 34/75 identical, mean KL 2.7e-3 on the 3,590 common positions (noise scale 5.0e-3) |
| Sampled qeval / T=0 qeval / needle, v0.6 | 577/600 sampled with `SPEC_ACCEPT=rejection` (v0.5 577/600; the dh32f reference 578/600, standard error 0.8 pp); 71/75 T=0; 96K needle correct |
| T=0 identity / KL, v0.5 | the same check, rerun on v0.5: 34/75 identical, mean KL 2.7e-3 on the common prefix (noise scale 5.0e-3) |
| Sampled qeval / T=0 qeval / needle, v0.5 | 577/600 sampled with `SPEC_ACCEPT=rejection`; 72/75 T=0; 96K needle correct |
| T=0 identity / KL, v0.4 | the same check: 34/75 identical, mean KL 2.7e-3 on the common prefix (noise scale 5.0e-3, as below) |
| Sampled qeval / T=0 qeval / needle, v0.4 | 578/600 sampled, identical per seed; 72/75 T=0; 96K needle correct |
| T=0 identity / KL, v0.3 | top-20 logprobs on the 75 qeval prompts vs an earlier configuration (GEMV split + INT4 draft heads + DSel probe 32), prefix-cached pass: 34/75 identical, mean KL 2.7e-3 on the common prefix; for scale, one configuration at concurrency 1 vs 4: 30/75 identical, mean KL 5.0e-3 |
| Sampled qeval / T=0 qeval / needle, v0.3 | 578/600 sampled; 72/75 T=0; 96K needle correct |
| T=0 identity, v0.1 configuration | top-20 logprobs on the 75 qeval prompts (up to 512 tokens) bit-identical to the previous configuration (GEMV split + INT4 heads + probe 32): 75/75 prompts, 7,551 positions. For scale, the same configuration run at concurrency 1 vs 4 differs with mean KL 5e-3 |
| T=0 identity / KL, v0.2 | top-20 logprobs on the 75 qeval prompts vs an earlier configuration (GEMV split + INT4 draft heads + DSel probe 32), second (prefix-cached) pass: 34/75 identical, mean KL 2.7e-3 on the common prefix; for scale, the same configuration at concurrency 1 vs 4: 30/75 identical, mean KL 5.0e-3. Text differs at T=0 because a cached prompt now drafts with DFlash2 (drafting class) |
| Sampled qeval, v0.2 | 578/600 (T=1.0, top_p 0.95, 8 seeds × 75 tasks), identical per seed to the previous configurations. Single T=0 run: 71/75, the same four misses as v0.1 |
| Sampled qeval, v0.1 configuration | 578/600 (T=1.0, top_p 0.95, 8 seeds × 75 tasks), identical per seed to the previous configuration. Single T=0 run: 71/75 (baseline 73/75); the extra misses are a character count off by one and one arithmetic answer, borderline tasks that flip between runs |

## RigMark, v0.2 (one full run)

[RigMark](https://github.com/alexellis/rigmark) 1.0.0, thinking on, `reasoning_effort: low`, one full run; settings in
[Reproducing](../README.md#reproducing). Each cell is RigMark's median for that workload.

| Workload | c1 decode tok/s |
|---|---:|
| prose | 35.5 |
| code | 54.8 |
| structured | 66.6 |

Concurrency, aggregate tok/s **[per-stream decode tok/s]**.

| Concurrency | C1 | C2 | C4 |
|---|---:|---:|---:|
| aggregate [per-stream] | 38.8 [42.5] | 54.3 [31.0] | 67.6 [20.9] |

| Prefill, cold | 8K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|
| tok/s | 1369 | 1370 | 1312 | 1157 |

## RigMark, v0.3 (one full run)

| Workload | c1 decode tok/s |
|---|---:|
| prose | 36.2 |
| code | 56.2 |
| structured | 65.8 |

| Concurrency | C1 | C2 | C4 |
|---|---:|---:|---:|
| aggregate [per-stream] | 43.1 [47.8] | 55.6 [31.6] | 68.6 [20.7] |

Decode screens (RigMark c1 decode only, two runs each, same boot procedure): v0.3 prose 36.80 / 36.78, code 54.79 /
54.68, structured 65.76 / 65.67; v0.2 prose 35.73 / 35.39, code 52.48 / 52.16, structured 67.40 / 67.34.
