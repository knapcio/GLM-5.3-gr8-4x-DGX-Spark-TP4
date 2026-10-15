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
requests sent at once; one boot per release.

| Release | Prompt type | c1 | c2 | c4 |
|---|---|---:|---:|---:|
| v0.3 | prose | 36.2 [37.3] | 49.2 [26.1] | 63.3 [17.7] |
| v0.3 | code | 46.3 [47.4] | 59.2 [31.5] | 75.5 [21.4] |
| v0.3 | structured | 70.6 [73.2] | 65.7 [35.7] | 85.0 [25.0] |
| v0.2 (2026-10-15) | prose | 34.9 [35.8] | 48.9 [25.9] | 62.8 [17.6] |
| v0.2 (2026-10-15) | code | 43.4 [44.3] | 58.8 [31.2] | 74.8 [21.3] |
| v0.2 (2026-10-15) | structured | 69.5 [72.0] | 65.2 [35.4] | 84.3 [24.8] |

## RigMark: releases and baseline

[RigMark](https://github.com/alexellis/rigmark) 1.0.0, thinking on, `reasoning_effort: low`, decode tok/s; settings in
the README's Reproducing section. RigMark sends single-turn requests.

| c1 decode | prose | code | structured |
|---|---:|---:|---:|
| v0.2, full run | 35.5 | 54.8 | 66.6 |
| v0.1, full run | 35.4 | 54.9 | 66.7 |
| v0.1, decode screens (two runs) | 35.76 / 35.74 | 52.51 / 52.56 | 67.51 / 67.36 |
| baseline (`d9188e3` + MTU fix), full run | 29.0 | 48.1 | 60.6 |

| Concurrency, aggregate decode tok/s | C1 | C2 | C4 |
|---|---:|---:|---:|
| v0.2 | 38.8 | 54.3 | 67.6 |
| v0.1 | 39.4 | 55.7 | 67.5 |
| baseline (`d9188e3` + MTU fix) | 34.7 | 50.0 | 63.7 |

| Prefill, cold, tok/s | 8K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|
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

## Quality checks against earlier configurations

| Check | Result |
|---|---|
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
