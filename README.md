# Full GLM-5.3 on 4× NVIDIA DGX Spark with gr8 (TP4)

A recipe for serving full [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) on four DGX Sparks with tensor parallelism
4, using the [gr8](https://github.com/kindlingai/gr8) engine by kindlingai at a pinned upstream commit, plus thirty
small patches (three of them are upstream's own open PR [#98](https://github.com/kindlingai/gr8/pull/98), backported).
Many of the earlier releases' levers are now upstream code: the pinned commit includes the fabric MTU fix, the GEMV
splits, the INT4 draft heads and drafter body, the route overlap, the three rounds of decode attention kernels, the MTP
confidence stop, DFlash2 after prefix reuse, the parallel drafter step and the step timeline (upstream PRs #82–#90).
The weights are kindlingai's EXLR8 quantization
[kindlingai/glm-5.3-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4) (width manifest `home`),
with GLM-5.3's own MTP layer and the [incoai/GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) drafter.

This repository holds only the patches, the build, an example configuration and the results. It contains no engine
fork, no weights and no images: the engine comes from upstream at `c53b5b8`, the weights from Hugging Face, the base
image from upstream's registry. Every patch is behind a flag that defaults to off, so the patched engine behaves as
upstream until `G6ENV_X` turns the levers on. The exceptions are the launcher's share-limit rule (0011, part of the
launcher), the S2 drafter-shape fixes (0015, always on but invisible to the B=8 drafter: they only add capture shapes
and row headroom), and 0016, which turns SELX prefill selection on by default from 53248-token contexts; 0012 changes
the buckets only when `SHARE_LIMIT` is not a power of two. Upstream at `c53b5b8` turns some of its merged levers on by
default when serving (`G6_QA_SPLIT=2`, `G6_DRAFT_PAR=2`, `G6_MTP_KSTOP=2` and others, upstream #96); this recipe sets
the values it measured explicitly in `G6ENV_X`. The serving setup also compacts the host memory on every box before
each start ([Build and run](#build-and-run), step 5).

**One serving flag is not bit-exact:** `G6_ATTN_POSIDX=1` (0026, 0028) sums the same attention keys in another order at
contexts up to 2048 tokens. It is deterministic, but its output is not byte-identical to upstream's (exactness class
E1, [Levers](#levers)). To get upstream's summation order, remove `G6_ATTN_POSIDX=1` from `G6ENV_X`.

Release **v0.9**, 2026-10-19. The tables below are this release's measurements. What changed in each release, and by
how much, is in [CHANGELOG.md](CHANGELOG.md); A/B measurements against earlier releases and the upstream baseline are
in [docs/history.md](docs/history.md).

## speedbench

[speedbench](https://github.com/knapcio/speedbench) 0.2, full mode: 73 frozen prompts in six categories (prose, code,
structured, agentic with tools, multilingual, long documents of a few thousand tokens; prompt set speedbench-v1), c1,
temperature 0, 256 output tokens, `reasoning_effort: low`, each arm from fresh boots with the host memory compacted first (v0.8, v0.9, v0.9, v0.8), the
change paired per prompt. Decode tok/s, median per category (first run of each arm), change with 95 % confidence
interval:

| Category | v0.8 | v0.9 | change [95 % CI] |
|---|---:|---:|---:|
| prose | 43.6 | 42.9 | +0.5 % [−3.3, +4.5] |
| code | 53.7 | 54.1 | +1.3 % [−3.9, +6.7] |
| structured | 55.6 | 57.7 | +3.5 % [+0.1, +7.0] |
| agentic | 51.3 | 50.0 | −1.0 % [−5.5, +3.7] |
| multilingual | 43.5 | 45.4 | +2.9 % [−1.0, +7.0] |
| long context | 45.9 | 48.1 | +0.6 % [−2.1, +3.4] |
| all 73 prompts | 47.8 | 48.6 | +1.2 % [−1.2, +3.6] |

v0.9 is about 1 % faster than v0.8 overall, which is below what this design can resolve (detectable effect at 80 %
power: 3.5 %). The per-category medians and the paired changes can point in different directions because v0.9's
POSIDX and two-stage draft head change the generated text: each prompt's speed then also depends on its new text.
v0.9's main changes are the upstream rebase and the host-memory headroom (`G6_CP_STEP_ROWS`), not speed.

## RigMark — thinking on, effort low

Measured 2026-10-19 on the v0.9 configuration with [RigMark](https://github.com/alexellis/rigmark) 1.0.0 (settings in
[Reproducing](#reproducing)), temperature 0, `reasoning_effort: low` (the model's chat template has no thinking-off
mode). Each cell is RigMark's median for that workload, and the table gives the median of three
full runs, each from a fresh boot with the host memory compacted first. The three runs agree within 0.1 tok/s on c1
prose decode.

| c1 decode | prose | code | structured |
|---|---:|---:|---:|
| tok/s | 40.6 | 60.6 | 75.5 |

Concurrency, aggregate end-to-end tok/s **[per-stream decode tok/s]**. At c ≥ 2 the sessions share verify steps and
draft with MTP (the DFlash2 drafter serves a session that is alone), so code and structured, where DFlash2 drafts long
blocks, gain less from concurrency than prose.

| Concurrency | C1 | C2 | C4 |
|---|---:|---:|---:|
| aggregate [per-stream] | 47.0 [52.4] | 59.9 [34.0] | 76.6 [23.4] |

Cold prefill, input tok/s (unique prompts, no prefix reuse; median of the same three full runs, no separate warm-up
prefill). A 96K-token needle
prompt: correct answer.

| Prompt length | 8K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|
| cold prefill tok/s | 1381 | 1382 | 1327 | 1226 |

The decode table by prompt type and concurrency from [`bench/conc.py`](bench/conc.py) and every release's results are
in [docs/history.md](docs/history.md).

## Multi-turn conversations

RigMark sends single-turn requests. In a multi-turn conversation, turns whose prompt hits the prefix cache draft with
DFlash2 (`G6_DFL_REUSE=2`, upstream #83) instead of staying on MTP for their whole life as in upstream's default. The
measured A/B is in [docs/history.md](docs/history.md#multi-turn-prefix-cache-hits).

## Levers

Rows marked "upstream (#NN)" are levers this recipe contributed that are now part of upstream gr8 at the pinned commit;
the recipe still serves them with the flags shown. Patch numbers are those of this release; the v0.8 numbering is
mapped in [CHANGELOG.md](CHANGELOG.md).

| Patch | What it does | Flags (serving) | Exactness | Credit |
|---|---|---|---|---|
| upstream ([#82](https://github.com/kindlingai/gr8/pull/82)) | Negotiate the RDMA queue-pair MTU from both ports' active MTU instead of assuming one; logs the chosen MTU per QP. Needed to boot on a switched fabric | always on | bit-exact | knapcio; merged upstream |
| upstream ([#84](https://github.com/kindlingai/gr8/pull/84)) | Split the replicated GEMVs every rank holds in full (layer input q_a / kv_a / indexer k and weights; the indexer's wq_b; the MTP draft layer's eh_proj, q_a, kv_a, indexer) by 16-output tiles across the four ranks, then all-gather the columns per layer on a fabric of their own | `G6_QA_SPLIT=1` `G6_IDXQ_SPLIT=1` `G6_MTP_SPLIT=1` | bit-exact | knapcio; merged upstream |
| upstream ([#85](https://github.com/kindlingai/gr8/pull/85)) | MTP draft head and DFlash2 drafter head on INT4 g32 records made at boot from the target's MXFP8 head (45 % fewer head bytes per draft step); the target head and verify are unchanged | `G6_DRAFT_HEAD_I4=1` `G6_DFL_HEAD_I4=1` | drafting | knapcio; merged upstream; DFlash2 by incoai |
| — | DSel probe interval (upstream knob, no patch): while a request drafts with MTP, DFlash2 drafts a scoring probe every 32nd round instead of every 4th | `G6_S2_DSEL_PROBE=32` | drafting | DSel by kindlingai; setting by knapcio |
| — | Draft acceptance at temperature > 0 (upstream launcher option, no patch): rejection sampling instead of coupled sampling. Both give the target's output distribution; rejection keeps more drafts here. A seed's text then depends on the drafting configuration. Temperature 0 is unchanged | `SPEC_ACCEPT=rejection` | sampling: same distribution | upstream gr8; setting by knapcio |
| 0001–0003 | DSel experiments: start DFlash2 requests on MTP, a selector-walk correction coefficient, per-position observation clocks | off | — | knapcio |
| 0004 | Overrides for DSel's drafting-cost constants (ms per round). Not used in serving since v0.3: the cost table of 0005 replaces these constants | off | drafting | knapcio; DSel by kindlingai |
| upstream ([#86](https://github.com/kindlingai/gr8/pull/86)) | MoE: run the trellis shared expert on a forked stream during the router GEMV and plan kernel, so its weight reads fill idle DRAM time | `G6_ROUTE_OVL=1` | bit-exact | knapcio; merged upstream |
| upstream ([#87](https://github.com/kindlingai/gr8/pull/87)) | Decode attention, round 1: a scores kernel with several KV slots per warp and a transpose reduction; a packed, branch-free context kernel | `G6_ATTN_SC2=4` `G6_ATTN_CTX2=4` | bit-exact | knapcio; merged upstream |
| upstream ([#87](https://github.com/kindlingai/gr8/pull/87)) | Decode attention, round 2: the transpose reduction done eagerly (fewer registers), a context kernel with 32-bit row indices and warp-ballot batch states, the selector's final sort as a rank by count, and an indexer that reads each key tile once for a session's rows | `G6_ATTN_SC2_TR=2` `G6_ATTN_CTX3=2` `G6_ATTN_SEL2=16` `G6_IDX_GRP=16` | bit-exact | knapcio; merged upstream |
| upstream ([#88](https://github.com/kindlingai/gr8/pull/88)) | MTP confidence stop: a request's solo MTP rounds verify 1–3 drafts, stopping the chain where the MTP head's cumulative probability falls below a per-position threshold; a session that `G6_DFL_REUSE` promoted verifies at most `kmax` of the shared chain's drafts | `G6_MTP_KSTOP=1` `G6_MTP_KSTOP_KMAX=3` `G6_MTP_KSTOP_TAU=0,0.5,0.75` | drafting | knapcio; merged upstream |
| upstream ([#83](https://github.com/kindlingai/gr8/pull/83)) | DFlash2 after prefix reuse. Upstream's default runs a request on MTP only for its whole life when its prompt hit the prefix cache inside the drafter's 2,048-position window or it was not alone at its prompt tail (upstream KNOWN-ISSUES #1). Mode 1 lets prefix-hit turns keep DFlash2, by continuing the drafter's context ring from the previous turn of the same conversation or by warming it up over a few MTP rounds; mode 2 also promotes a session that is alone on MTP | `G6_DFL_REUSE=2` | drafting | knapcio; merged upstream; DFlash2 by incoai |
| upstream ([#87](https://github.com/kindlingai/gr8/pull/87)) | Decode attention, round 3: the absorb step stores each head's query and rope lane-transposed, so the scores kernel reads a head's query with 5 vector loads instead of 18 (`QT`); the step's front kernels (KV append, index append and index preparation on a full layer; KV append and index reuse on a shared layer) run in one launch (`FRONT`) | `G6_ATTN_QT=1` `G6_ATTN_FRONT=1` | bit-exact | knapcio; merged upstream |
| upstream ([#89](https://github.com/kindlingai/gr8/pull/89)) | DFlash2 drafter step in parallel: in a DFlash2 round of a request that decodes alone, the drafter choice is made before MTP's first form (it reads nothing the first form writes), the DFlash2 drafter step runs on its own CUDA stream under the first form (and the MTP chain), and the round waits for it at the end. The same graphs and inputs as serial, so every step's bytes are unchanged | `G6_DRAFT_PAR=1` | bit-exact | knapcio; merged upstream; DFlash2 by incoai |
| upstream ([#90](https://github.com/kindlingai/gr8/pull/90)) | Decode step timeline: per step kind and row count, the GPU span, host input preparation, launch and fence wait, and the GPU idle between steps, printed every N main steps on every rank. Used to measure the DSel cost table (0005); off in serving | off (`G6_S2_TIMELINE=N` when measuring) | diagnostics only | knapcio; merged upstream |
| 0005 | DSel cost model: the drafter choice prices DFlash2 and MTP from one frozen cost table (verify ms by row count, drafting ms per DFlash2 round, MTP round and MTP chain step, a probe's extra, and the MTP rounds' prior under the confidence stop) instead of fixed per-round constants and the built-in verify table, and prices an MTP round as the confidence stop runs it, not as K drafts. `tools/s2_dsel_table.py` writes the table from a timeline boot (see [Reproducing](#reproducing)) | `G6_S2_DSEL_MODEL=1` `G6_S2_DSEL_TABLE=…` | drafting | knapcio; DSel by kindlingai |
| 0006 | `tools/s2_dsel_table.py`: `--w` (the estimates' prior weight) and the measurement boot procedure in its usage text | — (tool) | — (tool, not in the engine) | knapcio |
| 0007 | DSel cost model, table key `r`: price DFlash2 from the request's DFlash2 rounds and scored probes as they ran (committed tokens over their table cost, aged by `r` per MTP round) instead of its position-rate estimate, so a request that a dip moved to MTP can come back to DFlash2 | `r:0.95` in `G6_S2_DSEL_TABLE` | drafting | knapcio; DSel by kindlingai |
| 0008 | `tools/s2_dsel_table.py` usage text: the measurement boot removes the model and `G6_DRAFT_PAR` and appends the diagnostics | — | tool docs | knapcio |
| upstream ([#85](https://github.com/kindlingai/gr8/pull/85)) | DFlash2 drafter body on INT4 g32 records made at boot from its MXFP8 records (fc and, per layer, kv / aconv / q / o / mconv / gu / down, same splits): the drafter body's records take 382 MB instead of 701 MB | `G6_DFL_BODY_I4=1` | drafting | knapcio; merged upstream; DFlash2 by incoai |
| 0009 | Decode: paced L2 prefetch of the layer's next weights (input projections, q_b, the attention's absorb and value-projection weights, o_proj, router, shared expert) from one warp on a side stream, in the windows where DRAM is idle: the two all-reduce waits, the column gathers and the latency-bound attention. Small chunks at a set rate (after [dsgx10](https://github.com/blaesus/dsgx10)) instead of a burst, stopped when the collective arrives; a prefetch writes nothing | `G6_L2PF=15` | bit-exact | knapcio; pacing after blaesus/dsgx10 |
| 0010 | MoE decode kernel: a split table per step row count (`G6_K1_SPLIT`: fc1 CTA shape and fc2 n-split, same summation order), and a translation prefetch at the start of each expert's fc1 (`G6_K1_TLBW`), because a step's routed experts touch about 5,300 2 MB pages per rank, more than the GPU's translation cache covers (about 4 GB). Also in the patch, off: the weight image from `cuMemCreate` chunks (`G6_IMG_VMM`) and the `k1_audit` bench | `G6_K1_SPLIT=1:/4,3:2x1,4-32:/4` `G6_K1_TLBW=2048` | bit-exact | knapcio |
| 0011 | The launcher's `--share-limit` takes any multiple of 64 from 4096, not only powers of two, so a context-share limit such as 98304 (the cap is 212224 at TP=4) is accepted | `SHARE_LIMIT=65536` (a power of two; the rule matters for other values) | — (launcher) | knapcio |
| 0012 | Decode buckets: nine slots instead of eight, so a share limit off the power-of-two doublings (98304) no longer merges the 2048 bucket away and short-context steps keep their own graph. With no limit, or a power-of-two limit, the bucket list, the graphs and the outputs are those of before | (with `SHARE_LIMIT`; inert at 65536) | bit-exact with a power-of-two limit; a ninth-bucket configuration regroups verify rows and can change T=0 output | knapcio |
| 0013 | CP decode: past R, the other ranks' head groups run as one 48-head attention call (one `k_vget`, one call whose merge writes each group's partial, the three puts in the S1 order) instead of three separate chains: 8 launches a layer instead of 18. TP=4 only, and not with `G6_ATTN_FLIP` | `G6_CP_ONECHAIN=1` | bit-exact | knapcio |
| 0014 | S2 per-session MTP confidence stop for shared rounds (KStopB): the shared chain still drafts K drafts; each session verifies its own, usually smaller, count of them — the most j with the running product of draft probabilities past the per-position thresholds. Thresholds are per active session count (`G6_MTP_KSTOPB_TAU_S<n>`, frozen values measured 2026-10-17); `G6_MTP_LPLOG=1` logs every shared round's verification and timing. Extra boot cost: additional main widths captured (~115 MiB per extra width per rank) | `G6_MTP_KSTOPB=1` `G6_MTP_LPLOG=1` `G6_MTP_KSTOPB_TAU_S1..S4` | drafting | knapcio |
| 0015 | S2 drafter-shape fixes for B=12 drafters (measured with the AlayaNeW DFlash2 drafter): boot captures every DFlash2 first-form row count (1..dfl_k+1, not just the prompt chunk), the per-step row budget includes the verify width, and the acceptance histogram ceiling is widened to 16 entries, past anything a 12-block round can produce. With the B=8 drafter this changes nothing observable: the shapes were already covered by the prompt chunk | always on (no flag) | bit-exact for the B=8 drafter | knapcio; AlayaNeW drafter MIT |
| 0016 | Prefill index-tile selection (SELX) turns on by default from 53248-token contexts (was env-only, default off). `G6_PF_SELX_CTX` still overrides; values 32769..1048576 | default on (set `G6_PF_SELX_CTX` to override) | prefill (kernel selection; below the threshold identical to the flag off) | knapcio |
| 0017–0019 | kv_b (the attention's UK / UV absorb and value-projection weights) read as the release's own block FP8 (E4M3 + one f32 scale per 128 × 128 block) instead of the BF16 copy the loader makes: half the bytes on every decode step. The kernels take FP8 as a template parameter and round each weight to the same BF16 value the BF16 loader stores, so the products are the same. Upstream's open PR [#98](https://github.com/kindlingai/gr8/pull/98), its three commits as published | `G6_KVB=fp8` | bit-exact | spectra-kindling ([#98](https://github.com/kindlingai/gr8/pull/98)) |
| 0020 | 0009's L2 prefetch reads kv_b at its stored width under `G6_KVB=fp8` (the E4M3 bytes plus the f32 block scales, not BF16-sized ranges); prefetch queue ranges 16 → 20 so the scales do not push the router or shared-expert ranges out | with `G6_L2PF` and `G6_KVB=fp8` | prefetch only (no arithmetic) | knapcio |
| 0021 | 0009's L2 prefetch reports a range dropped by a full queue instead of dropping it silently | with `G6_L2PF` | prefetch only | knapcio |
| 0022 | Router GEMV with one output tile per CTA (16 CTAs instead of 8). Not used in serving | off (`G6_ROUTER_TPC=1`) | bit-exact | knapcio |
| 0023 | CP step rows: a decode step past R (`SHARE_LIMIT`) carries at most n rows, and the CP fabric is sized for n rows. Upstream #94 lets several sessions past R share steps and sizes the CP fabric for the widest S2 step (24 rows at 8 sessions and MTP 2: 1.358 GB pinned per rank, 0.453 GB before), which left the head box at about 0.5 GiB of available memory and hung it (`NV_ERR_NO_MEMORY`). With n = 8 the fabric is back to 0.453 GB: available memory after boot 3.3 GiB on the head, 2.5–2.8 GiB on the others. While a session past R decodes alongside others whose rows do not fit in n, a round takes at most n / (MTP + 1) sessions, those past R first, and the sessions take turns; rounds with no session past R keep the full batch. Proposed upstream as [#106](https://github.com/kindlingai/gr8/pull/106) | `G6_CP_STEP_ROWS=8` | scheduling (drafting class): unchanged while not engaged; engaged, it changes which sessions share a step (see Exactness classes) | knapcio |
| 0024 | `batch_dsel_test`: the start-on-MTP case uses the cost table set before the reset case's | — (test) | test only | knapcio |
| 0025 | Decode attention context over 64- or 128-slot partitions instead of 256 (4× or 2× the CTAs, shorter serial chains, the partials summed in partition order). Measured no gain; not used in serving | off (`G6_ATTN_CTX3_SP=64\|128`) | E1 | knapcio |
| 0026, 0028 | Positional index at contexts up to 2048 tokens: at a decode bucket of 2048 keys or fewer the top-2048 selection keeps every key at or before the row's position anyway, so the indexer only decides the keys' order. The attention lists the keys by position instead and skips the indexer's scores, the selector and that layer's idx_q GEMV (the idx_q gather still runs, so every rank's collectives keep their order); under `G6_ATTN_FRONT` the front kernel has a form of its own (0028). Larger buckets, the CP path and prefill are unchanged. **Not bit-exact:** the same keys and the same scores, but normalize and the context sum them in position order instead of the indexer's order, so the low bits differ from upstream's. Deterministic. To turn it off, remove `G6_ATTN_POSIDX=1` from `G6ENV_X` | `G6_ATTN_POSIDX=1` | **E1** (not byte-identical to upstream) | knapcio |
| 0027 | attnk bench: the E1 arms of 0025 and 0026 (determinism, what must not change, the error against stock, timing) and `engine/attn/tests/attnk_r10a.sh`, the sweep | — (bench) | bench only | knapcio |
| 0029 | MTP first form, last row only: a first-form draft step of a + 1 rows exists to write layer 78's caches for its first a rows, and only the last row's token, log-probability and hidden state are read. The caches are written in the attention block, before the MoE, so the other rows skip layer 78's routed experts (they keep the shared expert); the attention, the head and every collective run as before | `G6_MTP_LASTROW=1` | bit-exact | knapcio |
| 0030 | MTP draft head in two stages: a 2-bit g32 screen of the rank's vocab slice (74.5 MB per rank at TP=4, made at boot from the MXFP8 head; the INT4 head is 134 MB), then the exact MXFP8 logits of the screen's 64 best rows per draft row. Draft steps of up to 8 rows; wider steps keep the INT4 head. The DFlash2 head is unchanged. 74.5 MB more device memory per rank | `G6_DRAFT_HEAD2=1` | drafting | knapcio |

The measured contribution of each lever is in [docs/history.md](docs/history.md#per-lever-contributions).

**Exactness classes.** *Bit-exact*: the patched path produces the same bytes as upstream's for the same inputs, with
the serving flags above. Evidence: the MTU fix changes transport setup only; the GEMV split was byte-compared with
m4sim (the engine's single-GPU simulator of all ranks) on the plain, graph, MTP, S2 and CP paths; the route overlap
gave identical T=0 output on the four-box cluster (and is byte-compared by its own test,
`engine/kernels/tests/k1_ovl_test.cu`); attention rounds 1 and 2 are byte-identical in the attnk bench at contexts 2K /
8K / 32K × 1 / 3 / 8 / 12 rows and gave identical T=0 output on the cluster; round 3's serving flags (`G6_ATTN_QT=1`,
`G6_ATTN_FRONT=1`) are byte-identical in the attnk bench and gave identical T=0 output on the cluster; the parallel
drafter step (`G6_DRAFT_PAR=1`) runs the same graphs on the same inputs, gives the same tokens, steps and statistics as
serial in its host test (`batch_draftpar_test`) and gave identical T=0 output on the cluster. The paced L2 prefetch
(0009) and the MoE split table and translation prefetch (0010) gave byte-identical outputs in their one-GPU benches
(`l2pf_bench`, `k1_audit`) and identical T=0 top-20 logprobs on the 75 qeval prompts on the cluster (75/75, KL 0, each
configuration from a fresh boot). The MTP last-row routing (0029) keeps every byte the step's readers use: the kept
rows are byte-compared against the full plan by `engine/kernels/tests/mtp_lastrow_test.cu`, and m4sim with
`G6_MTP_LASTROW_CHECK=1` (`scripts/mtp_lastrow_check.sh`) compares the caches and last rows flag off vs on. `mtp_lastrow_test` passed 400/400 cases on one GB10; the m4sim
check ships with the patch and was not run for this release. Exactness covers these flags only.

*E1*: deterministic, a different fp summation order, not byte-identical to upstream. The same inputs give the same
bytes on every run, and the terms summed are the same as upstream's, but they are added in another order, so the
low bits of the logits differ from upstream's and T=0 text can diverge from upstream's over long generations. In
serving this is `G6_ATTN_POSIDX=1` only (0026, 0028), at contexts up to 2048 tokens; `G6_ATTN_CTX3_SP` (0025) is also
E1 and is off. Evidence: the attnk bench (Quality) and the qeval KL and scores below.

*Diagnostics only*: the decode step timeline (upstream #90) adds CUDA events and host clocks around the step launches
and prints them; it is off in serving and used only to measure the DSel cost table.

*Drafting*: deterministic, and the target model, its weights and its precision are unchanged; every committed token is
verified by the target. These levers change only what is drafted and when. Upstream gr8's verify step is not
row-count invariant (its fp32 summation order depends on how many rows a step verifies), so a drafting change can
regroup verify rows and change the low bits of the logits. The output is deterministic for a given configuration but
not guaranteed byte-identical to upstream over long generations. The CP step-row cap (0023) is in this class for
scheduling: while it is not engaged (no session past R, or every decoding session's rows fit in 8) the steps are
upstream's; engaged, the sessions take turns, which changes which rows share a step, so the T=0 output of a mixed long
+ short workload can differ. A session that decodes alone is unaffected.

## Quality

| Check | Result |
|---|---|
| T=0 identity / KL | v0.9, top-20 logprobs on the 75 qeval prompts vs the dh32f configuration, prefix-cached pass: 35/75 prompts identical, mean KL 2.830e-03 on the 3,622 common positions (max 0.465), against v0.8's 2.734e-03 measured the same way; for scale, one configuration run at concurrency 1 vs 4 differs by about 5.0e-03 (details in [docs/history.md](docs/history.md)) |
| Positional index (`G6_ATTN_POSIDX=1`, E1), attnk bench | deterministic (repeated calls byte-identical); error vs stock within the fp summation reorder; byte-identical to stock past 2048 keys, where it is inert (`attnk_r10a.sh` sweep: every check passed) |
| Sampled qeval | v0.9 578/600 on each of the three runs (T=1.0, top_p 0.95, 8 seeds × 75 tasks, rejection sampling: `SPEC_ACCEPT=rejection`; the dh32f reference scored 578/600) |
| T=0 qeval | v0.9 71, 71 and 70/75 on the three runs (v0.8 72/75 on each of its three) |
| Needle | 96K-token needle: correct answer |
| CP step rows (`G6_CP_STEP_ROWS=8`) | host test `cp_rows_test`: random requests (past R from the start, crossing R, insert rules, grow stalls) at n = 8 and 12: every request emits its solo tokens, no step past R wider than n, waits bounded, two ranks agree on every step; the T=0 identity probe on the cluster gives the same output with the cap as without |
| kv_b FP8 exactness (`G6_KVB=fp8`, 0017–0021) | `kvb8_test` on one GB10 (v0.8): every decode and prefill case byte-identical to the BF16 form of the same weights (every absorb / value-projection variant, alone and under QT); identical T=0 output on the four-box cluster on seven fresh boots. The three commits are unchanged in v0.9 |
| GEMV split exactness, m4sim | every combination of `G6_QA_SPLIT` / `G6_IDXQ_SPLIT` / `G6_MTP_SPLIT` byte-identical to flags off on the plain, graph, MTP, S2 and CP paths: 30/30 |
| Attention exactness, attnk bench (rounds 1 and 2, serving flags) | byte-identical to stock at T 2K / 8K / 32K × M 1 / 3 / 8 / 12 |
| Attention exactness, round 3 (`G6_ATTN_QT=1` `G6_ATTN_FRONT=1`) | byte-identical to stock in the attnk bench; identical T=0 output on the four-box cluster |
| Parallel drafter step (`G6_DRAFT_PAR=1`) | same tokens, steps and statistics as serial in `batch_draftpar_test`; identical T=0 output on the four-box cluster |
| Route overlap exactness | identical T=0 output on the four-box cluster; byte-compared by `engine/kernels/tests/k1_ovl_test.cu` |
| L2 prefetch and MoE split / translation prefetch exactness (`G6_L2PF`, `G6_K1_SPLIT`, `G6_K1_TLBW`) | byte-identical in `l2pf_bench` and `k1_audit`; T=0 top-20 logprobs on the 75 qeval prompts identical to the configuration without them, 75/75 prompts, KL 0 (fresh boot each) |
| CP one-chain attention (`G6_CP_ONECHAIN=1`) | the merged call's partials byte-identical to the three separate calls (`engine/attn/tests/cpgroups.cu`); identical T=0 decode at 70K / 80K / 90K-token prompts (same output SHA as the flags-off boot) |
| Ninth decode bucket (`SHARE_LIMIT=98304`) | with a power-of-two limit the buckets, graphs and outputs are unchanged (host tests of `context_buckets`); a ninth bucket regroups verify rows, so its T=0 output is not comparable byte for byte (drafting class) |

qeval is the 75-task quality screen from the earlier
[GLM-5.3 recipe](https://github.com/knapcio/GLM-5.3-4x-DGX-Spark-TP4) (`bench/qeval.py`). The m4sim exactness runs used
a development harness that runs the split gathers on a loopback; it is not included in this patch series. T=0 identity
and KL checks against earlier configurations are in [docs/history.md](docs/history.md#quality-checks-against-earlier-configurations).

## Reproducing

- **Engine flags** (serving): `G6ENV_X`, `SPEC_ACCEPT` and `SHARE_LIMIT` in [docker/env.example](docker/env.example),
  the same lines on every box:

  ```bash
  G6ENV_X="G6_QA_SPLIT=1 G6_IDXQ_SPLIT=1 G6_MTP_SPLIT=1 G6_DRAFT_HEAD_I4=1 G6_DFL_HEAD_I4=1 G6_S2_DSEL_PROBE=32 G6_ROUTE_OVL=1 G6_MTP_KSTOP=1 G6_MTP_KSTOP_KMAX=3 G6_MTP_KSTOP_TAU=0,0.5,0.75 G6_ATTN_SC2=4 G6_ATTN_SC2_TR=2 G6_ATTN_CTX2=4 G6_ATTN_CTX3=2 G6_ATTN_SEL2=16 G6_IDX_GRP=16 G6_MTP_KSTOPB=1 G6_MTP_LPLOG=1 G6_MTP_KSTOPB_TAU_S1=0.7,0.7 G6_MTP_KSTOPB_TAU_S2=0.7,0.8 G6_MTP_KSTOPB_TAU_S3=0.7,0.88 G6_MTP_KSTOPB_TAU_S4=0.7,0.75 G6_DFL_REUSE=2 G6_ATTN_QT=1 G6_ATTN_FRONT=1 G6_DRAFT_PAR=1 G6_S2_DSEL_MODEL=1 G6_S2_DSEL_TABLE=v2:46.63,v3:54.71,v4:63.99,v5:71.54,v6:79.37,v7:88.23,v8:96.82,d:7.285,m:2.039,e:1.155,p:5.026,g:1.107,k:2.177,n:2.486,w:1,r:0.95 G6_DFL_BODY_I4=1 G6_L2PF=15 G6_K1_SPLIT=1:/4,3:2x1,4-32:/4 G6_K1_TLBW=2048 G6_CP_ONECHAIN=1 G6_KVB=fp8 G6_CP_STEP_ROWS=8 G6_ATTN_POSIDX=1 G6_DRAFT_HEAD2=1 G6_MTP_LASTROW=1"
  SPEC_ACCEPT=rejection
  SHARE_LIMIT=65536
  ```

  `G6_S2_DSEL_MCOST` is no longer set: with `G6_S2_DSEL_MODEL=1` the drafter choice prices both drafters from
  `G6_S2_DSEL_TABLE` and ignores the per-round cost constants. `SHARE_LIMIT=65536` is the launcher's default at TP=4,
  written out because `G6_CP_STEP_ROWS` acts on the steps past it. For upstream's attention summation order, remove
  `G6_ATTN_POSIDX=1`.

- **DSel cost table** (`G6_S2_DSEL_TABLE`): measured on the four boxes of this recipe with `tools/s2_dsel_table.py`
  in the patched tree (added by patches 0005 and 0006; its usage text has the procedure). The values are milliseconds
  on this hardware (GB10, this fabric, the stock clock cap, these kernels): on other hardware, or after changing the
  kernels or the K-stop settings, measure it again, because a table from other machines misprices the drafters.
  The served table above was measured on the v0.3 configuration, before `G6_DFL_BODY_I4`, `G6_L2PF`, the MoE levers of
  0010 and the later levers, which shorten the steps it prices (the comparison with a table measured with
  `G6_DFL_BODY_I4` on is in [CHANGELOG.md](CHANGELOG.md)). Measurement boot:

  1. Boot the serving configuration being priced with the model off and the timeline on: the `G6ENV_X` line above
     without `G6_S2_DSEL_MODEL=1` and `G6_S2_DSEL_TABLE=…`, with `G6_DRAFT_PAR=0` instead of `G6_DRAFT_PAR=1`
     (removing the flag is not enough: upstream's serving default at this commit is 2), plus
     `G6_S2_TIMELINE=200 G6_MTP_LPLOG=1`. `G6_DRAFT_PAR` must be off while measuring (the drafter step would overlap
     the first form; the tool refuses such logs), and `G6_S2_ROUNDLOG` must not be set (its copies lengthen the main
     steps).
  2. Run solo decode that visits both drafters: RigMark decode screens at c1 (prose, code and structured).
  3. Pass every rank's engine log (stderr, all four boxes) to the tool and check the derivation it prints on stderr:
     each verify row count the drafters reach (2–4 on MTP, 2–8 on DFlash2) from enough steps, and enough MTP rounds
     for `g`, `k`, `n`:

     ```bash
     python3 tools/s2_dsel_table.py rank0.log rank1.log rank2.log rank3.log --w 1 --r 0.95
     ```

     It prints the table on stdout, one line; put it in `G6_S2_DSEL_TABLE` (with `G6_S2_DSEL_MODEL=1` and
     `G6_DRAFT_PAR=1` back) on every box. Every rank must read the same table: it is part of the S2 runtime signature.

- **speedbench** 0.2 ([github.com/knapcio/speedbench](https://github.com/knapcio/speedbench)) against the head's API, one
  run per fresh boot, arms alternated A, B, B, A:

  ```bash
  python3 -m speedbench run --mode full --url http://127.0.0.1:8100 --metrics --label <arm> --block <boot> --out <arm>-<boot>.json
  python3 -m speedbench compare --a A-boot1.json A-boot4.json --b B-boot2.json B-boot3.json
  ```
- **Decode table**: `python3 bench/conc.py --prompts <rigmark>/prompts.json` against the head's API (rounds 3,
  256 output tokens).
- **Optional context-share limit**: `SHARE_LIMIT=98304` instead of 65536 (the launcher's `--share-limit`, patches 0011
  and 0012) speeds up long-context decode
  ([docs/history.md](docs/history.md#long-context-one-chain-and-the-ninth-bucket-v06)) and costs maximum context
  (310K → 212K). Not part of the serving configuration.
- **Decode over 24 prompts** (used for the release A/B in [CHANGELOG.md](CHANGELOG.md)): `python3 bench/decset.py`
  against the head's API; 12 prose, 6 code and 6 structured prompts in the script, c1, T=0, 384 output tokens.

- **RigMark** 1.0.0 ([github.com/alexellis/rigmark](https://github.com/alexellis/rigmark)) against the head's API. Full
  runs:

  ```bash
  rigmark run --base-url http://127.0.0.1:8100 --model glm-5.3 \
    --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}' \
    --prefill-depths 8192,32768,65536,131072
  ```

  Decode screens: the same command with `--skip-prefill --skip-concurrency`. Each cell is RigMark's median decode
  tok/s for that workload.
- **Multi-turn**: [`bench/multiturn.py`](bench/multiturn.py), stdlib Python against the local API; usage in its
  docstring.
- **Hardware**: the four boxes at the stock GPU clock cap, no clock changes.
- **Attention exactness** (upstream #87's kernels; the positional index of 0026 and 0028): the attnk bench in the
  patched tree compares every arm against the stock chain. On an idle Spark (not while serving), with CUDA 13:

  ```bash
  ./scripts/apply.sh gr8-patched          # the patched source tree (the E1 arms and attnk_r10a.sh come with the patches)
  cd gr8-patched/engine/attn/tests
  eval "$(sed -n '/^K1H2_FLAGS=/,/"$/p' ../../../scripts/build.sh)"
  nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a $K1H2_FLAGS -DG6_NR=4 -DDEC_S2 -DPF_F12 \
    -o attnk attnk.cu ../bind.cu -lcublas
  for M in 1 3 8 12; do for TL in "2048 1000" "8192 8000" "32768 32000"; do
    ATTNK_ARMS="base QT=1,FRONT=1" ./attnk $TL $M 2 > attnk.log || echo "FAIL (exit $?): T L = $TL, M = $M"
    grep '^EXACT' attnk.log
  done; done
  ./attnk_r10a.sh ./attnk attnk_r10a.log  # the E1 arms (POSIDX, CTX3_SP) against the served attention flags
  ```

  `base` is the bench's default base, rounds 1 and 2 with their serving flags
  (`SC2=4,TR=2,CTX2=4,CTX3=2,SEL2=16,IDX=16`); the second arm adds round 3's serving flags. Each arm's bytes are
  compared against the stock chain. Every `EXACT` line must end in `IDENTICAL` and no `FAIL` line may appear.
  `attnk_r10a.sh` checks each E1 arm for determinism and for what must not change (the selected keys, each key's
  score, the appended rows, the error words), prints its error against stock and its timing, and ends with
  `all checks passed` or the failing lines.
- **GEMV split exactness**: measured with m4sim and a development harness that is not included (see Quality).

## Known limits

- **`G6_ATTN_POSIDX=1` is not bit-exact** (E1). At contexts up to 2048 tokens the attention sums the same keys in
  position order instead of the indexer's order: deterministic, but the low bits differ from upstream's, and T=0 text
  can diverge from upstream's. Remove `G6_ATTN_POSIDX=1` from `G6ENV_X` to get upstream's order. Under it a nonfinite
  indexer score no longer drops its key and no longer raises the attention's error word 4 (the positional path does
  not compute the indexer scores).
- **`G6_CP_STEP_ROWS=8` makes sessions take turns past `SHARE_LIMIT`.** While a session past R decodes alongside
  others and their rows do not fit in 8 rows (more than two sessions at MTP 2), a round takes at most two sessions,
  those past R first, and every session gets one round per cycle. Per-session decode is slower in that window:
  with S such sessions a session waits about ceil(S / 2) − 1 rounds between its own (three rounds with eight long
  sessions); not measured separately for this release. Without the cap the head
  box can run out of host memory.
- **Upstream's steering profiles were not measured with these flags.** The pinned commit adds steering support
  (upstream #69–#81, #45); this recipe neither uses nor tested it.
- **Code decode varies more than prose** between runs and between short screens and full runs.
- **Verify numerics are not row-count invariant in upstream gr8.** A step that verifies 3 rows and one that verifies 8
  can sum in a different order. T=0 output in upstream also depends on prefix-cache state (a cold prompt and a cached
  one can differ). Compare T=0 runs only under the same cache state.
- **T=0 output depends on what the server ran before** (upstream behaviour): the same configuration gave identical
  top-20 logprobs on 39 of the 75 qeval prompts from a fresh boot and after a full RigMark run, whose concurrency
  section leaves timing-dependent state. Compare T=0 runs only from the same state.
- **`SPEC_ACCEPT=rejection` ties a seed's text to the drafting configuration.** The output distribution is the
  target's, as with upstream's default coupled sampling, but a fixed seed gives other text when the drafters, their
  weights or the drafter choice change (coupled sampling gives the same text with or without speculation).
- **TP=4 only.** Only the fp8-KV engine (`bin/g6m4`) is rebuilt, for TP=4 alone. Upstream's g6m4 runs every TP width
  and its `bin3/` and `bin6/` link to `bin/`, so this image does not run other widths. `--kv-format int4` runs
  upstream's unpatched `g6m4-int4` and was not tested with these flags.
- **Same flags on every box.** `G6ENV_X` is a cluster-wide launcher option, so use one `docker/.env` everywhere. The
  split flags add collectives at engine start, before the engines compare their settings, so ranks started with
  different split flags by other means can hang the boot instead of failing.
- **DFlash2 after prefix reuse (`G6_DFL_REUSE`) acts on a session that is alone**: mode 1 on a prefix-hit turn that
  upstream would give DFlash2 apart from the hit (alone at its prompt tail), mode 2 on the only active session. With
  several concurrent requests upstream's choice of drafter stands. A prefix-hit turn continues the drafter's context
  only while the drafter's ring still holds that conversation's previous turn; otherwise it warms the drafter up over
  MTP rounds (`G6_DFL_REUSE_MIN`, default 256 rows) before it drafts with DFlash2.
- **The DSel cost table is specific to these machines.** `G6_S2_DSEL_TABLE` holds measured milliseconds; on other
  hardware, or with other kernels or K-stop settings, measure it again (Reproducing). The drafter choice only decides
  which drafter proposes tokens and every committed token is verified by the target. Because verify numerics depend on
  the number of rows in a step (upstream behaviour), a different drafter choice can also change the generated text at
  T=0, within the same class as batching; a wrong table mainly costs speed.
- **The parallel drafter step (`G6_DRAFT_PAR`) acts on a session that is alone** drafting with DFlash2; batched rounds
  run as in upstream.
- **`G6_CP_ONECHAIN=1` is TP=4 only and refuses `G6_ATTN_FLIP`**: the boot dies if either does not hold.
- **Never set `G6_ATTN_P11=1`**: it takes precedence over `G6_ATTN_SC2` / `G6_ATTN_CTX2` / `G6_ATTN_CTX3` and replaces
  the measured attention kernels.

## Build and run

Requirements: four DGX Sparks (GB10, DGX OS with the NVIDIA container toolkit), a RoCE fabric between them as upstream
gr8 expects, and about 90 GB of disk per box for its part of the model.

1. Build the image (linux/arm64; on a Spark or any arm64 builder with Docker Buildx). The Dockerfile fetches upstream
   gr8 at `c53b5b8`, applies `patches/` with `scripts/apply.sh`, checks the resulting source tree, builds the TP=4
   engine and places it in the pinned upstream image:

   ```bash
   docker buildx build --platform linux/arm64 -t gr8-glm53-recipe:v0.9 --load .
   ```

   Use the same image on all four boxes (build it on each, or `docker save` / `docker load`).

2. On every box, get upstream's compose file and put this recipe's settings next to it:

   ```bash
   git clone https://github.com/kindlingai/gr8 && git -C gr8 checkout c53b5b8
   cp docker/env.example gr8/docker/.env     # set PEERS, GR8_CACHE, GR8_HOME; keep G6ENV_X as is
   ```

3. On every box, create the cache and state directories, owned by the container's uid 1000 (Docker would create a
   missing one owned by root, and the container refuses to start), and set the same paths in `gr8/docker/.env`:

   ```bash
   sudo mkdir -p /srv/gr8/cache /srv/gr8/home
   sudo chown 1000:1000 /srv/gr8/cache /srv/gr8/home
   ```

   Or leave `GR8_CACHE` / `GR8_HOME` unset to use Docker named volumes.

4. Prepare, once per box (this box's part of the model and the drafter), as in upstream's
   [docker/README.md](https://github.com/kindlingai/gr8/blob/c53b5b869818950a96a9b72c647d4b156b057127/docker/README.md):

   ```bash
   docker compose -f gr8/docker/compose.yaml run --rm gr8 prepare
   ```

5. **Before starting, restrict the network.** The container uses host networking and the head's OpenAI-compatible
   front end listens on `0.0.0.0:8100` with no authentication. Run it only on a trusted network, firewall port 8100
   (and the cluster's control ports) to the machines that need it, and use Tailscale or a VPN for remote access.
   Never forward these ports on a public router. Then, on every box, compact the host memory and start:

   ```bash
   sudo sh -c 'sync; echo 1 > /proc/sys/vm/compact_memory'
   docker compose -f gr8/docker/compose.yaml up -d
   ```

   Compaction is a one-shot request to the kernel (no setting changes, no cache dropped). GB10's GPU allocates from
   host memory, and on a fragmented box its allocations get small physical pages; RigMark prose c1 measured 38.3 tok/s
   on v0.7 as previously served and 39.4 with compaction before the boot (same release, same day; see
   [docs/history.md](docs/history.md#v08-compaction-and-kv_b-fp8)). Repeat it before every restart of the engine.

The head (first address in `PEERS`) serves the API. `scripts/apply.sh DIR` on its own produces the patched source tree
for reading or for a bare-metal build; `scripts/make-patches.sh` documents how `patches/` was made.

## Licences

- **Engine and this repository**: the patches, scripts and documentation are MIT ([LICENSE](LICENSE)), like upstream
  gr8, whose notice is in [LICENSES/MIT-gr8.txt](LICENSES/MIT-gr8.txt). The MIT licence does not cover any weights.
- **EXLR8 weights** (not included): the
  [GLM-5.3 licence](https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE), plus CC BY 4.0 for kindlingai's
  quantization additions; see the
  [model card](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4#licence).
- **DFlash2 drafter** (not included): **CC BY-NC-ND 4.0**, noncommercial. Downloading and converting it locally, as
  `prepare` does, does not lift that restriction; commercial use needs a separate licence from incoai (see its
  [model card](https://huggingface.co/incoai/GLM-5.3-DFlash2#license)).

See [CREDITS.md](CREDITS.md) and [NOTICE](NOTICE).
