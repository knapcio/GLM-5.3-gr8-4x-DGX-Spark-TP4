# Full GLM-5.3 on 4× NVIDIA DGX Spark with gr8 (TP4)

A recipe for serving full [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) on four DGX Sparks with tensor parallelism
4, using the [gr8](https://github.com/kindlingai/gr8) engine by kindlingai at a pinned upstream commit, plus fourteen
small patches. The weights are kindlingai's EXLR8 quantization
[kindlingai/glm-5.3-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4) (width manifest `home`),
with GLM-5.3's own MTP layer and the [incoai/GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) drafter.

This repository holds only the patches, the build, an example configuration and the results. It contains no engine
fork, no weights and no images: the engine comes from upstream at `6448a16`, the weights from Hugging Face, the base
image from upstream's registry. Every patch except the fabric MTU fix (0001, always on) is behind a flag that defaults
to off, so the patched engine behaves as upstream until `G6ENV_X` turns the levers on.

Release **v0.2**, 2026-10-15: v0.1 plus patch 0014, DFlash2 after prefix reuse (`G6_DFL_REUSE=2`). The v0.1 rows below
are the v0.1 measurements of 2026-10-15; see [CHANGELOG.md](CHANGELOG.md).

## Results

[RigMark](https://github.com/alexellis/rigmark) 1.0.0, thinking on, `reasoning_effort: low`, decode tok/s; settings in
[Reproducing](#reproducing). The baseline is upstream gr8 `d9188e3` plus the MTU fix only (the engine without this
recipe's speed patches), measured on the same machines. The recipe runs on upstream `6448a16`, so the rows compare
complete stacks; see [Known limits](#known-limits). RigMark sends single-turn requests; the multi-turn case that patch
0014 targets is measured separately in [Multi-turn](#multi-turn-prefix-cache-hits).

| c1 decode | prose | code | structured |
|---|---:|---:|---:|
| **v0.2**, full run | **35.5** | **54.8** | **66.6** |
| v0.1, full run | 35.4 | 54.9 | 66.7 |
| v0.1, decode screens (two runs) | 35.76 / 35.74 | 52.51 / 52.56 | 67.51 / 67.36 |
| baseline (`d9188e3` + MTU fix), full run | 29.0 | 48.1 | 60.6 |

| Concurrency, aggregate decode tok/s | C1 | C2 | C4 |
|---|---:|---:|---:|
| **v0.2** | **38.8** | **54.3** | **67.6** |
| v0.1 | 39.4 | 55.7 | 67.5 |
| baseline (`d9188e3` + MTU fix) | 34.7 | 50.0 | 63.7 |

| Prefill, cold, tok/s | 8K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|
| **v0.2** | 1369 | 1370 | 1312 | 1157 |
| v0.1 | 1367 | 1373 | 1311 | 1181 |
| baseline (`d9188e3` + MTU fix) | 1361 | 1375 | 1307 | 1188 |

Prefill (v0.1) is within 1 % of the baseline: the levers act on decode. A 96K-token needle prompt returns the same
answer as the baseline. v0.2: prefill within 3 % of the baseline (128K: 1157 vs 1188), same 96K needle answer.

### Multi-turn (prefix-cache hits)

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

Mean of the per-turn ratios (`bench/multiturn.py compare`): prefix-hit turns (A2, A3, B2, A4) ×1.20 decode, fresh turns ×1.09 (driven by C1; A1 and B1 are ×1.01), replayed turn ×1.01. Single run: no variance is established.

- This is one run of a small script ([`bench/multiturn.py`](bench/multiturn.py), stdlib Python against the local API), not a
  benchmark; all rows come from the same pair of runs.
- The turns are code-heavy. We expect prose turns to gain less, because DSel picks MTP for prose anyway; this was not
  measured.
- A turn whose conversation no longer owns the drafter's context ring (A4) warms the drafter up over a few MTP rounds
  instead of continuing it, and gains little.
- 0014 is a drafting lever: T=0 text on these turns differs between v0.1 and v0.2 (see the exactness classes below).

## Levers

Measured contribution is RigMark c1 decode on short screens, each step on top of the levers above it, so the steps use
different control runs and do not add up exactly to the totals above.

| Patch | What it does | Flags (serving) | Exactness | Measured | Credit |
|---|---|---|---|---|---|
| 0001 | Negotiate the RDMA queue-pair MTU from both ports' active MTU instead of assuming one; logs the chosen MTU per QP | always on | bit-exact | needed to boot on a switched fabric; no speed effect | knapcio |
| 0002, 0003 | Split the replicated GEMVs every rank holds in full (layer input q_a / kv_a / indexer k and weights; the indexer's wq_b; the MTP draft layer's eh_proj, q_a, kv_a, indexer) by 16-output tiles across the four ranks, then all-gather the columns per layer on a fabric of their own | `G6_QA_SPLIT=1` `G6_IDXQ_SPLIT=1` `G6_MTP_SPLIT=1` | bit-exact | prose +6.5 % | knapcio |
| 0004, 0005 | MTP draft head and DFlash2 drafter head on INT4 g32 records made at boot from the target's MXFP8 head (45 % fewer head bytes per draft step); the target head and verify are unchanged | `G6_DRAFT_HEAD_I4=1` `G6_DFL_HEAD_I4=1` | drafting | prose +0.6 %, code +1.5 % | knapcio; DFlash2 by incoai |
| — | DSel probe interval (upstream knob, no patch): while a request drafts with MTP, DFlash2 drafts a scoring probe every 32nd round instead of every 4th | `G6_S2_DSEL_PROBE=32` | drafting | prose +3 % | DSel by kindlingai; setting by knapcio |
| 0006–0008 | DSel experiments: start DFlash2 requests on MTP, a selector-walk correction coefficient, per-position observation clocks | off | — | not used | knapcio |
| 0009 | Overrides for DSel's drafting-cost constants (ms per round); serving lowers the MTP first-step cost from 5.5 to 2.9 | `G6_S2_DSEL_MCOST=2.9` | drafting | prose +2.2 %, code −1.9 % | knapcio; DSel by kindlingai |
| 0010 | MoE: run the trellis shared expert on a forked stream during the router GEMV and plan kernel, so its weight reads fill idle DRAM time | `G6_ROUTE_OVL=1` | bit-exact | prose +1.7 % | knapcio |
| 0011 | Decode attention, round 1: a scores kernel with several KV slots per warp and a transpose reduction; a packed, branch-free context kernel | `G6_ATTN_SC2=4` `G6_ATTN_CTX2=4` | bit-exact | prose +3.4–4.7 %, code / structured +4–7 %; −8.6 ms/step at c4 (12 rows) | knapcio |
| 0012 | Decode attention, round 2: the transpose reduction done eagerly (fewer registers), a context kernel with 32-bit row indices and warp-ballot batch states, the selector's final sort as a rank by count, and an indexer that reads each key tile once for a session's rows | `G6_ATTN_SC2_TR=2` `G6_ATTN_CTX3=2` `G6_ATTN_SEL2=16` `G6_IDX_GRP=16` | bit-exact | prose +0.7 % at 1K context; at 32K context up to −3.1 ms/step at c1, −10.3 ms/step at c4 | knapcio |
| 0013 | MTP confidence stop: a request's solo MTP rounds verify 1–3 drafts, stopping the chain where the MTP head's cumulative probability falls below a per-position threshold | `G6_MTP_KSTOP=1` `G6_MTP_KSTOP_KMAX=3` `G6_MTP_KSTOP_TAU=0,0.5,0.75` | drafting | prose +1.2 %, code −2.1 % | knapcio |
| 0014 | DFlash2 after prefix reuse. Upstream runs a request on MTP only for its whole life when its prompt hit the prefix cache inside the drafter's 2,048-position window or it was not alone at its prompt tail (upstream KNOWN-ISSUES #1). Mode 1 lets prefix-hit turns keep DFlash2, by continuing the drafter's context ring from the previous turn of the same conversation or by warming it up over a few MTP rounds; mode 2 also promotes a session that is alone on MTP | `G6_DFL_REUSE=2` | drafting | multi-turn script: prefix-hit turns ×1.20, fresh ×1.09 (see [Multi-turn](#multi-turn-prefix-cache-hits)); RigMark single-turn rows similar to v0.1 in these single full runs (it sends fresh prompts); variance not established | knapcio; DFlash2 by incoai |

**Exactness classes.** *Bit-exact*: the patched path produces the same bytes as upstream's for the same inputs, with
the serving flags above. Evidence: the MTU fix changes transport setup only; the GEMV split was byte-compared with
m4sim (the engine's single-GPU simulator of all ranks) on the plain, graph, MTP, S2 and CP paths; the route overlap
gave identical T=0 output on the four-box cluster (and is byte-compared by its own test,
`engine/kernels/tests/k1_ovl_test.cu`); both attention rounds are byte-identical in the attnk bench at contexts 2K / 8K
/ 32K × 1 / 3 / 8 / 12 rows and gave identical T=0 output on the cluster. Exactness covers these flags only: patch
0011 also contains `G6_ATTN_CTX2=9`, an experimental mode that changes the summation order; it is off and not part of
this recipe.

*Drafting*: deterministic, and the target model, its weights and its precision are unchanged; every committed token is
verified by the target. These levers change only what is drafted and when. Upstream gr8's verify step is not
row-count invariant (its fp32 summation order depends on how many rows a step verifies), so a drafting change can
regroup verify rows and change the low bits of the logits. The output is deterministic for a given configuration but
not guaranteed byte-identical to upstream over long generations.

## Quality

| Check | Result |
|---|---|
| GEMV split exactness, m4sim | every combination of `G6_QA_SPLIT` / `G6_IDXQ_SPLIT` / `G6_MTP_SPLIT` byte-identical to flags off on the plain, graph, MTP, S2 and CP paths: 30/30 |
| Attention exactness, attnk bench (rounds 1 and 2, serving flags) | byte-identical to stock at T 2K / 8K / 32K × M 1 / 3 / 8 / 12 |
| T=0 identity, v0.1 configuration | top-20 logprobs on the 75 qeval prompts (up to 512 tokens) bit-identical to the previous configuration (GEMV split + INT4 heads + probe 32): 75/75 prompts, 7,551 positions. For scale, the same configuration run at concurrency 1 vs 4 differs with mean KL 5e-3 |
| T=0 identity / KL, v0.2 | top-20 logprobs on the 75 qeval prompts vs the previous configuration, second (prefix-cached) pass: 34/75 identical, mean KL 2.7e-3 on the common prefix; for scale, the same configuration at concurrency 1 vs 4: 30/75 identical, mean KL 5.0e-3. Text differs at T=0 because a cached prompt now drafts with DFlash2 (drafting class) |
| Sampled qeval, v0.2 | 578/600 (T=1.0, top_p 0.95, 8 seeds × 75 tasks), identical per seed to the previous configurations. Single T=0 run: 71/75, the same four misses as v0.1 |
| Sampled qeval, v0.1 configuration | 578/600 (T=1.0, top_p 0.95, 8 seeds × 75 tasks), identical per seed to the previous configuration. Single T=0 run: 71/75 (baseline 73/75); the extra misses are a character count off by one and one arithmetic answer, borderline tasks that flip between runs |

qeval is the 75-task quality screen from the earlier
[GLM-5.3 recipe](https://github.com/knapcio/GLM-5.3-4x-DGX-Spark-TP4) (`bench/qeval.py`). The m4sim exactness runs used
a development harness that runs the split gathers on a loopback; it is not included in this patch series.

## Reproducing

- **Engine flags** (serving, v0.2): `G6ENV_X` in [docker/env.example](docker/env.example), the same line on every box:

  ```bash
  G6ENV_X="G6_QA_SPLIT=1 G6_IDXQ_SPLIT=1 G6_MTP_SPLIT=1 G6_DRAFT_HEAD_I4=1 G6_DFL_HEAD_I4=1 G6_S2_DSEL_PROBE=32 G6_ROUTE_OVL=1 G6_S2_DSEL_MCOST=2.9 G6_ATTN_SC2=4 G6_ATTN_SC2_TR=2 G6_ATTN_CTX2=4 G6_ATTN_CTX3=2 G6_ATTN_SEL2=16 G6_IDX_GRP=16 G6_MTP_KSTOP=1 G6_MTP_KSTOP_KMAX=3 G6_MTP_KSTOP_TAU=0,0.5,0.75 G6_DFL_REUSE=2"
  ```

  v0.1 was measured with the same line without `G6_DFL_REUSE=2`.
- **Benchmark**: RigMark 1.0.0 ([github.com/alexellis/rigmark](https://github.com/alexellis/rigmark)) against the
  head's API. Full runs:

  ```bash
  rigmark run --base-url http://127.0.0.1:8100 --model glm-5.3 \
    --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}' \
    --prefill-depths 8192,32768,65536,131072
  ```

  Decode screens: the same command with `--skip-prefill --skip-concurrency`. Each cell is RigMark's median decode
  tok/s for that workload.
- **Hardware**: the four boxes at the stock GPU clock cap, no clock changes. Baseline and recipe ran on the same
  machines.
- **Attention exactness** (patches 0011–0012): the attnk bench in the patched tree compares every arm byte for byte
  against the stock chain and exits 3 on any difference. On an idle Spark (not while serving), with CUDA 13:

  ```bash
  ./scripts/apply.sh gr8-patched          # the patched source tree (attnk.cu exists only after the patches)
  cd gr8-patched/engine/attn/tests
  eval "$(sed -n '/^K1H2_FLAGS=/,/"$/p' ../../../scripts/build.sh)"
  nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a $K1H2_FLAGS -DG6_NR=4 -DDEC_S2 -DPF_F12 \
    -o attnk attnk.cu ../bind.cu -lcublas
  for M in 1 3 8 12; do for TL in "2048 1000" "8192 8000" "32768 32000"; do
    ATTNK_ARMS="base TR=2,CTX3=2,SEL2=16,IDX=16" ./attnk $TL $M 2 > attnk.log || echo "FAIL (exit $?): T L = $TL, M = $M"
    grep '^EXACT' attnk.log
  done; done
  ```

  `base` is round 1 (`SC2=4,TR=1,CTX2=4`); the second arm adds round 2's serving flags. Every `EXACT` line must end in
  `IDENTICAL` and no `FAIL` line may appear.
- **GEMV split exactness**: measured with m4sim and a development harness that is not included (see Quality).

## Known limits

- **Code decode varies more than prose** between runs and between short screens and full runs.
- **Verify numerics are not row-count invariant in upstream gr8.** A step that verifies 3 rows and one that verifies 8
  can sum in a different order. T=0 output in upstream also depends on prefix-cache state (a cold prompt and a cached
  one can differ). Compare T=0 runs only under the same cache state.
- **The baseline is upstream `d9188e3`, the recipe pins `6448a16`.** The upstream changes in between are engine-side
  (fabric relay back-pressure, giving host heap back after load, S2 prefill and notice scheduling); the front end,
  launcher and loader are identical. The gains above are of the complete stack, not of the patches alone.
- **Only the TP=4 engine with the FP8 KV latent (`bin/g6m4`) is rebuilt.** TP=3, TP=6 and `--kv-format int4` keep
  upstream's binaries and were not tested with these flags.
- **Same flags on every box.** `G6ENV_X` is a cluster-wide launcher option, so use one `docker/.env` everywhere. The
  split flags add collectives at engine start, before the engines compare their settings, so ranks started with
  different split flags by other means can hang the boot instead of failing.
- **DFlash2 after prefix reuse (0014) acts on a session that is alone**: mode 1 on a prefix-hit turn that upstream
  would give DFlash2 apart from the hit (alone at its prompt tail), mode 2 on the only active session. With several
  concurrent requests upstream's choice of drafter stands. A prefix-hit turn continues the drafter's context only while
  the drafter's ring still holds that conversation's previous turn; otherwise it warms the drafter up over MTP rounds
  (`G6_DFL_REUSE_MIN`, default 256 rows) before it drafts with DFlash2.
- **Never set `G6_ATTN_P11=1`**: it takes precedence over `G6_ATTN_SC2` / `G6_ATTN_CTX2` / `G6_ATTN_CTX3` and replaces
  the measured attention kernels.

## Build and run

Requirements: four DGX Sparks (GB10, DGX OS with the NVIDIA container toolkit), a RoCE fabric between them as upstream
gr8 expects, and about 90 GB of disk per box for its part of the model.

1. Build the image (linux/arm64; on a Spark or any arm64 builder with Docker Buildx). The Dockerfile fetches upstream
   gr8 at `6448a16`, applies `patches/` with `scripts/apply.sh`, checks the resulting source tree, builds the TP=4
   engine and places it in the pinned upstream image:

   ```bash
   docker buildx build --platform linux/arm64 -t gr8-glm53-recipe:v0.2 --load .
   ```

   Use the same image on all four boxes (build it on each, or `docker save` / `docker load`).

2. On every box, get upstream's compose file and put this recipe's settings next to it:

   ```bash
   git clone https://github.com/kindlingai/gr8 && git -C gr8 checkout 6448a16
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
   [docker/README.md](https://github.com/kindlingai/gr8/blob/6448a16661abd81e9540bc073b3cfa92d7234565/docker/README.md):

   ```bash
   docker compose -f gr8/docker/compose.yaml run --rm gr8 prepare
   ```

5. **Before starting, restrict the network.** The container uses host networking and the head's OpenAI-compatible
   front end listens on `0.0.0.0:8100` with no authentication. Run it only on a trusted network, firewall port 8100
   (and the cluster's control ports) to the machines that need it, and use Tailscale or a VPN for remote access.
   Never forward these ports on a public router. Then start:

   ```bash
   docker compose -f gr8/docker/compose.yaml up -d
   ```

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
