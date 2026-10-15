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

Release **v0.2**, 2026-10-15. The tables below are this release's measurements. What changed in each release, and by
how much, is in [CHANGELOG.md](CHANGELOG.md); A/B measurements against earlier releases and the upstream baseline are
in [docs/history.md](docs/history.md).

## Decode by prompt type and concurrency — thinking on, effort low

Measured 2026-10-15 on the v0.2 configuration, one boot.

Decode, aggregate tok/s **[per-stream tok/s]**, RigMark's prose, code and structured prompts, 256 output tokens,
temperature 0, `reasoning_effort: low` (the model's chat template has no thinking-off mode), medians of three rounds per
cell; c identical requests sent at once. Measured with [`bench/conc.py`](bench/conc.py).

At c ≥ 2 the sessions share verify steps and draft with MTP (the DFlash2 drafter serves a session that is alone), so
structured and code, where DFlash2 drafts long blocks, gain less from concurrency than prose; structured c2 aggregate
is below c1.

| Prompt type | c1 | c2 | c4 |
|---|---:|---:|---:|
| prose | 34.9 [35.8] | 48.9 [25.9] | 62.8 [17.6] |
| code | 43.4 [44.3] | 58.8 [31.2] | 74.8 [21.3] |
| structured | 69.5 [72.0] | 65.2 [35.4] | 84.3 [24.8] |

## Prefill

Cold prefill, input tok/s, measured with [RigMark](https://github.com/alexellis/rigmark) 1.0.0 (unique prompts, no
prefix reuse). A 96K-token needle prompt returns the correct answer.

| Prompt length | 8K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|
| cold prefill tok/s | 1369 | 1370 | 1312 | 1157 |

RigMark decode results (used to measure each lever) are in [docs/history.md](docs/history.md).

## Multi-turn conversations

RigMark sends single-turn requests. In a multi-turn conversation, turns whose prompt hits the prefix cache draft with
DFlash2 (patch 0014, `G6_DFL_REUSE=2`) instead of staying on MTP for their whole life as in upstream. The measured A/B
is in [docs/history.md](docs/history.md#multi-turn-prefix-cache-hits).

## Levers

| Patch | What it does | Flags (serving) | Exactness | Credit |
|---|---|---|---|---|
| 0001 | Negotiate the RDMA queue-pair MTU from both ports' active MTU instead of assuming one; logs the chosen MTU per QP. Needed to boot on a switched fabric | always on | bit-exact | knapcio |
| 0002, 0003 | Split the replicated GEMVs every rank holds in full (layer input q_a / kv_a / indexer k and weights; the indexer's wq_b; the MTP draft layer's eh_proj, q_a, kv_a, indexer) by 16-output tiles across the four ranks, then all-gather the columns per layer on a fabric of their own | `G6_QA_SPLIT=1` `G6_IDXQ_SPLIT=1` `G6_MTP_SPLIT=1` | bit-exact | knapcio |
| 0004, 0005 | MTP draft head and DFlash2 drafter head on INT4 g32 records made at boot from the target's MXFP8 head (45 % fewer head bytes per draft step); the target head and verify are unchanged | `G6_DRAFT_HEAD_I4=1` `G6_DFL_HEAD_I4=1` | drafting | knapcio; DFlash2 by incoai |
| — | DSel probe interval (upstream knob, no patch): while a request drafts with MTP, DFlash2 drafts a scoring probe every 32nd round instead of every 4th | `G6_S2_DSEL_PROBE=32` | drafting | DSel by kindlingai; setting by knapcio |
| 0006–0008 | DSel experiments: start DFlash2 requests on MTP, a selector-walk correction coefficient, per-position observation clocks | off | — | knapcio |
| 0009 | Overrides for DSel's drafting-cost constants (ms per round); serving lowers the MTP first-step cost from 5.5 to 2.9 | `G6_S2_DSEL_MCOST=2.9` | drafting | knapcio; DSel by kindlingai |
| 0010 | MoE: run the trellis shared expert on a forked stream during the router GEMV and plan kernel, so its weight reads fill idle DRAM time | `G6_ROUTE_OVL=1` | bit-exact | knapcio |
| 0011 | Decode attention, round 1: a scores kernel with several KV slots per warp and a transpose reduction; a packed, branch-free context kernel | `G6_ATTN_SC2=4` `G6_ATTN_CTX2=4` | bit-exact | knapcio |
| 0012 | Decode attention, round 2: the transpose reduction done eagerly (fewer registers), a context kernel with 32-bit row indices and warp-ballot batch states, the selector's final sort as a rank by count, and an indexer that reads each key tile once for a session's rows | `G6_ATTN_SC2_TR=2` `G6_ATTN_CTX3=2` `G6_ATTN_SEL2=16` `G6_IDX_GRP=16` | bit-exact | knapcio |
| 0013 | MTP confidence stop: a request's solo MTP rounds verify 1–3 drafts, stopping the chain where the MTP head's cumulative probability falls below a per-position threshold | `G6_MTP_KSTOP=1` `G6_MTP_KSTOP_KMAX=3` `G6_MTP_KSTOP_TAU=0,0.5,0.75` | drafting | knapcio |
| 0014 | DFlash2 after prefix reuse. Upstream runs a request on MTP only for its whole life when its prompt hit the prefix cache inside the drafter's 2,048-position window or it was not alone at its prompt tail (upstream KNOWN-ISSUES #1). Mode 1 lets prefix-hit turns keep DFlash2, by continuing the drafter's context ring from the previous turn of the same conversation or by warming it up over a few MTP rounds; mode 2 also promotes a session that is alone on MTP | `G6_DFL_REUSE=2` | drafting | knapcio; DFlash2 by incoai |

The measured contribution of each lever is in [docs/history.md](docs/history.md#per-lever-contributions).

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
| Sampled qeval | 578/600 (T=1.0, top_p 0.95, 8 seeds × 75 tasks) |
| T=0 qeval | 71/75, single run |
| Needle | a 96K-token needle prompt returns the correct answer |
| GEMV split exactness, m4sim | every combination of `G6_QA_SPLIT` / `G6_IDXQ_SPLIT` / `G6_MTP_SPLIT` byte-identical to flags off on the plain, graph, MTP, S2 and CP paths: 30/30 |
| Attention exactness, attnk bench (rounds 1 and 2, serving flags) | byte-identical to stock at T 2K / 8K / 32K × M 1 / 3 / 8 / 12 |
| Route overlap exactness | identical T=0 output on the four-box cluster; byte-compared by `engine/kernels/tests/k1_ovl_test.cu` |

qeval is the 75-task quality screen from the earlier
[GLM-5.3 recipe](https://github.com/knapcio/GLM-5.3-4x-DGX-Spark-TP4) (`bench/qeval.py`). The m4sim exactness runs used
a development harness that runs the split gathers on a loopback; it is not included in this patch series. T=0 identity
and KL checks against earlier configurations are in [docs/history.md](docs/history.md#quality-checks-against-earlier-configurations).

## Reproducing

- **Engine flags** (serving): `G6ENV_X` in [docker/env.example](docker/env.example), the same line on every box:

  ```bash
  G6ENV_X="G6_QA_SPLIT=1 G6_IDXQ_SPLIT=1 G6_MTP_SPLIT=1 G6_DRAFT_HEAD_I4=1 G6_DFL_HEAD_I4=1 G6_S2_DSEL_PROBE=32 G6_ROUTE_OVL=1 G6_S2_DSEL_MCOST=2.9 G6_ATTN_SC2=4 G6_ATTN_SC2_TR=2 G6_ATTN_CTX2=4 G6_ATTN_CTX3=2 G6_ATTN_SEL2=16 G6_IDX_GRP=16 G6_MTP_KSTOP=1 G6_MTP_KSTOP_KMAX=3 G6_MTP_KSTOP_TAU=0,0.5,0.75 G6_DFL_REUSE=2"
  ```

- **Decode table**: `python3 bench/conc.py --prompts <rigmark>/prompts.json` against the head's API (rounds 3,
  256 output tokens).

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
