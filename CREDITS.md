# Credits

- **kindlingai — Matt Mastracci and the gr8 contributors**: the [gr8](https://github.com/kindlingai/gr8) engine this
  recipe runs unmodified apart from `patches/` (TP=4 runtime, S2 batched decode, the RoCE fabric, MTP and DFlash2
  drafting with the DSel per-round drafter choice, the launcher, the front end and the container), pinned at
  `c53b5b869818950a96a9b72c647d4b156b057127`. The engine is MIT; the patches keep upstream's files, headers and licence.
  Also the EXLR8 weights
  [kindlingai/glm-5.3-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4), which are not MIT:
  the [GLM-5.3 licence](https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE) plus CC BY 4.0 for the quantization
  additions ([model card](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4#licence)).
- **turboderp and the [exllamav3](https://github.com/turboderp-org/exllamav3) contributors**: the EXL3 format and
  trellis quantizer the EXLR8 weights are encoded with; **Cornell RelaxML**: [QTIP](https://github.com/Cornell-RelaxML/qtip),
  which EXL3's trellis quantizer streamlines; **davidsyoung**:
  [GLM-5.3-EXL3-TR3-3.25bpw](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw), whose per-expert bit
  distribution the `home` manifest keeps. (As credited by upstream gr8.)
- **Z.ai**: [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) and its native multi-token-prediction layer, used as the
  MTP drafter.
- **incoai**: the [GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) drafter, CC BY-NC-ND 4.0
  (noncommercial). Not redistributed, modified or converted here; each box fetches and converts its own copy with
  upstream gr8's tooling, which does not lift the noncommercial restriction.
- **Alex Ellis (alexellis) and RigMark contributors**: [RigMark](https://github.com/alexellis/rigmark), the decode,
  concurrency and prefill benchmark behind the RigMark results in the README. Measurements only; no RigMark code is
  included. The multi-turn numbers come from this repository's own `bench/multiturn.py`.
- **NVIDIA**: the CUDA images and toolchain the image is built from.
- **blaesus**: [dsgx10](https://github.com/blaesus/dsgx10) (MIT), from-scratch inference engines for GB10 boxes, whose
  paced side-stream L2 prefetch (small chunks from a token bucket instead of a burst) patch 0009 follows.
- **spectra-kindling** (kindlingai): upstream gr8's open PR [#98](https://github.com/kindlingai/gr8/pull/98), kv_b as
  the release's block FP8 (`G6_KVB=fp8`), carried here as patches 0017–0019 with authorship kept (still open upstream at
  the pinned commit); MIT like the rest of gr8.
- **knapcio**: the other twenty-seven patches (DSel experiments and cost overrides, the DSel cost model and its table
  tool, the paced L2 prefetch and its FP8 targets, the MoE split table and translation prefetch, the launch share-limit
  rule and the ninth decode bucket, the one-chain CP attention call, the per-session confidence stop in shared rounds,
  the drafter-shape fixes, the SELX default, the CP step-row cap, the positional attention index, the MTP last-row
  routing, the two-stage MTP draft head, and the experiments and tests that are off: the router tile split, the
  64/128-slot context partitions, the attnk round-10 arms and a DSel host-test fix); the levers now merged upstream
  (the fabric MTU negotiation, the replicated-GEMV splits, the INT4 draft heads and drafter body, the shared-expert
  stream overlap, three rounds of decode attention kernels, the MTP confidence stop with its K-stop cap, DFlash2 after
  prefix reuse, the parallel DFlash2 drafter step and the decode step timeline, upstream PRs #82–#90); the serving
  configuration, the measured DSel cost table, the measurements and the quality checks (m4sim and attnk exactness
  runs, T=0 identity, qeval). qeval comes from the earlier
  [GLM-5.3 recipe](https://github.com/knapcio/GLM-5.3-4x-DGX-Spark-TP4).
