# Credits

- **kindlingai — Matt Mastracci and the gr8 contributors**: the [gr8](https://github.com/kindlingai/gr8) engine this
  recipe runs unmodified apart from `patches/` (TP=4 runtime, S2 batched decode, the RoCE fabric, MTP and DFlash2
  drafting with the DSel per-round drafter choice, the launcher, the front end and the container), pinned at
  `6448a16661abd81e9540bc073b3cfa92d7234565`. The engine is MIT; the patches keep upstream's files, headers and licence.
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
- **knapcio**: the twenty-one patches (fabric MTU negotiation, replicated-GEMV splits, INT4 draft heads, DSel
  experiments and cost overrides, the shared-expert stream overlap, three rounds of decode attention kernels, the MTP
  confidence stop, DFlash2 after prefix reuse, the parallel DFlash2 drafter step, the decode step timeline, the DSel
  cost model and its table tool), the serving configuration, the measured DSel cost table, the measurements and the
  quality checks (m4sim and attnk exactness runs, T=0 identity, qeval). qeval comes from the earlier
  [GLM-5.3 recipe](https://github.com/knapcio/GLM-5.3-4x-DGX-Spark-TP4).
