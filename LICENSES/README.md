# Licence boundaries

`../LICENSE` (MIT) covers what this repository authors: `patches/` as changes, `scripts/`, `Dockerfile`, `docker/`
and the documentation.

`MIT-gr8.txt` is upstream gr8's licence, retained because the patches apply to and contain context lines from gr8
sources. The patched tree, like upstream, is MIT.

Not included and not relicensed (MIT covers none of them):

- the EXLR8 weights: the [GLM-5.3 licence](https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE) plus CC BY 4.0
  for kindlingai's quantization additions
  ([model card](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4#licence));
- the incoai DFlash2 drafter: CC BY-NC-ND 4.0, noncommercial, also after local conversion
  ([model card](https://huggingface.co/incoai/GLM-5.3-DFlash2#license));
- the container images the Dockerfile builds from (upstream gr8, NVIDIA CUDA), which keep their own terms.
