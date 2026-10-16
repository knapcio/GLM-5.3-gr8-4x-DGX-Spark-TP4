# The serving image for this recipe: GB10 (DGX Spark), linux/arm64, CUDA 13, sm_121a, TP=4.
#
#   docker buildx build --platform linux/arm64 -t gr8-glm53-recipe:v0.6 --load .
#
# It is the measured setup: the published upstream gr8 image, pinned by digest, with its TP=4 engine binary
# (/opt/gr8/bin/g6m4) replaced by one built from kindlingai/gr8 at the pinned commit plus patches/, and the
# launcher's options.py replaced by the patched one (patch 0027: --share-limit accepts any multiple of 64).
# Everything else in the image (front end, loader, shims, entrypoint, uid 1000, environment) is upstream's, unchanged.
# The build stages follow upstream's Dockerfile at the pinned commit (toolchain, Rust loader and KV allocator, engine).
ARG CUDA_VERSION=13.0.1
ARG UBUNTU=ubuntu24.04
ARG GR8_RUNTIME=ghcr.io/kindlingai/gr8@sha256:7c6c21b697a0095de0fc40f0bcfa6d92b90d3d386af0644010dd50e20862e87c

FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS toolchain
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# GB10 is arm64: refuse any other target (build with --platform linux/arm64)
RUN [ "$(uname -m)" = aarch64 ] || { echo "build with --platform linux/arm64" >&2; exit 1; }
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ binutils libibverbs-dev ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

# kindlingai/gr8 at the pinned commit + patches/, checked against the expected source tree (scripts/apply.sh)
FROM toolchain AS source
COPY patches /recipe/patches
COPY scripts/apply.sh /recipe/scripts/apply.sh
RUN /recipe/scripts/apply.sh /src

FROM toolchain AS rust
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# upstream pins the front end's compiler in frontend/g6front-rs/rust-toolchain.toml; the same version builds the loader
ARG RUST_VERSION=1.94.1
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal --default-toolchain "$RUST_VERSION"
WORKDIR /src

FROM rust AS loader
COPY --from=source /src/engine/loader-rs engine/loader-rs
RUN mkdir -p /out \
 && cargo build --release --locked --manifest-path engine/loader-rs/Cargo.toml \
 && cp engine/loader-rs/target/release/libg6load.a /out/

FROM rust AS kvman
COPY --from=source /src/engine/kvman-rs engine/kvman-rs
RUN mkdir -p /out \
 && cargo build --release --locked --manifest-path engine/kvman-rs/Cargo.toml \
 && cp engine/kvman-rs/target/release/libg6kvman.a /out/

# As upstream's engine-src stage: engine sources only, no cargo; build.sh links the two Rust libraries built above.
FROM toolchain AS engine
WORKDIR /src
COPY --from=source /src/engine engine
COPY --from=source /src/tools/health/gpuprobe.cu tools/health/gpuprobe.cu
COPY --from=source /src/scripts/build.sh scripts/build.sh
# the fabric MTU negotiation's host test (patch 0001)
RUN g++ -std=c++17 -O2 -Wall -Wextra engine/comms/fabric_mtu_test.cpp -o /tmp/fabric_mtu_test && /tmp/fabric_mtu_test
COPY --from=loader /out/libg6load.a engine/loader-rs/target/release/libg6load.a
COPY --from=kvman /out/libg6kvman.a engine/kvman-rs/target/release/libg6kvman.a
RUN G6_NR=4 scripts/build.sh /out/bin

FROM ${GR8_RUNTIME} AS runtime
COPY --from=engine /out/bin/g6m4 /opt/gr8/bin/g6m4
# patch 0027: the patched launcher options (otherwise the image rejects --share-limit 98304)
COPY --from=source /src/tools/launch/options.py /opt/gr8/tools/launch/options.py
