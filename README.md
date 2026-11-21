# Kintsugi-27B engine

This source is the engine of [Kintsugi-27B](https://github.com/professorpalmer/Kintsugi-27B). It is a fork of
llama.cpp. The release bundle runs it with its own launcher and tool layer, which are in the Kintsugi-27B repo. This
source adds:

- **Escha W2 / EXL3 trellis weights**: the dense Escha op (`ggml_escha_mul_mat`, CPU and CUDA), the exllamav3 CUDA
  GEMV kernels (`ggml/src/ggml-cuda/exl3`), a tensor-core prefill, and a fast int8 read of long prompts
  (`GGML_ESCHA_PREFILL=i8`; `recon` = the exact read).
- **Tiered K/V cache** (`--kv-vram-cells N`): the first N cells of each K/V tensor are in VRAM, the other cells are in
  pinned system RAM, in one CUDA address range. A copy-engine staging buffer reads the system-RAM part when a sequence
  gets that deep. The attention kernels do not change.
- **Two-stage output head** (`LLAMA_HEAD2=N`): a small q4_0 copy of the output head in VRAM reads all rows; the exact
  Q8_0 rows in system RAM score the N best candidates of each row again.
- **Drafting**: the model's own MTP head at every depth (`--spec-type draft-mtp`, `--spec-draft-window`,
  `--spec-draft-n-max-tail`), and lookup drafting from the prompt (`ngram-mod`).
- **Batch-invariant kernels** (`GGML_CUDA_BATCH_INVARIANT=1`): the output for a token does not change with the batch
  that it runs in.
- **Settings for agent harnesses**: `--reasoning-effort-allow`, `--reasoning-effort-fallback`,
  `--reasoning-max-tokens-floor`.

## Build notes

The Kintsugi-27B release engine is this source built without the Mirai S kernels (`GGML_CUDA_MIRAI=OFF`; the loader
then refuses Mirai S files with a clear message). Release flags:

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75-real;86-real;89-real" -DGGML_CUDA_MIRAI=OFF -DLLAMA_BUILD_UI=OFF -DGGML_CUDA_GRAPHS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF -DGGML_RPC=OFF -DGGML_BLAS=OFF -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF
cmake --build build --target llama-server llama-perplexity test-backend-ops
```

- Linux: also give `-DLLAMA_USE_PREBUILT_UI=OFF`. Without it, the `llama-ui-assets` step tries to download the
  prebuilt web UI and the build stops there.
- One GPU family only: set `CMAKE_CUDA_ARCHITECTURES` to it (for example `75-real` for RTX 20, `89-real` for RTX 40).
- NVIDIA driver: the tiered K/V cache maps system RAM into the GPU's address space (`cuMemCreate` on a host location).
  Drivers of the 580 family and newer take the default host type. Older drivers (570) refuse it; the engine then uses
  the host NUMA type and writes one WARN line ("... HOST_NUMA ... this driver refuses CU_MEM_LOCATION_TYPE_HOST"). If
  the driver refuses both, the load stops with "update the NVIDIA driver to 580 or newer". Test switch:
  `GGML_CUDA_HOSTMEM_FORCE=host|numa|pinned` ("pinned" = no host type: the stop message).
- The Windows bundle carries the CUDA 13 runtime, which needs a 580+ driver; a CUDA 12.x build (12.2 or newer) runs on
  older drivers.

## Where it comes from

- **llama.cpp** by the ggml authors and all its contributors (ggml-org, MIT). The upstream README is
  `README-upstream.md`.
- **PrismML's fork** ([PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp), MIT) is the direct base.
- **The Mirai S port** by alesha-pro ([alesha-pro/llama.cpp-mirai-s](https://github.com/alesha-pro/llama.cpp-mirai-s))
  of Mirai Labs' trellis codec is in this source. The Kintsugi build leaves its kernels out (`GGML_CUDA_MIRAI=OFF`).
  alesha-pro also gave the control-vector projection mode (`--cvec-mode project`).
- **Ajay9o9 (AjHomelab), llama.cpp-escha**: his commits are the base of the Escha op (CPU and CUDA), with his
  authorship kept.
- **turboderp (exllamav3)**: the EXL3 trellis codec and the CUDA GEMV kernels in `ggml/src/ggml-cuda/exl3` (MIT; the
  licence is in that folder).
- **Escha Labs**: the Escha W2 format.
- **sudoingX**: the batch-invariant mode and the planar activation layout.

The full list of credits is `CREDITS.md` in the [Kintsugi-27B repo](https://github.com/professorpalmer/Kintsugi-27B).

## Licence

MIT, the same as llama.cpp: see `LICENSE`. The licences of third-party code are in `licenses/`, in `vendor/`, and in
`ggml/src/ggml-cuda/exl3/LICENSE`. No person or company named here endorses or sponsors this project.
