# llama.cpp

> [!NOTE]
> **This fork adds native support for [Kev](https://github.com/jaredpalmer/kev) System One decision models.** A Kev GGUF loads like any other model, and `llama-server` exposes the TypeSafe-compatible `POST /v1/systemone` endpoint plus a `/studio` page for editing state and questions. No Python at inference time. Everything else is stock upstream llama.cpp.
>
> Jump to [Kev in 5 minutes](#kev-in-5-minutes) for install, model download and a first run.

## Kev in 5 minutes

Kev answers typed questions about a piece of state and returns calibrated probabilities instead of text (prefill only, nothing generated). Useful for classification, routing, moderation, scoring and tool-call gating. Full reference: [docs/kev.md](docs/kev.md).

### 1. Install this fork

Stock llama.cpp packages (`brew`, `winget`, `conda-forge`) do **not** include Kev support. Use one of these:

```sh
# Linux x64 - latest pre-built release (see the releases page for macOS/Windows/arm64/Vulkan/CUDA/SYCL assets)
TAG=$(curl -s https://api.github.com/repos/espetro/llama.cpp/releases | grep -m1 '"tag_name"' | cut -d'"' -f4)
curl -L -o llama-kev.tar.gz https://github.com/espetro/llama.cpp/releases/download/$TAG/llama-$TAG-bin-ubuntu-x64.tar.gz
tar xf llama-kev.tar.gz && export PATH="$PWD/llama-$TAG:$PATH"
```

```sh
# any platform - mise; kev releases are pre-releases, so prerelease=true is required
mise use -g "github:espetro/llama.cpp[asset_pattern=llama-*-bin-ubuntu-x64.tar.gz,prerelease=true]@latest"
# macOS arm64: asset_pattern=llama-*-bin-macos-arm64.tar.gz    Windows: llama-*-bin-win-cpu-x64.zip
# mise hides releases younger than 24 h; pass an explicit @kev-<tag> to take a fresh one
```

```sh
# from source
git clone -b kev https://github.com/espetro/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build -j --target llama-server llama-decide llama-quantize
```

Released assets: macOS arm64/x64, Linux x64/arm64 (CPU, Vulkan, CUDA 12.8 and 13.4), SYCL, OpenVINO, Snapdragon, Windows (CPU, Vulkan, CUDA, SYCL, OpenCL), Android, iOS xcframework. The CPU builds are the ones tested with Kev so far. On macOS, a tarball downloaded with a browser needs `xattr -d com.apple.quarantine`. All `kev-*` releases are on the [releases page](https://github.com/espetro/llama.cpp/releases).

### 2. Get a model

The 0.8B bundle from [taigrr/kev-0.8b-gguf](https://huggingface.co/taigrr/kev-0.8b-gguf) works as is (4B and 9B are at `taigrr/kev-4b-gguf` and `taigrr/kev-9b-gguf`):

```sh
pip install -U huggingface_hub
hf download taigrr/kev-0.8b-gguf model-f16.gguf head.json --local-dir kev-0.8b
```

### 3. Run it

```sh
llama-server -m kev-0.8b/model-f16.gguf --kev-head kev-0.8b/head.json
```

```sh
curl localhost:8080/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
  "questions": {
    "refund":     {"type": "noul",   "instructions": "Should we refund?"},
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"returns": "Refunds, exchanges", "shipping": "Delays", "billing": "Charges"}}
  }
}'
```

```json
{"answers":{"refund":{"type":"noul","noul":0.4431},
            "department":{"type":"choice","choice":"shipping","confidence":0.3088,
                          "probabilities":{"returns":0.2019,"shipping":0.5392,"billing":0.2589}}},
 "latency_ms":344.2}
```

Same request from the CLI, without a server:

```sh
llama-decide -m kev-0.8b/model-f16.gguf --kev-head kev-0.8b/head.json --json request.json
```

### 4. Play with it in the browser

Open http://localhost:8080/studio while the server runs: edit the state, add `noul` / `choice` / `score` questions, re-run on every change, and read per-option probability bars with confidence labels (automate / review / escalate). The page also shows the matching curl and Python snippets for the request you built.

### 5. Optional: one self-contained GGUF

Packing the head into the model removes `--kev-head` and lets you quantize:

```sh
python tools/kev/kev_pack.py --gguf kev-0.8b/model-f16.gguf --head kev-0.8b/head.json --out kev-0.8b-f16.gguf
llama-quantize kev-0.8b-f16.gguf kev-0.8b-q8_0.gguf q8_0
llama-server -m kev-0.8b-q8_0.gguf
```

The head tensors stay F32 through quantization. Measured against Kev's Python reference on the 0.8B fixtures: max probability delta 0.0005 (F16) / 0.011 (Q8_0), 0 argmax flips.

### 6. Optional: no server at all

The 0.8B model also runs client side, compiled with emscripten (`tools/kev/wasm/build.sh`, about 1 GB live in the tab, 2.1 s for 3 questions with 4 threads). [examples/kev-web](examples/kev-web) is the static page for it: the runtime and the GGUF are fetched only when you press Load, then cached by the browser. See [docs/kev.md](docs/kev.md#browser-wasm).

### 7. Try Kev without installing anything

Kev's authors run a hosted Gradio demo on free ZeroGPU with the original Python stack: [huggingface.co/spaces/jaredpalmer/kev](https://huggingface.co/spaces/jaredpalmer/kev) (0.8B and 4B, ready-made examples). Good for a first look at 4B; this fork is the path for running Kev yourself.

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
