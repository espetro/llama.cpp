#!/bin/bash
# Builds the Kev browser bundle (kev.js, kev.wasm, kev-wasm.js, index.html) into build-wasm-kev/bin.
# Needs an activated emsdk (source /path/to/emsdk/emsdk_env.sh) plus cmake and ninja.
#
#   tools/kev/wasm/build.sh            # threaded, needs COOP/COEP when served
#   KEV_WASM_THREADS=0 tools/kev/wasm/build.sh   # single thread, no COOP/COEP needed

set -e

BUILD_DIR=${BUILD_DIR:-build-wasm-kev}
THREADS=${KEV_WASM_THREADS:-1}

FLAGS="-fwasm-exceptions"
LINK_FLAGS=""
if [ "$THREADS" != "0" ]; then
    FLAGS="$FLAGS -pthread"
    LINK_FLAGS="-pthread -sPTHREAD_POOL_SIZE=5"
fi

# EMSCRIPTEN_SYSTEM_PROCESSOR: emsdk reports x86, which loses the wasm SIMD quant kernels
# LLAMA_WASM_MEM64=OFF: wasm32, since browsers cap a tab well below what memory64 allows
emcmake cmake -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm \
    -DCMAKE_C_FLAGS="$FLAGS" \
    -DCMAKE_CXX_FLAGS="$FLAGS" \
    -DCMAKE_EXE_LINKER_FLAGS="$LINK_FLAGS" \
    -DLLAMA_WASM_MEM64=OFF \
    -DLLAMA_BUILD_HTML=OFF \
    -DGGML_OPENMP=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF

cmake --build "$BUILD_DIR" --target llama-decide-wasm -j "$(nproc)"

ls -la "$BUILD_DIR/bin"
