// Kev in the browser: loads kev.js/kev.wasm, keeps the GGUF in the emscripten FS
// and exposes systemOne() with the same request and response shape as /v1/systemone.
//
//   import { loadKev } from "./kev-wasm.js";
//   const kev = await loadKev({ model: "kev-0.8b-q8_0.gguf", threads: 4 });
//   const answers = await kev.systemOne({ state: "...", questions: { ... } });

import createKev from "./kev.js";

// model is a GGUF URL or the bytes of one
export async function loadKev({ model, rowCap = 1024, threads = 0, onProgress } = {}) {
    if (!model) {
        throw new Error("loadKev needs a model URL or model bytes");
    }

    // threads need COOP/COEP headers, a page without them can only run single-threaded
    if (threads > 1 && !self.crossOriginIsolated) {
        throw new Error("threads need COOP/COEP headers: Cross-Origin-Opener-Policy: same-origin, " +
                        "Cross-Origin-Embedder-Policy: require-corp");
    }
    const nthreads = threads || (self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1);

    // llama.cpp logs go to stderr, console.error would mark every load line as a page error
    const mod = await createKev({ printErr: (text) => console.log(text) });

    // a threaded build cannot start a worker without SharedArrayBuffer, it would block forever
    if (mod.PThread && typeof SharedArrayBuffer === "undefined") {
        throw new Error("this page is not cross-origin isolated, the threaded build cannot run here: " +
                        "serve it with COOP/COEP headers, reload without bypassing the service worker, " +
                        "or use the single thread build (KEV_WASM_THREADS=0)");
    }
    mod.FS.writeFile("/model.gguf", typeof model === "string" ? await fetchModel(model, onProgress) : model);

    const rc = mod.ccall("kev_init", "number", ["string", "number", "number"], ["/model.gguf", rowCap, nthreads]);
    if (rc !== 0) {
        throw new Error(`kev_init failed with code ${rc} (see the console for the reason)`);
    }

    return {
        threads: nthreads,
        systemOne(request) {
            const ptr = mod.ccall("kev_system_one", "number", ["string"], [JSON.stringify(request)]);
            try {
                const response = JSON.parse(mod.UTF8ToString(ptr));
                if (response.error) {
                    throw new Error(response.error);
                }
                return response;
            } finally {
                mod._free(ptr);
            }
        },
    };
}

async function fetchModel(url, onProgress) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`cannot fetch ${url}: ${response.status}`);
    }

    const total = Number(response.headers.get("content-length")) || 0;
    if (!onProgress || !response.body) {
        return new Uint8Array(await response.arrayBuffer());
    }

    const reader = response.body.getReader();
    const chunks = [];
    let read = 0;
    for (;;) {
        const { done, value } = await reader.read();
        if (done) {
            break;
        }
        chunks.push(value);
        read += value.length;
        onProgress(read, total);
    }

    const bytes = new Uint8Array(read);
    let offset = 0;
    for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.length;
    }
    return bytes;
}
