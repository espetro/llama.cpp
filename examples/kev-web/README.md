# kev-web

Static page that runs Kev-0.8B in the tab: the wasm runtime and the GGUF are only fetched when the user clicks
**Load model**, the model then stays in the browser cache. Deployed from `.github/workflows/kev-pages.yml` to
[espetro.github.io/llama.cpp](https://espetro.github.io/llama.cpp/). The default model is the 466 MB demo quant at
[espetro/kev-0.8b-demo-gguf](https://huggingface.co/espetro/kev-0.8b-demo-gguf); pass `?model=` for the
higher-calibrated q8_0 at [espetro/kev-0.8b-gguf](https://huggingface.co/espetro/kev-0.8b-gguf).

The page needs three build outputs next to it (`kev.js`, `kev.wasm`, `kev-wasm.js`), so serve it from an assembled
directory, not from the source tree:

```sh
source /path/to/emsdk/emsdk_env.sh
tools/kev/wasm/build.sh

mkdir -p site && cp examples/kev-web/* site/
cp build-wasm-kev/bin/kev.js build-wasm-kev/bin/kev.wasm tools/kev/wasm/kev-wasm.js site/
python3 -m http.server -d site 8099
```

`coi.js` registers a service worker that adds the COOP/COEP headers threads need, because GitHub Pages cannot send
headers itself. The first visit loads without isolation and reloads once; a browser that refuses the worker still
works on one thread, about 3x slower.

`?model=<url>` points the page at another packed GGUF (needs CORS), `?threads=N` and `?rowcap=N` override the
defaults. Row capacity drives the context (`4 * rowCap` tokens) and therefore the memory the tab asks for.
