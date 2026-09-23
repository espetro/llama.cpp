# Kev (System One) decision models

[Kev](https://github.com/jaredpalmer/kev) is a family of small Jev-style decision models (Qwen3.5 0.8B / 4B / 9B + LoRA + a trained pointer head). It runs prefill-only and returns calibrated probabilities for typed questions instead of generating text. This fork runs Kev natively: the pointer head and calibration temperature live inside the GGUF, and `llama-server`, `llama-decide` and `common/decision` execute the decision on top of the stock backbone.

## Quick start

```sh
llama-server -m kev-0.8b-q8_0.gguf
```

Then:

```sh
curl localhost:8080/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"returns": "Exchanges, refunds", "shipping": "Delays, lost packages", "billing": "Charges, invoices"}},
    "escalate":   {"type": "noul", "instructions": "Should a human agent take over right away?"},
    "frustration":{"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}'
```

```json
{"model":"kev-0.8b-q8_0.gguf",
 "answers":{
  "department":{"type":"choice","choice":"returns","confidence":0.2682,"probabilities":{"billing":0.1531,"returns":0.5121,"shipping":0.3348}},
  "escalate":{"type":"noul","noul":0.6753},
  "frustration":{"type":"score","score":1.3218,"legend":{"0":"Calm","1":"Frustrated","2":"Very angry"},"probabilities":{"0":0.0644,"1":0.5495,"2":0.3861},"confidence":0.7747}},
 "latency_ms":513.9}
```

Open `http://localhost:8080/studio` for an editor with presets, per-question probability bars, confidence labels and copyable curl/Python snippets.

The request and response follow the TypeSafe `/v1/systemone` contract, so the TypeSafe SDK and Kev's own `tests/test_api.py` work by repointing `base_url`. Every response carries an `x-typesafe-request-id` header. `GET /v1/models` includes a `kev` object (head dim, temperature, source) when a head is loaded.

Ordinary chat, completion and embedding routes are unchanged. Without a Kev head, `/v1/systemone` and `/studio` return 404.

## Question types

| type     | criteria                       | answer                                                        |
|----------|--------------------------------|---------------------------------------------------------------|
| `noul`   | optional `{true: ..., false: ...}` | `noul`: probability of the true option                     |
| `choice` | `{key: description, ...}`      | `choice`, `probabilities`, `confidence = (max - 1/K) / (1 - 1/K)` |
| `score`  | `["level 0", "level 1", ...]`  | `score` (expected level), `legend`, `probabilities`, `confidence` |

Confidence gating is the intended production pattern: automate above your threshold (for example 0.85), escalate the rest to a frontier model or a human.

## CLI

```sh
llama-decide -m kev-0.8b-q8_0.gguf --json request.json         # one TypeSafe request, JSON answer on stdout
llama-decide -m kev-0.8b-q8_0.gguf --json request.json --bench 20
llama-decide -m kev-0.8b-q8_0.gguf --check reference.json      # parity against a Kev/gojev reference fixture
```

## Models

Two ways to get a GGUF with a head:

1. **Embedded head (preferred).** Pack the head into the backbone GGUF once:

   ```sh
   python tools/kev/kev_pack.py --gguf model-f16.gguf --head head.json --manifest manifest.json --out kev-0.8b-f16.gguf
   llama-quantize kev-0.8b-f16.gguf kev-0.8b-q8_0.gguf q8_0
   ```

   `model-f16.gguf` + `head.json` come from a [gojev](https://github.com/taigrr/gojev) bundle (`taigrr/kev-{0.8b,4b,9b}-gguf` on Hugging Face) or from gojev's `tools/kev-convert` run on a Kev checkpoint. Quantization keeps the `dec.head_*` tensors in F32. The packed file still loads in stock llama.cpp as a plain Qwen3.5 model.

2. **Sidecar head.** Keep the gojev layout and pass the head explicitly:

   ```sh
   llama-server -m model-f16.gguf --kev-head head.json
   ```

Measured against Kev's Python reference on the 0.8B fixtures: max |dp| 0.0005 (F16) / 0.011 (Q8_0), 0 argmax flips.

## Browser (WASM)

The same decision path builds with emscripten as `kev.js`/`kev.wasm`, so a 0.8B q8_0 model runs fully client side:

```sh
source /path/to/emsdk/emsdk_env.sh
tools/kev/wasm/build.sh                         # threaded build, needs COOP/COEP when served
KEV_WASM_THREADS=0 tools/kev/wasm/build.sh      # single thread, works on any static host

cp kev-0.8b-q8_0.gguf build-wasm-kev/bin/model.gguf
tools/kev/wasm/serve.py build-wasm-kev/bin      # sends the COOP/COEP headers threads need
```

`index.html` is a small demo of the JS API; `?model=<url>`, `?threads=N` and `?rowcap=N` override its defaults.

```js
import { loadKev } from "./kev-wasm.js";
const kev = await loadKev({ model: "model.gguf", rowCap: 1024, threads: 4 });
const response = kev.systemOne({ state: "...", questions: { ... } });
```

`rowCap` matters in a tab: the context is `4 * rowCap` tokens, and the model default (8192) asks for more memory than a browser gives. 1024 rows on 0.8B q8_0 need about 1 GB live. Measured on this box, 3 questions on a short state: 2.1 s with 4 threads, 7.1 s single threaded, 0.36 s native. Probabilities match the native run to |dp| 0.009 (different SIMD kernels), same argmax. Bigger models are impractical in a tab - the download alone is 814 MB for 0.8B q8_0, and q4_k_m drifts 0.15 which defeats the calibration.

## GGUF format

Four F32 tensors: `dec.head_q.{weight,bias}`, `dec.head_k.{weight,bias}` (`[n_embd, head_dim]`).

Metadata: `kev.version`, `kev.temperature`, `kev.head_dim`, `kev.tokens.{state,question,opt_start,opt_end,decide}` (token ids of `<|fim_prefix|>`, `<|fim_middle|>`, `<|box_start|>`, `<|box_end|>`, `<|fim_suffix|>`), `kev.limits.{max_state,max_row}`, `kev.source`.

## How it runs

Each request is encoded as one row per question: `<state> ... <question> ... <opt_start> option <opt_end> ... <decide>`. The shared state is decoded once into sequence 0 and copied to a branch sequence per question with `llama_memory_seq_cp`; branches decode causally in one batch (this is also what makes the hybrid Gated DeltaNet backbone correct, since no custom attention mask is needed). The hidden states at every option end and at the decide token are read back as per-token embeddings, projected by the pointer head, scaled by `1/sqrt(head_dim)` and divided by the calibration temperature, then softmaxed per question.

The server uses a small dedicated decision context (4 sequences, row capacity `min(kev.limits.max_row, n_ctx_seq)`) next to the normal one, so decisions and chat can share a loaded model. Requests are serialized through a mutex; states longer than the row capacity minus the longest question branch are truncated.

Code: `common/decision.{h,cpp}` (encoding, execution, pointer head, answer formatting), `tools/kev/` (CLI, packer, WASM wrapper), `tools/server/server-decision.{h,cpp}` and `tools/server/public_kev/studio.html` (embedded at build time).

## Licenses

Kev weights and head: Apache-2.0 (jaredpalmer/kev). The gojev converter and fixtures used for parity: 0BSD. The C++ decision code in this fork is original and MIT like the rest of llama.cpp.
