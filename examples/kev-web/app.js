// Kev web demo: the wasm runtime and the model are only fetched when the user asks for them,
// so the page itself stays small. ?model=<url>, ?threads=N and ?rowcap=N override the defaults.

import { PRESETS } from "./presets.js";

const DEFAULT_MODEL = "https://huggingface.co/espetro/kev-0.8b-demo-gguf/resolve/main/kev-0.8b-demo-q4km-im.gguf";
const CACHE_NAME = "kev-models";

const params = new URLSearchParams(location.search);
const el = (id) => document.getElementById(id);

let kev;

el("model-url").value = params.get("model") || DEFAULT_MODEL;
fillPresets();

el("load").addEventListener("click", load);
el("run").addEventListener("click", run);
el("preset").addEventListener("change", () => applyPreset(Number(el("preset").value)));

async function load() {
    const url = el("model-url").value.trim();
    el("load").disabled = true;
    el("model-url").disabled = true;

    try {
        // the check belongs before the download, the runtime cannot start without isolation
        if (!self.crossOriginIsolated) {
            throw new Error("this page is not cross-origin isolated: the service worker that adds the COOP/COEP " +
                            "headers is missing, reload the page normally (a hard reload bypasses it)");
        }

        setLoaderStatus("fetching the runtime...");
        const { loadKev } = await import("./kev-wasm.js");

        const model = await fetchModel(url, el("cache").checked);
        setLoaderStatus("starting the model...");
        kev = await loadKev({
            model,
            threads: Number(params.get("threads")) || 0,
            rowCap: Number(params.get("rowcap")) || 1024,
        });

        el("loader").hidden = true;
        el("app").hidden = false;
        setStatus(`ready - ${kev.threads} thread(s)${kev.threads === 1 ? ", no cross-origin isolation" : ""}`);
    } catch (err) {
        setLoaderStatus(err.message, true);
        el("load").disabled = false;
        el("model-url").disabled = false;
    }
}

async function fetchModel(url, useCache) {
    const cache = useCache && self.caches ? await caches.open(CACHE_NAME).catch(() => null) : null;

    let response = cache ? await cache.match(url) : undefined;
    if (response) {
        setLoaderStatus("reading the model from the browser cache...");
    } else {
        setLoaderStatus("downloading the model...");
        response = await fetch(url);
        if (!response.ok) {
            throw new Error(`cannot fetch ${url}: ${response.status}`);
        }
        if (cache) {
            // put() needs its own body, the clone is consumed by the cache
            cache.put(url, response.clone()).catch((err) => console.warn("model not cached:", err));
        }
    }

    const total = Number(response.headers.get("content-length")) || 0;
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
        if (total) {
            el("progress").style.width = `${(100 * read / total).toFixed(1)}%`;
        }
        setLoaderStatus(`${(read / 1e6).toFixed(0)}${total ? ` / ${(total / 1e6).toFixed(0)}` : ""} MB`);
    }

    const bytes = new Uint8Array(read);
    let offset = 0;
    for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.length;
    }
    return bytes;
}

function run() {
    let questions;
    try {
        questions = JSON.parse(el("questions").value);
    } catch (err) {
        setStatus(`questions are not valid JSON: ${err.message}`, true);
        return;
    }

    el("run").disabled = true;
    setStatus("deciding...");
    // the decode is synchronous, let the browser paint the status first
    setTimeout(() => {
        try {
            const response = kev.systemOne({ state: el("state").value, questions });
            setStatus(`${response.latency_ms} ms on ${kev.threads} thread(s)`);
            render(response.answers);
            el("raw").textContent = JSON.stringify(response, null, 2);
        } catch (err) {
            setStatus(err.message, true);
        } finally {
            el("run").disabled = false;
        }
    }, 0);
}

function render(answers) {
    const root = el("answers");
    root.replaceChildren();

    for (const [id, answer] of Object.entries(answers)) {
        const box = document.createElement("div");
        box.className = "answer";
        box.innerHTML = `<h3>${id} <span>${summary(answer)}</span></h3>`;

        const top = answer.type === "choice" ? answer.choice : null;
        for (const [name, prob] of Object.entries(answer.probabilities || {})) {
            const label = answer.legend ? `${name} - ${answer.legend[name]}` : name;
            const row = document.createElement("div");
            row.className = name === top ? "row top" : "row";
            row.innerHTML = `<span class="name">${label}</span><div class="bar"><div style="width:${(100 * prob).toFixed(1)}%"></div></div>` +
                            `<span class="value">${prob.toFixed(3)}</span>`;
            box.appendChild(row);
        }
        root.appendChild(box);
    }
}

function summary(answer) {
    if (answer.type === "noul") {
        return `p(yes) ${answer.noul.toFixed(3)}`;
    }
    if (answer.type === "choice") {
        return `${answer.choice}, confidence ${answer.confidence.toFixed(3)}`;
    }
    return `score ${answer.score.toFixed(3)}, confidence ${answer.confidence.toFixed(3)}`;
}

function fillPresets() {
    const select = el("preset");
    PRESETS.forEach((preset, i) => select.add(new Option(preset.name, String(i))));
    applyPreset(0);
}

function applyPreset(index) {
    el("state").value = PRESETS[index].state;
    el("questions").value = JSON.stringify(PRESETS[index].questions, null, 2);
}

function setLoaderStatus(text, isError) {
    el("loader-status").textContent = text;
    el("loader-status").className = isError ? "error" : "";
}

function setStatus(text, isError) {
    el("status").textContent = text;
    el("status").className = isError ? "error" : "";
}
