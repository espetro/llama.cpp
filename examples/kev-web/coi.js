// GitHub Pages cannot send COOP/COEP headers, so a service worker adds them to every response.
// Without them the page is not cross-origin isolated and the wasm build falls back to one thread.
// The same file runs as the page script and as the worker.

if (typeof window === "undefined") {
    self.addEventListener("install", () => self.skipWaiting());
    self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

    self.addEventListener("fetch", (event) => {
        const request = event.request;
        if (request.cache === "only-if-cached" && request.mode !== "same-origin") {
            return;
        }

        event.respondWith(fetch(request).then((response) => {
            if (response.status === 0) {
                return response;
            }
            const headers = new Headers(response.headers);
            headers.set("Cross-Origin-Embedder-Policy", "require-corp");
            headers.set("Cross-Origin-Opener-Policy", "same-origin");
            headers.set("Cross-Origin-Resource-Policy", "cross-origin");
            return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
        }));
    });
} else if (!window.crossOriginIsolated && window.isSecureContext && navigator.serviceWorker) {
    navigator.serviceWorker.register(document.currentScript.src).then((registration) => {
        // the first load is not isolated yet, one reload puts the page under the worker
        registration.addEventListener("updatefound", () => window.location.reload());
        if (registration.active && !navigator.serviceWorker.controller) {
            window.location.reload();
        }
    }).catch((err) => console.warn("no cross-origin isolation, single thread only:", err));
}
