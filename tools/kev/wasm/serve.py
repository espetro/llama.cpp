#!/usr/bin/env python3
# Serves the build output with the COOP/COEP headers that WASM threads need.
#
#   cp kev-0.8b-q8_0.gguf build-wasm-kev/bin/model.gguf
#   tools/kev/wasm/serve.py build-wasm-kev/bin
#   open http://localhost:8099

import argparse
import functools
import http.server
import socketserver


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


parser = argparse.ArgumentParser()
parser.add_argument("directory", nargs="?", default="build-wasm-kev/bin")
parser.add_argument("--port", type=int, default=8099)
args = parser.parse_args()

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", args.port), functools.partial(Handler, directory=args.directory)) as httpd:
    print(f"serving {args.directory} on http://localhost:{args.port}")
    httpd.serve_forever()
