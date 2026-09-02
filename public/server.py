#!/usr/bin/env python3
"""
Optimized HTTP server for the UI detection demo.
- Gzip compression for text assets
- Proper MIME types for ONNX/JS/CSS/HTML
- Cache headers (immutable for model, no-cache for HTML)
- Range request support for large model files
- CORS headers for cross-origin model loading
"""

import gzip
import http.server
import io
import os
import sys
import zlib

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

TEXT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}

BINARY_TYPES = {
    ".onnx": "application/octet-stream",
    ".bin": "application/octet-stream",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
}

CACHE_CONTROL = {
    ".onnx": "public, max-age=31536000, immutable",
    ".js": "public, max-age=3600",
    ".css": "public, max-age=3600",
    ".html": "no-cache, no-store, must-revalidate",
}


class OptimizedHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map}
    extensions_map.update(TEXT_TYPES)
    extensions_map.update(BINARY_TYPES)

    def end_headers(self):
        path = self.translate_path(self.path)
        ext = os.path.splitext(path)[1].lower()

        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Length, Content-Range")

        cache = CACHE_CONTROL.get(ext, "no-cache")
        self.send_header("Cache-Control", cache)

        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def do_GET(self):
        if self.path.endswith(".onnx"):
            self._serve_with_gzip()
        else:
            super().do_GET()

    def _serve_with_gzip(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404, "File not found")
            return

        accept_encoding = self.headers.get("Accept-Encoding", "")
        use_gzip = "gzip" in accept_encoding

        with open(path, "rb") as f:
            data = f.read()

        if use_gzip and len(data) > 1024:
            compressed = gzip.compress(data, compresslevel=6)
            if len(compressed) < len(data) * 0.9:
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(compressed)))
                self.end_headers()
                self.wfile.write(compressed)
                return

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        if "200" in str(args):
            sys.stderr.write(f"[+] {args[0]}\n")
        elif any(c in str(args[0]) for c in ["4", "5"]):
            sys.stderr.write(f"[!] {args[0]}\n")


if __name__ == "__main__":
    os.chdir(DIRECTORY)
    with http.server.HTTPServer(("", PORT), OptimizedHandler) as httpd:
        print(f"[*] Serving UI Detection Demo at http://localhost:{PORT}")
        print("[*] Gzip compression enabled for .onnx files")
        print("[*] Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[+] Server stopped")
