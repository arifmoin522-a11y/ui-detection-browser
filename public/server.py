#!/usr/bin/env python3
"""
Simple HTTP server for the UI detection demo.
Serves static files with proper MIME types for ONNX model loading.
"""

import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".onnx": "application/octet-stream",
        ".tfjs": "application/octet-stream",
        ".json": "application/json",
        ".js": "application/javascript",
        ".mjs": "application/javascript",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


if __name__ == "__main__":
    os.chdir(DIRECTORY)
    with http.server.HTTPServer(("", PORT), Handler) as httpd:
        print(f"[*] Serving UI Detection Demo at http://localhost:{PORT}")
        print("[*] Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[+] Server stopped")
