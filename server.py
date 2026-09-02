#!/usr/bin/env python3
import http.server
import os
import socket

PORT = int(os.environ.get("PORT", "8000"))
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=DIR, **kw)
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map}
    extensions_map[".onnx"] = "application/octet-stream"
    extensions_map[".js"] = "application/javascript"
    extensions_map[".css"] = "text/css"
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

class ReuseTCPServer(http.server.HTTPServer):
    allow_reuse_address = True
    address_family = socket.AF_INET

if __name__ == "__main__":
    with ReuseTCPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"[*] Serving at http://localhost:{PORT}")
        print(f"[*] Also accessible at http://0.0.0.0:{PORT}")
        httpd.serve_forever()
