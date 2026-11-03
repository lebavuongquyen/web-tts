import http.server
import os
import sys

PORT = 8085
DIRECTORY = "/home/quyen/08_TTS/rust_tts_runtime/wasm"

class DualHeaderHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # BẮT BUỘC ĐỂ TRÌNH DUYỆT BẬT SHAREDARRAYBUFFER ĐA LUỒNG CHO WEBASSEMBLY
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

def run_multithreaded_server():
    os.chdir(DIRECTORY)
    server_address = ("0.0.0.0", PORT)
    with http.server.ThreadingHTTPServer(server_address, DualHeaderHandler) as httpd:
        print(f"🚀 WASM Multi-thread High-Performance Server running on http://0.0.0.0:{PORT} with COOP/COEP enabled!")
        httpd.serve_forever()

if __name__ == "__main__":
    run_multithreaded_server()
