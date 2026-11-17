
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

HTML = b"<!doctype html><title>inner</title><body style='font:20px sans-serif'>" \
       b"<h1 id='ok'>INNER LOADED</h1></body>"

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        if self.path.startswith("/c_inner"):
            # frame-ancestors names ONLY the immediate parent (middle)
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; frame-ancestors http://localhost:18811")
        self.end_headers()
        self.wfile.write(HTML)
    def log_message(self, *a): pass

for port in (18811, 18813):
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
print("ready", flush=True)
threading.Event().wait()
