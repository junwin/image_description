"""Temporarily expose exactly one JPEG to Instagram through a pyngrok tunnel."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import quote
import secrets


@contextmanager
def serve_image(image_path):
    """Yield a public HTTPS URL; close HTTP server and tunnel on exit.

    Requires an ngrok authtoken in the user's ngrok configuration.
    A random, unguessable path is used so other local files are inaccessible.
    """
    from pyngrok import ngrok

    image = Path(image_path).resolve()
    route = "/" + secrets.token_urlsafe(24) + ".jpg"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != route:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(image.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with image.open("rb") as source:
                while True:
                    chunk = source.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

        def log_message(self, fmt, *args):
            pass  # avoid leaking the temporary public URL in logs

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    public_url = None
    try:
        tunnel = ngrok.connect(addr=server.server_address[1], proto="http")
        public_url = tunnel.public_url
        if not public_url.startswith("https://"):
            raise RuntimeError("ngrok did not return a public HTTPS URL")
        yield public_url.rstrip("/") + route
    finally:
        if public_url:
            ngrok.disconnect(public_url)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
