"""Temporary HTTPS hosting for a single MP4 Reel, with byte-range support."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import secrets
import re


@contextmanager
def serve_video(video_path):
    from pyngrok import ngrok

    video = Path(video_path).resolve()
    route = "/" + secrets.token_urlsafe(24) + ".mp4"

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_video(head_only=True)

        def do_GET(self):
            self.send_video(head_only=False)

        def send_video(self, head_only):
            if self.path != route:
                self.send_error(404)
                return
            size = video.stat().st_size
            start, end = 0, size - 1
            raw_range = self.headers.get("Range")
            if raw_range:
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", raw_range.strip())
                if not match:
                    self.send_error(416)
                    return
                start = int(match.group(1))
                end = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
                if start >= size or end < start:
                    self.send_error(416)
                    return
            self.send_response(206 if raw_range else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            if raw_range:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not head_only:
                with video.open("rb") as handle:
                    handle.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = handle.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)

        def log_message(self, fmt, *args):
            pass

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
        try:
            if public_url:
                ngrok.disconnect(public_url)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
