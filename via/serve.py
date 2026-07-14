import logging
import os
import socketserver
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler

from via.utils import setup_logging

logger = logging.getLogger(__name__)

# Shared static assets (the franka*.obj meshes) live in the UI/ folder next to
# this file. Per-run pages are served from the demo's save dir; this dir is the
# fallback so the meshes resolve without copying them into every demo folder.
ASSET_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "UI")


class DemoRequestHandler(SimpleHTTPRequestHandler):
    """Serve files from the run's demo dir (self.directory), falling back to
    ASSET_DIR for anything not found there (e.g. the franka .obj meshes). Adds
    permissive CORS headers."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def translate_path(self, path):
        # Resolve against the primary dir first; if the file isn't there, remap
        # the same (already-sanitized) relative path under the shared ASSET_DIR.
        local = super().translate_path(path)
        if os.path.exists(local):
            return local
        rel = os.path.relpath(local, self.directory)
        return os.path.join(ASSET_DIR, rel)


class _ForkSafeHTTPServer(HTTPServer):
    """HTTPServer whose server_bind skips socket.getfqdn().

    We run this server in a forked-but-not-exec'd child (record_sim starts it via
    mp.Process with the 'fork' start method). Stock HTTPServer.server_bind calls
    socket.getfqdn(host) purely to set self.server_name -- which this server never
    uses. On macOS that reverse-DNS lookup routes through Network.framework, whose
    lazy os_log init is NOT fork-safe: in the child it segfaults ("multi-threaded
    process forked" / "crashed on child side of fork") before the port ever binds,
    so the UI page is unreachable. Skipping getfqdn removes that crashing path.
    See record_sim._prime_fork_dns for the sibling workaround on the websocket
    servers, which resolve "localhost" via getaddrinfo rather than getfqdn."""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)  # bind + getsockname; no getfqdn
        host, port = self.server_address[:2]
        self.server_name = host or "localhost"
        self.server_port = port


def http_server(serve_dir=None, port=8100):
    """Serve `serve_dir` (default: the shared asset dir) on `port`, with CORS and
    an ASSET_DIR fallback for the shared meshes. Blocks forever — run in a
    subprocess."""
    setup_logging()  # runs as a forked subprocess / __main__; ensure a handler exists
    root = os.path.abspath(serve_dir) if serve_dir else ASSET_DIR
    handler = partial(DemoRequestHandler, directory=root)
    httpd = _ForkSafeHTTPServer(("", port), handler)
    logger.info(f"Starting HTTP server on port {port} (root: {root})")
    httpd.serve_forever()


if __name__ == "__main__":
    http_server()
