"""A thin web page over the same configs. Standard library only, bound to 127.0.0.1.

GET /              the page
GET /api/state     registries (with broken components and why) and the starting config
POST /api/run      a config as JSON -> runs it, returns metrics, figure URLs and the config as TOML
GET /runs/<n>/<f>  a figure from run n
GET /guide.pdf     the guide, docs/spikelab-guide.pdf

The page cannot load plugins or pick the output folder: those come from the config it started with.
Every run from the page is checked against spikelab/limits.py first (refused by name if over a cap), then runs
in a child process killed at the time limit. One run at a time; a second is answered 409 busy.

Three checks keep another site in the owner's browser from driving this server:
Host must be this machine's loopback, an Origin (if sent at all) must be this server's own,
and POST /api/run must be application/json, which no cross-site form or no-cors fetch can send.

Public mode (off unless a public host is given) is for serving behind Cloudflare Access. It still binds
127.0.0.1; Host and Origin also accept the public hostname over https, and every request must carry a valid
Access assertion for an allowed email (spikelab/access.py), else 403 with no detail.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config, limits, registry
from .registry import ComponentError

HOST = "127.0.0.1"
MAX_BODY = 64 * 1024
PAGE = (Path(__file__).parent / "web.html").read_text()
GUIDE = Path(__file__).parent.parent / "docs" / "spikelab-guide.pdf"


class Busy(Exception):
    pass


class App:
    def __init__(self, config_path: str):
        self.base = config.load(config_path)
        self.start = config.resolve(self.base)
        self.root = Path(self.start["out"]).parent / "web"
        self.caps, self.seconds = limits.CAPS, limits.SECONDS
        self.lock = threading.Lock()
        self.n = 0

    def state(self) -> dict:
        return {"registry": registry.describe(), "config": self.start}

    def run(self, posted: dict) -> dict:
        cfg = {k: v for k, v in posted.items() if k not in ("plugins", "out")}
        cfg["plugins"] = self.base.get("plugins", [])
        cfg = config.resolve(cfg)
        limits.check(cfg, self.caps)
        if not self.lock.acquire(blocking=False):
            raise Busy("busy: another run is going, try again when it finishes")
        try:
            self.n += 1
            n = self.n
            out = self.root / str(n)
            res = self._child(cfg, out)
        finally:
            self.lock.release()
        return {"final": res["final"], "seconds": res["seconds"],
                "figures": [f"/runs/{n}/{f}" for f in res["figures"]],
                "toml": (out / "config.toml").read_text()}

    def _child(self, cfg: dict, out: Path) -> dict:
        """Run cfg in spikelab.worker; kill its whole process group at the time limit."""
        out.mkdir(parents=True, exist_ok=True)
        (out / "request.toml").write_text(config.dumps(cfg))
        env = {**os.environ, "OMP_NUM_THREADS": "2"}
        p = subprocess.Popen([sys.executable, "-m", "spikelab.worker", str(out / "request.toml"), str(out)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                             start_new_session=True)
        try:
            _, stderr = p.communicate(timeout=self.seconds)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.communicate()
            raise ComponentError(f"stopped: the run went over the {self.seconds:g} s time limit")
        if p.returncode != 0:
            lines = stderr.strip().splitlines()
            raise ComponentError(lines[-1] if lines else f"the run failed (exit {p.returncode})")
        return json.loads((out / "metrics.json").read_text())


def make_handler(app: App, port: int, public_host: str | None = None, access=None):
    allowed = {f"{HOST}:{port}", f"localhost:{port}"}
    origins = {f"http://{h}" for h in allowed}
    if public_host:
        allowed.add(public_host)
        origins.add(f"https://{public_host}")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _caller_ok(self):
            if access is not None and access.email(self.headers) is None:
                self._send(403, {"error": "forbidden"})
                return False
            if self.headers.get("Host") not in allowed:
                self._send(403, {"error": "loopback only"})
                return False
            origin = self.headers.get("Origin")  # absent for curl and for a same-origin GET
            if origin is not None and origin not in origins:
                self._send(403, {"error": "cross-site request"})
                return False
            return True

        def do_GET(self):
            if not self._caller_ok():
                return
            if self.path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if self.path == "/api/state":
                return self._send(200, app.state())
            if self.path == "/guide.pdf" and GUIDE.is_file():
                return self._send(200, GUIDE.read_bytes(), "application/pdf")
            m = re.fullmatch(r"/runs/(\d+)/(raster|membrane|weights|learning)\.png", self.path)
            if m:
                f = app.root / m[1] / f"{m[2]}.png"
                if f.is_file():
                    return self._send(200, f.read_bytes(), "image/png")
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._caller_ok():
                return
            if self.path != "/api/run":
                return self._send(404, {"error": "not found"})
            # application/json forces a preflight, so a cross-site no-cors POST cannot reach this.
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                return self._send(415, {"error": "send application/json"})
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return self._send(413, {"error": "config too large"})
            try:
                res = app.run(json.loads(self.rfile.read(n)))
            except Busy as e:
                return self._send(409, {"error": str(e)})
            except ComponentError as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:  # a bad value must not take the server down; say what broke
                return self._send(400, {"error": f"{type(e).__name__}: {e}"})
            self._send(200, res)

        def log_message(self, fmt, *args):
            pass

    return Handler


def make_server(config_path: str, port: int, public_host: str | None = None, access=None) -> ThreadingHTTPServer:
    """public_host and access go together: both set is public mode, both None is local mode."""
    if (public_host is None) != (access is None):
        raise ValueError("public mode needs both a public host and an Access verifier")
    app = App(config_path)
    srv = ThreadingHTTPServer((HOST, port), None)
    srv.RequestHandlerClass = make_handler(app, srv.server_address[1], public_host, access)
    srv.app = app
    return srv


def serve(config_path: str, port: int = 8765, public_host: str | None = None, access=None) -> None:
    srv = make_server(config_path, port, public_host, access)
    mode = f", public as https://{public_host}/ behind Cloudflare Access" if public_host else ""
    print(f"spikelab on http://{HOST}:{srv.server_address[1]}/{mode}  (Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
