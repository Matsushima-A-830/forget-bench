"""デモ UI のローカル HTTP サーバー(標準ライブラリだけ。外部のスクリプトや CDN は読まない)。

127.0.0.1 だけで待ち受ける。認証はないので、外向きに公開しない。
GPU を使うバックエンドでは同時に1つの操作しか処理しない(ロック)。
"""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any

import yaml

from forget_bench.demo.session import DemoSession, StrategySpec


def load_demo_config(path: str) -> dict[str, Any]:
    raw = yaml.safe_load(open(path, encoding="utf-8"))
    for key in ("backend", "strategies"):
        if key not in raw:
            raise ValueError(f"デモ設定に {key} がない: {path}")
    return raw


def build_session(cfg: dict[str, Any], backend_override: str | None = None) -> DemoSession:
    from forget_bench.backends import make_backend
    from forget_bench.retrieval import make_embedder

    bcfg = cfg["backend"]
    backend = make_backend(backend_override or bcfg["name"], **({} if backend_override else bcfg.get("params", {})))
    specs = [StrategySpec(s["name"], s.get("label"), s.get("params", {})) for s in cfg["strategies"]]
    if backend_override:  # dummy などで UI だけ確かめるとき、KV 層の戦略は動かないので外す
        specs = [s for s in specs if not s.name.startswith("kv_")]
    embedder = None
    if any(s.name in ("rag", "hybrid") for s in specs):
        ecfg = (cfg.get("retrieval") or {}).get("embedder") or {"name": "ollama", "params": {"model": "bge-m3"}}
        embedder = make_embedder(ecfg["name"], **ecfg.get("params", {}))
    primary = cfg.get("primary")
    if primary and primary not in [s.display for s in specs]:
        primary = None
    return DemoSession(backend, specs, lang=cfg.get("lang", "ja"), primary=primary, embedder=embedder,
                       max_tokens=int(cfg.get("max_tokens", 256)))


class _Handler(BaseHTTPRequestHandler):
    session: DemoSession
    lock: threading.Lock

    def log_message(self, fmt: str, *args: Any) -> None:  # アクセスログは出さない(発話の内容を端末に流さない)
        pass

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, status: int = 200) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            html = resources.files("forget_bench.demo").joinpath("static/index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/api/state":
            with self.lock:
                self._json(self.session.state())
        else:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "JSON ではない"}, HTTPStatus.BAD_REQUEST)
            return
        s = self.session
        try:
            with self.lock:
                extra: dict[str, Any] = {}
                if self.path == "/api/chat":
                    s.chat(str(body.get("text", "")))
                elif self.path == "/api/ask":
                    extra["answers"] = s.ask_all(str(body.get("text", "")))
                elif self.path == "/api/forget":
                    s.forget_value(str(body.get("value", "")))
                elif self.path == "/api/session":
                    s.new_session()
                elif self.path == "/api/load":
                    s.load_scenario(int(body.get("seed", 1000)), int(body.get("sessions", 3)))
                elif self.path == "/api/reset":
                    s.reset()
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    return
                self._json({**s.state(), **extra})
        except ValueError as e:
            self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)


def make_server(session: DemoSession, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"session": session, "lock": threading.Lock()})
    return ThreadingHTTPServer((host, port), handler)


def serve(session: DemoSession, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = make_server(session, host, port)
    print(f"デモ: http://{host}:{server.server_address[1]}/  (Ctrl+C で終了)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
