#!/usr/bin/env python3
"""server — the promptsmith web app, hosted on the laptop, opened from the desktop.

The browser captures the mic, downsamples to 16 kHz mono and streams raw PCM to
the laptop every ~1.5s. A per-session worker keeps a live transcript (bounded
incremental passes) and re-runs Jev classification every few seconds while you
talk. Finish re-transcribes the whole recording with the accurate model; Review
hands transcript + classification + system state to the reviewer agent and
returns the improved prompt (also saved to out/).

API (JSON unless noted; send X-Promptsmith-Token when a token is configured):
  POST /api/session                 -> {id}
  POST /api/session/<id>/audio      body: raw int16 LE PCM @16 kHz -> session snapshot
  GET  /api/session/<id>            -> session snapshot
  POST /api/session/<id>/finish     -> {transcript, classification}
  POST /api/review {transcript, classification?} -> {classification, review, prompt, saved}
  GET  /api/config                  -> reviewer/whisper settings
"""

import json
import os
import secrets
import socket
import ssl
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config
import pipeline
import stt

MAX_BODY = 4 * 1024 * 1024
CLASSIFY_EVERY = 6.0  # s between live Jev classifications
IDLE_STOP = 120.0  # s without audio before a session's worker exits
SESSION_TTL = 3600.0

TOKEN = ""  # set by serve(); empty = no auth


# ── sessions ──────────────────────────────────────────────────────────────────
class Session:
    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.live = stt.LiveTranscript(config.WHISPER_LIVE)
        self.classification = {}
        self.stt_error = ""
        self.version = 0
        self.updated = time.time()
        self.closed = False
        self._wake = threading.Event()
        self._worker = None
        self._lock = threading.Lock()
        self._classifying = False
        self._classified_text = ""
        self._classified_at = 0.0

    def add_audio(self, pcm: bytes) -> bool:
        ok = self.live.append(pcm)
        self.updated = time.time()
        with self._lock:
            if not self.closed and (self._worker is None or not self._worker.is_alive()):
                self._worker = threading.Thread(target=self._loop, daemon=True)
                self._worker.start()
        self._wake.set()
        return ok

    def _loop(self):
        while not self.closed:
            try:
                changed = self.live.step()
                self.stt_error = ""
            except Exception as e:
                changed, self.stt_error = False, f"transcription failed: {e}"
            if changed:
                self.version += 1
            self._maybe_classify()
            if not changed:
                if time.time() - self.updated > IDLE_STOP:
                    return
                self._wake.wait(0.4)
                self._wake.clear()

    def _maybe_classify(self):
        text = self.live.text
        if (self._classifying or not text or text == self._classified_text
                or time.time() - self._classified_at < CLASSIFY_EVERY):
            return
        self._classifying = True
        threading.Thread(target=self._classify, args=(text,), daemon=True).start()

    def _classify(self, text: str):
        try:
            self.classification = pipeline.classify(text)
        finally:
            self._classified_text, self._classified_at = text, time.time()
            self._classifying = False
            self.version += 1

    def finish(self) -> dict:
        """Stop live work and re-transcribe the whole recording with the accurate model."""
        self.closed = True
        self._wake.set()
        if self._worker is not None:
            self._worker.join(timeout=60)
        audio = self.live.audio()
        text = stt.transcribe(audio, config.WHISPER_FINAL) if len(audio) else ""
        text = text or self.live.text
        self.classification = pipeline.classify(text)
        self.version += 1
        return {"transcript": text, "classification": self.classification}

    def snapshot(self) -> dict:
        return {"id": self.id, "version": self.version, "seconds": round(self.live.seconds, 1),
                "transcript": self.live.text, "classification": self.classification,
                "error": self.stt_error, "recording": not self.closed}


_sessions = {}
_sessions_lock = threading.Lock()


def _new_session() -> Session:
    s = Session()
    with _sessions_lock:
        for sid in [k for k, v in _sessions.items() if time.time() - v.updated > SESSION_TTL]:
            _sessions.pop(sid).closed = True
        _sessions[s.id] = s
    return s


# ── HTTP ──────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    server_version = "promptsmith/3"

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _authorized(self) -> bool:
        if not TOKEN or secrets.compare_digest(self.headers.get("X-Promptsmith-Token", ""), TOKEN):
            return True
        self._json({"error": "unauthorized — open the URL with ?token=… printed by the server"}, 401)
        return False

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("request body too large")
        return self.rfile.read(n) if n else b""

    def _session(self, parts):
        s = _sessions.get(parts[2]) if len(parts) > 2 else None
        if s is None:
            self._json({"error": "unknown session — press Record to start a new one"}, 404)
        return s

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(config.WEB_DIR, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if not path.startswith("/api/"):
            return self._json({"error": "not found"}, 404)
        if not self._authorized():
            return
        parts = path.strip("/").split("/")
        if parts == ["api", "config"]:
            return self._json({"reviewer": config.REVIEWER, "review_model": config.REVIEW_MODEL,
                               "whisper_live": config.WHISPER_LIVE, "whisper_final": config.WHISPER_FINAL})
        if len(parts) == 3 and parts[1] == "session":
            s = self._session(parts)
            return s and self._json(s.snapshot())
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if not self._authorized():
            return
        parts = path.strip("/").split("/")
        try:
            body = self._body()
            if parts == ["api", "session"]:
                return self._json({"id": _new_session().id})
            if len(parts) == 4 and parts[1] == "session":
                s = self._session(parts)
                if s is None:
                    return
                if parts[3] == "audio":
                    if s.closed:
                        return self._json({"error": "session already finished"}, 409)
                    full = not s.add_audio(body)
                    return self._json({**s.snapshot(), "full": full})
                if parts[3] == "finish":
                    return self._json(s.finish())
            if parts == ["api", "review"]:
                req = json.loads(body or b"{}")
                text = (req.get("transcript") or "").strip()
                if not text:
                    return self._json({"error": "nothing to review — record or type a prompt first"}, 400)
                cls = req.get("classification") or None
                return self._json(pipeline.run(text, cls, backend=req.get("backend", "")))
            self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, *a):
        pass  # quiet


# ── TLS / startup ─────────────────────────────────────────────────────────────
def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _ensure_cert(cert, key):
    """Generate a self-signed cert for the LAN IP + localhost if missing."""
    if os.path.exists(cert) and os.path.exists(key):
        return
    os.makedirs(os.path.dirname(cert) or config.CERT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(key) or config.CERT_DIR, exist_ok=True)
    ip = _lan_ip()
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-keyout", key, "-out", cert,
         "-days", "825", "-nodes", "-subj", "/CN=promptsmith",
         "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def make_server(host: str, port: int, https: bool = True, cert: str = "", key: str = ""):
    srv = ThreadingHTTPServer((host, int(port)), Handler)
    srv.daemon_threads = True
    if https:
        cert = cert or os.path.join(config.CERT_DIR, "cert.pem")
        key = key or os.path.join(config.CERT_DIR, "key.pem")
        _ensure_cert(cert, key)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    return srv


def serve(host="0.0.0.0", port=8721, https=True, cert="", key="", token=""):
    """Run the web app until Ctrl-C."""
    global TOKEN
    TOKEN = token or config.env("PROMPTSMITH_TOKEN")
    srv = make_server(host, port, https, cert, key)
    # Load whisper in the background so the first chunk/finish isn't a multi-second stall.
    threading.Thread(target=stt.preload, args=(config.WHISPER_LIVE, config.WHISPER_FINAL), daemon=True).start()

    scheme = "https" if https else "http"
    q = f"?token={TOKEN}" if TOKEN else ""
    print("═" * 64)
    print("  promptsmith — Ctrl-C to stop")
    print(f"  open on your desktop:  {scheme}://{_lan_ip()}:{port}/{q}")
    print(f"  local:                 {scheme}://127.0.0.1:{port}/{q}")
    if https:
        print("  self-signed cert — accept the browser warning once (the mic needs HTTPS)")
    else:
        print("  plain HTTP — browsers only allow the mic on localhost; use HTTPS for the desktop")
    print(f"  whisper live={config.WHISPER_LIVE} final={config.WHISPER_FINAL} · "
          f"reviewer={config.REVIEWER} ({config.REVIEW_MODEL if config.REVIEWER == 'llm' else config.HERMES_CMD})")
    print("═" * 64, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
    finally:
        srv.server_close()


if __name__ == "__main__":
    import promptsmith

    promptsmith.main(["--live"] + __import__("sys").argv[1:])
