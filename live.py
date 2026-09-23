#!/usr/bin/env python3
"""promptsmith live — ephemeral streaming prompt server.

Runs ONLY while `promptsmith live` is active (Ctrl-C stops it). Serves a tiny
browser page you open on any machine (Windows/phone); the page captures the mic
and POSTs rolling audio chunks here. Each chunk is transcribed (faster-whisper,
cached model, incremental tail-only) and Omni-classified (per-realm scores +
coverage) so the prompt is "living" — it updates as you talk. `/done` runs the
omni-classify -> doc-ground -> structure+refine pipeline.

stdlib threaded server + faster-whisper + Jev (native TypeSafe). No daemon, no reboot.
"""

import base64
import difflib
import json
import os
import socket
import ssl
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from omni import _typesafe_key, ground_prompt, merge_omni_context, omni_classify
from promptsmith import DEFAULT_MODEL, _load_key, build_prompt

# ── shared state ──────────────────────────────────────────────────────────────
_state = {"chunks": [], "transcript": "", "classification": {}, "final": "", "seq": 0,
          "tx_len": 0}  # tx_len = chunks already folded into the transcript
_lock = threading.Lock()
_tx_lock = threading.Lock()  # serializes buffer mutation + transcription

WHISPER = "base"
MODEL = DEFAULT_MODEL

# ── whisper model cache (lazy module-level singletons) ───────────────────────
_WHISPER_MODELS = {}


def _whisper_model(model_size: str):
    if model_size not in _WHISPER_MODELS:
        from faster_whisper import WhisperModel

        _WHISPER_MODELS[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _WHISPER_MODELS[model_size]

# queued promptsmith settings
_CFG = {"improve": True, "model": MODEL, "whisper": WHISPER}


def _transcribe_bytes(data: bytes, model_size: str) -> str:
    """Transcribe arbitrary audio bytes (webm/wav/mp3 via PyAV)."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(data)
        path = f.name
    model = _whisper_model(model_size)
    segments, _ = model.transcribe(path)
    return " ".join(s.text.strip() for s in segments).strip()


# ── incremental transcription ────────────────────────────────────────────────
# MediaRecorder chunks after the first carry no webm init segment, so the new
# tail is transcribed as chunks[0] (head, cached) + new chunks; the known head
# text is stripped and only the new tail is appended to the transcript.
_audio = {"head": b"", "head_text": ""}


def _strip_head(text: str, head: str) -> str:
    """Remove the already-known head transcription from the front of `text`."""
    if not head:
        return text
    if text.startswith(head):
        return text[len(head):].strip()
    head_words, text_words = head.split(), text.split()
    blocks = difflib.SequenceMatcher(None, head_words, text_words).get_matching_blocks()
    match = max(blocks, key=lambda m: m.size, default=None)
    if match and match.a == 0 and match.size >= max(1, len(head_words) // 2):
        return " ".join(text_words[match.size:]).strip()
    return text  # give up stripping; appending whole may duplicate ~3s


def _update_transcript_locked():
    """Transcribe only the chunks not yet folded into the transcript.
    Must be called with _tx_lock held. Raises on transcription failure."""
    chunks = _state["chunks"]
    if not chunks:
        return
    if chunks[0] != _audio["head"]:
        _audio["head"] = chunks[0]
        _audio["head_text"] = _transcribe_bytes(chunks[0], _CFG["whisper"])
    new = chunks[_state["tx_len"]:]
    if not new:
        return
    if _state["tx_len"] == 0:
        _state["transcript"] = _transcribe_bytes(b"".join(chunks), _CFG["whisper"])
    else:
        text = _transcribe_bytes(chunks[0] + b"".join(new), _CFG["whisper"])
        tail = _strip_head(text, _audio["head_text"])
        if tail:
            _state["transcript"] = (_state["transcript"] + " " + tail).strip()
    _state["tx_len"] = len(chunks)


# ── HTML client (served at /) ─────────────────────────────────────────────────
PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>promptsmith live</title>
<style>
  :root { color-scheme: dark; }
  body { font: 15px/1.5 system-ui, sans-serif; background:#0f1115; color:#e8eaf0;
         max-width:820px; margin:2rem auto; padding:0 1.2rem; }
  h1 { font-size:1.1rem; font-weight:600; }
  button { background:#2c7; color:#07200f; border:0; padding:.6rem 1.1rem; border-radius:8px;
           font-weight:600; cursor:pointer; font-size:.95rem; }
  button.off { background:#3a3f4a; color:#cfd2da; }
  #status { color:#9aa3b2; font-size:.9rem; margin:.4rem 0 1rem; }
  .card { background:#171b22; border:1px solid #262b35; border-radius:12px; padding:1rem 1.2rem; margin:1rem 0; }
  .lbl { font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:#7a8391; margin-bottom:.3rem; }
  #transcript { white-space:pre-wrap; min-height:3em; }
  .pill { display:inline-block; padding:.15rem .55rem; border-radius:999px; font-size:.78rem;
          margin:.15rem .3rem .15rem 0; border:1px solid #333a46; }
  .pill.ok { color:#4ade80; border-color:#1f4d33; }
  .pill.off { color:#f87171; border-color:#5b2a2a; }
  #final { white-space:pre-wrap; font-size:.9rem; }
  #done { background:#8b5cf6; color:#0b0616; margin-top:.4rem; }
  a { color:#7aa2f7; }
</style></head><body>
<h1>🎙️ promptsmith <span style="color:#7a8391;font-weight:400">live</span></h1>
<div id="status">press <b>start</b> to open the mic; speak; watch it assemble.</div>
<button id="mic">start mic</button>
<div class="card"><div class="lbl">live transcript</div><div id="transcript">—</div></div>
<div class="card"><div class="lbl">Omni classification (live)</div>
  <div id="realms" style="font-size:.95rem">—</div>
  <div id="coverage" style="color:#9aa3b2;font-size:.82rem;margin-top:.4rem">—</div>
</div>
<div class="card"><div class="lbl">final prompt (after done)</div>
  <div id="final">press <b>done</b> to structure + refine.</div>
</div>
<button id="done">done → structure + refine</button>
<div style="margin-top:.5rem">
  <button id="copy">copy prompt</button>
  <button id="download">download .md</button>
</div>
<p id="note" style="color:#7a8391;font-size:.8rem;margin-top:.6rem">
  promptsmith only <i>crafts</i> the prompt into a file — it never sends it anywhere.
  Copy or download it, then dispatch to whichever agent you like.</p>
<script>
let rec=null, chunks=[], stream=null;
const $=id=>document.getElementById(id);
const post=async(path,body)=>{const r=await fetch(path,{method:'POST',body});return r.json();};

function paint(s){
  if(!s) return;
  $('transcript').textContent = s.transcript || '—';
  const c=s.classification||{};
  const rs=c.realm_scores||{};
  const names=Object.keys(rs);
  if(names.length){
    let show=names.filter(n=>rs[n]>=0.2).slice(0,3);
    if(!show.length) show=names.slice(0,1);
    $('realms').innerHTML='This prompt is about: '+show.map(n=>
      '<span class="pill ok" style="font-size:.85rem">'+n[0].toUpperCase()+n.slice(1)+' '+rs[n].toFixed(1)+'</span>'
    ).join(' ');
  } else { $('realms').textContent='—'; }
  const cov=c.coverage||{};
  const mark=v=>v>0.5?'✓':'✗';
  $('coverage').textContent = Object.keys(cov).length ?
    'goal '+mark(cov.goal_stated)+' · requirements '+mark(cov.has_requirements)+
    ' · context '+mark(cov.has_context)+' · completeness '+
    (cov.completeness==null?'—':cov.completeness.toFixed(1)+'/2') : '—';
}

$('mic').onclick=async()=>{
  if(rec){ rec.stop(); stream.getTracks().forEach(t=>t.stop()); rec=null; $('mic').textContent='start mic'; $('status').textContent='stopped.'; return; }
  stream=await navigator.mediaDevices.getUserMedia({audio:true});
  rec=new MediaRecorder(stream);
  rec.ondataavailable=async e=>{ if(e.data.size>0){ const r=await post('/chunk', e.data); paint(r); } };
  rec.start(3000);  // roll every 3s
  $('mic').textContent='stop mic'; $('mic').classList.add('off');
  $('status').textContent='recording — speak. (server transcribes + classifies each chunk)';
};

$('done').onclick=async()=>{
  $('status').textContent='classifying + grounding + refining…';
  const r=await post('/done','');
  if(r.classification) paint({classification:r.classification});
  $('final').textContent = r.final || ('error: '+(r.error||''));
  $('status').textContent='done. copy or download the prompt, then dispatch it wherever.'
};

$('copy').onclick=()=>{
  const t=$('final').textContent;
  const ta=document.createElement('textarea'); ta.value=t;
  ta.style.position='fixed'; ta.style.opacity='0'; document.body.appendChild(ta);
  ta.select();
  try{ document.execCommand('copy'); $('status').textContent='copied to clipboard ✔'; }
  catch(e){ $('status').textContent='copy blocked — select the text and Ctrl-C.'; }
  ta.remove();
};

$('download').onclick=()=>{
  const t=$('final').textContent;
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([t],{type:'text/markdown'}));
  a.download='prompt.md'; a.click();
};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode())

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE.encode())
        elif self.path == "/state":
            self._json({"transcript": _state["transcript"], "classification": _state["classification"],
                        "seq": _state["seq"]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        if self.path == "/chunk":
            with _tx_lock:
                _state["chunks"].append(body)
                if len(_state["chunks"]) > 60:  # ~3 min cap
                    dropped = len(_state["chunks"]) - 60
                    _state["chunks"] = _state["chunks"][-60:]
                    _state["tx_len"] = max(0, _state["tx_len"] - dropped)
                try:
                    _update_transcript_locked()
                except Exception:
                    pass  # keep last good transcript; retry with more audio next chunk
                tx = _state["transcript"]
                _state["seq"] += 1
            try:
                classification = omni_classify(tx, _typesafe_key()) if tx else {}
            except Exception:
                classification = {}
            with _lock:
                _state["classification"] = classification
            self._json({"transcript": tx, "classification": classification, "seq": _state["seq"]})
        elif self.path == "/done":
            with _tx_lock:
                try:
                    _update_transcript_locked()
                except Exception:
                    pass
                tx = _state["transcript"]
            try:
                classification = omni_classify(tx, _typesafe_key()) if tx else {}
            except Exception:
                classification = {}
            try:
                grounded = ground_prompt(tx, classification.get("realm_scores", {}))
            except Exception:
                grounded = tx
            final = build_prompt(grounded, _CFG["model"], _CFG["improve"])
            final = merge_omni_context(final, grounded)
            with _lock:
                _state["final"] = final
                _state["transcript"] = tx
                _state["classification"] = classification
            out = __import__("os").path.join(
                __import__("os").path.expanduser("~/tools/promptsmith/out"),
                "prompt-" + time.strftime("%Y%m%d-%H%M%S") + ".md",
            )
            __import__("os").makedirs(__import__("os").path.dirname(out), exist_ok=True)
            open(out, "w").write(final + "\n")
            self._json({"final": final, "classification": classification,
                        "grounded": grounded, "saved": out})
        else:
            self._json({"error": "unknown endpoint"}, 404)

    def log_message(self, *a):
        pass  # quiet


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _default_cert_dir():
    return os.path.join(os.path.expanduser("~"), "tools", "promptsmith", "certs")


def _ensure_cert(cert, key):
    """Generate a self-signed cert for the LAN IP + localhost if missing."""
    if os.path.exists(cert) and os.path.exists(key):
        return
    d = os.path.dirname(cert) or _default_cert_dir()
    os.makedirs(d, exist_ok=True)
    if not os.path.dirname(cert):
        cert = os.path.join(d, "cert.pem")
    if not os.path.dirname(key):
        key = os.path.join(d, "key.pem")
    ip = _lan_ip()
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", key, "-out", cert, "-days", "825", "-nodes",
            "-subj", "/CN=promptsmith",
            "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def serve(host="0.0.0.0", port=8721, model=None, whisper=None, improve=True,
          https=False, cert=None, key=None):
    """Run the ephemeral streaming server until Ctrl-C."""
    global _CFG
    _CFG = {"improve": improve, "model": model or MODEL, "whisper": whisper or WHISPER}
    ip = _lan_ip()
    srv = ThreadingHTTPServer((host, int(port)), Handler)
    scheme = "http"
    if https:
        cert = cert or os.path.join(_default_cert_dir(), "cert.pem")
        key = key or os.path.join(_default_cert_dir(), "key.pem")
        _ensure_cert(cert, key)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
        scheme = "https"
    print("═" * 64)
    print("  promptsmith live — ephemeral server (Ctrl-C to stop)")
    print(f"  open this on ANY machine on your LAN (Windows/phone):")
    print(f"      {scheme}://{ip}:{port}/")
    print(f"  (local: {scheme}://127.0.0.1:{port}/)")
    if https:
        print("  self-signed cert — accept the browser warning to enable the mic")
    print(f"  model={_CFG['model']}  whisper={_CFG['whisper']}  improve={_CFG['improve']}")
    print("═" * 64, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
    finally:
        srv.server_close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8721)
    ap.add_argument("--model")
    ap.add_argument("--whisper")
    ap.add_argument("--no-improve", action="store_true")
    ap.add_argument("--https", action="store_true", help="serve over HTTPS (self-signed) so the browser mic works cross-device")
    ap.add_argument("--cert", help="path to TLS cert (auto-generated if --https and missing)")
    ap.add_argument("--key", help="path to TLS key")
    args = ap.parse_args()
    serve(args.host, args.port, args.model, args.whisper, not args.no_improve,
          args.https, args.cert, args.key)