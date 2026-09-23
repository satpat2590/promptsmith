"""server.py — HTTP flow end to end with whisper, Jev and the reviewer mocked."""

import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

import pipeline
import review
import server
import stt
from stt import SAMPLE_RATE, Segment

CLASSIFICATION = {"realm_scores": {"edoras": 0.9, "omni": 0.1}, "coverage": {"goal_stated": 0.8}}


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(stt, "transcribe_segments",
                        lambda audio, size, fast=False: [Segment(0, len(audio) / SAMPLE_RATE, "rebalance the portfolio")])
    monkeypatch.setattr(stt, "transcribe", lambda audio, size="": "rebalance the Radagast portfolio")
    monkeypatch.setattr(pipeline, "classify", lambda text: CLASSIFICATION if text else {})
    monkeypatch.setattr(review, "_llm", lambda *a, **k: json.dumps({
        "verdict": "ready", "summary": "ok", "agent": "argus", "prompt": "# Goal\nRebalance."}))
    monkeypatch.setattr(server, "CLASSIFY_EVERY", 0)
    monkeypatch.setattr(server, "TOKEN", "")
    srv = server.make_server("127.0.0.1", 0, https=False)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def call(url, data=None, headers=None, method=None):
    if isinstance(data, dict):
        data = json.dumps(data).encode()
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method or ("POST" if data is not None else "GET"))
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_page_served(app):
    with urllib.request.urlopen(app + "/") as r:
        assert b"promptsmith" in r.read()


def test_full_flow(app, monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.config, "OUT_DIR", str(tmp_path))
    sid = call(app + "/api/session", data=b"")["id"]
    chunk = np.zeros(SAMPLE_RATE * 2, dtype="<i2").tobytes()
    call(f"{app}/api/session/{sid}/audio", data=chunk)

    deadline = time.time() + 5
    snap = {}
    while time.time() < deadline:
        snap = call(f"{app}/api/session/{sid}")
        if snap["transcript"] and snap["classification"]:
            break
        time.sleep(0.05)
    assert snap["transcript"] == "rebalance the portfolio"
    assert snap["classification"]["realm_scores"]["edoras"] == 0.9
    assert snap["seconds"] == 2.0

    fin = call(f"{app}/api/session/{sid}/finish", data=b"")
    assert fin["transcript"] == "rebalance the Radagast portfolio"  # accurate final pass wins

    with pytest.raises(urllib.error.HTTPError) as e:
        call(f"{app}/api/session/{sid}/audio", data=chunk)
    assert e.value.code == 409

    res = call(app + "/api/review", data={"transcript": fin["transcript"], "classification": fin["classification"]})
    assert res["review"]["verdict"] == "ready"
    assert "## Omni context" in res["prompt"]
    assert res["saved"].startswith(str(tmp_path))


def test_review_requires_text(app):
    with pytest.raises(urllib.error.HTTPError) as e:
        call(app + "/api/review", data={"transcript": "  "})
    assert e.value.code == 400


def test_unknown_session(app):
    with pytest.raises(urllib.error.HTTPError) as e:
        call(app + "/api/session/nope")
    assert e.value.code == 404


def test_token_required(app, monkeypatch):
    monkeypatch.setattr(server, "TOKEN", "s3cret")
    with pytest.raises(urllib.error.HTTPError) as e:
        call(app + "/api/config")
    assert e.value.code == 401
    assert call(app + "/api/config", headers={"X-Promptsmith-Token": "s3cret"})["reviewer"]
    with urllib.request.urlopen(app + "/") as r:  # the page itself stays public
        assert r.status == 200
