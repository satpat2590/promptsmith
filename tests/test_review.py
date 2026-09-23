"""review.py / pipeline.py — reviewer agent with mocked LLM / hermes backends."""

import json
import os
import subprocess
import sys
import urllib.request

import pytest
from conftest import ROOT, FakeResp

import config
import pipeline
import review

CLASSIFICATION = {"realm_scores": {"edoras": 0.9, "omni": 0.1}, "coverage": {"goal_stated": 0.8}}

REVIEW_JSON = {
    "verdict": "needs_clarification",
    "summary": "Valid, but the target portfolio is ambiguous.",
    "realm": "edoras", "agent": "argus",
    "issues": ["'Radagast' is not in the docs"], "reductions": ["removed filler"],
    "improvements": ["named argus"], "questions": ["Which portfolio?"],
    "prompt": "# Goal\nRebalance the portfolio.\n\n## Omni context\n- Realm: edoras\n- Responsible agent: argus",
}


def test_parse_review_plain_and_fenced():
    assert review.parse_review(json.dumps(REVIEW_JSON))["agent"] == "argus"
    fenced = "```json\n" + json.dumps(REVIEW_JSON) + "\n```"
    assert review.parse_review(fenced)["verdict"] == "needs_clarification"
    chatty = "Sure! Here you go:\n" + json.dumps(REVIEW_JSON) + "\nHope that helps."
    assert review.parse_review(chatty)["questions"] == ["Which portfolio?"]


def test_parse_review_falls_back_to_raw_text():
    r = review.parse_review("# Goal\njust markdown")
    assert r["verdict"] == "unknown"
    assert r["prompt"] == "# Goal\njust markdown"


def test_parse_review_normalizes_fields():
    r = review.parse_review(json.dumps({"verdict": "great", "issues": "one thing", "prompt": "p"}))
    assert r["verdict"] == "unknown"
    assert r["issues"] == ["one thing"]
    assert r["questions"] == []


def test_ensure_omni_context_appends_once():
    out = review.ensure_omni_context("# Goal\nx", ["edoras"], ["~/edoras/AGENTS.md"])
    assert "## Omni context" in out and "argus, paisa" in out and "~/edoras/AGENTS.md" in out
    assert review.ensure_omni_context(out, ["edoras"], []) == out
    assert review.ensure_omni_context("x", [], []) == "x"


def test_review_llm_backend(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        return FakeResp({"choices": [{"message": {"content": json.dumps(REVIEW_JSON)}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(config, "env", lambda name, default="": "key" if name == "OPENROUTER_API_KEY" else default)
    r = review.review("rebalance the Radagast portfolio", CLASSIFICATION, backend="llm", model="test/model")

    assert r["verdict"] == "needs_clarification"
    assert r["agent"] == "argus"
    assert "## Omni context" in r["prompt"]
    assert r["context"]["realms"] == ["edoras"]
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["body"]["model"] == "test/model"
    user = captured["body"]["messages"][1]["content"]
    assert "Radagast" in user and "JEV CLASSIFICATION" in user and "SYSTEM STATE" in user
    assert '"edoras": 0.9' in user


def test_review_adds_omni_context_when_missing(monkeypatch):
    reply = dict(REVIEW_JSON, prompt="# Goal\nRebalance.")
    monkeypatch.setattr(review, "_llm", lambda *a, **k: json.dumps(reply))
    r = review.review("rebalance", CLASSIFICATION, backend="llm")
    assert r["prompt"].rstrip().endswith("- Informed by: " + (", ".join(r["context"]["docs"]) or "(no local docs found)"))
    assert "## Omni context" in r["prompt"]


def test_review_hermes_backend(monkeypatch):
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="thinking...\n" + json.dumps(REVIEW_JSON), stderr="")

    monkeypatch.setattr(review.subprocess, "run", fake_run)
    monkeypatch.setattr(config, "HERMES_CMD", "hermes -z")
    r = review.review("rebalance", CLASSIFICATION, backend="hermes")
    assert captured["cmd"][:2] == ["hermes", "-z"]
    assert "do NOT carry out the task" in captured["cmd"][2]
    assert r["context"]["backend"] == "hermes"
    assert r["agent"] == "argus"


def test_review_unknown_backend():
    with pytest.raises(ValueError):
        review.review("x", CLASSIFICATION, backend="nope")


def test_pipeline_run_saves(monkeypatch, tmp_path):
    monkeypatch.setattr(review, "_llm", lambda *a, **k: json.dumps(REVIEW_JSON))
    out = tmp_path / "p.md"
    res = pipeline.run("rebalance", CLASSIFICATION, backend="llm", out=str(out))
    assert res["saved"] == str(out)
    assert "## Omni context" in out.read_text()


def test_pipeline_classify_never_raises(monkeypatch):
    def boom(_):
        raise OSError("offline")

    monkeypatch.setattr(pipeline.omni, "omni_classify", boom)
    assert "offline" in pipeline.classify("hello")["error"]
    assert pipeline.classify("   ") == {}


# ── end-to-end smoke (needs real API keys) ────────────────────────────────────
@pytest.mark.skipif(not (config.env("OPENROUTER_API_KEY") and config.env("TYPESAFE_API_KEY")),
                    reason="OPENROUTER_API_KEY / TYPESAFE_API_KEY not available")
def test_cli_smoke(tmp_path):
    out = tmp_path / "prompt.md"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "promptsmith.py"),
                        "--text", "rebalance the portfolio", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "## Omni context" in out.read_text()
