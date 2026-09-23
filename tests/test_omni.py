#!/usr/bin/env python3
"""Tests for omni.py — offline (mocked TypeSafe/OpenRouter) where possible."""

import json
import os
import subprocess
import sys
import urllib.request

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import omni  # noqa: E402


class FakeResp:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _typesafe_payload(edoras_score=0.9, other_score=0.1):
    answers = {f"realm_{name}": {"noul": other_score} for name in omni.OMNI_REALMS}
    answers["realm_edoras"] = {"noul": edoras_score}
    answers.update({
        "goal_stated": {"noul": 0.8},
        "has_requirements": {"noul": 0.7},
        "has_context": {"noul": 0.6},
        "completeness": {"score": 1.5},
    })
    return {"answers": answers}


def _llm_payload(text):
    return {"choices": [{"message": {"content": text}}]}


# ── omni_classify ─────────────────────────────────────────────────────────────
def test_omni_classify_realm_scores(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        return FakeResp(_typesafe_payload())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    res = omni.omni_classify("rebalance the Radagast portfolio", api_key="test-key")

    assert "realm_scores" in res and "coverage" in res
    assert res["realm_scores"]["edoras"] == 0.9
    # edoras is the top score (dict sorted desc)
    assert next(iter(res["realm_scores"])) == "edoras"
    # all six realms scored
    assert set(res["realm_scores"]) == set(omni.OMNI_REALMS)
    # coverage preserved
    assert res["coverage"]["goal_stated"] == 0.8
    assert res["coverage"]["completeness"] == 1.5
    # request shape: state + jev model + per-realm questions
    assert captured["url"] == omni.TYPESAFE_URL
    assert captured["body"]["model"] == "jev-latest"
    assert "omni_realms" in captured["body"]["state"]
    assert "realm_edoras" in captured["body"]["questions"]
    assert "completeness" in captured["body"]["questions"]


# ── fetch_docs ────────────────────────────────────────────────────────────────
def test_fetch_docs_edoras():
    path = os.path.expanduser("~/edoras/AGENTS.md")
    if not os.path.exists(path):
        pytest.skip("~/edoras/AGENTS.md not present on this machine")
    text = omni.fetch_docs("edoras")
    assert text.strip(), "fetch_docs returned empty text"
    with open(path, encoding="utf-8", errors="replace") as f:
        first_line = next(l.strip() for l in f if l.strip())
    assert first_line in text
    assert len(text) <= omni.DOC_CHAR_CAP + 4096  # capped (~12k chars)


def test_fetch_docs_missing_realm():
    assert omni.fetch_docs("nonsense-realm") == ""


# ── ground_prompt ─────────────────────────────────────────────────────────────
GROUNDED = (
    "Rebalance the Radagast portfolio per the mandate.\n\n"
    "## Omni context\n"
    "- Realm: edoras\n- Responsible agent: argus\n- Informed by: ~/edoras/AGENTS.md"
)


def test_ground_prompt_embeds_realm_and_context(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode())
        return FakeResp(_llm_payload(GROUNDED))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = omni.ground_prompt("rebalance the Radagast portfolio",
                             {"edoras": 0.9, "omni": 0.1}, model="test/model")

    assert "## Omni context" in out
    assert "edoras" in out
    # system prompt names the top realm + its agents
    sysmsg = captured["body"]["messages"][0]["content"]
    assert "edoras" in sysmsg
    assert "argus" in sysmsg
    assert captured["body"]["model"] == "test/model"
    # transcript was fed to the LLM
    assert "Radagast" in captured["body"]["messages"][1]["content"]


def test_ground_prompt_no_scores_returns_transcript():
    assert omni.ground_prompt("hello world", {}) == "hello world"


def test_merge_omni_context():
    final = "# Goal\nDo the thing."
    merged = omni.merge_omni_context(final, GROUNDED)
    assert "## Omni context" in merged
    assert merged.startswith(final)
    # already present -> unchanged
    assert omni.merge_omni_context(merged, GROUNDED) == merged


# ── end-to-end smoke (needs real API keys) ────────────────────────────────────
def _key_available(name):
    if os.environ.get(name, "").strip():
        return True
    try:
        for line in open(os.path.expanduser("~/.hermes/.env")):
            if line.startswith(name + "=") and line.split("=", 1)[1].strip().strip('"').strip("'"):
                return True
    except OSError:
        pass
    return False


@pytest.mark.skipif(
    not (_key_available("OPENROUTER_API_KEY") and _key_available("TYPESAFE_API_KEY")),
    reason="OPENROUTER_API_KEY / TYPESAFE_API_KEY not available",
)
def test_cli_smoke(tmp_path):
    out = tmp_path / "prompt.md"
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "promptsmith.py"),
         "--text", "rebalance the portfolio", "--no-improve", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.exists() and out.read_text().strip()
    assert "## Omni context" in out.read_text()
