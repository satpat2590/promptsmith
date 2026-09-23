"""omni.py — realm registry, Jev classification (mocked), docs, system state."""

import json
import os
import urllib.request

import pytest
from conftest import FakeResp

import omni


def typesafe_payload(edoras=0.9, other=0.1):
    answers = {f"realm_{name}": {"noul": other} for name in omni.OMNI_REALMS}
    answers["realm_edoras"] = {"noul": edoras}
    answers.update({
        "goal_stated": {"noul": 0.8}, "has_requirements": {"noul": 0.7}, "has_context": {"noul": 0.6},
        "has_done_criteria": {"noul": 0.2}, "single_task": {"noul": 0.9}, "completeness": {"score": 1.5},
    })
    return {"answers": answers}


def test_realms_loaded_from_json():
    assert set(omni.OMNI_REALMS) == {"edoras", "atma", "soma", "raga", "smriti", "omni"}
    assert omni.OMNI_REALMS["edoras"]["agents"] == ["argus", "paisa"]


def test_omni_classify_realm_scores(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data.decode())
        return FakeResp(typesafe_payload())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    res = omni.omni_classify("rebalance the Radagast portfolio", api_key="test-key")

    assert next(iter(res["realm_scores"])) == "edoras"
    assert res["realm_scores"]["edoras"] == 0.9
    assert set(res["realm_scores"]) == set(omni.OMNI_REALMS)
    assert res["coverage"]["goal_stated"] == 0.8
    assert res["coverage"]["has_done_criteria"] == 0.2
    assert res["coverage"]["completeness"] == 1.5
    assert captured["url"] == "https://api.typesafe.ai/v1/systemone"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["body"]["model"] == "jev-latest"
    assert "omni_realms" in captured["body"]["state"]
    assert {"realm_edoras", "completeness", "single_task"} <= set(captured["body"]["questions"])


def test_omni_classify_tolerates_missing_answers(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=0: FakeResp({"answers": {"realm_omni": {"noul": 0.5}}}))
    res = omni.omni_classify("x", api_key="k")
    assert res["realm_scores"]["omni"] == 0.5
    assert res["realm_scores"]["edoras"] == 0.0
    assert res["coverage"] == {}


def test_omni_classify_requires_key(monkeypatch):
    monkeypatch.setattr(omni.config, "env", lambda name, default="": "")
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        omni.omni_classify("x")


def test_pick_realms():
    assert omni.pick_realms({"edoras": 0.9, "omni": 0.4, "atma": 0.36}) == ["edoras", "omni"]
    assert omni.pick_realms({"soma": 0.2, "raga": 0.1}) == ["soma"]
    assert omni.pick_realms({}) == []


def test_collect_docs_caps_and_skips_missing(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("# Agents\nargus runs the book.\n")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("A" * 5000)
    (docs / "sub").mkdir()
    (docs / "sub" / "b.md").write_text("B" * 5000)
    (docs / "sub" / "deeper").mkdir()
    (docs / "sub" / "deeper" / "c.md").write_text("never read")
    monkeypatch.setitem(omni.OMNI_REALMS, "test", {
        "meaning": "t", "repo": str(tmp_path), "agents": [],
        "docs": [str(tmp_path / "AGENTS.md"), str(tmp_path / "missing.md"), str(docs)]})

    got = omni.collect_docs("test", cap=8000)
    paths = [p for p, _ in got]
    assert paths[0].endswith("AGENTS.md")
    assert not any("deeper" in p for p in paths)
    assert sum(len(t) for _, t in got) <= 8000
    assert "argus runs the book" in omni.fetch_docs("test")


def test_fetch_docs_edoras_on_real_machine():
    path = os.path.expanduser("~/edoras/AGENTS.md")
    if not os.path.exists(path):
        pytest.skip("~/edoras/AGENTS.md not present on this machine")
    text = omni.fetch_docs("edoras")
    with open(path, encoding="utf-8", errors="replace") as f:
        first_line = next(l.strip() for l in f if l.strip())
    assert first_line in text


def test_fetch_docs_unknown_realm():
    assert omni.fetch_docs("nonsense-realm") == ""


def test_system_state_reports_git_repo(tmp_path, monkeypatch):
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "--allow-empty", "-m", "add rebalancer"], check=True)
    (tmp_path / "dirty.txt").write_text("x")
    monkeypatch.setitem(omni.OMNI_REALMS, "test", {"meaning": "t", "repo": str(tmp_path), "agents": [], "docs": []})
    state = omni.system_state(["test"])
    assert "add rebalancer" in state
    assert "1 uncommitted change" in state


def test_whisper_vocabulary_has_agents():
    vocab = omni.whisper_vocabulary()
    assert "Edoras" in vocab and "Argus" in vocab and "Veltiosi" in vocab
