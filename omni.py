#!/usr/bin/env python3
"""omni — Omni-ecosystem grounding for promptsmith.

OMNI_REALMS maps the six realms of the Omni ecosystem (meaning, repo, bound
agents, docs). omni_classify() scores a transcript against every realm via
TypeSafe System One (Jev); fetch_docs() loads a realm's local documentation;
ground_prompt() rewrites a raw transcript into a doc-grounded prompt.
"""

import json
import os
import urllib.request

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_GROUND_MODEL = "deepseek/deepseek-chat"  # NON-reasoning: reasoning models (v4-*) burn max_tokens on `reasoning` and return empty content on 12KB docs
DOC_CHAR_CAP = 12000  # per realm

OMNI_REALMS = {
    "edoras": {"meaning": "financial (Tolkien)", "repo": "~/edoras",
               "agents": ["argus", "paisa"],
               "docs": ["~/edoras/AGENTS.md", "~/edoras/docs", "~/edoras/README.md",
                        "~/.hermes/profiles/argus/SOUL.md", "~/.hermes/profiles/paisa/SOUL.md",
                        "~/edoras-operations-journal/AGENTS.md"]},
    "atma":   {"meaning": "the self / growth (आत्मन्)", "repo": "~/atma",
               "agents": ["gyani"],
               "docs": ["~/atma", "~/.hermes/profiles/gyani/SOUL.md", "~/gyani/SOUL.md"]},
    "soma":   {"meaning": "the body (σῶμα)", "repo": "~/whoop-sync",
               "agents": [],
               "docs": ["~/whoop-sync", "~/whoop-sync/README.md"]},
    "raga":   {"meaning": "sound × emotion × body (राग)", "repo": "~/whoop-sync",
               "agents": [],
               "docs": ["~/whoop-sync/spotify_etl.py", "~/whoop-sync/README.md"]},
    "smriti": {"meaning": "memory (स्मृति)", "repo": "~/veltiosi",
               "agents": ["veltiosi"],
               "docs": ["~/veltiosi", "~/.hermes/profiles/veltiosi/SOUL.md", "~/Obsidian"]},
    "omni":   {"meaning": "the glue (ὅμος)", "repo": "~/omni",
               "agents": ["satya"],
               "docs": ["~/omni", "~/omni/net/agent_services.json"]},
}


def _typesafe_key() -> str:
    k = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if k:
        return k
    try:
        for line in open(os.path.expanduser("~/.hermes/.env")):
            if line.startswith("TYPESAFE_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def _realms_summary() -> str:
    lines = []
    for name, r in OMNI_REALMS.items():
        agents = ", ".join(r["agents"]) or "(none)"
        lines.append(f"- {name}: {r['meaning']}; bound agents: {agents}")
    return "\n".join(lines)


# ── classification ────────────────────────────────────────────────────────────
def omni_classify(transcript: str, api_key: str = "") -> dict:
    """Score the transcript against all six Omni realms + coverage (TypeSafe Jev).

    Returns {"realm_scores": {<realm>: 0..1, ...} sorted desc, "coverage": {...}}.
    """
    api_key = api_key or _typesafe_key()
    questions = {}
    for name, r in OMNI_REALMS.items():
        agents = ", ".join(r["agents"]) or "none"
        questions[f"realm_{name}"] = {
            "type": "noul",
            "instructions": f"Does this prompt concern the {name} realm ({r['meaning']})? "
                            f"Consider the bound agents ({agents}).",
        }
    questions.update({
        "goal_stated": {"type": "noul", "instructions": "Has the user stated a clear goal or objective?"},
        "has_requirements": {"type": "noul", "instructions": "Has the user stated specific requirements, features, or constraints?"},
        "has_context": {"type": "noul", "instructions": "Has the user given background or context?"},
        "completeness": {
            "type": "score",
            "instructions": "How complete is this prompt so far?",
            "criteria": ["fragmentary — just started", "partial — several parts stated", "complete — ready to finalize"],
        },
    })
    body = {
        "state": {
            "transcript": transcript,
            "task": "the user is dictating a prompt out loud",
            "omni_realms": _realms_summary(),
        },
        "model": "jev-latest",
        "questions": questions,
    }
    req = urllib.request.Request(
        TYPESAFE_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 # Cloudflare (error 1010) bans the default python-urllib UA
                 "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) promptsmith/2.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        a = json.loads(r.read())["answers"]
    realm_scores = {name: round(a[f"realm_{name}"]["noul"], 2) for name in OMNI_REALMS}
    realm_scores = dict(sorted(realm_scores.items(), key=lambda kv: kv[1], reverse=True))
    coverage = {
        "goal_stated": round(a["goal_stated"]["noul"], 2),
        "has_requirements": round(a["has_requirements"]["noul"], 2),
        "has_context": round(a["has_context"]["noul"], 2),
        "completeness": round(a["completeness"]["score"], 2),
    }
    return {"realm_scores": realm_scores, "coverage": coverage}


# ── doc fetching ──────────────────────────────────────────────────────────────
def _md_files_one_level(path: str) -> list:
    """*.md files in `path` and in its immediate subdirectories (one level)."""
    files = []
    try:
        names = sorted(os.listdir(path))
    except OSError:
        return files
    for name in names:
        full = os.path.join(path, name)
        if os.path.isfile(full) and name.endswith(".md"):
            files.append(full)
        elif os.path.isdir(full):
            try:
                subs = sorted(os.listdir(full))
            except OSError:
                continue
            for sub in subs:
                sf = os.path.join(full, sub)
                if os.path.isfile(sf) and sub.endswith(".md"):
                    files.append(sf)
    return files


def _obsidian_key_files(path: str) -> list:
    """Cap the Obsidian vault to key files only (README / MOCs) — never crawl it."""
    files = []
    try:
        names = sorted(os.listdir(path))
    except OSError:
        return files
    for name in names:
        full = os.path.join(path, name)
        low = name.lower()
        if os.path.isfile(full) and name.endswith(".md") and ("readme" in low or "moc" in low):
            files.append(full)
        elif os.path.isdir(full) and ("moc" in low or "readme" in low):
            try:
                subs = sorted(os.listdir(full))
            except OSError:
                continue
            for sub in subs:
                sf = os.path.join(full, sub)
                if os.path.isfile(sf) and sub.endswith(".md"):
                    files.append(sf)
    return files


def fetch_docs(realm_name: str) -> str:
    """Read the realm's docs (AGENTS.md / SOUL.md / README.md / docs *.md, one level
    deep). Missing paths are skipped. Total text capped at ~DOC_CHAR_CAP chars."""
    realm = OMNI_REALMS.get(realm_name)
    if not realm:
        return ""
    parts, total = [], 0
    for p in realm["docs"]:
        path = os.path.expanduser(p)
        if not os.path.exists(path):
            continue
        if os.path.isfile(path):
            files = [path]
        elif os.path.basename(path.rstrip("/")) == "Obsidian":
            files = _obsidian_key_files(path)
        else:
            files = _md_files_one_level(path)
        for f in files:
            try:
                text = open(f, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            chunk = f"\n\n--- {f} ---\n{text}"
            if total + len(chunk) > DOC_CHAR_CAP:
                chunk = chunk[: max(0, DOC_CHAR_CAP - total)]
            if chunk:
                parts.append(chunk)
                total += len(chunk)
            if total >= DOC_CHAR_CAP:
                return "".join(parts)
    return "".join(parts)


# ── doc-grounded refinement ───────────────────────────────────────────────────
def ground_prompt(transcript: str, realm_scores: dict, model: str = None) -> str:
    """Rewrite `transcript` grounded in the docs of the top-scoring realm(s).

    Picks realms with score >= 0.35 (at minimum the top-1 realm). Returns the
    grounded prompt text, or the transcript unchanged if nothing can be picked.
    """
    from promptsmith import _llm

    model = model or os.environ.get("PROMPTSMITH_GROUND_MODEL", DEFAULT_GROUND_MODEL)
    picks = [n for n, s in realm_scores.items() if s >= 0.35]
    if not picks and realm_scores:
        picks = [next(iter(realm_scores))]
    picks = [p for p in picks if p in OMNI_REALMS]
    if not picks or not transcript.strip():
        return transcript

    realm_bits, doc_bits = [], []
    for name in picks:
        r = OMNI_REALMS[name]
        agents = ", ".join(r["agents"]) or "none"
        realm_bits.append(f"the {name} realm ({r['meaning']}), whose bound agents are {agents}")
        docs = fetch_docs(name)
        if docs:
            doc_bits.append(f"=== documentation for realm '{name}' ===\n{docs}")
    realms_desc = "; and ".join(realm_bits)
    docs_text = "\n\n".join(doc_bits) or "(no local documentation found)"

    system = (
        "You ground a user's spoken prompt in the Omni ecosystem. "
        f"The prompt concerns {realms_desc}. "
        "Use the documentation below to refine the prompt so it uses correct domain terminology, "
        "names the real agents/entities/services from the docs, and is actionable by the right agent. "
        "Preserve the user's intent and wording. Do not invent requirements. "
        "Append an '## Omni context' section naming the realm, the responsible agent, and which "
        "doc(s) informed the refinement."
    )
    user = f"TRANSCRIPT:\n{transcript}\n\n{docs_text}"
    out = _llm(system, user, model)
    if not out or "## Omni context" not in out:  # reasoning models can spill an empty body; retry once
        out = _llm(system, user, model)
    if not out:
        return transcript
    return out


def merge_omni_context(final: str, grounded: str) -> str:
    """Guarantee the final prompt carries the grounded '## Omni context' section
    (the structure/refine passes may otherwise rewrite it away)."""
    marker = "## Omni context"
    if marker in final or marker not in grounded:
        return final
    section = grounded[grounded.index(marker):]
    lines = section.splitlines()
    end = len(lines)
    for i, line in enumerate(lines[1:], 1):
        if line.startswith("## ") or line.startswith("# "):
            end = i
            break
    section = "\n".join(lines[:end]).strip()
    return final.rstrip() + "\n\n" + section + "\n"
