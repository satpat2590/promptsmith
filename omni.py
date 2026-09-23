"""omni — the Omni ecosystem: realm registry, Jev classification, docs and live system state.

OMNI_REALMS (loaded from realms.json) maps each realm to its meaning, repo, bound
agents and docs. omni_classify() scores a transcript against every realm plus a few
prompt-quality signals via TypeSafe System One (Jev). collect_docs() and
system_state() gather what the reviewer agent needs to judge the prompt against
the system as it is right now.
"""

import json
import os
import subprocess
import urllib.request

import config

# Cloudflare (error 1010) bans the default python-urllib UA.
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) promptsmith/3.0"


def load_realms(path: str = config.REALMS_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


OMNI_REALMS = load_realms()

# Prompt-quality signals asked alongside the realm questions. `noul` answers are 0..1.
COVERAGE_QUESTIONS = {
    "goal_stated": {"type": "noul", "instructions": "Has the user stated a clear goal or objective?"},
    "has_requirements": {"type": "noul", "instructions": "Has the user stated specific requirements, features, or constraints?"},
    "has_context": {"type": "noul", "instructions": "Has the user given background or context?"},
    "has_done_criteria": {"type": "noul", "instructions": "Has the user said how to tell when the task is done?"},
    "single_task": {"type": "noul", "instructions": "Is this one focused task rather than several unrelated requests?"},
    "completeness": {
        "type": "score",
        "instructions": "How complete is this prompt so far?",
        "criteria": ["fragmentary — just started", "partial — several parts stated", "complete — ready to finalize"],
    },
}


def _realms_summary() -> str:
    return "\n".join(
        f"- {name}: {r['meaning']}; bound agents: {', '.join(r['agents']) or '(none)'}"
        for name, r in OMNI_REALMS.items()
    )


# ── classification (Jev) ──────────────────────────────────────────────────────
def omni_classify(transcript: str, api_key: str = "") -> dict:
    """Score the transcript against every realm + coverage signals.

    Returns {"realm_scores": {<realm>: 0..1} sorted desc, "coverage": {<signal>: float}}.
    """
    api_key = api_key or config.env("TYPESAFE_API_KEY")
    if not api_key:
        raise RuntimeError("TYPESAFE_API_KEY not set (export it or put it in ~/.hermes/.env)")
    questions = {
        f"realm_{name}": {
            "type": "noul",
            "instructions": f"Does this prompt concern the {name} realm ({r['meaning']})? "
                            f"Consider the bound agents ({', '.join(r['agents']) or 'none'}).",
        }
        for name, r in OMNI_REALMS.items()
    }
    questions.update(COVERAGE_QUESTIONS)
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
        config.TYPESAFE_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        answers = json.loads(r.read())["answers"]

    def val(key, kind):
        v = (answers.get(key) or {}).get(kind)
        return round(float(v), 2) if v is not None else None

    scores = {name: val(f"realm_{name}", "noul") or 0.0 for name in OMNI_REALMS}
    coverage = {k: val(k, q["type"]) for k, q in COVERAGE_QUESTIONS.items()}
    return {
        "realm_scores": dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)),
        "coverage": {k: v for k, v in coverage.items() if v is not None},
    }


def pick_realms(realm_scores: dict, threshold: float = config.REALM_THRESHOLD, limit: int = 2) -> list:
    """Realms scoring >= threshold (at most `limit`), else the single top realm."""
    picks = [n for n, s in realm_scores.items() if s >= threshold and n in OMNI_REALMS]
    if not picks:
        picks = [n for n in realm_scores if n in OMNI_REALMS][:1]
    return picks[:limit]


# ── docs ──────────────────────────────────────────────────────────────────────
def _md_files(path: str, keep=lambda name: True) -> list:
    """*.md files in `path` and its immediate subdirectories (one level), filtered by `keep`."""
    files = []
    try:
        names = sorted(os.listdir(path))
    except OSError:
        return files
    for name in names:
        full = os.path.join(path, name)
        if os.path.isfile(full) and name.endswith(".md") and keep(name):
            files.append(full)
        elif os.path.isdir(full) and not name.startswith(".") and keep(name):
            try:
                files += [os.path.join(full, s) for s in sorted(os.listdir(full))
                          if s.endswith(".md") and os.path.isfile(os.path.join(full, s))]
            except OSError:
                continue
    return files


def _obsidian_keep(name: str) -> bool:
    # Only README / MOC notes — never crawl the whole vault.
    low = name.lower()
    return "readme" in low or "moc" in low


def collect_docs(realm_name: str, cap: int = config.DOC_CHAR_CAP) -> list:
    """[(path, text)] for the realm's docs; missing paths skipped, total text capped."""
    realm = OMNI_REALMS.get(realm_name)
    if not realm:
        return []
    out, total, seen = [], 0, set()
    for p in realm["docs"]:
        path = os.path.expanduser(p)
        if os.path.isfile(path):
            files = [path]
        elif os.path.isdir(path):
            keep = _obsidian_keep if os.path.basename(path.rstrip("/")) == "Obsidian" else (lambda n: True)
            files = _md_files(path, keep)
        else:
            continue
        for f in files:
            if f in seen:
                continue
            seen.add(f)
            try:
                with open(f, encoding="utf-8", errors="replace") as fh:
                    text = fh.read(cap - total)
            except OSError:
                continue
            if text:
                out.append((f, text))
                total += len(text)
            if total >= cap:
                return out
    return out


def fetch_docs(realm_name: str) -> str:
    """The realm's docs as one text blob (see collect_docs)."""
    return "".join(f"\n\n--- {path} ---\n{text}" for path, text in collect_docs(realm_name))


# ── live system state ─────────────────────────────────────────────────────────
def _run(cmd, cwd=None, timeout=5) -> str:
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           shell=isinstance(cmd, str))
        return (r.stdout or r.stderr).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f"(failed: {e})"


def _repo_state(repo: str) -> str:
    path = os.path.expanduser(repo)
    if not os.path.isdir(path):
        return f"{repo}: (not present on this machine)"
    if not os.path.isdir(os.path.join(path, ".git")):
        return f"{repo}: present (not a git repo)"
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=path)
    dirty = _run(["git", "status", "--porcelain"], cwd=path)
    log = _run(["git", "log", "-8", "--date=short", "--pretty=%ad %h %s"], cwd=path)
    changed = len([l for l in dirty.splitlines() if l.strip()])
    return f"{repo}: branch {branch}, {changed} uncommitted change(s)\n  recent commits:\n" + \
        "\n".join("    " + l for l in log.splitlines())


def system_state(realms: list) -> str:
    """A short, read-only snapshot of the parts of the system the prompt touches."""
    lines = []
    repos = []
    for name in realms:
        repo = OMNI_REALMS.get(name, {}).get("repo")
        if repo and repo not in repos:
            repos.append(repo)
    for repo in repos:
        lines.append(_repo_state(repo))
    profiles = os.path.expanduser("~/.hermes/profiles")
    if os.path.isdir(profiles):
        lines.append("hermes profiles (agents): " + ", ".join(sorted(os.listdir(profiles))))
    if config.STATE_CMD:
        lines.append(f"$ {config.STATE_CMD}\n" + _run(config.STATE_CMD, timeout=15)[:4000])
    return "\n".join(lines) or "(no system state available)"


def whisper_vocabulary() -> list:
    """Proper nouns whisper should spell correctly (realms, agents, extras from env)."""
    words = ["Omni", "Hermes", "Jev", "promptsmith"]
    for name, r in OMNI_REALMS.items():
        words.append(name.capitalize())
        words += [a.capitalize() for a in r["agents"]]
    words += [w.strip() for w in config.EXTRA_VOCAB.split(",") if w.strip()]
    return list(dict.fromkeys(words))
