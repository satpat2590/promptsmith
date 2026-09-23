"""review — the agent that judges a dictated prompt and pads it out.

Given the transcript, its Jev classification, the relevant realm docs and a
snapshot of current system state, the reviewer decides whether the prompt is
valid as-is, what can be cut, what should be added, and writes the improved
prompt. Two backends:

  llm     one OpenRouter chat call (fast, default)
  hermes  a one-shot Hermes agent run (`PROMPTSMITH_HERMES_CMD`, default `hermes -z`),
          which can use its own tools to check the system before answering

The reviewer never carries out the task — promptsmith only crafts prompts.
"""

import json
import re
import shlex
import subprocess
import urllib.request

import config
import omni

REVIEW_SYSTEM = """\
You are promptsmith's reviewer. The user dictated a prompt out loud for their agent fleet
(the Omni ecosystem). You receive: the raw transcript, a Jev classification of it
(per-realm relevance 0..1 and prompt-quality signals 0..1; completeness is 0..2), the docs of
the most relevant realm(s), and a snapshot of the current system state.

Your job is to gauge the prompt's validity against the system as it is now, then produce a
better prompt:
- VALIDITY: does it refer to repos, agents, services and files that actually exist? Is any part
  already done (see recent commits)? Is it aimed at the right realm and agent? Low Jev coverage
  signals point at what is missing.
- REDUCE: drop filler, repetition, speech disfluencies, and requests the system already satisfies.
- IMPROVE: use correct domain terminology and the real names of agents/entities/services from the
  docs; add context the docs provide; make implied acceptance criteria explicit.
- Preserve the user's intent and voice. Never invent requirements — anything unresolved goes under
  Open questions. Fix obvious speech-to-text errors (misheard names) using the docs.
- Do not carry out the task yourself.

Reply with ONLY a JSON object, no prose around it:
{
  "verdict": "ready" | "needs_clarification" | "invalid",
  "summary": "one or two sentences on the prompt's validity",
  "realm": "<primary realm>",
  "agent": "<agent best suited to run it, or empty>",
  "issues": ["problems found against the docs/system state"],
  "reductions": ["what you removed or would remove, and why"],
  "improvements": ["what you added or changed, and why"],
  "questions": ["what the user must answer before an agent starts"],
  "prompt": "the improved prompt in markdown with sections: # Goal, # Context, # Constraints, # Done when, # Open questions, and a final '## Omni context' section naming the realm, the responsible agent and the doc(s) that informed it"
}"""

HERMES_PREAMBLE = (
    "You are acting as a prompt REVIEWER, not an executor. You may inspect the system with "
    "read-only actions (read files, list directories, git log/status) to verify the prompt "
    "against reality, but do NOT modify anything and do NOT carry out the task.\n\n"
)

VERDICTS = ("ready", "needs_clarification", "invalid")


# ── backends ──────────────────────────────────────────────────────────────────
def _llm(system: str, user: str, model: str = config.REVIEW_MODEL, max_tokens: int = 4000) -> str:
    key = config.env("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set (export it or put it in ~/.hermes/.env)")
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        config.OPENROUTER_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": omni.USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        content = json.loads(r.read())["choices"][0]["message"].get("content") or ""
    return content.strip()


def _hermes(system: str, user: str) -> str:
    cmd = shlex.split(config.HERMES_CMD) + [HERMES_PREAMBLE + system + "\n\n" + user]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=config.HERMES_TIMEOUT)
    if r.returncode != 0 and not r.stdout.strip():
        raise RuntimeError(f"hermes exited {r.returncode}: {r.stderr.strip()[-500:]}")
    return r.stdout.strip()


# ── parsing ───────────────────────────────────────────────────────────────────
def parse_review(raw: str) -> dict:
    """Pull the JSON object out of the reviewer's reply (tolerates fences/prose)."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    data = None
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
            except ValueError:
                data = None
    if not isinstance(data, dict):
        return {"verdict": "unknown", "summary": "The reviewer did not return structured output.",
                "issues": [], "reductions": [], "improvements": [], "questions": [], "prompt": raw.strip()}
    for k in ("issues", "reductions", "improvements", "questions"):
        v = data.get(k) or []
        data[k] = [str(x) for x in v] if isinstance(v, list) else [str(v)]
    for k in ("summary", "realm", "agent", "prompt"):
        data[k] = str(data.get(k) or "").strip()
    if data.get("verdict") not in VERDICTS:
        data["verdict"] = "unknown"
    return data


def ensure_omni_context(prompt: str, realms: list, doc_paths: list, agent: str = "") -> str:
    """Guarantee the prompt ends with an '## Omni context' section."""
    if "## Omni context" in prompt or not realms:
        return prompt
    agents = agent or ", ".join(a for r in realms for a in omni.OMNI_REALMS[r]["agents"]) or "(none bound)"
    lines = ["## Omni context",
             f"- Realm: {', '.join(realms)}",
             f"- Responsible agent: {agents}",
             f"- Informed by: {', '.join(doc_paths) if doc_paths else '(no local docs found)'}"]
    return prompt.rstrip() + "\n\n" + "\n".join(lines) + "\n"


# ── entry point ───────────────────────────────────────────────────────────────
def review(transcript: str, classification: dict, backend: str = "", model: str = "") -> dict:
    """Run the reviewer. Returns the parsed review plus `prompt` (final) and `context`."""
    backend = backend or config.REVIEWER
    scores = (classification or {}).get("realm_scores") or {}
    realms = omni.pick_realms(scores)

    docs, doc_paths = [], []
    for name in realms:
        for path, text in omni.collect_docs(name):
            docs.append(f"--- {path} ---\n{text}")
            doc_paths.append(path)
    realm_lines = [f"- {n}: {omni.OMNI_REALMS[n]['meaning']}; agents: "
                   f"{', '.join(omni.OMNI_REALMS[n]['agents']) or '(none)'}" for n in realms]

    user = "\n\n".join([
        f"TRANSCRIPT:\n{transcript}",
        "JEV CLASSIFICATION:\n" + json.dumps(classification or {"note": "classification unavailable"}, indent=1),
        "MOST RELEVANT REALMS:\n" + ("\n".join(realm_lines) or "(none — classification unavailable)"),
        "SYSTEM STATE (snapshot taken now):\n" + omni.system_state(realms),
        "DOCS:\n" + ("\n\n".join(docs) or "(no local documentation found)"),
    ])

    if backend == "hermes":
        raw = _hermes(REVIEW_SYSTEM, user)
    elif backend == "llm":
        model = model or config.REVIEW_MODEL
        raw = _llm(REVIEW_SYSTEM, user, model)
        if not raw:  # reasoning models can return an empty body; retry once
            raw = _llm(REVIEW_SYSTEM, user, model)
    else:
        raise ValueError(f"unknown reviewer backend: {backend!r} (use 'llm' or 'hermes')")
    if not raw:
        raise RuntimeError("the reviewer returned an empty reply")

    result = parse_review(raw)
    result["prompt"] = ensure_omni_context(result["prompt"] or transcript, realms, doc_paths, result.get("agent", ""))
    result["context"] = {"realms": realms, "docs": doc_paths, "backend": backend}
    return result
