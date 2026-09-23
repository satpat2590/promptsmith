# promptsmith — Omni-grounded classification + doc-driven refinement

## Context

`promptsmith` (`~/tools/promptsmith/`) is a voice→prompt tool. A browser captures the
mic, POSTs rolling 3s audio chunks to a local server, which transcribes
(faster-whisper), classifies, and (on "done") turns dictation into a structured
prompt. Entry points:

- `~/tools/promptsmith/live.py` — the streaming HTTP server (`promptsmith --live`,
  `--https`). Contains `_transcribe_bytes`, `_jev_classify`, the `Handler`, the HTML
  `PAGE`, and `serve()`.
- `~/tools/promptsmith/promptsmith.py` — record/transcribe/structure/refine/dispatch,
  plus the `--live` entry that imports `live.serve`.
- `~/bin/promptsmith` — bash wrapper running `promptsmith.py` under the Hermes venv.

The server currently works but has two defects the owner wants fixed, plus a
redesigned classification/refinement workflow. Implement ALL of this in one pass.

---

## 1. BUG FIX — mic "stops picking up voice" after a while

Root cause (confirmed by reading `live.py`):
1. `_transcribe_bytes` instantiates a **new `WhisperModel` on every chunk** — a
   seconds-long CPU load each time.
2. The server re-transcribes the **entire accumulated buffer** on every chunk
   (`data = b"".join(_state["chunks"])`), so work grows with total speech.
3. `serve()` uses the single-threaded `HTTPServer`, so one slow transcribe blocks
   the next chunk's POST; requests queue and the UI transcript freezes.

Required fixes:
- **Cache the WhisperModel** once (module-level lazy singleton) instead of reloading
  per chunk.
- **Thread the server**: import and use `ThreadingHTTPServer` in place of
  `HTTPServer` (still bound `0.0.0.0:<port>`, same TLS wrap).
- **Stop unbounded re-transcription**: transcribe only the *new* audio since the
  last successful transcription (track a byte offset / keep the already-transcribed
  prefix and append the new tail), OR transcribe a bounded rolling window (~last
  45s). Do NOT re-transcribe the full accumulated history every chunk.
- Keep the chunk cap and `_state` shape working; the transcript must keep updating
  during a >60s session (this is the acceptance test).

## 2. REDESIGN — Omni-grounded classification (replace `_jev_classify`)

The current classification asks 5 hardcoded coverage questions and is confusing.
Replace it with a classification grounded in the **Omni ecosystem** so the UI can
say *where the prompt lies in Omni terms*.

Omni = six realms. Embed this as a data constant (e.g. in a new
`~/tools/promptsmith/omni.py`):

```
OMNI_REALMS = {
  "edoras": {"meaning": "financial (Tolkien)",   "repo": "~/edoras",
             "agents": ["argus","paisa"],
             "docs": ["~/edoras/AGENTS.md","~/edoras/docs","~/edoras/README.md",
                      "~/.hermes/profiles/argus/SOUL.md","~/.hermes/profiles/paisa/SOUL.md",
                      "~/edoras-operations-journal/AGENTS.md"]},
  "atma":   {"meaning": "the self / growth (आत्मन्)", "repo": "~/atma",
             "agents": ["gyani"],
             "docs": ["~/atma","~/.hermes/profiles/gyani/SOUL.md","~/gyani/SOUL.md"]},
  "soma":   {"meaning": "the body (σῶμα)",       "repo": "~/whoop-sync",
             "agents": [],
             "docs": ["~/whoop-sync","~/whoop-sync/README.md"]},
  "raga":   {"meaning": "sound × emotion × body (राग)", "repo": "~/whoop-sync",
             "agents": [],
             "docs": ["~/whoop-sync/spotify_etl.py","~/whoop-sync/README.md"]},
  "smriti": {"meaning": "memory (स्मृति)",        "repo": "~/veltiosi",
             "agents": ["veltiosi"],
             "docs": ["~/veltiosi","~/.hermes/profiles/veltiosi/SOUL.md","~/Obsidian"]},
  "omni":   {"meaning": "the glue (ὅμος)",       "repo": "~/omni",
             "agents": ["satya"],
             "docs": ["~/omni","~/omni/net/agent_services.json"]},
}
```

Classification call (keep the existing TypeSafe `https://api.typesafe.ai/v1/systemone`
POST shape from `_jev_classify` — `{"state": {...}, "model": "jev-latest", "questions": {...}}`,
`Authorization: Bearer <TYPESAFE_API_KEY>`; `_typesafe_key()` already loads the key):
- `state` = `{"transcript": <text>, "task": "the user is dictating a prompt out loud",
  "omni_realms": <serialized OMNI_REALMS names + meanings + bound agents>}`.
- `questions`:
  - One per realm, keyed `realm_<name>`, `{"type":"noul","instructions":"Does this
    prompt concern the <name> realm (<meaning>)? Consider the bound agents (<agents>)."}`
  - Keep coverage: `goal_stated`, `has_requirements`, `has_context` (noul) and
    `completeness` (score) exactly as the current code does.
- Return `{"realm_scores": {<realm>: 0..1 ...} sorted desc, "coverage": {...}}`.

The UI (`PAGE`) must render the top realm(s) (e.g. "This prompt is about: Edoras 0.9,
Omni 0.3") in place of the confusing pills, plus a compact coverage line
(goal/requirements/context/completeness). Keep it dark-themed, match existing CSS
tokens.

## 3. REDESIGN — doc-driven refinement ("modify the prompt using relevant docs")

New step, run on `/done` (and on `--text`/`--audio` in `promptsmith.py`), between
classification and the existing structure/refine:

`ground_prompt(transcript, realm_scores, model)`:
1. Pick the top realm(s): score ≥ 0.35, or at minimum the top-1 realm.
2. `fetch_docs(realm_name)`: read the realm's `docs` paths — `AGENTS.md`, `SOUL.md`,
   `README.md`, and `docs/*.md` (recursively, one level). For `~/Obsidian` cap to
   key subdirs only (e.g. `README`/MOCs) — do NOT crawl the whole vault. Cap total
   text to ~12,000 chars per realm. If a path is missing, skip it.
3. Build a system prompt: "You ground a user's spoken prompt in the Omni ecosystem.
   The prompt concerns the `<realm>` realm (<meaning>), whose bound agents are
   `<agents>`. Use the documentation below to refine the prompt so it uses correct
   domain terminology, names the real agents/entities/services from the docs, and is
   actionable by the right agent. Preserve the user's intent and wording. Do not
   invent requirements. Append an '## Omni context' section naming the realm, the
   responsible agent, and which doc(s) informed the refinement." Feed [transcript +
   docs] to the LLM.
4. Return the grounded prompt text.

Model for this pass: a FAST model, configurable via env `PROMPTSMITH_GROUND_MODEL`,
default `deepseek/deepseek-v4-flash-0731`. Use the existing `_llm` helper (from
`promptsmith`, OpenRouter) or an equivalent urllib call.

Final `/done` pipeline: transcribe → omni-classify → ground (docs) → structure+refine
(`build_prompt`) → save to `out/prompt-*.md`. Return `{"final": ..., "classification":
..., "grounded": ..., "saved": ...}` so the UI can show the classification summary and
the final prompt.

## 4. Config / env

- `PROMPTSMITH_GROUND_MODEL` (default `deepseek/deepseek-chat` — must be NON-reasoning;
  v4-* reasoning models return empty content on the 12KB docs payload) — the
  doc-grounded pass.
- `PROMPTSMITH_MODEL` (existing) — structure/refine.
- `TYPESAFE_API_KEY` (existing) — classification.
- `OPENROUTER_API_KEY` (existing) — LLM.

## 5. Files to create / modify

- CREATE `~/tools/promptsmith/omni.py` — `OMNI_REALMS`, `omni_classify()`,
  `fetch_docs()`, `ground_prompt()`.
- MODIFY `~/tools/promptsmith/live.py` — cache model, ThreadingHTTPServer, bounded
  transcription, call `omni_classify` + `ground_prompt`, update `PAGE`.
- MODIFY `~/tools/promptsmith/promptsmith.py` — wire `omni_classify` + `ground_prompt`
  into the `--text`/`--audio`/`--live` paths; print the classification summary and
  grounded prompt.
- CREATE `~/tools/promptsmith/tests/test_omni.py` — see below.

## 6. Testing (required)

Ship a test that runs WITHOUT a mic and WITHOUT network where possible:
- Mock the TypeSafe and OpenRouter calls (monkeypatch `urllib.request.urlopen` or the
  module functions) to return canned `{"answers": ...}` and canned LLM text.
- Test `omni_classify("rebalance the Radagast portfolio")` → `realm_scores["edoras"]`
  is the top score.
- Test `fetch_docs("edoras")` returns non-empty text containing content from
  `~/edoras/AGENTS.md` (or gracefully skips if the file is absent on CI).
- Test `ground_prompt(...)` embeds the top realm and produces output with the
  `## Omni context` section.
- A smoke test: `promptsmith --text "rebalance the portfolio" --no-improve` runs
  end-to-end (may be skipped if no API keys; guard with `@pytest.mark.skipif` on
  missing env vars).
- Run `python -m pytest tests/test_omni.py -q` under the Hermes venv
  (`~/.hermes/hermes-agent/venv/bin/python`) and report pass/fail.

## 7. Acceptance criteria

1. `promptsmith --live --https` still starts, serves the page, and the transcript
   keeps updating during a >60s continuous session (no more freeze/stall).
2. Live classification shows per-realm scores ("Edoras 0.9") + coverage, not the old
   five pills.
3. `/done` (and `--text`) produce a prompt with an `## Omni context` section naming
   the realm + responsible agent + informing doc(s).
4. `pytest tests/test_omni.py` passes.
5. HTTPS/cert behavior from the existing `--https` flag is preserved (do not touch
   `_ensure_cert`, `_default_cert_dir`, or the cert dir).

## 8. Off-limits

- Do NOT modify `~/bin/promptsmith`, `~/.hermes/plugins/promptsmith/__init__.py`, or
  anything under `~/tools/promptsmith/certs/`.
- Do NOT change the TypeSafe/OpenRouter endpoint URLs or the existing env-var names
  used elsewhere.
- Do NOT crawl the entire `~/Obsidian` vault (cap as specified).
- Do NOT make promptsmith auto-dispatch to any agent — it remains craft-only.

Report: files changed, `git diff --stat`, and `pytest` results.
