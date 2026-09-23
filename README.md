# promptsmith

Dictate a structured prompt by voice, have it refined, dispatch it to an agent.

```
mic -> faster-whisper -> structure (LLM) -> refine (LLM) -> prompt.md
                                                        -> dispatch (hermes / claude / cortex / ...)
```

## Usage
```bash
promptsmith                            # record: Enter to start, Enter to stop
promptsmith --seconds 90               # record a fixed 90s (auto-stop)
promptsmith --text "..."               # skip the mic — paste dictation or test
promptsmith --no-improve               # stop after the structure pass
promptsmith --out foo.md               # write to a specific file
promptsmith --dispatch hermes          # build, then send to a target
promptsmith --dispatch "claude -p"     # ...or any command (prompt passed via a temp file)
```

## Live mode (streaming — Level 2)

```bash
promptsmith --live            # starts an EPHEMERAL server; Ctrl-C stops it
promptsmith --live --port 9000
```

Prints a URL like `http://<laptop-ip>:8721/`. Open it on ANY machine on the LAN
(the Windows desktop, a phone) → the page captures the mic via the browser and
POSTs rolling 3s audio chunks back. Each chunk is transcribed (faster-whisper) and
**Jev-classified** (goal/requirements/context coverage, completeness, needs-clarification),
so the prompt is "living" — it updates as you speak. The **Done** button runs the
full structure+refine and shows the final prompt (also saved to `out/`).

The server exists only while `--live` runs — no daemon, no reboot, nothing left
listening after Ctrl-C.

## Pipeline
1. **record** — sounddevice captures the mic (needs PortAudio; see below).
2. **transcribe** — faster-whisper (`--whisper`, default `base`).
3. **structure** — an LLM turns rambling dictation into a prompt with
   `Goal / Context / Constraints / Done / Open questions`, faithful to intent.
4. **refine** — a second LLM pass tightens wording, adds acceptance criteria, surfaces
   hidden assumptions. Skip with `--no-improve`.
5. **dispatch** — sends the final prompt to Hermes, Claude Code, Cortex, OpenCode, or any
   shell command.

## Config
| Env | Default | Notes |
|-----|---------|-------|
| `OPENROUTER_API_KEY` | (from ~/.hermes/.env) | LLM auth |
| `PROMPTSMITH_MODEL` | `deepseek/deepseek-v4-pro-0813` | structure + refine model |
| `PROMPTSMITH_WHISPER` | `base` | whisper size (`base`/`small`/`medium`/`large-v3`) |

## Dependencies (already in `~/.hermes/hermes-agent/venv`)
- `faster-whisper`, `sounddevice`, `numpy`
- PortAudio C lib — see `multimodal-capabilities` skill (§ "Mic capture + wake word");
  no-sudo install via conda + `~/.local/lib` symlink + `LD_LIBRARY_PATH`.

The `~/bin/promptsmith` wrapper runs the tool with the Hermes venv (which has the deps).