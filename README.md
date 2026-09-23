# promptsmith

Speak a prompt on your desktop; the laptop transcribes it, Jev classifies it, and a
reviewer agent checks it against the current state of your system and pads it out
into a prompt your fleet can act on.

```
desktop browser (mic)                               laptop (promptsmith + Hermes gateway)
  16 kHz PCM every 1.5s  ───────────────────────►  live transcript (faster-whisper, fast model)  
                                                   Jev classification every ~6s (Omni realms + coverage)
  [Finish] ─────────────────────────────────────►  accurate re-transcription of the full recording
  (fix anything misheard in the textbox)
  [Review] ─────────────────────────────────────►  reviewer agent: transcript + Jev + realm docs
                                                     + system-state snapshot (git state, agents)
  ◄───────────────────────────────────────────────  verdict · issues · reductions · improvements ·
                                                     questions · improved prompt (saved to out/)
```

promptsmith only **crafts** prompts. It never runs them. Copy the result into the
Hermes desktop app (or use `--dispatch` from the CLI).

## Quick start

On the laptop:

```bash
pip install -r requirements.txt         # or use the Hermes venv, which already has them
promptsmith --live                      # HTTPS by default; Ctrl-C stops it
```

It prints something like `https://192.168.1.20:8721/`. Open that on the desktop, accept
the self-signed certificate once (browsers only allow the mic over HTTPS), then:

1. **Record**: speak. The live transcript is a draft, and Jev scores update as you go.
2. **Finish**: the whole recording is re-transcribed with the more accurate model, and
   the review runs automatically.
3. Fix any misheard words in the transcript box and hit **Review with agent** to re-run.
4. **Copy** the improved prompt into Hermes.

You can also just type or paste into the transcript box and hit Review.

To stop other machines on the LAN from using it, set a token:
`promptsmith --live --token s3cret` (or `PROMPTSMITH_TOKEN`). Open the printed
`?token=` URL once and the browser remembers the token.

## CLI

```bash
promptsmith --text "rebalance the portfolio"   # skip speech
promptsmith --audio clip.wav                   # transcribe a file
promptsmith                                    # record on this machine (Enter to start/stop)
promptsmith --reviewer hermes                  # review with a Hermes agent run
promptsmith --no-review                        # transcribe + classify only
promptsmith --dispatch hermes                  # hand the final prompt to a target afterwards
```

## The reviewer agent

The reviewer gets the transcript, the Jev classification, the docs of the top realm(s)
(see `realms.json`) and a snapshot of current system state: branch, uncommitted changes
and recent commits of each realm's repo, the Hermes profiles on disk, and the output of
`PROMPTSMITH_STATE_CMD` if you set one. It returns:

- **verdict**: `ready` / `needs_clarification` / `invalid`
- **issues**: things that don't match the docs or system (missing repo, work already done, wrong agent)
- **reductions** and **improvements**: what it cut and what it added, and why
- **questions**: what you must answer before an agent starts
- **prompt**: the improved prompt (Goal / Context / Constraints / Done when / Open questions / Omni context)

Backends (`PROMPTSMITH_REVIEWER` or `--reviewer`):

| backend  | what runs | notes |
|----------|-----------|-------|
| `llm` (default) | one OpenRouter chat call | fast; only sees the snapshot promptsmith collects |
| `hermes` | `PROMPTSMITH_HERMES_CMD <prompt>` (default `hermes -z`) | a real agent run that can inspect the system with its own tools; told to stay read-only and not execute the task |

## Config

Settings come from the environment, `<repo>/.env` or `~/.hermes/.env`, in that order.

| Env | Default | Notes |
|-----|---------|-------|
| `OPENROUTER_API_KEY` | | `llm` reviewer |
| `TYPESAFE_API_KEY` | | Jev classification |
| `PROMPTSMITH_REVIEWER` | `llm` | `llm` or `hermes` |
| `PROMPTSMITH_REVIEW_MODEL` | `deepseek/deepseek-chat` | use a non-reasoning model; reasoning models can return empty content on docs-sized inputs. Falls back to `PROMPTSMITH_GROUND_MODEL` |
| `PROMPTSMITH_HERMES_CMD` | `hermes -z` | any command that takes the prompt as its last argument |
| `PROMPTSMITH_HERMES_TIMEOUT` | `600` | seconds |
| `PROMPTSMITH_STATE_CMD` | | extra shell command added to the system-state snapshot |
| `PROMPTSMITH_WHISPER` | `base.en` | live-preview model (speed matters) |
| `PROMPTSMITH_WHISPER_FINAL` | `small.en` | Finish-pass model (accuracy matters); try `medium.en` or `distil-large-v3` if the laptop can take it |
| `PROMPTSMITH_WHISPER_DEVICE` / `_COMPUTE` | `cpu` / `int8` | `cuda` / `float16` on a GPU |
| `PROMPTSMITH_LANGUAGE` | `en` | `auto` to detect |
| `PROMPTSMITH_VOCAB` | | extra comma-separated names whisper should spell right (realm and agent names are already included) |
| `PROMPTSMITH_REALMS` | `realms.json` | realm registry |
| `PROMPTSMITH_OUT` | `out/` | where prompts are saved |
| `PROMPTSMITH_TOKEN` | | require a token from browsers |

## Layout

| file | role |
|------|------|
| `promptsmith.py` | CLI entry point |
| `server.py` | web app: sessions, live worker, HTTP API, TLS |
| `web/index.html` | browser client (mic capture, downsampling, UI) |
| `stt.py` | faster-whisper wrapper + bounded incremental `LiveTranscript` |
| `omni.py` | realm registry, Jev classification, docs, system-state snapshot |
| `review.py` | reviewer agent (llm / hermes backends) |
| `pipeline.py` | classify → review → save, shared by CLI and server |
| `realms.json` | the six Omni realms: meaning, repo, agents, docs |

## Tests

```bash
python -m pytest -q
```

Whisper, Jev, OpenRouter and Hermes are mocked. The CLI smoke test runs only when real
API keys are present.
