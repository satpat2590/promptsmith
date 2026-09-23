#!/usr/bin/env python3
"""promptsmith — dictate a structured prompt by voice, refine it, dispatch it.

Pipeline:
    mic -> faster-whisper (STT) -> omni-classify (Jev) -> doc-ground (LLM)
        -> structure (LLM) -> improve (LLM) -> stdout + file
    [ -> dispatch to Hermes / Claude Code / Cortex / any command ]

Dependencies (all already in ~/.hermes/hermes-agent/venv):
    faster-whisper, sounddevice, numpy. LLM via OpenRouter (OPENROUTER_API_KEY).

Usage:
    promptsmith                           # record (Enter to start/stop) -> full pipeline
    promptsmith --seconds 90              # record for a fixed duration (auto-stop)
    promptsmith --text "..."              # skip the mic; feed raw text (paste/testing)
    promptsmith --no-improve              # stop after the structure pass
    promptsmith --out foo.md              # also write the final prompt to a file
    promptsmith --dispatch hermes         # after building, dispatch to a target
    promptsmith --dispatch "claude -p"    # ...or any command (prompt passed via a temp file)

Env:
    OPENROUTER_API_KEY      (loads from ~/.hermes/.env if not exported)
    TYPESAFE_API_KEY        (omni classification; loads from ~/.hermes/.env too)
    PROMPTSMITH_MODEL       model for structure+improve (default deepseek/deepseek-v4-pro-0813)
    PROMPTSMITH_GROUND_MODEL model for the doc-grounded pass (default deepseek/deepseek-v4-flash-0731)
    PROMPTSMITH_WHISPER     whisper size (default base)
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

DEFAULT_MODEL = "deepseek/deepseek-v4-pro-0813"
SAMPLE_RATE = 44100

# ── dispatch targets ──────────────────────────────────────────────────────────
# Each target is a shell command template; `{file}` is the saved prompt path.
# `hermes`/`claude`/`opencode` read the prompt as an argument (through $(cat)).
DISPATCH_TARGETS = {
    "hermes": 'hermes -z "$(cat {file})"',
    "claude": 'claude -p "$(cat {file})"',
    "opencode": 'opencode run "$(cat {file})"',
    "cortex": 'cortex "$(cat {file})"',
}


def _load_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        return os.environ["OPENROUTER_API_KEY"].strip()
    for path in (os.path.expanduser("~/.hermes/.env"),):
        try:
            for line in open(path, encoding="utf-8"):
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            continue
    return ""


# ── 1. record ────────────────────────────────────────────────────────────────
def record(out_path: str, seconds: float = 0) -> None:
    """Record from the default mic. `seconds`>0 = fixed duration; else toggle on Enter."""
    import numpy as np
    import sounddevice as sd

    chunks = []
    if seconds > 0:
        print(f"🎙️ Recording for {seconds:.0f}s — speak now…", flush=True)
        audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        chunks = [audio]
    else:
        print("🎙️ Press Enter to START recording, then Enter again to STOP.", flush=True)
        input()
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")

        def cb(indata, frames, t, status):
            chunks.append(indata.copy())

        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=cb)
        stream.start()
        print("● recording… press Enter to STOP", flush=True)
        input()
        stream.stop()
        stream.close()

    if not chunks:
        raise RuntimeError("no audio captured")
    audio = np.concatenate(chunks).flatten()
    import wave

    w = wave.open(out_path, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SAMPLE_RATE)
    w.writeframes((audio * 32767).astype(np.int16).tobytes())
    w.close()
    print(f"   captured {len(audio)/SAMPLE_RATE:.1f}s -> {out_path}")


# ── 2. transcribe ─────────────────────────────────────────────────────────────
def transcribe(wav_path: str, model_size: str) -> str:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(wav_path)
    text = " ".join(s.text.strip() for s in segments).strip()
    if not text:
        raise RuntimeError("no speech detected in the audio")
    return text


# ── 3/4. LLM passes ───────────────────────────────────────────────────────────
def _llm(system: str, user: str, model: str) -> str:
    import json
    import urllib.request

    key = _load_key()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set (export it or put it in ~/.hermes/.env)")
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.3,
        "max_tokens": 4000,
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


STRUCTURE_SYSTEM = (
    "You convert raw, rambling speech dictation into a complete, well-structured prompt for an "
    "AI agent. Organize it faithfully — preserve the user's actual intent, details, and wording. "
    "Do NOT invent requirements they didn't state. Use these sections:\n"
    "# Goal\n# Context / Background\n# Constraints\n# What 'done' looks like\n# Open questions\n"
    "Put anything missing or ambiguous under 'Open questions' rather than guessing. "
    "Write in the user's voice (first person where they used it)."
)

IMPROVE_SYSTEM = (
    "You are a senior prompt engineer. Improve this prompt draft: tighten wording, resolve "
    "ambiguity, add concrete acceptance criteria where the intent implies them, remove redundancy, "
    "and surface hidden assumptions. Append an '## Assumptions to verify' section flagging anything "
    "the agent must confirm before starting. Keep all real constraints and specifics. Output the "
    "improved prompt only."
)


def build_prompt(text: str, model: str, improve: bool) -> str:
    draft = _llm(STRUCTURE_SYSTEM, text, model)
    if not improve:
        return draft
    return _llm(IMPROVE_SYSTEM, draft, model)


# ── 5. dispatch ───────────────────────────────────────────────────────────────
def dispatch(prompt: str, target: str) -> None:
    cmd = DISPATCH_TARGETS.get(target, target)  # named target or raw command
    if "{file}" not in cmd:
        # no file placeholder -> assume the caller manages prompt delivery
        cmd = f'{cmd} "$(cat {{file}})"'
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(prompt)
        path = f.name
    full = cmd.format(file=path)
    print(f"\n🚀 dispatching to [{target}]:\n  $ {full}\n")
    subprocess.run(full, shell=True, check=False)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Dictate a structured prompt by voice, refine it, dispatch it.")
    ap.add_argument("--text", help="skip the mic; feed raw text (paste or test)")
    ap.add_argument("--audio", help="transcribe an existing audio file (recorded on any machine) instead of the mic")
    ap.add_argument("--seconds", type=float, default=0, help="fixed recording duration (0 = toggle on Enter)")
    ap.add_argument("--no-improve", action="store_true", help="skip the refine pass")
    ap.add_argument("--out", help="write the final prompt to this file")
    ap.add_argument("--dispatch", help="target: hermes|claude|opencode|cortex|any shell command")
    ap.add_argument("--model", default=os.environ.get("PROMPTSMITH_MODEL", DEFAULT_MODEL))
    ap.add_argument("--whisper", default=os.environ.get("PROMPTSMITH_WHISPER", "base"))
    ap.add_argument("--live", action="store_true", help="run the ephemeral streaming server (browser mic -> laptop)")
    ap.add_argument("--host", default="0.0.0.0", help="bind host for --live")
    ap.add_argument("--port", type=int, default=8721, help="port for --live")
    ap.add_argument("--https", action="store_true", help="serve --live over HTTPS (self-signed) so the browser mic works cross-device")
    ap.add_argument("--cert", help="TLS cert path for --https (auto-generated if missing)")
    ap.add_argument("--key", help="TLS key path for --https")
    args = ap.parse_args()

    if args.live:
        from live import serve

        serve(args.host, args.port, args.model, args.whisper, not args.no_improve,
              args.https, args.cert, args.key)
        return 0

    if args.text:
        raw = args.text
    elif args.audio:
        raw = transcribe(args.audio, args.whisper)
        print(f"\n📝 heard:\n{raw}\n")
    else:
        wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        record(wav, args.seconds)
        raw = transcribe(wav, args.whisper)
        print(f"\n📝 heard:\n{raw}\n")

    print("⏳ classifying (omni realms)…")
    from omni import ground_prompt, merge_omni_context, omni_classify

    try:
        classification = omni_classify(raw)
    except Exception as e:
        classification = {}
        print(f"   (classification skipped: {e})")
    scores = classification.get("realm_scores", {})
    if scores:
        top = ", ".join(f"{k.capitalize()} {v:.2f}" for k, v in list(scores.items())[:3])
        print(f"🧭 this prompt is about: {top}")
        cov = classification.get("coverage", {})
        if cov:
            print(f"   coverage: goal {cov.get('goal_stated', '—')} · "
                  f"requirements {cov.get('has_requirements', '—')} · "
                  f"context {cov.get('has_context', '—')} · "
                  f"completeness {cov.get('completeness', '—')}/2")

    print("⏳ grounding in realm docs…")
    try:
        grounded = ground_prompt(raw, scores)
    except Exception as e:
        grounded = raw
        print(f"   (grounding skipped: {e})")
    if grounded != raw:
        print(f"\n📚 grounded prompt:\n{grounded}\n")

    print("⏳ structuring + refining…\n")
    final = build_prompt(grounded, args.model, not args.no_improve)
    final = merge_omni_context(final, grounded)

    out = args.out or os.path.join(
        os.path.expanduser("~/tools/promptsmith/out"),
        f"prompt-{time.strftime('%Y%m%d-%H%M%S')}.md",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(final + "\n")

    print("═" * 60)
    print(final)
    print("═" * 60)
    print(f"\n💾 saved: {out}")

    if args.dispatch:
        dispatch(final, args.dispatch)
    else:
        print("   (re-run with --dispatch <target> to send it: hermes | claude | opencode | cortex | any $cmd)")
    return 0


if __name__ == "__main__":
    sys.exit(main())