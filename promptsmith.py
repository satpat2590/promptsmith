#!/usr/bin/env python3
"""promptsmith — speak a prompt, have Jev classify it and an agent review and pad it out.

    voice -> faster-whisper -> Jev (Omni realms + coverage) -> reviewer agent -> prompt.md

Usage:
    promptsmith --live                    # web app for the desktop browser (HTTPS by default)
    promptsmith --live --http             # plain HTTP (mic then only works on localhost)
    promptsmith                           # record on this machine (Enter to start/stop)
    promptsmith --seconds 90              # record for a fixed duration
    promptsmith --audio clip.wav          # transcribe an existing recording
    promptsmith --text "..."              # skip speech entirely
    promptsmith --reviewer hermes         # review with a one-shot Hermes agent instead of an LLM call
    promptsmith --no-review               # stop after transcription + classification
    promptsmith --dispatch hermes         # after building, hand the prompt to a target

Settings live in the environment, <repo>/.env or ~/.hermes/.env — see README.md.
"""

import argparse
import subprocess
import sys
import tempfile

import config

# Each target is a shell command template; `{file}` is the saved prompt path.
DISPATCH_TARGETS = {
    "hermes": 'hermes -z "$(cat {file})"',
    "claude": 'claude -p "$(cat {file})"',
    "opencode": 'opencode run "$(cat {file})"',
    "cortex": 'cortex "$(cat {file})"',
}


def record(seconds: float = 0):
    """Record from this machine's default mic at 16 kHz. Returns a float32 array."""
    import numpy as np
    import sounddevice as sd

    from stt import SAMPLE_RATE

    if seconds > 0:
        print(f"🎙️ Recording for {seconds:.0f}s — speak now…", flush=True)
        audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        return audio.flatten()
    chunks = []
    print("🎙️ Press Enter to START recording, then Enter again to STOP.", flush=True)
    input()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        callback=lambda indata, *_: chunks.append(indata.copy())):
        print("● recording… press Enter to STOP", flush=True)
        input()
    if not chunks:
        raise RuntimeError("no audio captured")
    return np.concatenate(chunks).flatten()


def dispatch(prompt: str, target: str) -> None:
    cmd = DISPATCH_TARGETS.get(target, target)  # named target or raw command
    if "{file}" not in cmd:
        cmd = f'{cmd} "$(cat {{file}})"'
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(prompt)
    full = cmd.format(file=f.name)
    print(f"\n🚀 dispatching to [{target}]:\n  $ {full}\n")
    subprocess.run(full, shell=True, check=False)


def _print_classification(c: dict) -> None:
    if c.get("error"):
        print(f"   ({c['error']})")
        return
    scores = c.get("realm_scores") or {}
    if scores:
        print("🧭 realms: " + ", ".join(f"{k.capitalize()} {v:.2f}" for k, v in list(scores.items())[:3]))
    cov = c.get("coverage") or {}
    if cov:
        print("   coverage: " + " · ".join(f"{k} {v}" for k, v in cov.items()))


def _print_review(r: dict) -> None:
    print(f"\n🔎 verdict: {r.get('verdict', 'unknown').upper()}"
          + (f"  →  agent: {r['agent']}" if r.get("agent") else ""))
    if r.get("summary"):
        print(f"   {r['summary']}")
    for key, title in (("issues", "issues"), ("reductions", "reduced"),
                       ("improvements", "improved"), ("questions", "questions for you")):
        if r.get(key):
            print(f"   {title}:")
            for item in r[key]:
                print(f"     - {item}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Speak a prompt; Jev classifies it; an agent reviews and improves it.")
    src = ap.add_argument_group("input (default: record from this machine's mic)")
    src.add_argument("--text", help="skip speech; use this text as the transcript")
    src.add_argument("--audio", help="transcribe an existing audio file")
    src.add_argument("--seconds", type=float, default=0, help="fixed recording duration (0 = toggle on Enter)")
    ap.add_argument("--whisper", help=f"whisper model size (default {config.WHISPER_FINAL})")
    ap.add_argument("--reviewer", choices=("llm", "hermes"), help=f"review backend (default {config.REVIEWER})")
    ap.add_argument("--model", help=f"LLM for the llm reviewer (default {config.REVIEW_MODEL})")
    ap.add_argument("--no-review", "--no-improve", dest="no_review", action="store_true",
                    help="stop after transcription + classification")
    ap.add_argument("--out", help="write the final prompt to this file (default out/prompt-<time>.md)")
    ap.add_argument("--dispatch", help="after building, send to: hermes|claude|opencode|cortex|any shell command")
    web = ap.add_argument_group("web app")
    web.add_argument("--live", "--serve", dest="live", action="store_true", help="run the web app for the desktop browser")
    web.add_argument("--host", default="0.0.0.0")
    web.add_argument("--port", type=int, default=8721)
    web.add_argument("--http", action="store_true", help="serve plain HTTP instead of self-signed HTTPS")
    web.add_argument("--https", action="store_true", help=argparse.SUPPRESS)  # default now; kept for old scripts
    web.add_argument("--cert", default="", help="TLS cert path (auto-generated if missing)")
    web.add_argument("--key", default="", help="TLS key path")
    web.add_argument("--token", default="", help="require this token from browsers (or set PROMPTSMITH_TOKEN)")
    args = ap.parse_args(argv)

    if args.whisper:
        config.WHISPER_FINAL = args.whisper
    if args.reviewer:
        config.REVIEWER = args.reviewer

    if args.live:
        from server import serve

        serve(args.host, args.port, not args.http, args.cert, args.key, args.token)
        return 0

    import pipeline

    if args.text:
        raw = args.text.strip()
    else:
        import stt

        audio = args.audio or record(args.seconds)
        print("⏳ transcribing…", flush=True)
        raw = stt.transcribe(audio, config.WHISPER_FINAL)
        if not raw:
            print("no speech detected", file=sys.stderr)
            return 1
        print(f"\n📝 heard:\n{raw}\n")

    print("⏳ classifying with Jev…", flush=True)
    classification = pipeline.classify(raw)
    _print_classification(classification)

    if args.no_review:
        final = raw
        saved = pipeline.save(final, args.out)
    else:
        print("⏳ reviewing against current system state…", flush=True)
        result = pipeline.run(raw, classification, model=args.model or "", out=args.out or "")
        _print_review(result["review"])
        final, saved = result["prompt"], result["saved"]

    print("\n" + "═" * 60)
    print(final)
    print("═" * 60)
    print(f"\n💾 saved: {saved}")

    if args.dispatch:
        dispatch(final, args.dispatch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
