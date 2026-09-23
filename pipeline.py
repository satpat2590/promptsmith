"""pipeline — transcript -> Jev classification -> agent review -> saved prompt.

Shared by the CLI and the web server so both run exactly the same steps.
"""

import os
import time

import config
import omni
import review as reviewer


def classify(transcript: str) -> dict:
    """Jev classification; never raises (errors come back under "error")."""
    if not transcript.strip():
        return {}
    try:
        return omni.omni_classify(transcript)
    except Exception as e:  # network / key problems shouldn't kill the pipeline
        return {"error": f"classification unavailable: {e}"}


def save(prompt: str, path: str = "") -> str:
    path = path or os.path.join(config.OUT_DIR, f"prompt-{time.strftime('%Y%m%d-%H%M%S')}.md")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(prompt.rstrip() + "\n")
    return path


def run(transcript: str, classification: dict = None, backend: str = "", model: str = "",
        out: str = "") -> dict:
    """Classify (unless given), review and save. Returns {classification, review, prompt, saved}."""
    if classification is None:
        classification = classify(transcript)
    result = reviewer.review(transcript, classification, backend=backend, model=model)
    saved = save(result["prompt"], out)
    return {"classification": classification, "review": result, "prompt": result["prompt"], "saved": saved}
