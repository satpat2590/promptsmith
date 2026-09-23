"""config — paths, env/secret loading and tunables shared by every promptsmith module.

Secrets and settings resolve in this order: the process environment, then
`<repo>/.env`, then `~/.hermes/.env` (so the Hermes gateway's keys just work).
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
_ENV_FILES = (os.path.join(ROOT, ".env"), os.path.expanduser("~/.hermes/.env"))


def env(name: str, default: str = "") -> str:
    """Read a setting from the environment or the first .env file that defines it."""
    val = os.environ.get(name, "").strip()
    if val:
        return val
    for path in _ENV_FILES:
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("export "):
                        line = line[len("export "):]
                    if line.startswith(name + "="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
        except OSError:
            continue
    return default


# ── paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = env("PROMPTSMITH_OUT", os.path.join(ROOT, "out"))
CERT_DIR = os.path.join(ROOT, "certs")
WEB_DIR = os.path.join(ROOT, "web")
REALMS_FILE = env("PROMPTSMITH_REALMS", os.path.join(ROOT, "realms.json"))

# ── speech-to-text ────────────────────────────────────────────────────────────
# The live model only drives the preview while you talk; the final model re-transcribes
# the whole recording once you hit Finish, so accuracy comes from the final one.
WHISPER_LIVE = env("PROMPTSMITH_WHISPER", "base.en")
WHISPER_FINAL = env("PROMPTSMITH_WHISPER_FINAL", "small.en")
WHISPER_DEVICE = env("PROMPTSMITH_WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = env("PROMPTSMITH_WHISPER_COMPUTE", "int8")
LANGUAGE = env("PROMPTSMITH_LANGUAGE", "en")  # "auto" = let whisper detect
EXTRA_VOCAB = env("PROMPTSMITH_VOCAB", "")  # comma-separated names whisper should spell right

# ── LLM / agent ───────────────────────────────────────────────────────────────
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
# Non-reasoning by default: reasoning models (deepseek v4-*) can spend the whole token
# budget on hidden reasoning and return empty content on a docs-sized payload.
REVIEW_MODEL = env("PROMPTSMITH_REVIEW_MODEL", env("PROMPTSMITH_GROUND_MODEL", "deepseek/deepseek-chat"))
# "llm" = OpenRouter chat call; "hermes" = one-shot Hermes agent run (it can use its own
# tools to inspect the system before answering).
REVIEWER = env("PROMPTSMITH_REVIEWER", "llm")
HERMES_CMD = env("PROMPTSMITH_HERMES_CMD", "hermes -z")
HERMES_TIMEOUT = int(env("PROMPTSMITH_HERMES_TIMEOUT", "600"))
# Optional shell command whose output is added to the system-state snapshot
# (e.g. "hermes status"). Runs on the laptop with a short timeout.
STATE_CMD = env("PROMPTSMITH_STATE_CMD", "")

DOC_CHAR_CAP = 12000  # per realm
REALM_THRESHOLD = 0.35
