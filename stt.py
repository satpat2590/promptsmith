"""stt — speech-to-text with faster-whisper.

Audio is handled as 16 kHz mono float32 numpy arrays end to end (the browser sends
raw PCM), so there is no container decoding and no webm-header juggling.

`LiveTranscript` keeps the live preview cheap: it only re-transcribes the audio
since the last *committed* segment, and commits finished segments once the
pending window grows past COMMIT_AFTER seconds, so each pass is bounded no matter
how long you talk. Accuracy comes from `transcribe()` over the whole recording
with the bigger final model once you finish.
"""

import threading
from dataclasses import dataclass

import config

SAMPLE_RATE = 16000

_models = {}
_models_lock = threading.Lock()


def _model(size: str):
    """Load each whisper size once; returns (model, lock) — a model runs one job at a time."""
    with _models_lock:
        if size not in _models:
            from faster_whisper import WhisperModel

            _models[size] = (WhisperModel(size, device=config.WHISPER_DEVICE,
                                          compute_type=config.WHISPER_COMPUTE), threading.Lock())
        return _models[size]


def preload(*sizes: str) -> None:
    for size in sizes:
        _model(size)


@dataclass
class Segment:
    start: float
    end: float
    text: str


def _initial_prompt() -> str:
    from omni import whisper_vocabulary

    return "Glossary: " + ", ".join(whisper_vocabulary()) + "."


def transcribe_segments(audio, size: str, fast: bool = False) -> list:
    """Transcribe a float32 16 kHz array (or an audio file path) into segments.

    fast=True is for the live preview (greedy decode); otherwise beam search.
    """
    model, lock = _model(size)
    kwargs = dict(
        beam_size=1 if fast else 5,
        vad_filter=True,  # drop silence -> no "Thank you." hallucinations
        condition_on_previous_text=False,  # avoids repetition loops on long dictation
        initial_prompt=_initial_prompt(),
    )
    if config.LANGUAGE and config.LANGUAGE != "auto":
        kwargs["language"] = config.LANGUAGE
    with lock:
        segments, _ = model.transcribe(audio, **kwargs)
        return [Segment(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]


def transcribe(audio, size: str = config.WHISPER_FINAL) -> str:
    return " ".join(s.text for s in transcribe_segments(audio, size)).strip()


def pcm16_to_float(pcm: bytes):
    import numpy as np

    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


class LiveTranscript:
    """Incrementally transcribed audio buffer (raw 16 kHz int16 PCM)."""

    COMMIT_AFTER = 12.0  # s of pending audio before finished segments are committed
    MAX_PENDING = 30.0  # s; past this, commit everything even mid-segment
    MAX_AUDIO = 20 * 60  # s; hard cap on one recording

    def __init__(self, size: str = config.WHISPER_LIVE):
        self.size = size
        self.pcm = bytearray()
        self.committed = 0  # samples already folded into committed_text
        self.seen = 0  # samples covered by the last pass
        self.committed_text = ""
        self.pending_text = ""
        self._lock = threading.Lock()

    @property
    def seconds(self) -> float:
        return len(self.pcm) / 2 / SAMPLE_RATE

    @property
    def text(self) -> str:
        return " ".join(t for t in (self.committed_text, self.pending_text) if t).strip()

    def append(self, pcm: bytes) -> bool:
        """Add PCM; returns False once the recording hits MAX_AUDIO."""
        with self._lock:
            room = self.MAX_AUDIO * SAMPLE_RATE * 2 - len(self.pcm)
            if room <= 0:
                return False
            self.pcm += pcm[: room - room % 2]
            return True

    def audio(self):
        with self._lock:
            return pcm16_to_float(bytes(self.pcm))

    def step(self) -> bool:
        """Transcribe any new audio. Returns True if the text may have changed."""
        with self._lock:
            total = len(self.pcm) // 2
            if total - self.seen < SAMPLE_RATE // 2:  # wait for >= 0.5s of new audio
                return False
            pending = pcm16_to_float(bytes(self.pcm[self.committed * 2: total * 2]))
            self.seen = total
        segs = transcribe_segments(pending, self.size, fast=True)
        dur = len(pending) / SAMPLE_RATE
        if dur > self.MAX_PENDING:
            self._commit(segs, total - self.committed)
        elif dur > self.COMMIT_AFTER and len(segs) > 1:
            self._commit(segs[:-1], int(segs[-2].end * SAMPLE_RATE))
            self.pending_text = segs[-1].text
        elif dur > self.COMMIT_AFTER and not segs:
            self.committed = total  # nothing but silence; skip it next time
            self.pending_text = ""
        else:
            self.pending_text = " ".join(s.text for s in segs)
        return True

    def _commit(self, segs, samples: int) -> None:
        words = " ".join(s.text for s in segs)
        self.committed_text = " ".join(t for t in (self.committed_text, words) if t)
        self.committed += samples
        self.pending_text = ""
