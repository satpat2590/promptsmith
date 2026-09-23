"""stt.LiveTranscript — incremental passes stay bounded; segments get committed."""

import numpy as np

import stt
from stt import SAMPLE_RATE, Segment


def pcm(seconds: float) -> bytes:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype="<i2").tobytes()


class FakeWhisper:
    """Returns one segment per 5s of audio it is given; records the lengths it saw."""

    def __init__(self):
        self.lengths = []
        self.n = 0

    def __call__(self, audio, size, fast=False):
        dur = len(audio) / SAMPLE_RATE
        self.lengths.append(dur)
        segs = []
        t = 0.0
        while t + 5 <= dur + 1e-6:
            self.n += 1
            segs.append(Segment(t, t + 5, f"w{self.n}"))
            t += 5
        return segs


def test_live_transcript_bounded(monkeypatch):
    fake = FakeWhisper()
    monkeypatch.setattr(stt, "transcribe_segments", fake)
    live = stt.LiveTranscript("tiny")
    for _ in range(40):  # 120s of talking in 3s chunks
        live.append(pcm(3))
        assert live.step()
    assert max(fake.lengths) <= live.COMMIT_AFTER + 3 + 1e-6  # never re-transcribes history
    assert live.committed > 0 and live.committed_text
    assert live.seconds == 120
    assert live.text.startswith(live.committed_text)


def test_live_transcript_waits_for_new_audio(monkeypatch):
    monkeypatch.setattr(stt, "transcribe_segments", FakeWhisper())
    live = stt.LiveTranscript("tiny")
    assert not live.step()
    live.append(pcm(0.2))
    assert not live.step()  # < 0.5s new audio
    live.append(pcm(1))
    assert live.step()
    assert not live.step()


def test_live_transcript_skips_long_silence(monkeypatch):
    monkeypatch.setattr(stt, "transcribe_segments", lambda audio, size, fast=False: [])
    live = stt.LiveTranscript("tiny")
    live.append(pcm(13))
    live.step()
    assert live.committed == 13 * SAMPLE_RATE
    assert live.text == ""


def test_live_transcript_cap(monkeypatch):
    live = stt.LiveTranscript("tiny")
    monkeypatch.setattr(live, "MAX_AUDIO", 2)
    assert live.append(pcm(1.5))
    assert live.append(pcm(1.5))  # truncated to the cap
    assert live.seconds == 2
    assert not live.append(pcm(1))


def test_pcm16_to_float():
    out = stt.pcm16_to_float(np.array([0, 16384, -32768], dtype="<i2").tobytes())
    assert out.dtype == np.float32
    assert list(out) == [0.0, 0.5, -1.0]
