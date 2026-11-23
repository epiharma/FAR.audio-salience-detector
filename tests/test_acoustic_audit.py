"""Independent regressions for confirmed DESIGN-003 audit defects."""

import importlib.util
from pathlib import Path

import pytest

from far_audio_salience import AcousticConfig
from far_audio_salience._rhythm import Rhythm

np = pytest.importorskip("numpy")


def test_late_impulse_cannot_reopen_expired_sequence():
    cfg = AcousticConfig()
    r = Rhythm()
    for frame in range(100, 3200, 300):
        r.observe(
            high=True,
            pulse_frame=frame,
            end=frame,
            rate=1000,
            profile=(1.0,),
            config=cfg,
        )
        r.observe(
            high=False,
            pulse_frame=frame + 1,
            end=frame + 1,
            rate=1000,
            profile=(1.0,),
            config=cfg,
        )
    old = r.sequence_id
    late = r.deadline + 100
    r.observe(
        high=True, pulse_frame=late, end=late, rate=1000, profile=(1.0,), config=cfg
    )
    assert r.sequence_id != old
    assert list(r.times) == [late]
    assert r.rate is None


def benchmark():
    pytest.importorskip("numpy")
    spec = importlib.util.spec_from_file_location(
        "audit_benchmark", Path(__file__).parents[1] / "benchmarks/acoustic.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("chunk", [80, 480, 960])
def test_level_step_is_not_new_band_activity(chunk):

    from far_audio_salience import AudioSalienceDetector, SourceSpec

    b = benchmark()
    rate = 48000
    x = np.asarray(
        np.concatenate(
            [b.tone(0.4, rate, 0.03), b.tone(0.3, rate, 0.7), b.tone(0.3, rate, 0.03)]
        ),
        dtype=np.float32,
    )
    d = AudioSalienceDetector(b.configuration())
    s = SourceSpec.microphone(label="level only", sample_rate_hz=rate)
    d.start_source(s)
    events = []
    for first in range(0, len(x), chunk):
        out = d.process(
            s.chunk(memoryview(x[first : first + chunk]), first_frame=first)
        )
        assert not out.refusals
        events.extend(
            c for result in out.salience for c in result.changes if c.revision == 1
        )
    assert not [c for c in events if c.kind == "sound_onset"], [
        (c.kind, c.start_frame, [(m.band_id, m.power) for m in c.bands]) for c in events
    ]
    assert [c.kind for c in events] == ["loudness_spike", "loudness_drop"]


def test_genuine_band_onset_and_spike_remain_independent():

    b = benchmark()
    rate = 48000
    x = np.concatenate(
        [
            b.tone(0.4, rate, 0.03, 200),
            b.tone(0.4, rate, 0.03, 200) + b.tone(0.4, rate, 0.7, 4000),
        ]
    )
    events, _ = b.run_audio(x, rate, AcousticConfig())
    assert len([e for e in events if e[0] == "sound_onset"]) == 1
    assert len([e for e in events if e[0] == "loudness_spike"]) == 1
    # Preserve the hum; redistribute energy so broadband RMS is unchanged.
    x = np.concatenate(
        [
            b.tone(0.4, rate, 0.2, 200),
            b.tone(0.4, rate, 0.19, 200)
            + b.tone(0.4, rate, np.sqrt(0.2**2 - 0.19**2), 4000),
        ]
    )
    events, _ = b.run_audio(x, rate, AcousticConfig())
    assert len([e for e in events if e[0] == "sound_onset"]) == 1
    assert not [e for e in events if e[0] in ("loudness_spike", "sound_offset")]


def test_materialization_failure_does_not_end_refusal_episode(monkeypatch):

    from far_audio_salience import AudioSalienceDetector, SourceSpec
    from far_audio_salience._history import _Index

    b = benchmark()
    cfg = b.configuration(acoustic=AcousticConfig(max_windows_per_call=1))
    d = AudioSalienceDetector(cfg)
    s = SourceSpec.microphone(label="atomic", sample_rate_hz=48000)
    d.start_source(s)
    x = np.zeros(1920, dtype=np.float32)
    refused = d.process(s.chunk(memoryview(x), first_frame=0))
    assert refused.refusals
    before = repr(d._episodes)

    def fail(*args):
        raise RuntimeError("materialization boundary")

    monkeypatch.setattr(_Index, "materialized", fail)
    with pytest.raises(RuntimeError, match="materialization"):
        d.process(s.chunk(memoryview(x[:480]), first_frame=0))
    assert repr(d._episodes) == before
    assert not d._acoustic.states


def test_ringing_impulse_is_one_onset_without_offset():
    b = benchmark()
    rate = 48000
    x = np.zeros(rate)
    t = np.arange(9600) / rate
    x[14400:24000] = 0.8 * np.exp(-t / 0.012) * np.sin(2 * np.pi * 1000 * t)
    events, _ = b.run_audio(x, rate, AcousticConfig())
    onsets = [e for e in events if e[0] == "sound_onset"]
    assert len(onsets) == 1
    assert 14400 <= round(onsets[0][1] * rate) <= 14880
    assert not [e for e in events if e[0] == "sound_offset"]


@pytest.mark.parametrize(
    "frequency,low,high,phase",
    [
        (200, 0.03, 0.7, 1.3),
        (200, 0.01, 0.99, 0.0),
        (200, 0.01, 0.99, 1.3),
        (1000, 0.01, 0.99, 0.0),
        (1000, 0.01, 0.99, 1.3),
    ],
)
def test_open_amplitude_edge_false_onset(frequency, low, high, phase):
    b = benchmark()
    t = np.arange(48000) / 48000
    amplitude = np.where(t < 0.4, low, np.where(t < 0.7, high, low))
    x = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    events, _ = b.run_audio(x, 48000, AcousticConfig())
    assert not [event for event in events if event[0] == "sound_onset"]
