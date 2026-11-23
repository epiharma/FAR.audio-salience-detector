"""Frozen labelled noise controls for the directional-presence correction."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from far_audio_salience import AcousticConfig, ChannelKey, FrequencyBand, SourceSpec
from far_audio_salience._acoustic import AcousticEngine, _Activity, _Stat

np = pytest.importorskip("numpy")
b = runpy.run_path(str(Path(__file__).parents[1] / "benchmarks/acoustic.py"))
RATE = 48000


def noise(seed, kind, length):
    x = np.random.default_rng(seed).uniform(-1, 1, length)
    if kind == "white":
        return x
    spectrum = np.fft.rfft(x)
    f = np.fft.rfftfreq(length, 1 / RATE)
    if kind == "pink":
        spectrum *= 1 / np.sqrt(np.maximum(f, 40))
    elif kind == "band":
        spectrum[(f < 500) | (f > 6000)] = 0
    y = np.fft.irfft(spectrum, n=length)
    return y / max(1e-12, np.max(np.abs(y)))


def run(x, layout="interleaved", sizes=(480,), acoustic=None):
    from far_audio_salience import AudioSalienceDetector

    d = AudioSalienceDetector(b["configuration"](acoustic=acoustic))
    s = SourceSpec.microphone(label="seeded noise", sample_rate_hz=RATE)
    d.start_source(s)
    out = []
    at = 0
    i = 0
    x = np.asarray(x, dtype=np.float32)
    while at < len(x):
        n = sizes[i % len(sizes)]
        block = np.ascontiguousarray(x[at : at + n])
        i += 1
        view = memoryview(block)
        batch = d.process(
            s.chunk(
                view if layout == "interleaved" else (view,),
                layout=layout,
                first_frame=at,
            )
        )
        assert not batch.refusals
        out.extend(batch.salience)
        at += len(block)
    d.close()
    return out


def firsts(results, kind):
    return [c for r in results for c in r.changes if c.kind == kind and c.revision == 1]


@pytest.mark.parametrize("factor", [2, 6, 10])
def test_positive_band_residual_does_not_establish_absence(factor):
    cfg = AcousticConfig()
    engine = AcousticEngine(cfg)
    s = SourceSpec.microphone(label="directional", sample_rate_hz=1000)
    state = SimpleNamespace(
        rate=1000,
        plan=SimpleNamespace(hop=5),
        current_power=3e-5,
        rhythm=SimpleNamespace(ever_tracked=False),
        group={},
        changes={},
        cursor=0,
    )
    activity = _Activity(initialized=True, active=True, observed_active=True)
    stat = _Stat(mean=1e-4, exposure=1, expected=True)
    changes = []
    for end in range(5, 105, 5):
        state.cursor = end
        engine._activity(
            state,
            s,
            ChannelKey(s.ref.source_id, s.channel_ids[0]),
            "low",
            activity,
            stat,
            factor * 1e-4 * 0.0025,
            0.01,
            0,
            end,
            1.0,
            changes,
            common_gain=0.0025,
        )
    assert activity.active and activity.fall is None
    assert not changes
    assert activity.presence_ambiguous


@pytest.mark.parametrize("kind", ["white", "pink", "band"])
@pytest.mark.parametrize("seed", [13, 79, 2026])
@pytest.mark.parametrize(
    "drop,recovery,low", [(19217, 43271, 0.01), (33599, 62401, 0.04)]
)
@pytest.mark.parametrize("initial", [0.1, 0.2, 0.3])
def test_stationary_noise_gain_controls(kind, seed, drop, recovery, low, initial):
    # Zero expected onsets: one fixed realization, gain only, all frames valid.
    x = noise(seed, kind, 96000)
    gain = np.full(len(x), initial)
    gain[drop:recovery] = low
    results = run(x * gain)
    assert not firsts(results, "sound_onset")
    assert not firsts(results, "sound_offset")


@pytest.mark.parametrize("kind", ["white", "pink", "band"])
def test_gradual_recovery_and_partition_equivalence(kind):
    x = noise(321, kind, 96000)
    t = np.arange(len(x)) / RATE
    x *= np.interp(t, [0, 0.6, 0.62, 1.0, 1.4, 2.0], [0.2, 0.2, 0.01, 0.01, 0.2, 0.2])
    a = run(x)
    c = run(x, "planar", (71, 1001, 480))

    def signature(rs):
        return [
            (v.kind, v.revision, v.phase, v.start_frame, v.confirmed_frame, v.end_frame)
            for r in rs
            for v in r.changes
        ]

    assert signature(a) == signature(c)
    assert not firsts(a, "sound_onset")
    assert not firsts(a, "sound_offset")


@pytest.mark.parametrize("offset", [0, 480, 9600])
@pytest.mark.parametrize("amplitude", [0.003, 0.08])
def test_new_high_band_during_gain_recovery(offset, amplitude):
    t = np.arange(96000) / RATE
    # Existing low-band random content; independent 8 kHz component is new.
    x = noise(84, "band", len(t)) * 0.2
    gain = np.where(t < 0.6, 1.0, np.where(t < 1.2, 0.1, 1.0))
    x *= gain
    begin = 57600 + offset
    x[begin:] += amplitude * np.sin(2 * np.pi * 8000 * t[begin:])
    cfg = AcousticConfig(
        bands=(FrequencyBand("background", 500, 6000), FrequencyBand("new", 7000, 9000))
    )
    rs = run(x, acoustic=cfg)
    onset = firsts(rs, "sound_onset")
    assert len(onset) == 1
    assert begin <= onset[0].start_frame <= begin + 1920
    assert onset[0].confirmed_frame <= begin + 3360


def test_noise_disappearance_is_observed_offset():
    x = noise(85, "white", 504000) * 0.1
    x[480000:] = 0
    rs = run(x)
    assert not firsts(rs, "sound_onset")
    assert any(
        r.baseline_status == "established"
        for r in rs
        if r.windows and r.windows[0].interval.stop_frame <= 480000
    )
    offsets = firsts(rs, "sound_offset")
    assert len(offsets) == 1
    assert 480000 <= offsets[0].start_frame <= 480960
    assert offsets[0].confirmed_frame <= 483840


@pytest.mark.parametrize("begin", [43200, 57600])
def test_strong_short_band_burst_on_noise(begin):
    t = np.arange(96000) / RATE
    x = noise(167, "band", len(t)) * np.where(
        t < 0.6, 0.2, np.where(t < 1.2, 0.02, 0.2)
    )
    x[begin : begin + 480] += 0.7 * np.sin(2 * np.pi * 8000 * t[begin : begin + 480])
    cfg = AcousticConfig(
        bands=(FrequencyBand("background", 500, 6000), FrequencyBand("new", 7000, 9000))
    )
    rs = run(x, acoustic=cfg)
    onsets = firsts(rs, "sound_onset")
    assert len(onsets) == 1
    assert begin <= onsets[0].start_frame <= begin + 480
    assert onsets[0].confirmed_frame <= begin + 960
    revisions = [c for r in rs for c in r.changes if c.change_id == onsets[0].change_id]
    assert revisions[-1].phase == "completed"
    assert 0 < revisions[-1].duration_frames < 2400
    assert not firsts(rs, "sound_offset")


def test_real_onset_and_loudness_spike_on_noise():
    t = np.arange(96000) / RATE
    x = noise(91, "band", len(t)) * np.where(t < 0.6, 0.8, np.where(t < 1.2, 0.04, 0.8))
    begin = 57600
    x[begin:] += 0.08 * np.sin(2 * np.pi * 8000 * t[begin:])
    cfg = AcousticConfig(
        bands=(FrequencyBand("background", 500, 6000), FrequencyBand("new", 7000, 9000))
    )
    rs = run(x, acoustic=cfg)
    onsets = firsts(rs, "sound_onset")
    spikes = firsts(rs, "loudness_spike")
    assert len(onsets) == 1 and len(spikes) == 1
    assert begin <= onsets[0].start_frame <= begin + 1920
    assert begin - 960 <= spikes[0].start_frame <= begin + 960


def test_already_active_noise_after_gap_and_invalid_block():
    from far_audio_salience import (
        AudioSalienceDetector,
        SourceInterval,
        UnavailableObservation,
    )

    d = AudioSalienceDetector(b["configuration"]())
    s = SourceSpec.microphone(label="recovery noise", sample_rate_hz=RATE)
    d.start_source(s)
    x = np.asarray(noise(548, "white", 96000) * 0.2, dtype=np.float32)
    results = []
    health = []
    for first in range(0, len(x), 480):
        if first == 28800:
            out = d.process(
                UnavailableObservation(
                    s.ref,
                    s.channel_ids,
                    "missing",
                    interval=SourceInterval(s.ref, first, first + 480),
                )
            )
            results.extend(out.salience)
            health.extend(out.health)
            continue
        block = x[first : first + 480].copy()
        if first == 57600:
            block[:] = np.nan
        out = d.process(s.chunk(memoryview(block), first_frame=first))
        assert not out.refusals
        results.extend(out.salience)
        health.extend(out.health)
    assert not firsts(results, "sound_onset")
    assert not firsts(results, "sound_offset")
    assert any(e.event_type == "invalid_audio" for e in health)
    assert any(e.event_type == "channels_recovered" for e in health)
    d.close()
