"""Adversarial labelled controls; no source labels enter the detector."""

import runpy
from pathlib import Path

import pytest

from far_audio_salience import AcousticConfig

np = pytest.importorskip("numpy")
b = runpy.run_path(str(Path(__file__).parents[1] / "benchmarks/acoustic.py"))
RATE = 48000


def detected(x):
    events, _ = b["run_audio"](x, RATE, AcousticConfig())
    return events


@pytest.mark.parametrize(
    "kind",
    [
        "noise_step",
        "noise_ramp",
        "multiband",
        "dip",
        "slow_ramp",
    ],
)
def test_same_pattern_gain_has_no_new_onset(kind):
    t = np.arange(RATE * 2) / RATE
    rng = np.random.default_rng(785)
    carrier = (
        rng.uniform(-1, 1, len(t))
        if kind.startswith("noise")
        else 0.5 * np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 4000 * t)
    )
    gain = np.where(t < 0.6, 0.2, np.where(t < 1.2, 0.01, 0.2))
    if kind == "noise_ramp":
        gain = np.interp(t, [0, 0.6, 1.0, 1.4, 2], [0.2, 0.2, 0.01, 0.2, 0.2])
    if kind == "slow_ramp":
        gain = np.interp(t, [0, 0.6, 1.2, 2], [0.03, 0.03, 0.7, 0.7])
    if kind == "dip":
        gain = np.where((t >= 0.6) & (t < 0.62), 0.01, 0.2)
    assert not [e for e in detected(gain * carrier) if e[0] == "sound_onset"]


@pytest.mark.parametrize("low", [0.0001, 0.001, 0.002])
def test_emergence_from_floor_is_not_vetoed(low):
    t = np.arange(RATE) / RATE
    x = np.where(t < 0.4, low, 0.2) * np.sin(2 * np.pi * 1000 * t)
    onset = [e for e in detected(x) if e[0] == "sound_onset"]
    if low**2 / 2 < 1e-6:
        assert len(onset) == 1
        assert 0.4 <= onset[0][1] <= 0.42
    else:
        assert not onset


def test_short_high_band_impulse_on_hum():
    t = np.arange(RATE) / RATE
    x = 0.03 * np.sin(2 * np.pi * 200 * t)
    begin = 19200
    stop = begin + 480
    x[begin:stop] += 0.7 * np.sin(2 * np.pi * 4000 * t[begin:stop])
    onset = [e for e in detected(x) if e[0] == "sound_onset"]
    assert len(onset) == 1
    assert 0.4 <= onset[0][1] <= 0.42


def test_new_broadband_component_with_small_rms_change():
    t = np.arange(RATE) / RATE
    rng = np.random.default_rng(28)
    x = 0.2 * np.sin(2 * np.pi * 200 * t)
    x[19200:] += 0.03 * rng.uniform(-1, 1, len(t) - 19200)
    onset = [e for e in detected(x) if e[0] == "sound_onset"]
    assert len(onset) == 1
    assert 0.4 <= onset[0][1] <= 0.44


@pytest.mark.parametrize("frequency", [200, 1000, 4000])
@pytest.mark.parametrize("levels", [(0.01, 0.3), (0.03, 0.7), (0.01, 0.99)])
@pytest.mark.parametrize("phase", [0.0, 1.3])
def test_frozen_eighteen_amplitude_controls(frequency, levels, phase):
    low, high = levels
    t = np.arange(RATE) / RATE
    amplitude = np.where(t < 0.4, low, np.where(t < 0.7, high, low))
    events = detected(amplitude * np.sin(2 * np.pi * frequency * t + phase))
    assert not [e for e in events if e[0] == "sound_onset"]


def test_frame_domain_layout_and_partition_equivalence():
    from far_audio_salience import AudioSalienceDetector, SourceSpec

    t = np.arange(RATE) / RATE
    gain = np.where(t < 0.4, 0.01, np.where(t < 0.7, 0.99, 0.01))
    first = gain * np.sin(2 * np.pi * 200 * t + 1.3)
    second = 0.03 * np.sin(2 * np.pi * 200 * t) + np.where(t < 0.4, 0, 0.2) * np.sin(
        2 * np.pi * 4000 * t
    )
    x = np.column_stack((first, second)).astype(np.float32)

    def run(layout, sizes):
        d = AudioSalienceDetector(b["configuration"](2))
        s = SourceSpec.microphone(label="layout", sample_rate_hz=RATE, channel_count=2)
        d.start_source(s)
        out = []
        at = 0
        i = 0
        while at < len(x):
            n = sizes[i % len(sizes)]
            block = x[at : at + n]
            i += 1
            data = (
                memoryview(np.ascontiguousarray(block).ravel())
                if layout == "interleaved"
                else tuple(
                    memoryview(np.ascontiguousarray(block[:, j])) for j in range(2)
                )
            )
            batch = d.process(s.chunk(data, first_frame=at, layout=layout))
            assert not batch.refusals
            for result in batch.salience:
                for c in result.changes:
                    out.append(
                        (
                            s.channel_ids.index(c.channel.channel_id),
                            c.kind,
                            c.revision,
                            c.phase,
                            c.start_frame,
                            c.confirmed_frame,
                            c.end_frame,
                            tuple(
                                (v.band_id, v.start_frame, v.end_frame) for v in c.bands
                            ),
                        )
                    )
            at += len(block)
        return sorted(out)

    assert run("interleaved", (480,)) == run("planar", (71, 401, 960))
