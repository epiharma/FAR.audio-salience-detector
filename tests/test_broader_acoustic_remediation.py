"""Regression controls for the frozen broader-validation failures."""

from pathlib import Path
from sys import path

import pytest

np = pytest.importorskip("numpy")


@pytest.fixture
def benchmark_modules(monkeypatch):
    monkeypatch.setattr(
        "sys.path", [str(Path(__file__).parents[1] / "benchmarks"), *path]
    )
    import broader_noise
    import broader_validation

    return broader_validation, broader_noise


def test_stationary_gain_recovery_and_gradual_tempo(benchmark_modules):
    harness, _ = benchmark_modules
    cases = (
        "development:43:16000:noise_gain",
        "heldout:809:16000:noise_gain",
        "development:43:48000:gradual_faster",
        "development:43:48000:competing_pulses",
    )
    for case in harness.fixtures():
        if case["name"] in cases:
            observed = harness.run_case(case)
            assert not observed["violations"]
            assert all(m["fp"] == m["fn"] == 0 for m in observed["metrics"].values())
            if case["family"] == "gradual_faster":
                revisions = [
                    r for r in observed["revisions"] if r["kind"] == "tempo_increase"
                ]
                assert len({r["change_id"] for r in revisions}) == 1
                assert [r["phase"] for r in revisions] == ["detected", "completed"]


@pytest.mark.parametrize("seed", [16031, 99277])
@pytest.mark.parametrize("amplitude", [0.02, 0.06])
def test_new_band_component_over_colored_background(benchmark_modules, seed, amplitude):
    harness, noise = benchmark_modules
    cases = [
        c
        for c in noise.fixtures()
        if c["seed"] == seed and c["family"] == f"colored_hum_component_{amplitude}"
    ]
    assert len(cases) == 2
    for case in cases:
        observed = harness.run_case(case)
        assert not observed["violations"]
        for kind in ("sound_onset", "sound_offset"):
            metric = observed["metrics"][kind]
            assert (metric["tp"], metric["fp"], metric["fn"]) == (1, 0, 0)
            [first] = [
                r
                for r in observed["revisions"]
                if r["kind"] == kind and r["revision"] == 1
            ]
            assert first["confirmed_frame"] >= first["start_frame"]
            assert "mid" in {b["band_id"] for b in first["bands"]}


@pytest.mark.parametrize("color", ["white", "pink", "brown"])
def test_unseen_smooth_gain_on_continuous_noise(benchmark_modules, color):
    harness, noise = benchmark_modules
    rate = 32000
    t = np.arange(rate * 2) / rate
    rng = np.random.default_rng(62013)
    white = rng.normal(size=t.size)
    frequency = np.fft.rfftfreq(t.size, 1 / rate)
    exponent = {"white": 0, "pink": 0.5, "brown": 1}[color]
    spectrum = np.fft.rfft(white) / np.maximum(frequency, 30) ** exponent
    spectrum[0] = 0
    carrier = np.fft.irfft(spectrum)
    gain = np.interp(
        t, [0, 0.5, 0.7, 1.0, 1.4, 2], [0.21, 0.21, 0.04, 0.04, 0.16, 0.16]
    )
    case = noise.make_case(
        color, 62013, rate, carrier * gain / np.max(abs(carrier)), [], ""
    )
    observed = harness.run_case(case)
    assert not observed["violations"]
    assert observed["metrics"]["sound_onset"]["fp"] == 0
    assert observed["metrics"]["sound_offset"]["fp"] == 0


def test_independent_louder_band_does_not_end_ongoing_component(benchmark_modules):
    from broader_remediation_cases import fixtures

    harness, _ = benchmark_modules
    case = next(c for c in fixtures() if c["family"] == "cross_band_dilution")
    observed = harness.run_case(case)
    assert not observed["violations"]
    assert all(m["fp"] == m["fn"] == 0 for m in observed["metrics"].values())
    [offset] = [
        r
        for r in observed["revisions"]
        if r["revision"] == 1
        and r["kind"] == "sound_offset"
        and "mid" in {b["band_id"] for b in r["bands"]}
    ]
    assert offset["start_frame"] >= round(1.1 * case["rate"])


def test_duplex_alignment_checks_independent_middle_marker(benchmark_modules):
    import controlled_audio_validation as capture

    _, markers, _ = capture.stimuli()
    rate, block = capture.RATE, capture.BLOCK
    total = 29 * rate
    adc = 100
    dac = adc + 0.035
    slope = 1.00008
    delay = round((dac - adc) * rate) + 90
    anchors = [(f, adc + f / rate, dac + f / rate) for f in range(0, total, block)]
    audio = np.zeros(total, dtype=np.float32)
    for f, template in markers:
        start = round(f * slope + delay)
        audio[start : start + len(template)] = template
    result = capture.align(audio, markers, anchors)
    assert result["valid"]
    assert result["slope"] == pytest.approx(slope, abs=1 / rate)
    assert result["intercept_frames"] == pytest.approx(delay, abs=1)

    f, template = markers[1]
    start = round(f * slope + delay)
    audio[start : start + len(template)] = 0
    audio[start + 2 * block : start + 2 * block + len(template)] = template
    assert (
        capture.align(audio, markers, anchors)["reason"]
        == "inconsistent_marker_frame_map"
    )
    invalid_anchors = [(f, 0, d) for f, _, d in anchors]
    assert (
        capture.align(audio, markers, invalid_anchors)["reason"]
        == "invalid_duplex_timestamp_evidence"
    )
