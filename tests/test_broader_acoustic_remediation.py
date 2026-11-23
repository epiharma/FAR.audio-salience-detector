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


def test_brown_noise_short_fit_does_not_override_conflicting_history(benchmark_modules):
    from broader_remediation_cases import fixtures

    harness, _ = benchmark_modules
    case = next(
        c for c in fixtures() if c["name"] == "colored:62013:32000:brown_smooth_gain"
    )
    observed = harness.run_case(case)
    assert not observed["violations"]
    assert observed["tempo_reasons"]["ambiguous_impulse_sequence"] > 0
    assert not [p for p in observed["predictions"] if p["kind"].startswith("tempo_")]


def test_stable_then_accelerating_and_slowing_pulses_over_brown_noise(
    benchmark_modules,
):
    from broader_remediation_cases import fixtures

    harness, noise = benchmark_modules
    background = next(
        c for c in fixtures() if c["name"] == "colored:62013:32000:brown_smooth_gain"
    )
    rate = background["rate"]
    stable = [0.4] * 10
    accelerating = [0.38 - 0.02 * i for i in range(10)]
    plateau = [0.2] * 10
    slowing = [0.4] * 12
    before_slowing = stable + accelerating + plateau
    intervals = before_slowing + slowing
    times = np.cumsum([0.3, *intervals])
    pulses = np.rint(times * rate).astype(int)
    samples = 0.05 * np.resize(background["samples"], pulses[-1] + rate // 10)
    samples[pulses] += 0.75
    case = noise.make_case("pulse_train_over_brown_noise", 62013, rate, samples, [], "")
    case["tempo"] = list(
        zip(times, [1 / stable[0], *(1 / i for i in intervals)], strict=True)
    )

    # Verify acquisition at the stable rate before any tempo change.
    prefix = dict(case, samples=samples[: pulses[len(stable)] + 1 + rate // 100])
    stable_result = harness.run_case(prefix)
    assert not stable_result["violations"]
    assert stable_result["tempo_availability"] > 0
    assert stable_result["tempo_mae_hz"] == pytest.approx(0, abs=0.001)
    assert not [
        p for p in stable_result["predictions"] if p["kind"].startswith("tempo_")
    ]

    result = harness.run_case(case)
    assert not result["violations"]
    tempo = [r for r in result["revisions"] if r["kind"].startswith("tempo_")]
    assert [(r["kind"], r["phase"]) for r in tempo] == [
        ("tempo_increase", "detected"),
        ("tempo_increase", "completed"),
        ("tempo_decrease", "detected"),
        ("tempo_decrease", "completed"),
    ]
    assert tempo[0]["change_id"] == tempo[1]["change_id"]
    assert tempo[2]["change_id"] == tempo[3]["change_id"]
    assert tempo[0]["change_id"] != tempo[2]["change_id"]
    first_slow_pulse = pulses[len(before_slowing) + 1]
    assert pulses[len(stable)] < tempo[0]["confirmed_frame"] < first_slow_pulse
    assert tempo[2]["confirmed_frame"] >= first_slow_pulse


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


def test_loudness_recovery_after_confirmed_drop_in_frozen_brown_noise(
    benchmark_modules,
):
    harness, noise = benchmark_modules
    case = next(
        c for c in noise.fixtures() if c["name"] == "colored:99277:48000:brown_gain"
    )
    observed = harness.run_case(case)
    assert not observed["violations"]
    spike = observed["metrics"]["loudness_spike"]
    # The unscheduled noise-envelope rise still meets both spike gates.
    assert (spike["tp"], spike["fp"], spike["fn"]) == (1, 1, 0)
    first, drop, recovery = (
        p for p in observed["predictions"] if p["kind"].startswith("loudness_")
    )
    assert [p["kind"] for p in (first, drop, recovery)] == [
        "loudness_spike",
        "loudness_drop",
        "loudness_spike",
    ]
    assert (first["confirmed"], drop["confirmed"], recovery["confirmed"]) == (
        28080,
        29760,
        53040,
    )
    assert first["change_id"] != recovery["change_id"]


def test_loudness_drop_rearms_after_short_confirmed_spike(benchmark_modules):
    harness, noise = benchmark_modules
    rate = 48000
    frames = np.arange(rate // 2)
    boundaries = (9600, 10800, 11760)
    gain = np.select(
        [frames < boundary for boundary in boundaries],
        (0.65, 0.018, 0.65),
        default=0.018,
    )
    kinds = ("loudness_drop", "loudness_spike", "loudness_drop")
    case = noise.make_case(
        "short_loudness_reversals",
        0,
        rate,
        gain * np.sin(2 * np.pi * 1000 * frames / rate),
        [(kind, frame / rate) for kind, frame in zip(kinds, boundaries, strict=True)],
        "A sustained carrier with two drops bracketing a 20 ms high interval.",
    )
    observed = harness.run_case(case)
    assert not observed["violations"]
    for kind, count in (("loudness_drop", 2), ("loudness_spike", 1)):
        metric = observed["metrics"][kind]
        assert (metric["tp"], metric["fp"], metric["fn"]) == (count, 0, 0)
    first, spike, second = [
        row
        for row in observed["revisions"]
        if row["kind"].startswith("loudness_") and row["revision"] == 1
    ]
    assert [row["kind"] for row in (first, spike, second)] == list(kinds)
    assert len({row["change_id"] for row in (first, spike, second)}) == 3
    for prior, reversal in ((first, spike), (spike, second)):
        [completed] = [
            r
            for r in observed["revisions"]
            if r["change_id"] == prior["change_id"] and r["revision"] > 1
        ]
        assert completed["phase"] == "completed"
        assert completed["end_frame"] == reversal["confirmed_frame"]


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


@pytest.mark.parametrize("seed", [62013, 86021])
def test_isolated_impulse_over_hum_keeps_one_onset_id(benchmark_modules, seed):
    from broader_remediation_cases import fixtures

    harness, _ = benchmark_modules
    case = next(c for c in fixtures() if c["family"] == "impulse" and c["seed"] == seed)
    observed = harness.run_case(case)
    assert not observed["violations"]
    metric = observed["metrics"]["sound_onset"]
    assert (metric["tp"], metric["fp"], metric["fn"]) == (1, 0, 0)
    assert observed["metrics"]["sound_offset"]["fp"] == 0
    [first] = observed["predictions"]
    assert (first["start"], first["confirmed"]) == (20800, 21120)
    revisions = observed["revisions"]
    assert {row["change_id"] for row in revisions} == {first["change_id"]}
    assert revisions[0]["phase"] == "detected"
    assert revisions[0]["end_frame"] is None
    assert revisions[-1]["phase"] == "completed"
    assert {band["band_id"] for band in revisions[-1]["bands"]} == {"mid", "high"}
    assert 0 < revisions[-1]["end_frame"] - first["start"] < 0.12 * case["rate"]


@pytest.mark.parametrize(
    "family", ["hum_gain", "pink_constant", "weak_narrowband", "short_burst"]
)
def test_isolated_peak_path_preserves_onset_controls(benchmark_modules, family):
    from broader_remediation_cases import fixtures

    harness, noise = benchmark_modules
    if family in {"pink_constant", "weak_narrowband"}:
        case = next(
            c for c in fixtures() if c["family"] == family and c["seed"] == 62013
        )
    else:
        rate = 32000
        frames = np.arange(rate)
        t = frames / rate
        if family == "hum_gain":
            gain = np.where((frames >= 20800) & (frames < 28000), 0.7, 0.03)
            samples, labels = gain * np.sin(2 * np.pi * 200 * t), []
        else:
            first, stop = 20807, 21127
            samples = 0.06 * np.sin(2 * np.pi * 213 * t + 1.37)
            samples[first:stop] += 0.7 * np.sin(2 * np.pi * 4000 * t[first:stop])
            labels = [("sound_onset", first / rate)]
        case = noise.make_case(family, 0, rate, samples, labels, "")
    observed = harness.run_case(case)
    assert not observed["violations"]
    for kind in ("sound_onset", "sound_offset"):
        metric = observed["metrics"][kind]
        assert (metric["tp"], metric["fp"], metric["fn"]) == (metric["positives"], 0, 0)


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
