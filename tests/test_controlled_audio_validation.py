"""Deterministic loopback and fault injection for the external live harness."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def adapter():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).parents[1] / "benchmarks"))
        import controlled_audio_validation

    return controlled_audio_validation


@pytest.fixture(scope="module")
def playback(adapter):
    samples, markers, _ = adapter.stimuli()
    return samples, markers


def loopback(adapter, playback, *, offset=0, slope=1.0):
    source, markers = playback
    rate = adapter.RATE
    frames = np.arange(29 * rate)
    # Resample the entire playback, including the chirps, at the capture rate.
    audio = np.interp(
        (frames - offset) / slope, np.arange(len(source)), source, left=0, right=0
    ).astype(np.float32)
    adc = 100.0
    # The clocks put the search center near the arrival. The waveform has an
    # additional 90-frame path delay when simulating a nonzero offset.
    dac = adc + (offset - 90 if offset else 0) / (slope * rate)
    anchors = [
        (f, adc + f / (slope * rate), dac + f / rate)
        for f in range(0, len(audio), adapter.BLOCK)
    ]
    return audio, markers, anchors


@pytest.mark.parametrize(
    "offset,slope",
    [(0, 1.0), (14000, 1.0), (3000, 1.00008)],
    ids=["aligned", "offset-outside-unanchored-search", "resampled-drift"],
)
def test_loopback_aligns_with_clock_guided_marker_search(
    adapter, playback, offset, slope
):
    audio, markers, anchors = loopback(adapter, playback, offset=offset, slope=slope)
    result = adapter.align(audio, markers, anchors)
    assert result["valid"]
    assert result["intercept_frames"] == pytest.approx(offset, abs=1)
    assert result["slope"] == pytest.approx(slope, abs=2 / (24 * adapter.RATE))
    assert all(marker["unique"] for marker in result["markers"])
    assert abs(result["middle_residual_frames"]) <= 1


@pytest.mark.parametrize("slope", [0.994, 1.006])
def test_loopback_rejects_excessive_drift_even_with_three_unique_peaks(
    adapter, playback, slope
):
    audio, markers, anchors = loopback(adapter, playback, offset=3000, slope=slope)
    result = adapter.align(audio, markers, anchors)
    assert not result["valid"]
    assert result["reason"] == "inconsistent_marker_frame_map"
    assert all(marker["unique"] for marker in result["markers"])
    assert abs(result["candidate_slope"] - 1) > adapter.SLOPE_ERROR


@pytest.mark.parametrize("fault", ["missing", "duplicate", "displaced"])
def test_middle_marker_is_required_unique_and_independent(adapter, playback, fault):
    audio, markers, anchors = loopback(adapter, playback)
    first, template = markers[1]
    if fault != "duplicate":
        audio[first : first + len(template)] = 0
    if fault != "missing":
        # The duplicate is resolvable but closer than both the old exclusion
        # mask and the middle-marker residual tolerance.
        shift = 300 if fault == "duplicate" else 2 * adapter.BLOCK
        audio[first + shift : first + shift + len(template)] += template
    result = adapter.align(audio, markers, anchors)
    assert not result["valid"]
    if fault == "displaced":
        assert result["reason"] == "inconsistent_marker_frame_map"
        assert abs(result["middle_residual_frames"]) == pytest.approx(shift)
    else:
        assert result["reason"] == "missing_or_ambiguous_marker"
        assert not result["markers"][1]["unique"]
        if fault == "duplicate":
            middle = result["markers"][1]
            assert middle["alternate_correlation"] == pytest.approx(
                middle["correlation"], rel=0.01
            )


@pytest.mark.parametrize(
    "clock, fault", [(1, "backward"), (1, "nonfinite"), (2, "stalled"), (2, "spacing")]
)
def test_bad_duplex_timestamps_reject_intact_markers(adapter, playback, clock, fault):
    audio, markers, anchors = loopback(adapter, playback)
    bad = list(anchors[1])
    bad[clock] = {
        "backward": anchors[0][clock] - 0.01,
        "nonfinite": float("nan"),
        "stalled": anchors[0][clock],
        "spacing": anchors[0][clock] + 0.03,
    }[fault]
    anchors[1] = tuple(bad)
    result = adapter.align(audio, markers, anchors)
    assert not result["valid"]
    assert result["reason"] == "invalid_duplex_timestamp_evidence"
    assert result["markers"] == []
    assert result["invalid_anchors"][0]["clock"] == ("adc" if clock == 1 else "dac")


def test_wasapi_first_adc_interval_replay(adapter, playback):
    source, markers = playback
    audio = np.pad(source, (0, 2 * adapter.RATE))
    # The trial retained the first ADC/DAC values and the signed ADC step.
    # Reconstruct that pair only. Remaining anchors are nominal controls.
    adc_start, dac_start = 203130.63438869998, 203130.6223887
    logged_step_frames = -463.329600635916
    anchors = [
        (first, adc_start + first / adapter.RATE, dac_start + first / adapter.RATE)
        for first in range(0, len(audio), adapter.BLOCK)
    ]
    first = anchors[1][0]

    control = adapter.align(audio, markers, anchors)
    assert control["valid"]
    assert len(control["markers"]) == 3
    assert abs(control["middle_residual_frames"]) <= 1

    adc_stop = adc_start + logged_step_frames / adapter.RATE
    anchors[1:] = [
        (frame, adc_stop + (frame - first) / adapter.RATE, dac)
        for frame, _, dac in anchors[1:]
    ]
    result = adapter.align(audio, markers, anchors)
    assert result["reason"] == "invalid_duplex_timestamp_evidence"
    assert not result["valid"]
    assert result["markers"] == []
    (failure,) = result["invalid_anchors"]
    assert failure["clock"] == "adc"
    assert failure["errors"] == 1
    assert failure["first_interval_start_frame"] == 0
    assert failure["frame_step"] == 480
    assert failure["timestamp_start_seconds"] == adc_start
    assert failure["timestamp_stop_seconds"] == adc_stop
    assert failure["timestamp_step_frames"] == pytest.approx(logged_step_frames)
    assert failure["failed_checks"] == ["strict_increase", "block_spacing"]


def test_callback_trace_preserves_separate_clocks(adapter, monkeypatch):
    receipt_ns = iter((10_000_000, 20_000_000))
    monkeypatch.setattr(
        adapter,
        "time",
        SimpleNamespace(
            monotonic_ns=receipt_ns.__next__,
            monotonic=adapter.time.monotonic,
            perf_counter_ns=adapter.time.perf_counter_ns,
        ),
    )

    @contextmanager
    def raw_stream(*, callback, **kwargs):
        status = SimpleNamespace(
            input_underflow=False,
            input_overflow=False,
            output_underflow=False,
            output_overflow=False,
        )
        block = bytearray(adapter.BLOCK * 4)
        for adc, dac, current in ((0.0, 50.01, 50.0), (float("nan"), 50.02, 50.01)):
            callback(
                block,
                block.copy(),
                adapter.BLOCK,
                SimpleNamespace(
                    inputBufferAdcTime=adc, outputBufferDacTime=dac, currentTime=current
                ),
                status,
            )
        yield SimpleNamespace(samplerate=adapter.RATE, active=False)

    sd = SimpleNamespace(
        RawStream=raw_stream,
        query_devices=lambda _: {"hostapi": 0},
        query_hostapis=lambda _: {"name": "simulated duplex"},
        __version__="test",
        get_portaudio_version=lambda: (0, "test backend"),
    )
    result = adapter.trial(sd, 0, 1, trace_callback_timing=True)
    assert result["alignment"]["reason"] == "incomplete_or_discontinuous_capture"
    assert result["labelled_accuracy"] == "NOT MEASURED"
    first, second = result["callback_timing"]
    assert first["first_frame"] == 0
    assert first["adc_seconds"] == 0.0
    assert first["dac_seconds"] == 50.01
    assert first["current_seconds"] == 50.0
    assert first["receipt_monotonic_ns"] == 10_000_000
    assert second["first_frame"] == adapter.BLOCK
    assert second["adc_seconds"] == "nan"
    assert second["receipt_monotonic_ns"] == 20_000_000
    assert all(row["queued"] for row in (first, second))
    json.dumps(result["callback_timing"], allow_nan=False)


@pytest.mark.parametrize("fault", ["missing", "repeated_frame", "short_capture"])
def test_delivered_frame_evidence_must_cover_capture(adapter, playback, fault):
    audio, markers, anchors = loopback(adapter, playback)
    if fault == "missing":
        anchors.pop(100)
    elif fault == "repeated_frame":
        anchors[100] = (anchors[99][0], *anchors[100][1:])
    else:
        audio = audio[:-1]
    result = adapter.align(audio, markers, anchors)
    assert not result["valid"]
    assert result["reason"] == "invalid_delivered_frame_evidence"
    assert result["markers"] == []
