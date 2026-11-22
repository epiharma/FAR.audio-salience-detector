from array import array
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest

from far_audio_salience import (
    AudioSalienceDetector,
    ChannelKey,
    ClockStamp,
    InterleavedAudio,
    SourceInterval,
    SourceSpec,
    SourceWatermark,
    StructuralInputError,
    UnavailableObservation,
)


def state(detector):
    data = dict(detector.__dict__)
    data["_baselines"] = detector._baselines.__dict__
    return deepcopy(data)


def source(detector, channels=1, label="mic", rate=10):
    spec = SourceSpec.microphone(
        label=label, sample_rate_hz=rate, channel_count=channels
    )
    assert not detector.start_source(spec).refusals
    return spec


def chunk(spec, values, start=0, **kwargs):
    return spec.chunk(memoryview(array("f", values)), first_frame=start, **kwargs)


def spans(output):
    return [
        (p.interval.start_frame, p.interval.stop_frame, p.channel_ids)
        for result in output.salience
        for p in result.header.provenance.spans
    ]


def test_layout_channel_reorder_and_borrowed_memory(reorder_config):
    detector = AudioSalienceDetector(reorder_config)
    mic = source(detector, channels=3)
    original = array("f", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    admitted = detector.process(mic.chunk(memoryview(original), first_frame=0))
    assert len(admitted.input_receipts[0].accepted) == 3
    original[:] = array("f", [0] * 6)
    # Same captured values, reordered into planar transport, must be duplicates.
    duplicate = detector.process(
        mic.chunk(
            (
                memoryview(array("f", [0.3, 0.6])),
                memoryview(array("f", [0.1, 0.4])),
                memoryview(array("f", [0.2, 0.5])),
            ),
            first_frame=0,
            layout="planar",
            channel_ids=(mic.channel_ids[2], mic.channel_ids[0], mic.channel_ids[1]),
        )
    )
    assert duplicate.outcome == "ignored"
    assert not duplicate.health
    assert not duplicate.input_receipts[0].accepted
    final = detector.end_source(mic.ref)
    assert len(final.salience) == 3
    assert detector.buffered_sample_count == 6


@pytest.mark.parametrize(
    "mutation",
    [
        "dtype",
        "length",
        "duplicate_ids",
        "shape",
        "noncontiguous",
        "clock",
        "time",
        "planar_count",
        "quality",
        "released",
    ],
)
def test_structural_rejection_changes_nothing(config, mutation):
    detector = AudioSalienceDetector(config)
    mic = source(detector, channels=2)
    item = chunk(mic, [0.1, 0.2, 0.3, 0.4])
    if mutation == "dtype":
        item = replace(
            item,
            audio=InterleavedAudio(mic.channel_ids, memoryview(array("d", [0] * 4))),
        )
    elif mutation == "length":
        item = replace(item, interval=SourceInterval(mic.ref, 0, 3))
    elif mutation == "duplicate_ids":
        item = replace(
            item, audio=replace(item.audio, channel_ids=(mic.channel_ids[0],) * 2)
        )
    elif mutation == "shape":
        item = replace(
            item,
            audio=replace(
                item.audio, samples=memoryview(bytearray(16)).cast("f", (2, 2))
            ),
        )
    elif mutation == "noncontiguous":
        item = replace(
            item,
            audio=replace(item.audio, samples=memoryview(array("f", [0] * 8))[::2]),
        )
    elif mutation == "clock":
        item = replace(item, measured_start=ClockStamp("not-the-source-clock", 0))
    elif mutation == "time":
        item = replace(item, interval=SourceInterval(mic.ref, 2, 1))
    elif mutation == "planar_count":
        from far_audio_salience import PlanarAudio

        item = replace(
            item, audio=PlanarAudio(mic.channel_ids, (memoryview(array("f", [0, 0])),))
        )
    elif mutation == "released":
        item.audio.samples.release()
    else:
        item = replace(item, quality="bad")
    before = state(detector)
    with pytest.raises(StructuralInputError):
        detector.process(item)
    assert state(detector) == before


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 1.1, -1.1])
def test_invalid_lane_isolated_and_no_fake_learning(config, bad):
    detector = AudioSalienceDetector(config)
    mic = source(detector, channels=2)
    output = detector.process(chunk(mic, [bad, 0.2, 0.1, 0.3]))
    assert [e.event_type for e in output.health] == ["invalid_audio"]
    assert len(output.salience) == 1
    assert output.salience[0].channels == (
        ChannelKey(mic.ref.source_id, mic.channel_ids[1]),
    )
    assert output.salience[0].salience_score is None
    assert output.salience[0].confidence is None
    assert output.salience[0].stream_discontinuity_score is None
    assert not detector.candidates()
    assert "not_implemented" in output.salience[0].unavailable_reasons


def test_silence_is_valid_unavailability_is_not_silence(config):
    detector = AudioSalienceDetector(config)
    mic = source(detector)
    assert detector.process(chunk(mic, [0, 0])).salience
    lost = detector.process(
        UnavailableObservation(
            mic.ref,
            mic.channel_ids,
            "missing",
            SourceInterval(mic.ref, 2, 4),
        )
    )
    assert lost.health[0].event_type == "channels_unavailable"
    assert not lost.salience
    recovered = detector.process(chunk(mic, [0, 0], start=4))
    assert recovered.health[0].event_type == "channels_recovered"


def test_independent_sources_and_subset_omissions(config):
    detector = AudioSalienceDetector(config)
    a = source(detector, channels=2, label="a", rate=10)
    b = source(detector, label="b", rate=17)
    assert a.clock_id != b.clock_id
    output = detector.process(chunk(a, [0.1, 0.2], channel_ids=(a.channel_ids[0],)))
    assert output.salience and not output.health
    for start in range(4):
        output = detector.process(
            chunk(a, [0.1], start=start + 2, channel_ids=(a.channel_ids[0],))
        )
        assert all(
            b.ref.source_id != c.source_id for e in output.health for c in e.channels
        )
    assert detector._lanes[ChannelKey(b.ref.source_id, b.channel_ids[0])].watermark == 0


def test_out_of_order_watermark_late_overlap_and_conflict(reorder_config):
    detector = AudioSalienceDetector(reorder_config)
    mic = source(detector)
    assert not detector.process(chunk(mic, [0.3] * 2, start=4)).salience
    assert not detector.process(chunk(mic, [0.1] * 2)).salience
    assert not detector.process(chunk(mic, [0.2] * 2, start=2)).salience
    output = detector.advance_watermark(SourceWatermark(mic.ref, mic.channel_ids, 6))
    assert spans(output) == [(0, 6, mic.channel_ids)]
    published = output.salience[0]
    duplicate = detector.process(chunk(mic, [0.2] * 2, start=2))
    assert duplicate.input_receipts[0].reason_codes == ("late",)
    assert not duplicate.salience
    tail = detector.process(chunk(mic, [0.3, 0.3, 0.4, 0.4], start=4))
    assert [
        (s.interval.start_frame, s.interval.stop_frame)
        for s in tail.input_receipts[0].accepted
    ] == [(6, 8)]
    conflict = detector.process(chunk(mic, [0.8, 0.8, 0.8], start=7))
    assert conflict.health[0].event_type == "conflicting_observation"
    assert not conflict.input_receipts[0].accepted  # Conservative: no tail salvage.
    assert published == output.salience[0]
    with pytest.raises(FrozenInstanceError):
        published.salience_score = 0.9


def test_comparison_expiration_never_claims_conflict(config):
    detector = AudioSalienceDetector(config)
    mic = source(detector)
    detector.process(chunk(mic, [0.1] * 10))
    detector.process(chunk(mic, [0.2] * 10, start=10))
    late = detector.process(chunk(mic, [0.9] * 2))
    assert late.input_receipts[0].reason_codes == ("late_unverifiable",)
    assert not late.health and not late.salience
    assert detector.buffered_sample_count == 10


def test_watermark_atomicity_and_other_source_isolation(reorder_config):
    detector = AudioSalienceDetector(reorder_config)
    a, b = source(detector, label="a"), source(detector, label="b")
    detector.process(chunk(a, [0.1, 0.1]))
    detector.process(chunk(b, [0.2, 0.2]))
    detector.advance_watermark(SourceWatermark(a.ref, a.channel_ids, 2))
    assert detector._lanes[ChannelKey(b.ref.source_id, b.channel_ids[0])].watermark == 0
    before = state(detector)
    with pytest.raises(StructuralInputError):
        detector.advance_watermark(SourceWatermark(a.ref, a.channel_ids, 1))
    assert state(detector) == before


def test_no_events_can_mutate_previous_batches(config):
    detector = AudioSalienceDetector(config)
    mic = source(detector)
    first = detector.process(chunk(mic, [0.1]))
    second = detector.process(chunk(mic, [0.2], start=2))
    combined = first.ordered + second.ordered
    sequences = [e.header.emission_sequence for e in combined]
    assert sequences == sorted(set(sequences))
    assert len({e.header.event_id for e in combined}) == len(combined)
    assert {e.header.run_id for e in combined} == {detector.run_id}
    # A consumer can notice an intermediate gap, but not prove receipt of a last event.
    assert (
        second.ordered[-1].header.emission_sequence
        > first.ordered[0].header.emission_sequence + 1
    )
