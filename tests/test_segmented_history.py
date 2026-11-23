"""Storage edge cases; no acoustic algorithm or wall-clock timing assertions."""

import gc
import math
import random
import weakref
from array import array
from copy import deepcopy
from dataclasses import replace

import pytest

from far_audio_salience import (
    AudioSalienceDetector,
    ChannelKey,
    ClockStamp,
    EligibilityPolicy,
    FeatureSpec,
    SourceWatermark,
)
from far_audio_salience._history import BLOCK_BYTES, _Index, _planned, _Timing

from .test_history_staging import evidence
from .test_streaming_contract import chunk, source, spans


def lane(detector):
    return next(iter(detector._lanes.values()))


@pytest.mark.parametrize("layout", ["interleaved", "planar"])
def test_owned_buffers_and_original_timing_survive_caller_release(
    reorder_config, layout
):
    detector = AudioSalienceDetector(reorder_config)
    mic = source(detector, channels=2, rate=44100)
    a, b = array("f", [0.125, 0.25, 0.375]), array("f", [0.5, 0.625, 0.75])
    backing = array("f", [v for pair in zip(a, b, strict=True) for v in pair])
    owners = (backing,) if layout == "interleaved" else (a, b)
    refs = [weakref.ref(owner) for owner in owners]
    views = tuple(memoryview(owner) for owner in owners)
    observation = mic.chunk(
        views[0] if layout == "interleaved" else views,
        first_frame=0,
        layout=layout,
        measured_start_ns=17,
        received_at=ClockStamp("receipt", 100),
    )
    detector.process(observation)
    for owner in owners:
        owner[:] = array("f", [0]) * len(owner)
    for view in views:
        view.release()
    del observation, views, owners, owner, view, backing, a, b
    gc.collect()
    assert all(ref() is None for ref in refs)
    data = [evidence(item.pending, 44100) for item in detector._lanes.values()]
    assert [row[1][0] for row in data] == [0.25, 0.625]
    assert all(row[2][1].ns == 17 + 2 * 10**9 // 44100 for row in data)
    assert all(row[0][2] == ClockStamp("receipt", 100) for row in data)


def test_nested_multi_span_overlap_admits_only_holes_and_quarantines_conflict(
    reorder_config,
):
    detector = AudioSalienceDetector(reorder_config)
    mic = source(detector)
    for start in (0, 4, 8):
        detector.process(chunk(mic, [0.125, 0.125], start=start))
    output = detector.process(chunk(mic, [0.125] * 12))
    accepted = output.input_receipts[0].accepted
    assert [(s.interval.start_frame, s.interval.stop_frame) for s in accepted] == [
        (2, 4),
        (6, 8),
        (10, 12),
    ]
    assert detector.buffered_sample_count == 12
    before = deepcopy(detector._lanes)
    values = [0.125] * 16
    values[9] = 0.5
    conflict = detector.process(chunk(mic, values))
    assert [e.event_type for e in conflict.health] == ["conflicting_observation"]
    assert not conflict.input_receipts[0].accepted
    assert detector._lanes == before
    result = detector.advance_watermark(SourceWatermark(mic.ref, mic.channel_ids, 12))
    assert spans(result) == [(0, 12, mic.channel_ids)]


@pytest.mark.parametrize("first,second", [(0.0, -0.0), (-0.0, 0.0)])
@pytest.mark.parametrize("finalized", [False, True])
def test_signed_zero_compares_numerically_without_changing_owned_bits(
    reorder_config, first, second, finalized
):
    detector = AudioSalienceDetector(reorder_config)
    mic = source(detector)
    detector.process(chunk(mic, [first] * 3))
    if finalized:
        detector.advance_watermark(SourceWatermark(mic.ref, mic.channel_ids, 3))
    output = detector.process(chunk(mic, [second] * 3))
    assert output.outcome == "ignored" and not output.health
    index = lane(detector).history if finalized else lane(detector).pending
    assert math.copysign(1, evidence(index, 10)[0][0]) == math.copysign(1, first)


@pytest.mark.parametrize("rate", [44100, 48000])
def test_trimming_and_compaction_preserve_anchor_and_legacy_end_rounding(config, rate):
    detector = AudioSalienceDetector(
        replace(config, comparison_retention_ns=2 * 10**9 // rate + 1)
    )
    mic = source(detector, rate=rate)
    output = detector.process(chunk(mic, [0.125] * 8, measured_start_ns=31))
    retained = evidence(lane(detector).history, rate)
    assert tuple(retained) == (6, 7)
    assert retained[6][1].ns == 31 + 6 * 10**9 // rate
    assert output.salience[0].timestamp.ns == 31 + 7 * 10**9 // rate + 10**9 // rate
    tail = detector.process(chunk(mic, [0.125] * 6, start=6, measured_start_ns=555))
    assert tail.input_receipts[0].accepted[0].interval.start_frame == 8
    assert (
        tail.salience[0].header.provenance.measurement_times[0].ns
        == 555 + 2 * 10**9 // rate
    )


@pytest.mark.parametrize(
    "retention,expected",
    [(0, ()), (200_000_000, (3, 4)), (500_000_000, (0, 1, 2, 3, 4))],
)
def test_exact_retention_boundaries_and_late_evidence(config, retention, expected):
    detector = AudioSalienceDetector(replace(config, comparison_retention_ns=retention))
    mic = source(detector)
    detector.process(chunk(mic, [0.125] * 5))
    assert tuple(evidence(lane(detector).history, 10)) == expected
    late = detector.process(chunk(mic, [0.75], start=0))
    if 0 in expected:
        assert late.health[0].event_type == "conflicting_observation"
    else:
        assert late.input_receipts[0].reason_codes == ("late_unverifiable",)
        assert not late.health


def test_invalid_original_prefix_invalidates_new_tail_without_backfill(reorder_config):
    detector = AudioSalienceDetector(reorder_config)
    mic = source(detector, channels=2)
    detector.process(chunk(mic, [0.125, 0.25] * 2))
    invalid = detector.process(
        chunk(mic, [float("nan"), 0.25, 0.125, 0.25, 0.125, 0.25, 0.125, 0.25])
    )
    assert len(invalid.health) == 1 and invalid.health[0].event_type == "invalid_audio"
    first_lane = next(iter(detector._lanes.values()))
    assert evidence(first_lane.pending, 10)[2][0] is None
    before = deepcopy(detector._lanes)
    duplicate = detector.process(chunk(mic, [0.125, 0.25] * 4))
    assert not duplicate.health and not duplicate.input_receipts[0].accepted
    assert detector._lanes == before
    finalized = detector.advance_watermark(SourceWatermark(mic.ref, mic.channel_ids, 4))
    assert [
        (s.interval.start_frame, s.interval.stop_frame)
        for r in finalized.salience
        for s in r.header.provenance.spans
    ] == [(0, 2), (0, 4)]


def test_sparse_large_coordinates_have_no_gap_sized_storage(reorder_config):
    detector = AudioSalienceDetector(reorder_config)
    mic = source(detector)
    result = detector.process(chunk(mic, [0.125], start=10**12))
    assert result.health[0].event_type == "channels_unavailable"
    assert len(lane(detector).pending.spans) == 1
    assert detector.buffered_sample_count == 1
    result = detector.advance_watermark(
        SourceWatermark(mic.ref, mic.channel_ids, 10**12 + 1)
    )
    assert len(result.salience) == 1
    assert len(lane(detector).history.spans) == 1


@pytest.mark.parametrize("pattern", ["forward", "reverse", "shuffled"])
def test_one_frame_fragmentation_remains_sorted_and_finalizes_one_run(
    reorder_config, pattern
):
    config = replace(
        reorder_config, limits=replace(reorder_config.limits, max_buffer_bytes=8192)
    )
    detector = AudioSalienceDetector(config)
    mic = source(detector, rate=48000)
    frames = list(range(512))
    if pattern == "reverse":
        frames.reverse()
    elif pattern == "shuffled":
        random.Random(73).shuffle(frames)
    for f in frames:
        assert not detector.process(chunk(mic, [0.125], start=f)).refusals
    assert lane(detector).pending.starts == tuple(range(512))
    result = detector.advance_watermark(SourceWatermark(mic.ref, mic.channel_ids, 512))
    assert spans(result) == [(0, 512, mic.channel_ids)]
    assert detector.buffered_sample_count == 512


def test_refusal_copies_no_new_payload_and_rolls_back_invalid_continuity(
    reorder_config, monkeypatch
):
    config = replace(
        reorder_config, limits=replace(reorder_config.limits, max_buffer_bytes=16)
    )
    detector = AudioSalienceDetector(config)
    mic = source(detector, channels=2)
    key = ChannelKey(mic.ref.source_id, mic.channel_ids[0])
    policy = EligibilityPolicy(0.5)
    detector.register_feature(
        FeatureSpec("independent", (key,), policy, policy, policy, True)
    )
    detector.register_candidate((key,))
    detector.process(chunk(mic, [0.125, 0.25] * 2))
    before = deepcopy(detector._lanes)
    baseline = detector.candidates()
    status = detector.feature_status("independent", ClockStamp("reference", 0))

    def forbidden(*args):
        raise AssertionError("payload copied before resource preflight")

    monkeypatch.setattr(_Index, "materialized", forbidden)
    refused = detector.process(chunk(mic, [float("nan"), 0.5], start=2))
    assert refused.refusals[0].resource.resource_id == "max_buffer_bytes"
    assert not any(e.event_type == "invalid_audio" for e in refused.health)
    assert detector._lanes == before and detector.candidates() == baseline
    assert detector.feature_status("independent", ClockStamp("reference", 0)) == status


def test_bounded_blocks_compact_expired_prefix_without_touching_old_snapshots(config):
    count = BLOCK_BYTES // 4 * 3 + 7
    config = replace(
        config,
        limits=replace(
            config.limits, max_chunk_bytes=count * 4, max_buffer_bytes=count * 8
        ),
        comparison_retention_ns=10**9,
    )
    detector = AudioSalienceDetector(config)
    mic = source(detector, rate=48000)
    result = detector.process(chunk(mic, [0.125] * count, measured_start_ns=0))
    assert len(result.salience) == 1
    old = lane(detector).history
    assert len(old.spans) == 4
    assert all(
        isinstance(s.backing, bytes) and len(s.backing) <= BLOCK_BYTES
        for s in old.spans
    )
    detector.config = replace(
        config, comparison_retention_ns=1_000_000_000 // 48000 + 1
    )
    detector.advance_watermark(SourceWatermark(mic.ref, mic.channel_ids, count))
    compact = lane(detector).history
    assert len(compact) == 1 and len(compact.spans[0].backing) == 4
    assert len(old) == count and len(old.spans[-1].backing) == 28
    assert evidence(compact, 48000)[count - 1] == evidence(old, 48000)[count - 1]


def test_interval_partition_and_coverage_match_set_reference():
    rng = random.Random(139)
    index = _Index()
    known = set()
    timing = _Timing(0, None, None)
    for _ in range(150):
        start, size = rng.randrange(100), rng.randrange(1, 15)
        stop = start + size
        uncovered = index.uncovered(start, stop)
        actual = {f for a, b in uncovered for f in range(a, b)}
        assert actual == set(range(start, stop)) - known
        index = index.added(_planned(uncovered, False, timing))
        known.update(range(start, stop))
        frontier = rng.randrange(120)
        before, after = index.split(frontier)
        assert set(evidence(before, 10)) == {f for f in known if f < frontier}
        assert set(evidence(after, 10)) == {f for f in known if f >= frontier}
        assert set(evidence(index.pruned(frontier), 10)) == {
            f for f in known if f >= frontier
        }
        assert len(index) == len(known)
        assert index.covers(start, stop)
