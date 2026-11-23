"""History sharing must preserve isolated mutable state and immutable evidence."""

from dataclasses import replace

import pytest

from far_audio_salience import AudioSalienceDetector, ClockStamp, SourceWatermark

from .test_streaming_contract import chunk, source


def evidence(index, rate):
    """Logical evidence snapshot, independent of descriptor partitioning."""
    result = {}
    for span in index.spans:
        values = memoryview(span.backing).cast("f") if span.valid else None
        for frame in range(span.start, span.stop):
            result[frame] = (
                None if values is None else values[span.offset + frame - span.start],
                span.timing.at(frame, rate),
                span.timing.received,
            )
    return result


@pytest.mark.parametrize("operation", ["process", "watermark"])
@pytest.mark.parametrize("value", [0.25, float("nan")])
def test_staging_reuses_immutable_evidence_without_mutating_prior_lane(
    reorder_config, operation, value
):
    detector = AudioSalienceDetector(reorder_config)
    mic = source(detector)
    detector.process(
        chunk(
            mic,
            [value, value],
            measured_start_ns=0,
            received_at=ClockStamp("receipt", 123),
        )
    )
    detector.advance_watermark(SourceWatermark(mic.ref, mic.channel_ids, 1))
    key = next(iter(detector._lanes))
    before = detector._lanes[key]
    history, pending = (
        evidence(before.history, mic.sample_rate_hz),
        evidence(before.pending, mic.sample_rate_hz),
    )
    old_history_span, old_pending_span = before.history.at(0), before.pending.at(1)
    if operation == "process":
        output = detector.process(chunk(mic, [0.5], start=2))
    else:
        output = detector.advance_watermark(
            SourceWatermark(mic.ref, mic.channel_ids, 2)
        )
    assert not output.refusals
    after = detector._lanes[key]
    assert evidence(before.history, mic.sample_rate_hz) == history
    assert evidence(before.pending, mic.sample_rate_hz) == pending
    assert before.watermark == 1 and before.high_water == 2
    retained = after.history.at(0)
    transferred = after.pending.at(1) or after.history.at(1)
    # Immutable backing/timing is shared while published old state remains intact.
    assert retained.backing is old_history_span.backing
    assert transferred.backing is old_pending_span.backing
    assert retained.timing is old_history_span.timing
    assert transferred.timing is old_pending_span.timing
    observed = evidence(after.history, mic.sample_rate_hz)[0]
    assert observed == (
        None if value != value else value,
        ClockStamp(mic.clock_id, 0),
        ClockStamp("receipt", 123),
    )


def test_capacity_refusal_rolls_back_all_staged_channels_and_pruning(config):
    config = replace(
        config,
        comparison_retention_ns=200_000_000,
        limits=replace(config.limits, max_buffer_bytes=16),
    )
    detector = AudioSalienceDetector(config)
    mic = source(detector, channels=2)
    detector.process(chunk(mic, [0.1, 0.2, 0.3, 0.4]))
    # Both lanes have retained history. Staging the next chunk would prune that
    # history and add pending data; refusing it must undo both changes in all lanes.
    detector.config = replace(config, max_reorder_delay_ns=1_000_000_000)
    before = dict(detector._lanes)
    contents = {
        key: (
            evidence(lane.history, mic.sample_rate_hz),
            evidence(lane.pending, mic.sample_rate_hz),
            lane.watermark,
            lane.high_water,
        )
        for key, lane in before.items()
    }
    refused = detector.process(chunk(mic, [0.5, 0.6, 0.7, 0.8, 0.1, 0.2], start=20))
    assert refused.refusals[0].resource.resource_id == "max_buffer_bytes"
    assert detector.buffered_sample_count == 4
    assert not refused.salience
    for key, lane in detector._lanes.items():
        assert lane is before[key]
        assert (
            evidence(lane.history, mic.sample_rate_hz),
            evidence(lane.pending, mic.sample_rate_hz),
            lane.watermark,
            lane.high_water,
        ) == contents[key]
    # History is still available for conflict detection after the refused call.
    conflict = detector.process(chunk(mic, [0.9, 0.2], start=0))
    assert any(e.event_type == "conflicting_observation" for e in conflict.health)
    assert detector.buffered_sample_count == 4


def test_watermark_pruning_keeps_prior_lane_evidence_intact(reorder_config):
    detector = AudioSalienceDetector(
        replace(reorder_config, comparison_retention_ns=100_000_000)
    )
    mic = source(detector)
    detector.process(chunk(mic, [0.1, 0.2, 0.3], measured_start_ns=0))
    detector.advance_watermark(SourceWatermark(mic.ref, mic.channel_ids, 1))
    before = next(iter(detector._lanes.values()))
    old_history = evidence(before.history, mic.sample_rate_hz)
    old_pending = evidence(before.pending, mic.sample_rate_hz)
    old_span = before.pending.at(2)
    detector.advance_watermark(SourceWatermark(mic.ref, mic.channel_ids, 3))
    after = next(iter(detector._lanes.values()))
    assert evidence(before.history, mic.sample_rate_hz) == old_history
    assert evidence(before.pending, mic.sample_rate_hz) == old_pending
    assert evidence(after.history, mic.sample_rate_hz) == {2: old_pending[2]}
    # Boundary compaction may copy the surviving suffix, while old evidence
    # remains untouched and original timing metadata stays shared.
    assert len(after.history.at(2).backing) == 4
    assert after.history.at(2).offset == 0
    assert len(old_span.backing) == 12
    assert after.history.at(2).timing is old_span.timing
    assert not after.pending
