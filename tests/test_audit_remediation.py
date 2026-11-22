"""Regressions for F1--F5; numeric policies here are test fixtures only."""

from dataclasses import replace

import pytest

from far_audio_salience import (
    AlignmentEvidence,
    AlignmentPolicy,
    AudioSalienceDetector,
    BaselineCompatibility,
    CandidateCommand,
    ChannelKey,
    ClockMapping,
    ClockStamp,
    EligibilityPolicy,
    FeatureSpec,
    ReadinessEvidence,
    RelationshipWindow,
    SourceInterval,
    SourceWatermark,
    UnavailableObservation,
    new_id,
)

from .test_baselines_persistence import candidate
from .test_streaming_contract import chunk, source


def _health(batch, kind):
    return [event for event in batch.health if event.event_type == kind]


def _single_feature(detector, mic, channel, name):
    key = ChannelKey(mic.ref.source_id, channel)
    policy = EligibilityPolicy(0.5)
    detector.register_feature(FeatureSpec(name, (key,), policy, policy, policy, True))


def _relationship(detector, confirmation=0, horizon=20):
    a, b = source(detector, label="a"), source(detector, label="b", rate=17)
    keys = tuple(ChannelKey(s.ref.source_id, s.channel_ids[0]) for s in (a, b))
    policy = EligibilityPolicy(0.5)
    feature = FeatureSpec(
        "joint",
        keys,
        policy,
        policy,
        policy,
        True,
        AlignmentPolicy(10, 4, confirmation),
        horizon,
    )
    detector.register_feature(feature)
    return a, b, feature


def _report(detector, sources, feature, at=0, *, start=0, stop=10, error=1):
    return detector.update_alignment(
        AlignmentEvidence(
            f"alignment-{at}-{error}",
            feature.channels,
            "ref",
            start,
            stop,
            ClockStamp("ref", at),
            error,
            tuple(ClockMapping(s.ref, s.clock_id, 0, 0, 1, 1) for s in sources),
        )
    )


def test_f1_original_audit_sequence_has_no_pre_loss_recovery(reorder_config):
    detector = AudioSalienceDetector(reorder_config)
    mic = source(detector)
    detector.process(chunk(mic, [0.1, 0.1]))
    detector.process(
        UnavailableObservation(
            mic.ref,
            mic.channel_ids,
            "missing",
            SourceInterval(mic.ref, 2, 4),
        )
    )
    output = detector.advance_watermark(SourceWatermark(mic.ref, mic.channel_ids, 2))
    assert not _health(output, "channels_recovered")


def test_f1_pre_loss_buffer_does_not_recover_and_later_audio_does(reorder_config):
    detector = AudioSalienceDetector(reorder_config)
    mic = source(detector, channels=2)
    a, b = mic.channel_ids
    detector.process(chunk(mic, [0.1, 0.2]))  # [0,1) pending on both channels.
    detector.process(
        UnavailableObservation(
            mic.ref,
            (a,),
            "missing",
            SourceInterval(mic.ref, 2, 4),
        )
    )
    # A delayed, valid pre-loss observation arrives after the loss notification.
    detector.process(chunk(mic, [0.1, 0.2], start=1))
    before_loss = detector.advance_watermark(SourceWatermark(mic.ref, (a, b), 2))
    assert not _health(before_loss, "channels_recovered")
    assert not any(
        ChannelKey(mic.ref.source_id, b) in e.channels for e in before_loss.health
    )
    detector.process(chunk(mic, [0.3, 0.4], start=4, channel_ids=(a,)))
    recovered = detector.advance_watermark(SourceWatermark(mic.ref, (a,), 6))
    events = _health(recovered, "channels_recovered")
    assert len(events) == 1
    span = events[0].header.provenance.spans[0]
    assert (span.interval.start_frame, span.interval.stop_frame) == (4, 6)
    assert span.channel_ids == (a,)


def test_f1_recovery_provenance_excludes_pre_loss_part_of_group(reorder_config):
    detector = AudioSalienceDetector(reorder_config)
    mic = source(detector)
    detector.process(chunk(mic, [0.1] * 6, measured_start_ns=0))
    # A report does not overwrite already admitted samples; recovery must still
    # cite only observations beyond its known loss boundary.
    detector.process(
        UnavailableObservation(
            mic.ref,
            mic.channel_ids,
            "missing",
            SourceInterval(mic.ref, 2, 4),
        )
    )
    output = detector.advance_watermark(SourceWatermark(mic.ref, mic.channel_ids, 6))
    event = _health(output, "channels_recovered")[0]
    span = event.header.provenance.spans[0]
    assert (span.interval.start_frame, span.interval.stop_frame) == (4, 6)
    assert event.header.provenance.measurement_times == (
        ClockStamp(mic.clock_id, 400_000_000),
    )


def test_f1_unknown_boundary_waits_for_explicit_resolution_or_new_epoch(reorder_config):
    detector = AudioSalienceDetector(reorder_config)
    mic = source(detector)
    detector.process(chunk(mic, [0.1, 0.1]))
    detector.process(UnavailableObservation(mic.ref, mic.channel_ids, "missing"))
    assert not _health(
        detector.advance_watermark(SourceWatermark(mic.ref, mic.channel_ids, 2)),
        "channels_recovered",
    )
    detector.process(chunk(mic, [0.2], start=2))
    assert not _health(
        detector.advance_watermark(SourceWatermark(mic.ref, mic.channel_ids, 3)),
        "channels_recovered",
    )
    restarted = mic.new_session()
    detector.reconfigure_source(restarted)
    detector.process(chunk(restarted, [0.3]))
    recovered = detector.advance_watermark(
        SourceWatermark(restarted.ref, restarted.channel_ids, 1)
    )
    assert len(_health(recovered, "channels_recovered")) == 1
    assert (
        recovered.health[0].header.provenance.spans[0].interval.source == restarted.ref
    )


def test_f1_reordered_loss_reports_do_not_lower_recovery_boundary(reorder_config):
    detector = AudioSalienceDetector(reorder_config)
    mic = source(detector)
    for stop in (6, 4):
        detector.process(
            UnavailableObservation(
                mic.ref,
                mic.channel_ids,
                "missing",
                SourceInterval(mic.ref, 2, stop),
            )
        )
    detector.process(chunk(mic, [0.1, 0.1], start=4))
    assert not _health(
        detector.advance_watermark(SourceWatermark(mic.ref, mic.channel_ids, 6)),
        "channels_recovered",
    )
    detector.process(chunk(mic, [0.1], start=6))
    assert (
        len(
            _health(
                detector.advance_watermark(
                    SourceWatermark(mic.ref, mic.channel_ids, 7)
                ),
                "channels_recovered",
            )
        )
        == 1
    )


def test_f1_unknown_loss_can_be_resolved_by_explicit_acquisition_boundary(
    reorder_config,
):
    detector = AudioSalienceDetector(reorder_config)
    mic = source(detector)
    detector.process(UnavailableObservation(mic.ref, mic.channel_ids, "missing"))
    detector.process(chunk(mic, [0.1, 0.1], start=4))
    detector.process(
        UnavailableObservation(
            mic.ref,
            mic.channel_ids,
            "missing",
            SourceInterval(mic.ref, 2, 4),
        )
    )
    recovered = detector.advance_watermark(SourceWatermark(mic.ref, mic.channel_ids, 6))
    assert len(_health(recovered, "channels_recovered")) == 1
    assert recovered.health[0].header.provenance.spans[0].interval.start_frame == 4


def test_f2_original_one_channel_invalid_duplicate_has_no_new_effect(config):
    detector = AudioSalienceDetector(config)
    mic = source(detector)
    _single_feature(detector, mic, mic.channel_ids[0], "single")
    bad = chunk(mic, [float("nan")])
    detector.process(bad)
    before = detector.feature_status("single", ClockStamp("query", 0))
    output = detector.process(bad)
    assert output.outcome == "ignored" and not output.ordered
    assert detector.feature_status("single", ClockStamp("query", 0)) == before


@pytest.mark.parametrize("delay", [0, 10_000_000_000])
def test_f2_ignored_invalid_duplicates_do_not_break_continuity(config, delay):
    detector = AudioSalienceDetector(replace(config, max_reorder_delay_ns=delay))
    mic = source(detector, channels=2)
    a, b = mic.channel_ids
    _single_feature(detector, mic, a, "a")
    _single_feature(detector, mic, b, "b")
    bad = chunk(mic, [float("nan"), 0.2])
    assert len(_health(detector.process(bad), "invalid_audio")) == 1
    before = detector.feature_status("a", ClockStamp("query", 0))
    count = detector.buffered_sample_count
    duplicate = detector.process(bad)
    assert duplicate.outcome == "ignored"
    assert not _health(duplicate, "invalid_audio")
    assert detector.feature_status("a", ClockStamp("query", 0)) == before
    assert detector.feature_status("b", ClockStamp("query", 0)).continuity_revision == 0
    assert detector.buffered_sample_count == count
    assert not detector.candidates()


def test_f2_late_unverifiable_invalid_input_is_not_a_new_break(config):
    detector = AudioSalienceDetector(config)
    mic = source(detector)
    _single_feature(detector, mic, mic.channel_ids[0], "single")
    detector.process(chunk(mic, [0.1]))
    detector.advance_watermark(SourceWatermark(mic.ref, mic.channel_ids, 20))
    before = detector.feature_status("single", ClockStamp("query", 0))
    late = detector.process(chunk(mic, [float("inf")]))
    assert "late_unverifiable" in late.input_receipts[0].reason_codes
    assert not late.ordered
    assert detector.feature_status("single", ClockStamp("query", 0)) == before


def test_f2_invalid_partial_overlap_breaks_only_for_new_tail(config):
    detector = AudioSalienceDetector(config)
    mic = source(detector)
    _single_feature(detector, mic, mic.channel_ids[0], "single")
    detector.process(chunk(mic, [float("nan")]))
    extended = chunk(mic, [float("nan"), 0.2])
    output = detector.process(extended)
    assert len(_health(output, "invalid_audio")) == 1
    span = output.health[0].header.provenance.spans[0]
    assert (span.interval.start_frame, span.interval.stop_frame) == (1, 2)
    before = detector.feature_status("single", ClockStamp("query", 0))
    assert not detector.process(extended).ordered
    assert detector.feature_status("single", ClockStamp("query", 0)) == before


def test_f3_compatibility_only_change_emits_once_and_has_accurate_outcome(config):
    detector = AudioSalienceDetector(config)
    mic = source(detector)
    before = candidate(detector, mic)
    evidence = ReadinessEvidence(
        before.candidate_id,
        1,
        False,
        False,
        (BaselineCompatibility("feature", "incompatible"),),
    )
    output = detector.report_readiness(evidence)
    assert output.outcome == "accepted"
    assert [e.event_type for e in output.baseline_lifecycle] == [
        "baseline_compatibility_changed"
    ]
    event = output.baseline_lifecycle[0]
    assert event.before == before and event.after == detector.candidates()[0]
    assert event.after.control_revision == before.control_revision
    assert event.header.provenance.source_clocks == ((mic.ref, mic.clock_id),)
    repeated = detector.report_readiness(evidence)
    assert repeated.outcome == "no_change" and not repeated.ordered
    newer = detector.report_readiness(replace(evidence, evidence_revision=2))
    assert newer.outcome == "accepted" and not newer.ordered


def test_f3_combined_compatibility_reactivation_readiness_has_unique_transitions(
    config,
):
    detector = AudioSalienceDetector(config)
    mic = source(detector)
    record = candidate(detector, mic)
    detector.command(CandidateCommand(new_id(), record.candidate_id, 0, "archive"))
    detector.command(CandidateCommand(new_id(), record.candidate_id, 1, "reactivate"))
    output = detector.report_readiness(
        ReadinessEvidence(
            record.candidate_id,
            1,
            True,
            False,
            (BaselineCompatibility("feature", "compatible"),),
        )
    )
    assert [e.event_type for e in output.baseline_lifecycle] == [
        "baseline_compatibility_changed",
        "baseline_reactivated",
        "candidate_ready",
    ]
    assert len({e.header.event_id for e in output.ordered}) == 3
    sequence = [e.header.emission_sequence for e in output.ordered]
    assert sequence == sorted(set(sequence))


@pytest.mark.parametrize(
    "start,stop,available",
    [(0, 5, True), (0, 10, True), (5, 15, False), (10, 15, False)],
)
def test_f4_delayed_windows_use_alignment_at_acquisition_time(
    config, start, stop, available
):
    detector = AudioSalienceDetector(config)
    a, b, feature = _relationship(detector, horizon=10)
    _report(detector, (a, b), feature)
    detector.open_relationship_window(
        RelationshipWindow("w", feature.feature_id, "ref", start, stop)
    )
    output = detector.advance_reference_time(ClockStamp("ref", stop + 10))
    result = output.salience[0]
    assert ("alignment_unavailable" not in result.unavailable_reasons) is available
    assert (
        result.salience_score is None
        and "not_implemented" in result.unavailable_reasons
    )
    if available:
        assert result.header.provenance.alignment_evidence_ids == ("alignment-0-1",)


@pytest.mark.parametrize("start,available", [(0, False), (5, True)])
def test_f4_window_must_not_cross_start_of_valid_evidence(config, start, available):
    detector = AudioSalienceDetector(config)
    a, b, feature = _relationship(detector)
    _report(detector, (a, b), feature, at=5, start=5, stop=100)
    detector.open_relationship_window(
        RelationshipWindow("w", feature.feature_id, "ref", start, 8)
    )
    result = detector.advance_reference_time(ClockStamp("ref", 28)).salience[0]
    assert ("alignment_unavailable" not in result.unavailable_reasons) is available


@pytest.mark.parametrize("start,available", [(0, False), (3, True)])
def test_f4_window_respects_hysteresis_qualification(config, start, available):
    detector = AudioSalienceDetector(config)
    a, b, feature = _relationship(detector, confirmation=3)
    _report(detector, (a, b), feature, at=0, stop=100)
    _report(detector, (a, b), feature, at=3, stop=100)
    detector.open_relationship_window(
        RelationshipWindow("w", feature.feature_id, "ref", start, 5)
    )
    result = detector.advance_reference_time(ClockStamp("ref", 25)).salience[0]
    assert ("alignment_unavailable" not in result.unavailable_reasons) is available


def test_f4_pending_window_preserves_closed_period_across_later_reports(config):
    detector = AudioSalienceDetector(config)
    a, b, feature = _relationship(detector, horizon=1000)
    _report(detector, (a, b), feature, stop=1000)
    detector.open_relationship_window(
        RelationshipWindow("old", feature.feature_id, "ref", 0, 1)
    )
    for at in range(2, 40):
        _report(
            detector,
            (a, b),
            feature,
            at=at,
            start=at,
            stop=1000,
            error=20 if at % 2 == 0 else 1,
        )
        assert len(detector._window_alignment) <= config.limits.max_relationship_windows
        assert (
            sum(s.period is not None for s in detector._sync.values())
            <= config.limits.max_features
        )
    result = detector.advance_reference_time(ClockStamp("ref", 1001)).salience[0]
    assert "alignment_unavailable" not in result.unavailable_reasons
    assert result.header.provenance.alignment_evidence_ids == ("alignment-0-1",)
    assert not detector._window_alignment


def test_f4_later_loss_invalidates_only_the_overlapping_window(config):
    detector = AudioSalienceDetector(config)
    a, b, feature = _relationship(detector)
    _report(detector, (a, b), feature, stop=100)
    for name, stop in (("before", 3), ("crossing", 5)):
        detector.open_relationship_window(
            RelationshipWindow(name, feature.feature_id, "ref", 0, stop)
        )
    _report(detector, (a, b), feature, at=4, stop=100, error=None)
    output = detector.advance_reference_time(ClockStamp("ref", 25))
    before, crossing = output.salience
    assert "alignment_unavailable" not in before.unavailable_reasons
    assert "alignment_unavailable" in crossing.unavailable_reasons
    assert not detector._windows and not detector._window_alignment


def test_f4_window_opened_after_expiry_uses_retained_period(config):
    detector = AudioSalienceDetector(config)
    a, b, feature = _relationship(detector)
    _report(detector, (a, b), feature)
    detector.advance_reference_time(ClockStamp("ref", 10))
    detector.open_relationship_window(
        RelationshipWindow("w", feature.feature_id, "ref", 0, 5)
    )
    _report(detector, (a, b), feature, at=11, start=11, stop=100)
    result = detector.advance_reference_time(ClockStamp("ref", 25)).salience[0]
    assert "alignment_unavailable" not in result.unavailable_reasons
    assert result.header.provenance.alignment_evidence_ids == ("alignment-0-1",)


def test_f5_capacity_release_ends_only_related_episode_once(config):
    config = replace(config, limits=replace(config.limits, max_relationship_windows=1))
    detector = AudioSalienceDetector(config)
    a, _, feature = _relationship(detector)
    oversized = detector.process(
        chunk(a, [0.1] * (config.limits.max_chunk_bytes // 4 + 1))
    )
    audio_episode = oversized.refusals[0].resource_episode_id
    detector.open_relationship_window(
        RelationshipWindow("w1", feature.feature_id, "ref", 0, 5)
    )
    refused = detector.open_relationship_window(
        RelationshipWindow("w2", feature.feature_id, "ref", 0, 5)
    )
    window_episode = refused.refusals[0].resource_episode_id
    freed = detector.advance_reference_time(ClockStamp("ref", 25))
    ends = [
        e.resource for e in freed.health if e.resource and e.resource.phase == "end"
    ]
    assert [e.episode_id for e in ends] == [window_episode]
    admitted = detector.open_relationship_window(
        RelationshipWindow("w3", feature.feature_id, "ref", 30, 35)
    )
    assert admitted.outcome == "accepted" and not admitted.health
    assert not detector.advance_reference_time(ClockStamp("ref", 26)).health
    again = detector.open_relationship_window(
        RelationshipWindow("w4", feature.feature_id, "ref", 30, 35)
    )
    assert again.health[0].resource.phase == "first"
    assert again.refusals[0].resource_episode_id != window_episode
    valid_audio = detector.process(chunk(a, [0.1]))
    assert [e.resource.episode_id for e in valid_audio.health if e.resource] == [
        audio_episode
    ]


def test_f5_reset_releases_window_capacity_without_repeated_end(config):
    config = replace(config, limits=replace(config.limits, max_relationship_windows=1))
    detector = AudioSalienceDetector(config)
    a, b, feature = _relationship(detector)
    detector.open_relationship_window(
        RelationshipWindow("w", feature.feature_id, "ref", 0, 5)
    )
    refused = detector.open_relationship_window(
        RelationshipWindow("overflow", feature.feature_id, "ref", 0, 5)
    )
    detector.end_source(a.ref)
    detector.end_source(b.ref)
    output = detector.reset_learning()
    ended = [e.resource for e in output.health if e.resource]
    assert len(ended) == 1 and ended[0].phase == "end"
    assert ended[0].episode_id == refused.refusals[0].resource_episode_id
    assert not detector.reset_learning().health
    assert not detector._windows and not detector._window_alignment
