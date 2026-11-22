from dataclasses import replace

import pytest

from far_audio_salience import (
    AlignmentEvidence,
    AlignmentPolicy,
    AudioSalienceDetector,
    ChannelKey,
    ClockMapping,
    ClockStamp,
    EligibilityPolicy,
    FeatureSpec,
    RelationshipWindow,
    StructuralInputError,
    evaluate_eligibility,
    feature_eligibility,
)

from .test_streaming_contract import chunk, source, state


def setup_features(detector):
    a, b = source(detector, label="a"), source(detector, label="b", rate=17)
    channels = (
        ChannelKey(a.ref.source_id, a.channel_ids[0]),
        ChannelKey(b.ref.source_id, b.channel_ids[0]),
    )
    assessment, learning = EligibilityPolicy(0.5), EligibilityPolicy(1.0)
    spec = FeatureSpec(
        "relationship",
        channels,
        assessment,
        learning,
        learning,
        True,
        AlignmentPolicy(10, 4, 3),
        20,
    )
    detector.register_feature(spec)
    detector.register_feature(
        FeatureSpec("independent", channels[:1], assessment, learning, learning, False)
    )
    return a, b, spec


def evidence(a, b, spec, at, error, stop=100):
    return AlignmentEvidence(
        f"evidence-{at}",
        spec.channels,
        "external-reference",
        0,
        stop,
        ClockStamp("external-reference", at),
        error,
        tuple(ClockMapping(s.ref, s.clock_id, 0, 0, 1, 1) for s in (a, b)),
    )


def test_union_coverage_nominal_duration_and_separate_learning():
    policy = EligibilityPolicy(0.8)
    result = evaluate_eligibility(
        start_ns=0,
        stop_ns=100,
        segments=((0, 30), (20, 40), (60, 100)),
        policy=policy,
        continuous_only=False,
    )
    assert result.valid_duration_ns == 80
    assert result.valid_coverage == 0.8 and result.eligible
    spec = FeatureSpec(
        "test", (), policy, EligibilityPolicy(1.0), EligibilityPolicy(0.9), False
    )
    decisions = feature_eligibility(
        spec, start_ns=0, stop_ns=100, segments=((0, 40), (60, 100))
    )
    assert decisions.assessment.eligible
    assert not decisions.established_learning.eligible
    assert not decisions.candidate_learning.eligible


def test_continuous_features_do_not_join_across_breaks():
    decision = evaluate_eligibility(
        start_ns=0,
        stop_ns=100,
        segments=((0, 40), (60, 100)),
        policy=EligibilityPolicy(0.5),
        continuous_only=True,
    )
    assert decision.valid_duration_ns == 40 and not decision.eligible
    fragmented = evaluate_eligibility(
        start_ns=0,
        stop_ns=100,
        segments=((0, 5), (10, 15)),
        policy=EligibilityPolicy(0.1, 6),
        continuous_only=False,
    )
    assert "insufficient_contiguous_duration" in fragmented.reason_codes


def test_alignment_unknown_hysteresis_expiry_and_atomicity(config):
    detector = AudioSalienceDetector(config)
    a, b, spec = setup_features(detector)
    assert not detector.alignment_available(
        spec.feature_id, ClockStamp("external-reference", 0)
    )
    assert detector.alignment_available("independent", ClockStamp("anything", 0))
    assert not detector.update_alignment(evidence(a, b, spec, 1, 2)).health
    recovered = detector.update_alignment(evidence(a, b, spec, 4, 2))
    assert recovered.health[0].event_type == "alignment_recovered"
    assert detector.alignment_available(
        spec.feature_id, ClockStamp("external-reference", 5)
    )
    assert not detector.update_alignment(evidence(a, b, spec, 5, 7)).health
    lost = detector.update_alignment(evidence(a, b, spec, 6, None))
    assert lost.health[0].event_type == "alignment_lost"
    detector.update_alignment(evidence(a, b, spec, 7, 2))
    assert detector.update_alignment(evidence(a, b, spec, 10, 2)).health
    expired = detector.advance_reference_time(ClockStamp("external-reference", 100))
    assert expired.health[0].reason_codes == ("alignment_expired",)
    assert detector.process(chunk(a, [0.1])).salience  # B never supplied audio.
    before = state(detector)
    with pytest.raises(StructuralInputError):
        detector.update_alignment(replace(evidence(a, b, spec, 11, 1), mappings=()))
    assert state(detector) == before


def test_bounded_explicit_relationship_deadline_does_not_hold_channels(config):
    detector = AudioSalienceDetector(config)
    a, _, spec = setup_features(detector)
    detector.open_relationship_window(
        RelationshipWindow("w", spec.feature_id, "ref", 0, 10)
    )
    assert detector.process(chunk(a, [0.1])).salience
    assert not detector.advance_reference_time(ClockStamp("ref", 29)).salience
    final = detector.advance_reference_time(ClockStamp("ref", 30))
    assert len(final.salience) == 1
    assert "not_implemented" in final.salience[0].unavailable_reasons
    assert "alignment_unavailable" in final.salience[0].unavailable_reasons
    assert final.salience[0].salience_score is None
    assert not detector._windows


def test_window_capacity_refusal_keeps_pending_windows(config):
    detector = AudioSalienceDetector(config)
    _, _, spec = setup_features(detector)
    for i in range(config.limits.max_relationship_windows):
        detector.open_relationship_window(
            RelationshipWindow(str(i), spec.feature_id, "ref", 0, 10)
        )
    before = dict(detector._windows)
    refusal = detector.open_relationship_window(
        RelationshipWindow("overflow", spec.feature_id, "ref", 0, 10)
    )
    assert refusal.refusals[0].code == "temporary_capacity"
    assert detector._windows == before


def test_closed_relationship_does_not_invent_reference_time(config):
    detector = AudioSalienceDetector(config)
    _, _, spec = setup_features(detector)
    detector.open_relationship_window(
        RelationshipWindow("w", spec.feature_id, "ref", 0, 10)
    )
    output = detector.close()
    unavailable = [
        r
        for r in output.salience
        if "source_closed_before_finalization" in r.unavailable_reasons
    ]
    assert len(unavailable) == 1 and unavailable[0].timestamp is None
    assert detector._reference_times == {}


def test_finalized_relationship_cannot_reopen_after_retention_cleanup(config):
    detector = AudioSalienceDetector(config)
    _, _, spec = setup_features(detector)
    window = RelationshipWindow("w", spec.feature_id, "ref", 0, 10)
    detector.open_relationship_window(window)
    detector.advance_reference_time(ClockStamp("ref", 30))
    before = state(detector)
    with pytest.raises(StructuralInputError):
        detector.open_relationship_window(window)
    assert state(detector) == before


def test_sync_continuity_break_is_feature_local(config):
    detector = AudioSalienceDetector(config)
    a, b, spec = setup_features(detector)
    detector.update_alignment(evidence(a, b, spec, 1, 2))
    detector.update_alignment(evidence(a, b, spec, 4, 2))
    before = detector.feature_status("independent", ClockStamp("external-reference", 4))
    detector.update_alignment(evidence(a, b, spec, 5, 20))
    after = detector.feature_status("independent", ClockStamp("external-reference", 5))
    assert after.continuity_revision == before.continuity_revision
    assert after.alignment_available
    assert (
        detector.feature_status(
            "relationship", ClockStamp("external-reference", 5)
        ).continuity_revision
        == 1
    )
