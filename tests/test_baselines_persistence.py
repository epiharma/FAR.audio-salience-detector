import json
from dataclasses import replace

import pytest

from far_audio_salience import (
    AudioSalienceDetector,
    BaselineCompatibility,
    CandidateCommand,
    ChannelKey,
    MicrophoneGeometry,
    MicrophonePosition,
    ReadinessEvidence,
    SnapshotError,
    StateError,
    new_id,
)

from .test_streaming_contract import chunk, source


def candidate(detector, mic, compatibility=None):
    output = detector.register_candidate(
        (ChannelKey(mic.ref.source_id, mic.channel_ids[0]),),
        compatibility=(BaselineCompatibility("feature", "compatible"),)
        if compatibility is None
        else compatibility,
    )
    return output.baseline_lifecycle[0].after


def command(record, action, *, command_id=None):
    return CandidateCommand(
        command_id or new_id(),
        record.candidate_id,
        record.control_revision,
        action,
        record.proposal_id,
    )


def ready(detector, record, revision=1, material=False):
    return detector.report_readiness(
        ReadinessEvidence(
            record.candidate_id,
            revision,
            True,
            material,
            (BaselineCompatibility("feature", "compatible"),),
        )
    )


def test_parallel_candidate_block_reject_and_promotion(config):
    detector = AudioSalienceDetector(config)
    mic = source(detector)
    a, b = candidate(detector, mic), candidate(detector, mic)
    blocked = detector.command(command(a, "block"))
    assert blocked.baseline_lifecycle[0].event_type == "promotion_blocked"
    ready(detector, a)
    a, b = detector.candidates()
    assert a.promotion_blocked and not b.promotion_blocked
    assert (
        detector.command(command(a, "approve")).refusals[0].code == "promotion_blocked"
    )
    detector.command(command(a, "unblock"))
    a = detector.candidates()[0]
    promoted = detector.command(command(a, "approve"))
    assert promoted.baseline_lifecycle[0].after.role == "established"
    ready(detector, b)
    b = detector.candidates()[1]
    rejected = detector.command(command(b, "reject"))
    assert rejected.baseline_lifecycle[0].event_type == "promotion_rejected"
    assert not ready(detector, b, revision=2).baseline_lifecycle
    assert ready(detector, b, revision=3, material=True).baseline_lifecycle
    assert len(detector.candidates()) == 2  # No deletion/merging.


def test_command_idempotency_conflict_stale_and_restart(config, tmp_path):
    detector = AudioSalienceDetector(config)
    mic = source(detector)
    record = candidate(detector, mic)
    block = command(record, "block")
    original = detector.command(block)
    repeated = detector.command(block)
    assert repeated.command_receipt == original.command_receipt
    assert not repeated.ordered
    conflict = detector.command(replace(block, action="unblock"))
    assert conflict.refusals[0].code == "command_id_conflict"
    assert (
        detector.command(command(record, "archive")).refusals[0].code
        == "stale_control_revision"
    )
    path = tmp_path / "state.json"
    detector.save_baselines(path)
    restored = AudioSalienceDetector.from_baselines(path, config=config)
    assert restored.run_id != detector.run_id
    assert restored.candidates() == detector.candidates()
    assert restored.command(block).command_receipt == original.command_receipt
    assert (
        restored.command(command(record, "unblock")).refusals[0].code
        == "stale_control_revision"
    )
    assert restored.sources == detector.sources
    assert not restored._active and restored.buffered_sample_count == 0


def test_receipt_expiry_still_guards_stale_commands(config, tmp_path):
    detector = AudioSalienceDetector(config)
    mic = source(detector)
    original = command(candidate(detector, mic), "block")
    detector.command(original)
    for i in range(6):
        detector.command(
            command(detector.candidates()[0], "unblock" if i % 2 == 0 else "block")
        )
    assert len(detector._baselines.receipts) == config.limits.max_receipts
    path = tmp_path / "state.json"
    detector.save_baselines(path)
    restored = AudioSalienceDetector.from_baselines(path, config=config)
    assert restored.command(original).refusals[0].code == "stale_control_revision"


def test_archive_reactivate_and_automatic_extension_readiness(config):
    detector = AudioSalienceDetector(replace(config, promotion_mode="automatic"))
    mic = source(detector)
    a = candidate(detector, mic)
    detector.command(command(a, "archive"))
    archived = detector.candidates()[0]
    assert archived.activity == "archived"
    detector.command(command(archived, "reactivate"))
    assert detector.candidates()[0].activity == "reactivating"
    output = ready(detector, detector.candidates()[0])
    assert [e.event_type for e in output.baseline_lifecycle] == [
        "baseline_compatibility_changed",
        "baseline_reactivated",
        "candidate_ready",
        "candidate_promoted",
    ]
    assert detector.candidates()[0].baseline_id == a.baseline_id


def test_both_resets_preserve_saved_files_and_control_tombstones(config, tmp_path):
    detector = AudioSalienceDetector(config)
    mic = source(detector)
    a = candidate(detector, mic)
    detector.command(command(a, "block"))
    with pytest.raises(StateError):
        detector.reset_learning()
    with pytest.raises(StateError):
        detector.factory_reset()
    detector.end_source(mic.ref)
    path = tmp_path / "state.json"
    detector.save_baselines(path)
    original_bytes = path.read_bytes()
    detector.reset_learning()
    assert not detector.candidates() and detector.sources
    assert detector._baselines.administration[a.candidate_id][1]
    assert detector._baselines.receipts
    assert path.read_bytes() == original_bytes
    detector.save_baselines(tmp_path / "reset.json")
    restored = AudioSalienceDetector.from_baselines(
        tmp_path / "reset.json", config=config
    )
    assert restored._baselines.administration[a.candidate_id][1]
    result = restored.command(CandidateCommand(new_id(), a.candidate_id, 1, "unblock"))
    assert not result.refusals
    detector.factory_reset()
    assert not detector.sources and not detector._baselines.administration
    assert not detector._baselines.receipts
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "damage", ["version", "duplicate", "role", "revision", "json", "oversize"]
)
def test_snapshot_rejects_corruption_and_limits(config, tmp_path, damage):
    detector = AudioSalienceDetector(config)
    mic = source(detector)
    candidate(detector, mic)
    path = tmp_path / "state.json"
    detector.save_baselines(path)
    payload = json.loads(path.read_text())
    if damage == "version":
        payload["version"] = 999
    elif damage == "duplicate":
        payload["sources"].append(payload["sources"][0])
    elif damage == "role":
        payload["baselines"][0]["role"] = "invented"
    elif damage == "revision":
        payload["baselines"][0]["control_revision"] = -1
    if damage == "json":
        path.write_text('{"version":1,"version":1}')
    elif damage == "oversize":
        path.write_bytes(b" " * (config.limits.max_snapshot_bytes + 1))
    else:
        path.write_text(json.dumps(payload))
    with pytest.raises(SnapshotError):
        AudioSalienceDetector.from_baselines(path, config=config)


def test_snapshot_write_limit_does_not_overwrite_previous(config, tmp_path):
    detector = AudioSalienceDetector(config)
    source(detector)
    path = tmp_path / "snapshot.json"
    detector.save_baselines(path)
    old = path.read_bytes()
    detector.config = replace(
        config, limits=replace(config.limits, max_snapshot_bytes=20)
    )
    with pytest.raises(SnapshotError):
        detector.save_baselines(path)
    assert path.read_bytes() == old


def test_geometry_and_rate_preserve_individual_compatibility(config):
    detector = AudioSalienceDetector(config)
    mic = source(detector)
    record = candidate(
        detector,
        mic,
        (
            BaselineCompatibility("individual", "compatible"),
            BaselineCompatibility("spatial", "compatible", True),
        ),
    )
    key = record.channels[0]
    detector.update_geometry(
        MicrophoneGeometry("g", "room", (MicrophonePosition(key, (0.0, 0.0, 0.0)),))
    )
    compatibility = detector.candidates()[0].compatibility
    assert compatibility[0].status == "compatible"
    assert compatibility[1].status == "revalidate"
    detector.process(chunk(mic, [0.1]))
    rate_change = detector.reconfigure_source(mic.new_session(sample_rate_hz=17))
    assert any(e.event_type == "sample_rate_changed" for e in rate_change.health)
    assert detector.candidates()[0].baseline_id == record.baseline_id
    assert all(c.status == "unknown" for c in detector.candidates()[0].compatibility)


def test_removed_channel_identity_and_baseline_survive_snapshot(config, tmp_path):
    detector = AudioSalienceDetector(config)
    mic = source(detector, channels=2)
    second = ChannelKey(mic.ref.source_id, mic.channel_ids[1])
    detector.register_candidate((second,))
    original = detector.candidates()[0]
    replacement = replace(mic.new_session(), channel_ids=(mic.channel_ids[0],))
    detector.reconfigure_source(replacement)
    path = tmp_path / "removed-channel.json"
    detector.save_baselines(path)
    restored = AudioSalienceDetector.from_baselines(path, config=config)
    assert restored.candidates()[0].candidate_id == original.candidate_id
    assert second.channel_id in restored._identities[second.source_id]
    recovered = replace(replacement.new_session(), channel_ids=mic.channel_ids)
    assert not restored.start_source(recovered).refusals
    assert restored.candidates()[0].compatibility == original.compatibility


def test_explicit_review_after_rejection_without_faking_new_stability(config):
    detector = AudioSalienceDetector(config)
    mic = source(detector)
    record = candidate(detector, mic)
    ready(detector, record)
    record = detector.candidates()[0]
    detector.command(command(record, "reject"))
    output = detector.report_readiness(
        ReadinessEvidence(
            record.candidate_id,
            1,
            True,
            False,
            record.compatibility,
            explicit_review=True,
        )
    )
    assert output.baseline_lifecycle[0].event_type == "candidate_ready"
    assert output.baseline_lifecycle[0].after.proposal_id != record.proposal_id


def test_baseline_events_have_source_clock_provenance(config):
    detector = AudioSalienceDetector(config)
    mic = source(detector)
    event = detector.register_candidate(
        (ChannelKey(mic.ref.source_id, mic.channel_ids[0]),)
    ).baseline_lifecycle[0]
    assert event.header.provenance.source_clocks == ((mic.ref, mic.clock_id),)


@pytest.mark.parametrize("remove_channel", [False, True])
def test_late_old_epoch_preserves_new_source_and_learning(config, remove_channel):
    from copy import deepcopy

    detector = AudioSalienceDetector(config)
    mic = source(detector, channels=2)
    old = chunk(mic, [0.1, 0.2], measured_start_ns=0)
    record = candidate(detector, mic)
    replacement = mic.new_session(sample_rate_hz=17)
    if remove_channel:
        replacement = replace(replacement, channel_ids=(mic.channel_ids[0],))
    detector.reconfigure_source(replacement)
    before = deepcopy(detector._lanes)
    baselines = detector.candidates()
    output = detector.process(old)
    assert output.outcome == "ignored"
    assert "late_unverifiable" in output.input_receipts[0].reason_codes
    assert not output.ordered
    assert detector._lanes == before and detector.candidates() == baselines
    assert detector.candidates()[0].candidate_id == record.candidate_id
