"""Candidate storage saturation must preserve valid streaming perception."""

from dataclasses import asdict, replace

import pytest

from far_audio_salience import AcousticConfig, CandidateCommand, ChannelKey

from .test_acoustic import changes, np, setup, tone

SKIP = ("candidate_allocation_skipped",)
WARM = np.concatenate([np.zeros(800), tone(0.35, frequency=200)]).astype(np.float32)
NOVEL = np.concatenate([np.zeros(800), tone(0.35, frequency=1000)]).astype(np.float32)


def limited(config, **acoustic):
    limits = replace(config.limits, max_baselines=1)
    return setup(
        replace(config, limits=limits),
        acoustic=AcousticConfig(
            candidate_seconds=0.02,
            candidate_ready_seconds=0.06,
            maturity_seconds=0.02,
            **acoustic,
        ),
    )


def submit(detector, source, pcm, *, first=0, sizes=(80,)):
    offset = 0
    index = 0
    batches = []
    while offset < len(pcm):
        stop = min(offset + sizes[index % len(sizes)], len(pcm))
        batch = detector.process(
            source.chunk(memoryview(pcm[offset:stop]), first_frame=first + offset)
        )
        assert batch.outcome == "accepted" and not batch.refusals
        (receipt,) = batch.input_receipts
        (span,) = receipt.accepted
        assert (span.interval.start_frame, span.interval.stop_frame) == (
            first + offset,
            first + stop,
        )
        assert batch.finalized[0].before_frame == first + stop
        batches.append(batch)
        index += 1
        offset = stop
    return batches


def skips(batches):
    return [e for b in batches for e in b.health if e.reason_codes == SKIP]


def test_saturation_preserves_admission_salience_learning_and_block(
    config, monkeypatch
):
    from far_audio_salience import _acoustic

    detector, source = limited(config)
    submit(detector, source, WARM)
    (record,) = detector.candidates()
    assert (
        detector.command(
            CandidateCommand("block", record.candidate_id, 0, "block")
        ).outcome
        == "accepted"
    )
    (blocked_record,) = detector.candidates()
    batches = submit(detector, source, NOVEL, first=len(WARM))
    health = skips(batches)
    episode = health[0].resource
    assert health[0].event_type == "capacity_exhausted"
    assert (episode.phase, episode.count, episode.hit.resource_id) == (
        "first",
        1,
        "max_baselines",
    )
    assert (episode.hit.observed, episode.hit.limit) == (2, 1)
    assert health[0].channels == blocked_record.channels
    assert health[0].header.provenance.spans
    assert health[1].resource.phase == "update"
    assert health[1].resource.count == 1 + config.reporting.aggregate_every
    assert {event.resource.episode_id for event in health} == {episode.episode_id}
    assert any(
        c.kind == "sound_onset"
        and c.phase == "detected"
        and c.confirmed_frame > len(WARM) + 800
        for c in changes([r for b in batches for r in b.salience])
    )
    assert any(b.salience and b.health and skips([b]) for b in batches)
    assert detector.candidates() == (blocked_record,)
    assert detector._baselines.administration == {record.candidate_id: (1, True)}

    # The steady unmatched window needs no change ID or candidate ID.
    def unexpected_id():
        pytest.fail("A skipped candidate allocation consumed an acoustic ID")

    first = len(WARM) + len(NOVEL)
    continued = tone(0.01, frequency=1000).astype(np.float32)
    with monkeypatch.context() as patch:
        patch.setattr(_acoustic, "new_id", unexpected_id)
        submit(detector, source, continued, first=first)

    key = blocked_record.channels[0]
    stats = detector._acoustic.candidates[record.candidate_id]["stats"]
    candidate_before = [s.exposure for s in stats.values()]
    matched = tone(0.35, frequency=200).astype(np.float32)
    submit(detector, source, matched, first=first + len(continued))
    assert all(
        s.exposure > had
        for s, had in zip(
            detector._acoustic.candidates[record.candidate_id]["stats"].values(),
            candidate_before,
            strict=True,
        )
    )
    assert detector.candidates() == (blocked_record,)
    primary_before = next(iter(detector._acoustic.models[key].values())).exposure
    submit(
        detector,
        source,
        np.zeros(2400, dtype=np.float32),
        first=first + len(continued) + len(matched),
    )
    assert next(iter(detector._acoustic.models[key].values())).exposure > primary_before


def test_full_metadata_store_leaves_missing_initial_assessment_unavailable(config):
    detector, source = limited(config)
    scope = (ChannelKey(source.ref.source_id, source.channel_ids[0]),)
    detector.register_candidate(scope)
    (batch,) = submit(detector, source, np.zeros(160, dtype=np.float32), sizes=(160,))
    (result,) = batch.salience
    assert result.windows
    assert result.acoustic_deviation is None
    assert result.salience_score is None and result.is_salient is None
    assert result.unavailable_reasons == ("component_unavailable",)


def test_capacity_episode_separate_from_registration_and_ends_on_authorized_reset(
    config,
):
    detector, source = limited(config)
    submit(detector, source, WARM)
    old_id = detector.candidates()[0].candidate_id
    health = skips(submit(detector, source, NOVEL, first=len(WARM)))
    episode = health[0].resource.episode_id
    refused = detector.register_candidate(detector.candidates()[0].channels)
    assert refused.outcome == "refused"
    assert refused.refusals[0].resource_episode_id != episode
    assert refused.health[0].event_type == "resource_limit_violation"
    detector.end_source(source.ref)
    assert not skips([detector.reset_learning()])
    assert old_id in detector._baselines.administration
    (ended,) = skips([detector.factory_reset()])
    assert ended.resource.phase == "end" and ended.resource.episode_id == episode
    assert not detector._baselines.administration
    fresh = source.new_session()
    assert detector.start_source(fresh).outcome == "accepted"
    assert not skips(submit(detector, fresh, WARM))
    assert detector.candidates()[0].candidate_id != old_id


def test_staged_capacity_skip_rolls_back_on_processing_limit(config):
    detector, source = limited(config, max_windows_per_call=1)
    submit(detector, source, WARM, sizes=(40,))
    submit(detector, source, NOVEL, first=len(WARM), sizes=(40,))
    before = asdict(detector._episodes[SKIP[0]])
    (record,) = detector.candidates()
    state = detector._acoustic
    cursor = len(WARM) + len(NOVEL)
    samples = tone(0.01, frequency=1000).astype(np.float32)
    refusal = detector.process(source.chunk(memoryview(samples), first_frame=cursor))
    assert refusal.refusals[0].resource.resource_id == "acoustic_windows_per_call"
    assert not skips([refusal])
    assert not refusal.salience and not refusal.baseline_lifecycle
    assert not refusal.input_receipts and not refusal.finalized
    assert detector._acoustic is state
    assert detector.candidates() == (record,)
    assert detector._lanes[record.channels[0]].watermark == cursor
    assert asdict(detector._episodes[SKIP[0]]) == before
    submit(detector, source, samples, first=cursor, sizes=(40,))
    assert detector._episodes[SKIP[0]].count == before["count"] + 2


def test_capacity_outcomes_equivalent_across_chunk_partitions(config):
    pcm = np.concatenate([WARM, NOVEL])
    summaries = []
    for sizes in ((80,), (1, 137, 23, 400)):
        detector, source = limited(config)
        batches = submit(detector, source, pcm, sizes=sizes)
        salience = [r for b in batches for r in b.salience]
        summaries.append(
            (
                [(w.interval.stop_frame, w.power) for r in salience for w in r.windows],
                [
                    (c.kind, c.start_frame, c.confirmed_frame, c.phase, c.revision)
                    for c in changes(salience)
                ],
                [
                    (
                        e.resource.phase,
                        e.resource.count,
                        e.header.provenance.spans[0].interval.stop_frame,
                    )
                    for e in skips(batches)
                ],
                len(detector.candidates()),
            )
        )
    assert summaries[0] == summaries[1]
    assert summaries[0][2] and summaries[0][3] == 1
