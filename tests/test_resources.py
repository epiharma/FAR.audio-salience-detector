from dataclasses import replace

from far_audio_salience import AudioSalienceDetector, SourceWatermark

from .test_streaming_contract import chunk, source


def test_hard_refusal_first_repeat_worse_aggregate_end(config):
    config = replace(config, limits=replace(config.limits, max_chunk_bytes=8))
    detector = AudioSalienceDetector(config)
    mic = source(detector)
    record = detector.process(chunk(mic, [0.1] * 2))
    before = detector.buffered_sample_count
    first = detector.process(chunk(mic, [0.1] * 3, start=2))
    assert first.refusals[0].code == "hard_limit"
    assert first.health[0].resource.phase == "first"
    assert first.health[0].resource.hit.exceedance == 4
    episode_id = first.refusals[0].resource_episode_id
    second = detector.process(chunk(mic, [0.1] * 3, start=2))
    assert not second.health
    assert second.refusals[0].resource_episode_id == episode_id
    worse = detector.process(chunk(mic, [0.1] * 8, start=2))
    assert worse.health[0].resource.phase == "worse"
    for _ in range(config.reporting.aggregate_every):
        aggregate = detector.process(chunk(mic, [0.1] * 8, start=2))
    assert aggregate.health[0].resource.phase == "update"
    assert detector.buffered_sample_count == before
    assert record.salience[0].salience_score is None
    ended = detector.process(chunk(mic, [0.1] * 2, start=2))
    assert any(e.resource and e.resource.phase == "end" for e in ended.health)
    assert not detector._episodes
    assert len(
        {
            e.header.event_id
            for batch in (first, worse, aggregate, ended)
            for e in batch.ordered
        }
    ) == sum(len(batch.ordered) for batch in (first, worse, aggregate, ended))


def test_temporary_capacity_keeps_reorder_state_and_can_resume(reorder_config):
    config = replace(
        reorder_config,
        limits=replace(reorder_config.limits, max_buffer_bytes=8),
        comparison_retention_ns=0,
    )
    detector = AudioSalienceDetector(config)
    mic = source(detector)
    detector.process(chunk(mic, [0.1] * 2))
    refused = detector.process(chunk(mic, [0.2], start=2))
    assert refused.refusals[0].code == "temporary_capacity"
    assert detector.buffered_sample_count == 2
    assert detector._lanes[next(iter(detector._lanes))].high_water == 2
    detector.advance_watermark(SourceWatermark(mic.ref, mic.channel_ids, 2))
    assert detector.buffered_sample_count == 0
    assert detector.process(chunk(mic, [0.2], start=2)).input_receipts[0].accepted


def test_candidate_limit_does_not_evict_experience(config):
    config = replace(config, limits=replace(config.limits, max_baselines=1))
    detector = AudioSalienceDetector(config)
    mic = source(detector)
    from far_audio_salience import ChannelKey

    scope = (ChannelKey(mic.ref.source_id, mic.channel_ids[0]),)
    detector.register_candidate(scope)
    before = detector.candidates()
    refused = detector.register_candidate(scope)
    assert refused.refusals[0].resource.resource_id == "max_baselines"
    assert detector.candidates() == before
    assert refused.health
