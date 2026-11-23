"""Numerical and streaming regressions for opt-in DESIGN-003 DSP."""

import json
from dataclasses import asdict, replace

import pytest

from far_audio_salience import (
    AcousticConfig,
    AudioSalienceDetector,
    FrequencyBand,
    SourceSpec,
    SourceWatermark,
    UnavailableObservation,
)
from far_audio_salience._acoustic_features import SpectralPlan

np = pytest.importorskip("numpy")


def setup(config, *, rate=8000, channels=1, acoustic=None, **kwargs):
    cfg = replace(
        config,
        acoustic=acoustic or AcousticConfig(),
        comparison_retention_ns=0,
        **kwargs,
    )
    d = AudioSalienceDetector(cfg)
    s = SourceSpec.microphone(
        label="fixture", sample_rate_hz=rate, channel_count=channels
    )
    assert d.start_source(s).outcome == "accepted"
    return d, s


def feed(d, s, samples, sizes=(80,)):
    x = np.asarray(samples, dtype=np.float32)
    results = []
    first = 0
    index = 0
    while first < len(x):
        n = sizes[index % len(sizes)]
        block = np.ascontiguousarray(x[first : first + n]).ravel()
        batch = d.process(s.chunk(memoryview(block), first_frame=first))
        assert not batch.refusals, batch.refusals
        results.extend(batch.salience)
        first += len(block) // len(s.channel_ids)
        index += 1
    return results


def tone(seconds, rate=8000, amplitude=0.2, frequency=1000):
    return amplitude * np.sin(
        2 * np.pi * frequency * np.arange(round(seconds * rate)) / rate
    )


def changes(results, kind=None):
    return [c for r in results for c in r.changes if kind is None or c.kind == kind]


def test_contract_defaults_serializable(config):
    d, s = setup(config)
    r = next(r for r in feed(d, s, np.zeros(800)) if r.windows)
    assert json.loads(json.dumps(asdict(r)))["changes"] == []
    assert r.pattern_novelty is None
    assert r.salience_score is None
    assert r.windows


@pytest.mark.parametrize("rate", [8000, 44100, 48000])
def test_fft_calibrated_tone_and_noise(rate):
    cfg = AcousticConfig(bands=(FrequencyBand("all", 0, rate / 2),))
    plan = SpectralPlan.create(np, cfg, rate)
    x = 0.2 * np.sin(2 * np.pi * 1000 * np.arange(plan.size) / rate)
    power, peak, bands, *_ = plan.extract(x)
    assert power == pytest.approx(0.02, rel=0.001)
    assert bands["all"][0] == pytest.approx(0.02, rel=0.001)
    assert peak <= 0.201
    rng = np.random.default_rng(83)
    estimates = [
        plan.extract(rng.normal(0, 0.1, plan.size))[2]["all"][0] for _ in range(150)
    ]
    assert np.mean(estimates) == pytest.approx(0.01, rel=0.05)


def test_partition_layout_and_zero_history(config):
    x = np.column_stack([tone(0.7), tone(0.7, frequency=200)])
    d, s = setup(config, channels=2)
    a = feed(d, s, x, (80,))
    d2, s2 = setup(config, channels=2)
    b = feed(d2, s2, x, (1, 137, 23, 400))

    def values(rs):
        return [
            (r.windows[0].interval.start_frame, r.windows[0].power)
            for r in rs
            if r.windows
        ]

    # Group by channel: cross-channel emission interleaving follows each call.
    for n in range(2):
        aa = [r for r in a if r.channels[0].channel_id == s.channel_ids[n]]
        bb = [r for r in b if r.channels[0].channel_id == s2.channel_ids[n]]
        assert values(aa) == values(bb)
    assert all(len(lane.history) == 0 for lane in d._lanes.values())
    d3, s3 = setup(config, channels=2)
    out = []
    for first in range(0, len(x), 80):
        planes = tuple(
            memoryview(np.ascontiguousarray(x[first : first + 80, n], dtype=np.float32))
            for n in range(2)
        )
        out.extend(
            d3.process(s3.chunk(planes, first_frame=first, layout="planar")).salience
        )
    for n in range(2):
        assert values(
            [r for r in a if r.channels[0].channel_id == s.channel_ids[n]]
        ) == values([r for r in out if r.channels[0].channel_id == s3.channel_ids[n]])


def test_sustained_onset_offset_revisions(config):
    d, s = setup(config)
    rs = feed(d, s, np.concatenate([np.zeros(2400), tone(0.4), np.zeros(2400)]))
    on = changes(rs, "sound_onset")
    off = changes(rs, "sound_offset")
    assert len({c.change_id for c in on}) == 1
    assert len({c.change_id for c in off}) == 1
    assert on[0].phase == "detected" and on[-1].phase == "completed"
    assert on[-1].duration_frames > 3000
    assert on[0].end_frame is None
    assert all(a.revision < b.revision for a, b in zip(on, on[1:]))
    assert len(on[-1].bands) >= 2
    assert on[0].bands != on[-1].bands


def test_click_has_completion_without_offset(config):
    d, s = setup(config)
    x = np.zeros(8000)
    x[3200] = 0.8
    rs = feed(d, s, x)
    on = changes(rs, "sound_onset")
    assert len({c.change_id for c in on}) == 1
    assert on[-1].phase == "completed"
    assert 0 < on[-1].duration_frames < 1000
    assert not changes(rs, "sound_offset")


def test_cold_hum_and_disappearance(config):
    d, s = setup(config)
    rs = feed(d, s, np.concatenate([tone(0.8, frequency=200), np.zeros(2400)]))
    assert not changes(rs, "sound_onset")
    assert len({c.change_id for c in changes(rs, "sound_offset")}) == 1
    assert (
        next(r for r in rs if r.windows).windows[0].bands[0].state == "observed_active"
    )


def test_gap_interrupts_not_offset(config):
    d, s = setup(config)
    rs = feed(d, s, np.concatenate([np.zeros(2400), tone(0.3)]))
    on = changes(rs, "sound_onset")
    batch = d.process(UnavailableObservation(s.ref, s.channel_ids, "missing"))
    interrupted = changes(batch.salience)
    assert interrupted and all(
        c.phase == "interrupted" and c.duration_frames is None for c in interrupted
    )
    assert interrupted[0].change_id == on[0].change_id
    next_block = np.asarray(tone(0.1), dtype=np.float32)
    recovery = d.process(s.chunk(memoryview(next_block), first_frame=5600))
    assert not changes(recovery.salience, "sound_onset")
    assert not changes(recovery.salience, "sound_offset")


def test_resource_rollback_and_borrowed_copy(config):
    d, s = setup(config, acoustic=AcousticConfig(max_windows_per_call=1))
    x = np.zeros(400, dtype=np.float32)
    batch = d.process(s.chunk(memoryview(x), first_frame=0))
    assert batch.outcome == "refused"
    assert not d._acoustic.states and not d._acoustic.models
    assert all(lane.watermark == 0 for lane in d._lanes.values())
    d, s = setup(config)
    x = np.zeros(80, dtype=np.float32)
    d.process(s.chunk(memoryview(x), first_frame=0))
    x[:] = 1
    state = next(iter(d._acoustic.states.values()))
    assert not np.any(state.ring.data[:80])


def test_invalid_channel_isolation(config):
    d, s = setup(config, channels=2)
    x = np.zeros((400, 2), dtype=np.float32)
    x[:, 0] = np.nan
    out = d.process(s.chunk(memoryview(x.ravel()), first_frame=0))
    assert len(out.health) == 1 and out.health[0].event_type == "invalid_audio"
    assert all(r.channels[0].channel_id == s.channel_ids[1] for r in out.salience)
    assert any(r.windows for r in out.salience)


def test_missing_numpy_is_clear(config, monkeypatch):
    from far_audio_salience import _acoustic_features

    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(_acoustic_features.importlib, "import_module", missing)
    AudioSalienceDetector(config)
    with pytest.raises(ValueError, match=r"\[dsp\]"):
        AudioSalienceDetector(replace(config, acoustic=AcousticConfig()))


def test_snapshot_roundtrip_legacy_and_transient_reset(config, tmp_path):
    d, s = setup(config)
    feed(d, s, tone(0.7, frequency=200))
    path = tmp_path / "learned.json"
    d.save_baselines(path)
    data = json.loads(path.read_text())
    assert data["version"] == 2
    restored = AudioSalienceDetector.from_baselines(path, config=d.config)
    assert restored._acoustic.models == d._acoustic.models
    assert not restored._acoustic.states
    metadata = AudioSalienceDetector.from_baselines(path, config=config)
    metadata.save_baselines(tmp_path / "metadata.json")
    assert (
        json.loads((tmp_path / "metadata.json").read_text())["acoustic"]
        == data["acoustic"]
    )
    del data["acoustic"]
    data["version"] = 1
    path.write_text(json.dumps(data))
    old = AudioSalienceDetector.from_baselines(path, config=d.config)
    assert not old._acoustic.models


def test_snapshot_rejects_nonfinite_and_unknown_schema(config, tmp_path):
    d, s = setup(config)
    feed(d, s, tone(0.1))
    path = tmp_path / "x.json"
    d.save_baselines(path)
    data = json.loads(path.read_text())
    stats = data["acoustic"]["channels"][0]["stats"]
    next(iter(stats.values()))["exposure"] = -1
    path.write_text(json.dumps(data))
    from far_audio_salience import SnapshotError

    with pytest.raises(SnapshotError):
        AudioSalienceDetector.from_baselines(path, config=d.config)


def test_sample_rate_nyquist_keeps_compatible_models(config):
    d, s = setup(config, rate=48000)
    feed(d, s, tone(0.5, rate=48000))
    saved = {key: dict(stats) for key, stats in d._acoustic.models.items()}
    d.end_source(s.ref)
    s2 = s.new_session(sample_rate_hz=8000)
    d.start_source(s2)
    rs = feed(d, s2, tone(0.05))
    row = next(r for r in rs if r.windows).windows[0]
    assert next(b for b in row.bands if b.band_id == "high").power is None
    assert saved.keys() == d._acoustic.models.keys()
    assert not changes(rs)


def test_loudness_step_and_dual_gate(config):
    d, s = setup(config)
    rs = feed(
        d,
        s,
        np.concatenate(
            [
                tone(0.4, amplitude=0.03),
                tone(0.3, amplitude=0.7),
                tone(0.3, amplitude=0.03),
            ]
        ),
    )
    for kind in ("loudness_spike", "loudness_drop"):
        assert len({c.change_id for c in changes(rs, kind)}) == 1
    d, s = setup(config)
    rs = feed(
        d, s, np.concatenate([tone(0.4, amplitude=0.0001), tone(0.3, amplitude=0.001)])
    )
    assert not changes(rs, "loudness_spike")


def test_independent_clocks_and_reorder(config):
    x = np.asarray(
        np.concatenate([np.zeros(2400), tone(0.2), np.zeros(1600)]), dtype=np.float32
    )
    d, s = setup(
        replace(config, limits=replace(config.limits, max_buffer_bytes=30000)),
        max_reorder_delay_ns=1_000_000_000,
    )
    second = SourceSpec.microphone(label="second", sample_rate_hz=8000)
    d.start_source(second)
    for first in reversed(range(0, len(x), 800)):
        assert not d.process(
            s.chunk(memoryview(x[first : first + 800]), first_frame=first)
        ).refusals
    out = d.advance_watermark(SourceWatermark(s.ref, s.channel_ids, len(x)))
    d2, s2 = setup(config)
    rs = feed(d2, s2, x, (800,))
    assert [r.windows[0].power for r in out.salience if r.windows] == [
        r.windows[0].power for r in rs if r.windows
    ]
    assert not any(k.source_id == second.ref.source_id for k in d._acoustic.states)


def test_partial_horizon_eligibility_and_no_missing_score_renormalization(config):
    d, s = setup(
        config, acoustic=AcousticConfig(assessment_min_coverage=0.5, temporal_weight=1)
    )
    rs = feed(d, s, np.zeros(800))
    windows = [r for r in rs if r.windows]
    assert windows[-1].windows[0].horizons[0][1] == 0
    assert windows[-1].windows[0].horizons[-1][1] is None
    assert all(
        r.salience_score is None and r.temporal_deviation is None for r in windows
    )


def test_candidate_block_and_parameters(config, tmp_path):
    from far_audio_salience import CandidateCommand

    cfg = AcousticConfig(
        candidate_seconds=0.05,
        candidate_ready_seconds=0.1,
        maturity_seconds=0.05,
        adaptation_seconds=1,
    )
    d, s = setup(config, acoustic=cfg)
    feed(d, s, np.concatenate([np.zeros(800), tone(0.5, frequency=200)]))
    records = list(d._baselines.records.values())
    assert len(records) == 1 and records[0].ready
    r = records[0]
    assert (
        d.command(CandidateCommand("block-test", r.candidate_id, 0, "block")).outcome
        == "accepted"
    )
    path = tmp_path / "candidate.json"
    d.save_baselines(path)
    restored = AudioSalienceDetector.from_baselines(path, config=d.config)
    assert restored._baselines.records[r.candidate_id].promotion_blocked
    assert restored._acoustic.candidates == d._acoustic.candidates


def test_band_onset_over_hum_and_opposite_directions(config):
    # Band swap preserves broadband RMS. No cross-band/global interpretation.
    d, s = setup(config, rate=48000)
    low = tone(0.4, rate=48000, frequency=200)
    high = tone(0.4, rate=48000, frequency=4000)
    rs = feed(d, s, np.concatenate([low, high]), (480,))
    onset = changes(rs, "sound_onset")
    offset = changes(rs, "sound_offset")
    assert onset and offset
    assert any(any(b.band_id == "high" for b in c.bands) for c in onset)
    assert any(any(b.band_id == "low" for b in c.bands) for c in offset)
    assert not set(c.change_id for c in onset) & set(c.change_id for c in offset)


def test_weak_confirmation_and_shutdown_interrupt(config):
    d, s = setup(config)
    rs = feed(d, s, np.concatenate([np.zeros(2400), tone(0.2, amplitude=0.005)]))
    on = changes(rs, "sound_onset")
    assert on and on[0].confirmed_frame - on[0].start_frame >= 200
    out = d.end_source(s.ref)
    assert changes(out.salience)
    assert all(
        c.phase == "interrupted" and c.end_frame is None for c in changes(out.salience)
    )


def test_rhythm_pipeline_tempo_and_merge(config):
    cfg = AcousticConfig(bands=())
    d, s = setup(config, acoustic=cfg)
    times = np.cumsum([0.3] + [0.3] * 10 + [0.15] * 12)
    x = np.zeros(round((times[-1] + 0.1) * 8000))
    for t in times:
        x[round(t * 8000) : round(t * 8000) + 8] = 0.7
    rs = feed(d, s, x, (400,))
    tempo = changes(rs, "tempo_increase")
    assert len({c.change_id for c in tempo}) == 1
    assert all(b.delta_db is None for c in tempo for b in c.bands)
    assert any(r.windows and r.windows[0].tempo_hz is not None for r in rs)
    # Lose distinguishability during a still-open tempo transition.
    d, s = setup(config, acoustic=cfg)
    times = np.cumsum([0.3] + [0.3] * 10 + [0.15] * 3)
    x = np.zeros(round((times[-1] + 0.4) * 8000))
    for t in times:
        x[round(t * 8000) : round(t * 8000) + 8] = 0.7
    x[round((times[-1] + 0.01) * 8000) :] = 0.7
    merged = feed(d, s, x, (400,))
    assert any(c.phase == "interrupted" for c in changes(merged, "tempo_increase"))


def test_revision_and_state_resource_refusals(config):
    d, s = setup(config, acoustic=AcousticConfig(max_state_bytes=1))
    out = d.process(s.chunk(memoryview(np.zeros(80, dtype=np.float32)), first_frame=0))
    assert out.refusals[0].resource.resource_id == "acoustic_state_bytes"
    assert not d._acoustic.states
    d, s = setup(config, acoustic=AcousticConfig(max_revisions_per_call=1))
    feed(d, s, np.zeros(2400))
    before = next(iter(d._lanes.values())).watermark
    out = d.process(
        s.chunk(memoryview(np.asarray(tone(0.04), dtype=np.float32)), first_frame=2400)
    )
    assert out.refusals[0].resource.resource_id == "acoustic_revisions_per_call"
    assert next(iter(d._lanes.values())).watermark == before
    assert not next(iter(d._acoustic.states.values())).changes


def test_overlap_duplicate_and_conflict_do_not_relearn(config):
    cfg = replace(
        config, comparison_retention_ns=100_000_000, acoustic=AcousticConfig()
    )
    d = AudioSalienceDetector(cfg)
    s = SourceSpec.microphone(label="overlap", sample_rate_hz=8000)
    d.start_source(s)
    x = np.asarray(tone(0.08), dtype=np.float32)
    d.process(s.chunk(memoryview(x[:400]), first_frame=0))
    before = {k: asdict(v) for k, v in next(iter(d._acoustic.models.values())).items()}
    duplicate = d.process(s.chunk(memoryview(x[:400]), first_frame=0))
    assert not duplicate.salience
    assert before == {
        k: asdict(v) for k, v in next(iter(d._acoustic.models.values())).items()
    }
    conflict = d.process(
        s.chunk(memoryview(np.zeros(400, dtype=np.float32)), first_frame=0)
    )
    assert any(h.event_type == "conflicting_observation" for h in conflict.health)
    overlap = d.process(s.chunk(memoryview(x[320:]), first_frame=320))
    assert (
        sum(
            span.interval.stop_frame - span.interval.start_frame
            for span in overlap.input_receipts[0].accepted
        )
        == 240
    )
    expected, s2 = setup(config)
    feed(expected, s2, x, (400,))
    assert [v.exposure for v in next(iter(d._acoustic.models.values())).values()] == [
        v.exposure for v in next(iter(expected._acoustic.models.values())).values()
    ]


def test_candidate_rejection_fresh_readiness_and_promotion(config):
    from far_audio_salience import CandidateCommand

    cfg = AcousticConfig(
        candidate_seconds=0.05,
        candidate_ready_seconds=0.1,
        maturity_seconds=0.05,
        adaptation_seconds=1,
    )
    d, s = setup(config, acoustic=cfg)
    feed(d, s, np.concatenate([np.zeros(800), tone(0.4, frequency=200)]))
    r = next(iter(d._baselines.records.values()))
    rejected = d.command(
        CandidateCommand("reject", r.candidate_id, 0, "reject", r.proposal_id)
    )
    assert not rejected.refusals
    old = r.proposal_id
    first = 4000
    for i in range(30):
        x = np.asarray(tone(0.01, frequency=200), dtype=np.float32)
        d.process(s.chunk(memoryview(x), first_frame=first))
        first += 80
    r = d._baselines.records[r.candidate_id]
    assert r.ready and r.proposal_id != old
    revision = d._baselines.administration[r.candidate_id][0]
    approved = d.command(
        CandidateCommand("approve", r.candidate_id, revision, "approve", r.proposal_id)
    )
    assert not approved.refusals
    assert d._baselines.records[r.candidate_id].role == "established"


def test_acoustic_config_rejects_unbounded_and_invalid_values():
    for kwargs in (
        {"hop_seconds": 0},
        {"hop_seconds": 1e-300},
        {"bands": (FrequencyBand("x", 0, 1), FrequencyBand("x", 1, 2))},
        {"horizons_seconds": (float("inf"),)},
        {"acoustic_weight": float("nan")},
        {"max_impulses": 2},
        {"assessment_min_coverage": 1.1},
    ):
        with pytest.raises(ValueError):
            AcousticConfig(**kwargs)


def test_measurement_time_and_full_window_cadence(config):
    d, s = setup(config)
    x = np.zeros(400, dtype=np.float32)
    out = d.process(s.chunk(memoryview(x), first_frame=0, measured_start_ns=1_000_000))
    assert len(out.salience) == 7
    for r in out.salience:
        end = r.windows[0].interval.stop_frame
        assert r.timestamp.clock_id == s.clock_id
        assert r.timestamp.ns == 1_000_000 + (end - 1) * 125_000
        assert r.header.provenance.measurement_times


def test_no_assessment_transition_when_contiguity_policy_fails(config):
    d, s = setup(config, acoustic=AcousticConfig(min_contiguous_seconds=0.1))
    rs = feed(d, s, np.concatenate([np.zeros(800), tone(0.2)]))
    assert not changes(rs)
    assert all(r.salience_score is None for r in rs)
    assert all(not w.established_learning_eligible for r in rs for w in r.windows)


def test_unexpected_dsp_failure_is_atomic(config, monkeypatch):
    from far_audio_salience._acoustic import AcousticEngine

    d, s = setup(config)

    def fail(*args):
        raise RuntimeError("injected DSP failure")

    monkeypatch.setattr(AcousticEngine, "_window", fail)
    with pytest.raises(RuntimeError, match="injected"):
        d.process(s.chunk(memoryview(np.zeros(400, dtype=np.float32)), first_frame=0))
    assert not d._acoustic.models and not d._acoustic.states
    assert next(iter(d._lanes.values())).watermark == 0
    assert d._sequence == 0


def test_closed_model_releases_transient_audio(config):
    d, s = setup(config)
    feed(d, s, tone(0.1))
    assert d._acoustic.states
    out = d.close()
    assert not out.refusals and not d._acoustic.states
    assert d.close().outcome == "no_change"
