"""Observed-IOI fixtures: never infer deceleration from absence."""

from far_audio_salience import AcousticConfig
from far_audio_salience._rhythm import Rhythm


def drive(intervals, cfg=None):
    cfg = cfg or AcousticConfig()
    r = Rhythm()
    frame = 100
    changes = []
    rates = []
    ids = []
    for seconds in [0, *intervals]:
        frame += round(seconds * 1000)
        c = r.observe(
            high=True,
            pulse_frame=frame,
            end=frame + 1,
            rate=1000,
            profile=(1.0, 0.0),
            config=cfg,
        )
        if c:
            changes.append(c)
        rates.append(r.rate)
        ids.append(r.sequence_id)
        r.observe(
            high=False,
            pulse_frame=frame + 2,
            end=frame + 2,
            rate=1000,
            profile=(1.0, 0.0),
            config=cfg,
        )
    return r, frame, changes, rates, ids


def test_stable_and_missing_beat():
    r, f, changes, rates, _ = drive([0.5] * 10 + [1.0] + [0.5] * 6)
    assert not changes
    assert r.rate == 2
    assert all(rate is None or rate == 2 for rate in rates)
    deadline = r.deadline
    r.observe(
        high=False,
        pulse_frame=f,
        end=deadline + 1,
        rate=1000,
        profile=(1.0, 0.0),
        config=AcousticConfig(),
    )
    assert r.rate is None and r.reason == "sequence_timeout"
    assert r.deadline is None


def test_abrupt_acceleration_and_deceleration():
    for intervals, direction in [
        ([0.5] * 10 + [0.25] * 10, 1),
        ([0.25] * 10 + [0.5] * 10, -1),
    ]:
        _, _, changes, _, _ = drive(intervals)
        assert len([c for c in changes if not c[2]]) == 1
        assert changes[0][0] == direction
        assert any(c[2] for c in changes)


def test_gradual_tempo_and_ambiguous_sequence():
    _, _, changes, _, _ = drive(
        [0.5] * 10
        + [0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30]
        + [0.3] * 8
    )
    assert any(c[0] == 1 and not c[2] for c in changes)
    r, _, _, _, _ = drive([0.2, 0.3] * 10)
    assert r.rate is None and r.reason == "ambiguous_impulse_sequence"


def test_merge_reacquires_new_sequence():
    r, f, _, _, ids = drive([0.3] * 10)
    old = ids[-1]
    cfg = AcousticConfig()
    for end in range(f + 10, f + 500, 10):
        r.observe(
            high=True,
            pulse_frame=end,
            end=end,
            rate=1000,
            profile=(1.0, 0.0),
            config=cfg,
        )
    assert r.rate is None and r.reason == "impulses_indistinguishable"
    r.observe(
        high=False,
        pulse_frame=f + 510,
        end=f + 510,
        rate=1000,
        profile=(1.0, 0.0),
        config=cfg,
    )
    r.observe(
        high=True,
        pulse_frame=f + 600,
        end=f + 600,
        rate=1000,
        profile=(1.0, 0.0),
        config=cfg,
    )
    assert r.sequence_id != old and len(r.times) == 1


def test_gradual_change_plateau_reversal_and_distinct_later_acceleration():
    _, _, changes, _, _ = drive(
        [0.5] * 10
        + [0.48 - 0.02 * i for i in range(13)]
        + [0.24] * 12
        + [0.4] * 12
        + [0.19] * 12
    )
    assert [direction for direction, _, finished in changes if not finished] == [
        1,
        -1,
        1,
    ]
    assert [direction for direction, _, finished in changes if finished] == [1, -1, 1]


def test_gradual_reversal_before_plateau_uses_observed_extreme():
    _, _, changes, _, _ = drive(
        [0.5] * 10
        + [0.48 - 0.02 * i for i in range(13)]
        + [0.245 + 0.015 * i for i in range(12)]
        + [0.41] * 12
    )
    assert [direction for direction, _, finished in changes if not finished] == [1, -1]
