"""Off-hop gain edges and nearby independent acoustic activity."""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

RATE = 32000
RISING_EDGE = 20807
FALLING_EDGE = 28019
LEVEL_KINDS = ("loudness_spike", "loudness_drop")
ACTIVITY_KINDS = ("sound_onset", "sound_offset")


@pytest.fixture(scope="module")
def harness():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).parents[1] / "benchmarks"))
        from broader_noise import make_case
        from broader_validation import run_case

    return make_case, run_case


def gain_only(*, first=RISING_EDGE, second=FALLING_EDGE, attenuation_first=False):
    frames = np.arange(RATE)
    t = frames / RATE
    low, high = (0.7, 0.03) if attenuation_first else (0.03, 0.7)
    gain = np.where((frames >= first) & (frames < second), high, low)
    return gain * np.sin(2 * np.pi * 213 * t + 1.37)


def test_exact_off_hop_probe_preserves_loudness(harness):
    make_case, run_case = harness
    case = make_case("off_hop_hum_gain_213hz", 0, RATE, gain_only(), [], "Gain only.")
    result = run_case(case)
    assert not result["violations"]
    assert [(e["kind"], e["start"], e["confirmed"]) for e in result["predictions"]] == [
        ("loudness_spike", 20320, 20960),
        ("loudness_drop", 28000, 28640),
    ]


@pytest.mark.parametrize("edge", [1, 7, 79, 152, 159])
@pytest.mark.parametrize("attenuation_first", [False, True])
def test_gain_edges_across_hop_do_not_create_activity(harness, edge, attenuation_first):
    make_case, run_case = harness
    samples = gain_only(
        first=20800 + edge,
        second=28000 + (edge + 19) % 160,
        attenuation_first=attenuation_first,
    )
    result = run_case(make_case("gain_sweep", edge, RATE, samples, [], "Gain only."))
    assert not result["violations"]
    assert len(result["predictions"]) == 2
    assert {e["kind"] for e in result["predictions"]} == set(LEVEL_KINDS)


def positive_case(make_case, kind):
    t = np.arange(RATE) / RATE
    hum = np.sin(2 * np.pi * 213 * t + 1.37)
    first, stop = RISING_EDGE, 29600
    if kind == "gain_and_weak_onset":
        samples = gain_only()
        samples[first:stop] += 0.005 * np.sin(2 * np.pi * 1337 * t[first:stop])
        labels = [("sound_onset", first / RATE), ("sound_offset", stop / RATE)]
    elif kind == "new_component":
        samples = 0.03 * hum
        samples[first:stop] += 0.06 * np.sin(2 * np.pi * 4000 * t[first:stop])
        labels = [("sound_onset", first / RATE), ("sound_offset", stop / RATE)]
    elif kind == "emergence":
        samples = np.where(np.arange(RATE) < first, 0.0001, 0.2) * hum
        labels = [("sound_onset", first / RATE)]
    else:
        samples = 0.03 * hum
        samples[first] += 0.7
        labels = [("sound_onset", first / RATE)]
    case = make_case(kind, 0, RATE, samples, labels, "")
    case["scored"] = ACTIVITY_KINDS
    return case


@pytest.mark.parametrize(
    "kind", ["gain_and_weak_onset", "new_component", "emergence", "isolated_impulse"]
)
def test_off_hop_correction_preserves_acoustic_onsets(harness, kind):
    make_case, run_case = harness
    result = run_case(positive_case(make_case, kind))
    assert not result["violations"]
    for event_kind in ACTIVITY_KINDS:
        metric = result["metrics"][event_kind]
        assert (metric["tp"], metric["fp"], metric["fn"], metric["duplicates"]) == (
            metric["positives"],
            0,
            0,
            0,
        )
    if kind == "isolated_impulse":
        onset = [r for r in result["revisions"] if r["kind"] == "sound_onset"]
        assert onset[0]["phase"] == "detected"
        assert onset[-1]["phase"] == "completed"
        assert len({r["change_id"] for r in onset}) == 1


@pytest.mark.parametrize("kind", ["gain_only", "gain_and_weak_onset"])
def test_off_hop_events_and_revisions_are_partition_invariant(harness, kind):
    make_case, run_case = harness
    case = (
        make_case(kind, 0, RATE, gain_only(), [], "")
        if kind == "gain_only"
        else positive_case(make_case, kind)
    )
    fixed = run_case(dict(case, split="development"))
    variable = run_case(case)
    assert not fixed["violations"] and not variable["violations"]
    assert fixed["revisions"]
    assert [
        {k: v for k, v in row.items() if k != "change_id"} for row in fixed["revisions"]
    ] == [
        {k: v for k, v in row.items() if k != "change_id"}
        for row in variable["revisions"]
    ]
