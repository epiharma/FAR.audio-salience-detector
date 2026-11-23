"""Validate the evaluation machinery, not favourable detector outcomes."""

import runpy
from pathlib import Path

import pytest

pytest.importorskip("numpy")
module = runpy.run_path(
    str(Path(__file__).parents[1] / "benchmarks/broader_validation.py")
)
match = module["match"]


def prediction(at, channel=0, kind="sound_onset"):
    return dict(start=at, confirmed=at + 2, channel=channel, kind=kind)


def label(at, channel=0):
    return dict(frame=at, early=10, late=10, channel=channel, kind="sound_onset")


def test_maximum_cardinality_not_greedy_nearest():
    result = match(
        [prediction(8), prediction(19)], [label(0), label(10)], "sound_onset", 0
    )
    assert (result["tp"], result["fp"], result["fn"]) == (2, 0, 0)


def test_duplicate_is_false_positive_not_another_true_positive():
    result = match(
        [prediction(8), prediction(19), prediction(20)],
        [label(0), label(10)],
        "sound_onset",
        0,
    )
    assert (result["tp"], result["fp"], result["duplicates"]) == (2, 1, 1)


def test_wrong_kind_channel_and_no_positive_subset():
    result = match([prediction(0)], [label(0, channel=1)], "sound_onset", 0)
    assert result["positives"] == 0 and result["fp"] == 1 and result["duplicates"] == 0
    result = match([prediction(0, kind="sound_offset")], [label(0)], "sound_onset", 0)
    assert result["tp"] == 0 and result["fn"] == 1
    p = prediction(0)
    p["emitted_through_frame"] = 80
    result = match([p], [label(0)], "sound_onset", 0)
    assert result["delay_frames"] == [2]
    assert result["delivery_delay_frames"] == [80]
