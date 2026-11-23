"""Reproduce stochastic-onset evidence (requires .[dev,dsp]).

Fixture assertions run unchanged; JSON records source-frame detection metrics.
--stress reuses the prior 120-second workload with bounded-state assertions.
"""

import argparse
import io
import json
import runpy
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path

from far_audio_salience._acoustic import AcousticEngine


def evaluate():
    tests = runpy.run_path(
        str(Path(__file__).parents[1] / "tests/test_stochastic_onset.py")
    )
    namespace = tests["run"].__globals__
    original = namespace["run"]
    records = []
    current = {}

    def recorded(*args, **kwargs):
        result = original(*args, **kwargs)
        events = [asdict(c) for r in result for c in r.changes if c.revision == 1]
        labels = list(current.get("onsets", []))
        tp = 0
        fp = 0
        timing = []
        for event in events:
            if event["kind"] != "sound_onset":
                continue
            at = event["start_frame"]
            index = next((i for i, f in enumerate(labels) if abs(at - f) <= 1920), None)
            if index is None:
                fp += 1
            else:
                target = labels.pop(index)
                tp += 1
                timing.append(
                    dict(
                        start_error_frames=at - target,
                        confirmation_delay_frames=event["confirmed_frame"] - target,
                    )
                )
        records.append(
            dict(
                **current,
                frames=len(args[0]),
                tp=tp,
                fp=fp,
                fn=len(labels),
                timing=timing,
                events=events,
            )
        )
        return result

    namespace["run"] = recorded
    try:
        for kind in ("white", "pink", "band"):
            for seed in (13, 79, 2026):
                for drop, recovery, low in ((19217, 43271, 0.01), (33599, 62401, 0.04)):
                    for initial in (0.1, 0.2, 0.3):
                        current = dict(
                            scenario="stationary_gain",
                            kind=kind,
                            seed=seed,
                            drop=drop,
                            recovery=recovery,
                            low=low,
                            initial=initial,
                            onsets=[],
                        )
                        tests["test_stationary_noise_gain_controls"](
                            kind, seed, drop, recovery, low, initial
                        )
            current = dict(
                scenario="gradual_recovery_layout", kind=kind, seed=321, onsets=[]
            )
            tests["test_gradual_recovery_and_partition_equivalence"](kind)
        for amplitude in (0.003, 0.08):
            for offset in (0, 480, 9600):
                current = dict(
                    scenario="new_band",
                    seed=84,
                    amplitude=amplitude,
                    offset=offset,
                    onsets=[57600 + offset],
                )
                tests["test_new_high_band_during_gain_recovery"](offset, amplitude)
        for begin in (43200, 57600):
            current = dict(scenario="short_burst", seed=167, onsets=[begin])
            tests["test_strong_short_band_burst_on_noise"](begin)
        current = dict(scenario="onset_and_spike", seed=91, onsets=[57600])
        tests["test_real_onset_and_loudness_spike_on_noise"]()
        current = dict(scenario="disappearance", seed=85, onsets=[], offsets=[480000])
        tests["test_noise_disappearance_is_observed_offset"]()
    finally:
        namespace["run"] = original
    return records


def stress():
    script = runpy.run_path(str(Path(__file__).with_name("acoustic_audit.py")))
    old = AcousticEngine.evaluate
    maxima = dict(impulses=0, profiles=0, groups=0, open_changes=0, history=0)

    def checked(self, *args, **kwargs):
        result = old(self, *args, **kwargs)
        for state in self.states.values():
            sizes = dict(
                impulses=len(state.rhythm.times),
                profiles=len(state.rhythm.profiles),
                groups=len(state.group),
                open_changes=len(state.changes),
                history=len(state.history),
            )
            assert sizes["impulses"] <= self.config.max_impulses
            assert sizes["profiles"] == sizes["impulses"]
            assert sizes["groups"] <= 6
            assert sizes["open_changes"] <= self.config.max_open_changes
            for k, v in sizes.items():
                maxima[k] = max(maxima[k], v)
        return result

    AcousticEngine.evaluate = checked
    try:
        output = io.StringIO()
        with redirect_stdout(output):
            script["stress"](120)
        data = json.loads(output.getvalue())
        data["bounded_state_maxima"] = maxima
        return data
    finally:
        AcousticEngine.evaluate = old


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress", action="store_true")
    args = parser.parse_args()
    print(json.dumps(stress() if args.stress else evaluate(), indent=2))
