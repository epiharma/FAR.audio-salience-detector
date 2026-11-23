"""Frozen holdout and explicit-label adversarial onset evaluation.

Run: python benchmarks/onset_remediation.py. Baseline replay disables only the
new confirmation/presence guards; it is checked against captured pre-edit traces.
"""

import json
import runpy
from pathlib import Path

import numpy as np

from far_audio_salience import AcousticConfig
from far_audio_salience._acoustic import AcousticEngine
from far_audio_salience._acoustic_features import SpectralPlan


def fixtures():
    t = np.arange(48000) / 48000
    for f in [200, 1000, 4000]:
        for lo, hi in [(0.01, 0.3), (0.03, 0.7), (0.01, 0.99)]:
            for phase in [0.0, 1.3]:
                a = np.where(t < 0.4, lo, np.where(t < 0.7, hi, lo))
                yield (
                    f"holdout:{f}:{lo}:{hi}:{phase}",
                    a * np.sin(2 * np.pi * f * t + phase),
                    [],
                )
    for kind in ["noise_step", "noise_ramp", "multiband", "dip", "slow_ramp"]:
        t = np.arange(96000) / 48000
        carrier = (
            np.random.default_rng(785).uniform(-1, 1, len(t))
            if kind.startswith("noise")
            else 0.5 * np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 4000 * t)
        )
        gain = np.where(t < 0.6, 0.2, np.where(t < 1.2, 0.01, 0.2))
        if kind == "noise_ramp":
            gain = np.interp(t, [0, 0.6, 1.0, 1.4, 2], [0.2, 0.2, 0.01, 0.2, 0.2])
        if kind == "slow_ramp":
            gain = np.interp(t, [0, 0.6, 1.2, 2], [0.03, 0.03, 0.7, 0.7])
        if kind == "dip":
            gain = np.where((t >= 0.6) & (t < 0.62), 0.01, 0.2)
        yield kind, carrier * gain, []
    t = np.arange(48000) / 48000
    for low in [0.0001, 0.001, 0.002]:
        yield (
            f"emergence:{low}",
            np.where(t < 0.4, low, 0.2) * np.sin(2 * np.pi * 1000 * t),
            [0.4] if low**2 / 2 < 1e-6 else [],
        )
    x = 0.03 * np.sin(2 * np.pi * 200 * t)
    x[19200:19680] += 0.7 * np.sin(2 * np.pi * 4000 * t[19200:19680])
    yield "short_impulse", x, [0.4]
    x = 0.2 * np.sin(2 * np.pi * 200 * t)
    x[19200:] += 0.03 * np.random.default_rng(28).uniform(-1, 1, len(t) - 19200)
    yield "new_broadband", x, [0.4]


def evaluate(baseline=False):
    b = runpy.run_path(str(Path(__file__).with_name("acoustic.py")))
    old_shape = SpectralPlan.level_invariant_bands
    old_activity = AcousticEngine._activity

    def original_activity(*args, **kwargs):
        kwargs["onset_power"] = None
        kwargs["common_gain"] = None
        return old_activity(*args, **kwargs)

    if baseline:
        SpectralPlan.level_invariant_bands = lambda *args: None
        AcousticEngine._activity = original_activity
    rows = []
    try:
        for name, x, labels in fixtures():
            events, _ = b["run_audio"](x, 48000, AcousticConfig())
            remaining = list(labels)
            tp = 0
            fp = 0
            delays = []
            for kind, start, confirmed in events:
                if kind != "sound_onset":
                    continue
                index = next(
                    (i for i, at in enumerate(remaining) if abs(start - at) <= 0.04),
                    None,
                )
                if index is None:
                    fp += 1
                else:
                    at = remaining.pop(index)
                    tp += 1
                    delays.append((confirmed - at) * 1000)
            rows.append(
                dict(
                    name=name,
                    frames=len(x),
                    labels=labels,
                    events=events,
                    tp=tp,
                    fp=fp,
                    fn=len(remaining),
                    decision_delay_ms=delays,
                )
            )
    finally:
        SpectralPlan.level_invariant_bands = old_shape
        AcousticEngine._activity = old_activity
    return rows


if __name__ == "__main__":
    print(json.dumps(dict(before=evaluate(True), after=evaluate()), indent=2))
