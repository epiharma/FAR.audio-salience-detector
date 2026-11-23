"""Additional frozen lifecycle waveforms, separate from the first 171 cases."""

import hashlib
import json
from pathlib import Path

import numpy as np
from broader_validation import KINDS, description, run_case


def fixtures():
    for rate in (16000, 48000):
        for family in ("merge_reacquire", "pause_timeout"):
            seconds = 7.8
            x = np.zeros(round(seconds * rate))
            initial = 0.4 + np.arange(12 if family == "merge_reacquire" else 10) * 0.25
            restarted = (4.4 if family == "merge_reacquire" else 4.15) + np.arange(
                12 if family == "merge_reacquire" else 10
            ) * 0.25
            for i, at in enumerate(initial):
                width = 0.01
                if family == "merge_reacquire" and i >= 8:
                    width = (0.05, 0.1, 0.18, 0.25)[i - 8]
                a = round(at * rate)
                b = a + round(width * rate)
                t = np.arange(a, b) / rate
                x[a:b] = 0.08 * np.sin(2 * np.pi * 800 * t)
            if family == "merge_reacquire":
                a = round(3.4 * rate)
                b = round(4.4 * rate)
                x[a:b] = 0.08 * np.sin(2 * np.pi * 800 * np.arange(a, b) / rate)
            for at in restarted:
                a = round(at * rate)
                b = a + round(0.01 * rate)
                x[a:b] = 0.08 * np.sin(2 * np.pi * 800 * np.arange(a, b) / rate)
            starts = list(initial[:4]) + (
                list(restarted[:4]) if family == "pause_timeout" else []
            )
            labels = [
                dict(
                    kind="sound_onset",
                    channel=0,
                    frame=round(at * rate),
                    early=round(0.02 * rate),
                    late=round(0.12 * rate),
                )
                for at in starts
            ]
            yield dict(
                name=f"lifecycle:{rate}:{family}",
                family=family,
                split="lifecycle",
                seed=None,
                rate=rate,
                channels=1,
                samples=x.astype(np.float32),
                labels=labels,
                scored=list(KINDS),
                tempo=None,
                notes="Observed continuous audio. Pulse widths merge to a continuous carrier, or pause beyond the fixed deadline. No injected missing data.",
            )


def main():
    cases = list(fixtures())
    manifest = Path(__file__).with_name(
        "acoustic-broader-lifecycle-manifest-2026-11-23.json"
    )
    frozen = json.loads(json.dumps([description(c) for c in cases]))
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding="utf-8")) == frozen
    else:
        manifest.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(
        json.dumps(
            dict(
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                cases=[run_case(c) for c in cases],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
