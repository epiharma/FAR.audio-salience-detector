"""Additional frozen colored-noise controls; no detector configuration changes."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from broader_validation import KINDS, config, description, run_case

from far_audio_salience import AudioSalienceDetector, SourceSpec


def fixtures():
    for seed in (16031, 99277):
        for rate in (16000, 48000):
            n = round(1.7 * rate)
            frequency = np.fft.rfftfreq(n, 1 / rate)
            rng = np.random.default_rng(seed + rate)
            for color, exponent in (("white", 0), ("pink", 0.5), ("brown", 1)):
                spectrum = np.fft.rfft(rng.normal(size=n))
                spectrum[0] = 0
                spectrum /= np.maximum(frequency, 20) ** exponent
                wave = np.fft.irfft(spectrum, n=n)
                wave *= 0.9 / np.max(np.abs(wave))
                assert np.mean(wave * wave) > 0.02
                gain = np.where(
                    np.arange(n) < round(0.6 * rate),
                    1.0,
                    np.where(np.arange(n) < round(1.1 * rate), 0.1, 1.0),
                )
                yield make_case(
                    f"{color}_gain",
                    seed,
                    rate,
                    wave * gain,
                    [("loudness_drop", 0.6), ("loudness_spike", 1.1)],
                    "One continuous colored-noise realization normalized to .9 peak; unchanged component presence across gain changes 1/.1/1.",
                )
            # Background occupies low frequencies; new component is in mid band.
            spectrum = np.fft.rfft(rng.normal(size=n))
            spectrum[(frequency < 40) | (frequency > 400)] = 0
            spectrum /= np.sqrt(np.maximum(frequency, 40))
            low = np.fft.irfft(spectrum, n=n)
            low /= np.sqrt(np.mean(low * low))
            t = np.arange(n) / rate
            background = 0.03 * low + 0.06 * np.sin(2 * np.pi * 213 * t)
            for amplitude in (0.02, 0.06):
                x = background.copy()
                first, stop = round(0.6 * rate), round(1.1 * rate)
                x[first:stop] += amplitude * np.sin(2 * np.pi * 1337 * t[first:stop])
                yield make_case(
                    f"colored_hum_component_{amplitude}",
                    seed,
                    rate,
                    x,
                    [("sound_onset", 0.6), ("sound_offset", 1.1)],
                    "Band-limited pink background RMS .03 plus .06-peak hum; .02/.06-peak mid-band component. Digital power ratio only, not field SNR.",
                )


def make_case(family, seed, rate, x, truth, notes):
    assert np.max(np.abs(x)) <= 1
    return dict(
        name=f"colored:{seed}:{rate}:{family}",
        family=family,
        split="additional_noise",
        seed=seed,
        rate=rate,
        channels=1,
        samples=x.astype(np.float32),
        labels=[
            dict(
                kind=kind,
                channel=0,
                frame=round(at * rate),
                early=round(0.02 * rate),
                late=round(0.12 * rate),
            )
            for kind, at in truth
        ],
        scored=list(KINDS),
        tempo=None,
        notes=notes,
    )


def grouping():
    rate = 48000
    d = AudioSalienceDetector(config())
    s = SourceSpec.microphone(label="grouping probe", sample_rate_hz=rate)
    d.start_source(s)
    x = np.zeros(rate, dtype=np.float32)
    t = np.arange(rate) / rate
    x[19200:38400] += (0.2 * np.sin(2 * np.pi * 213 * t[19200:38400])).astype(
        np.float32
    )
    x[19440:38400] += (0.08 * np.sin(2 * np.pi * 1337 * t[19440:38400])).astype(
        np.float32
    )
    snapshots = []
    original_first = None
    for first in range(0, len(x), 240):
        batch = d.process(
            s.chunk(memoryview(x[first : first + 240]), first_frame=first)
        )
        assert not batch.refusals
        for result in batch.salience:
            snapshots.extend(c for c in result.changes if c.kind == "sound_onset")
            if snapshots and original_first is None:
                original_first = asdict(snapshots[0])
    d.close()
    first = snapshots[0]
    assert len({c.change_id for c in snapshots}) == 1
    first_bands = first.bands
    assert asdict(snapshots[0]) == original_first
    rows = []
    previous = first
    for current in snapshots[1:]:
        assert current.revision == previous.revision + 1
        if {b.band_id for b in current.bands} - {b.band_id for b in previous.bands}:
            rows.append(
                dict(
                    revision=current.revision,
                    phase=current.phase,
                    membership_lag_ms=(
                        current.last_observed_frame - first.confirmed_frame
                    )
                    * 1000
                    / rate,
                    bands=[asdict(b) for b in current.bands],
                )
            )
        previous = current
    assert rows
    return dict(
        first_confirmation_ms=first.confirmed_frame * 1000 / rate,
        first_start_ms=first.start_frame * 1000 / rate,
        grouping_seconds=d.config.acoustic.grouping_seconds,
        first_bands=[asdict(b) for b in first_bands],
        final_bands=[asdict(b) for b in snapshots[-1].bands],
        revisions=len(snapshots),
        membership_revisions=rows,
        prior_snapshot_unchanged=asdict(snapshots[0]) == original_first,
    )


def main():
    cases = list(fixtures())
    path = Path(__file__).with_name("acoustic-broader-noise-manifest-2026-11-23.json")
    frozen = json.loads(json.dumps([description(c) for c in cases]))
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == frozen
    else:
        path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    # Freeze every label and sample hash before this set's first evaluation.
    results = [run_case(c) for c in cases]
    print(
        json.dumps(
            dict(
                manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                cases=results,
                grouping=grouping(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
