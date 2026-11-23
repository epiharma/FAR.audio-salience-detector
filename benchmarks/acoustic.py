"""Deterministic pilot accuracy + separately timed/traced DSP workload.

Run: python benchmarks/acoustic.py [--seconds 5] [--channels 1]
Synthetic labels test digital waveforms, not source classification or field accuracy.
"""

import argparse
import json
import os
import platform
import time
import tracemalloc
from collections import Counter
from dataclasses import replace

import numpy as np

from far_audio_salience import (
    AcousticConfig,
    AudioSalienceDetector,
    DetectorConfig,
    ResourceLimits,
    ResourceReporting,
    SourceSpec,
)

KINDS = (
    "sound_onset",
    "sound_offset",
    "tempo_increase",
    "tempo_decrease",
    "loudness_spike",
    "loudness_drop",
)


def configuration(channels=1, acoustic=None):
    return DetectorConfig(
        ResourceLimits(1, channels, 1_048_576, 4_194_304, 16, 32, 1_048_576, 1, 8),
        0,
        0,
        ResourceReporting(10, 4096),
        acoustic=acoustic or AcousticConfig(),
    )


def tone(seconds, rate, amplitude=0.2, frequency=1000):
    return amplitude * np.sin(
        2 * np.pi * frequency * np.arange(round(seconds * rate)) / rate
    )


def fixtures(rate):
    cfg = AcousticConfig()
    yield (
        "sustained",
        np.concatenate(
            [np.zeros(round(0.3 * rate)), tone(0.4, rate), np.zeros(round(0.3 * rate))]
        ),
        [
            ("sound_onset", 0.3),
            ("sound_offset", 0.7),
            ("loudness_spike", 0.3),
            ("loudness_drop", 0.7),
        ],
        cfg,
    )
    x = np.zeros(rate)
    x[round(0.4 * rate)] = 0.8
    yield "click", x, [("sound_onset", 0.4)], cfg
    yield (
        "cold_hum_loss",
        np.concatenate([tone(0.8, rate, frequency=200), np.zeros(round(0.3 * rate))]),
        [("sound_offset", 0.8), ("loudness_drop", 0.8)],
        cfg,
    )
    yield (
        "level_steps",
        np.concatenate(
            [tone(0.4, rate, 0.03), tone(0.3, rate, 0.7), tone(0.3, rate, 0.03)]
        ),
        [("loudness_spike", 0.4), ("loudness_drop", 0.7)],
        cfg,
    )
    # Disable extreme level transitions for isolated tempo evaluation.
    rhythm_cfg = replace(cfg, bands=(), spike_absolute_power=1, drop_absolute_power=1)
    for kind, old, new in [
        ("tempo_increase", 0.3, 0.15),
        ("tempo_decrease", 0.15, 0.3),
    ]:
        times = np.cumsum([0.3] + [old] * 10 + [new] * 12)
        x = np.zeros(round((times[-1] + 0.2) * rate))
        for t in times:
            x[round(t * rate) : round(t * rate) + round(0.001 * rate)] = 0.7
        labels = [("sound_onset", float(t)) for t in times[:4]] + [
            (kind, float(times[11]))
        ]
        yield kind, x, labels, rhythm_cfg
    yield "quiet", np.zeros(5 * rate), [], cfg
    yield "steady", tone(5, rate, frequency=200), [], cfg


def run_audio(x, rate, cfg, channels=1, *, traced=False, expected_tempo=None):
    d = AudioSalienceDetector(configuration(channels, cfg))
    s = SourceSpec.microphone(
        label="synthetic benchmark", sample_rate_hz=rate, channel_count=channels
    )
    d.start_source(s)
    latencies = []
    detected = []
    tempo = []
    tempo_errors = []
    refusals = 0
    cpu = time.process_time()
    wall = time.perf_counter()
    if traced:
        tracemalloc.start()
    for first in range(0, len(x), 480):
        samples = np.asarray(x[first : first + 480], dtype=np.float32)
        if channels > 1:
            samples = np.tile(samples[:, None], (1, channels)).ravel()
        tick = time.perf_counter_ns()
        out = d.process(s.chunk(memoryview(samples), first_frame=first))
        elapsed = time.perf_counter_ns() - tick
        latencies.append(elapsed / 1e6)
        refusals += len(out.refusals)
        for result in out.salience:
            detected.extend(
                (c.kind, c.start_frame / rate, c.confirmed_frame / rate)
                for c in result.changes
                if c.revision == 1
            )
            tempo.extend(w.tempo_hz for w in result.windows)
            if expected_tempo is not None:
                tempo_errors.extend(
                    abs(w.tempo_hz - expected_tempo(w.interval.stop_frame / rate))
                    for w in result.windows
                    if w.tempo_hz is not None
                )
    peak = tracemalloc.get_traced_memory()[1] if traced else None
    if traced:
        tracemalloc.stop()
    wall = time.perf_counter() - wall
    cpu = time.process_time() - cpu
    d.close()
    ordered = sorted(latencies)

    def percentile(q):
        return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]

    return detected, dict(
        p50_ms=percentile(0.5),
        p95_ms=percentile(0.95),
        p99_ms=percentile(0.99),
        worst_ms=max(ordered),
        wall_seconds=wall,
        cpu_seconds=cpu,
        audio_seconds=len(x) / rate,
        realtime_factor=(len(x) / rate) / wall,
        traced_peak_bytes=peak,
        refusals=refusals,
        tempo_availability=sum(v is not None for v in tempo) / max(1, len(tempo)),
        tempo_mean_absolute_error_hz=sum(tempo_errors) / len(tempo_errors)
        if tempo_errors
        else None,
    )


def evaluate(rate=48000):
    totals = {kind: Counter() for kind in KINDS}
    details = []
    valid_seconds = 0
    for name, x, labels, cfg in fixtures(rate):
        expected_tempo = None
        if name in ("tempo_increase", "tempo_decrease"):
            old, new = (0.3, 0.15) if name == "tempo_increase" else (0.15, 0.3)
            change_at = 0.3 + 10 * old + new

            def expected_tempo(t):
                return 1 / (old if t < change_at else new)

        detected, perf = run_audio(x, rate, cfg, expected_tempo=expected_tempo)
        remaining = list(labels)
        matches = []
        fp = Counter()
        for kind, at, confirmed in detected:
            tolerance = 1.5 if kind.startswith("tempo") else 0.12
            index = next(
                (
                    i
                    for i, (expected, t) in enumerate(remaining)
                    if expected == kind and abs(at - t) <= tolerance
                ),
                None,
            )
            if index is None:
                fp[kind] += 1
            else:
                _, target = remaining.pop(index)
                totals[kind]["tp"] += 1
                matches.append(
                    dict(
                        kind=kind,
                        start_error_ms=(at - target) * 1000,
                        decision_delay_ms=(confirmed - target) * 1000,
                    )
                )
        for kind, _ in remaining:
            totals[kind]["fn"] += 1
        for kind, n in fp.items():
            totals[kind]["fp"] += n
        valid_seconds += len(x) / rate
        details.append(
            dict(
                fixture=name,
                matches=matches,
                false_positives=dict(fp),
                missed=remaining,
                **perf,
            )
        )
    metrics = {}
    for kind, c in totals.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        metrics[kind] = dict(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=tp / (tp + fp) if tp + fp else None,
            recall=tp / (tp + fn) if tp + fn else None,
            false_events_per_valid_hour=fp * 3600 / valid_seconds,
        )
    return dict(
        valid_observed_seconds=valid_seconds,
        matching_tolerance_seconds=dict(activity=0.12, loudness=0.12, tempo=1.5),
        by_kind=metrics,
        fixtures=details,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--channels", type=int, default=1)
    args = p.parse_args()
    if not 0 < args.seconds <= 60 or not 1 <= args.channels <= 8:
        p.error("require 0 < seconds <= 60 and 1 <= channels <= 8")
    rate = 48000
    x = tone(args.seconds, rate, frequency=200)
    _, timing = run_audio(x, rate, AcousticConfig(), args.channels)
    _, memory = run_audio(x, rate, AcousticConfig(), args.channels, traced=True)
    sensitivity = []
    for confirmation in (0.02, 0.04, 0.08):
        cfg = replace(AcousticConfig(), offset_confirmation_seconds=confirmation)
        detections, _ = run_audio(
            np.concatenate([np.zeros(14400), tone(0.4, rate), np.zeros(14400)]),
            rate,
            cfg,
        )
        offset = next(
            (
                confirmed - 0.7
                for kind, _, confirmed in detections
                if kind == "sound_offset"
            ),
            None,
        )
        sensitivity.append(
            dict(
                offset_confirmation_seconds=confirmation,
                observed_offset_delay_seconds=offset,
            )
        )
    print(
        json.dumps(
            dict(
                environment=dict(
                    python=platform.python_version(),
                    numpy=np.__version__,
                    platform=platform.platform(),
                    processor=platform.processor(),
                    cpu_count=os.cpu_count(),
                ),
                workload=dict(
                    rate=rate,
                    frames=480,
                    channels=args.channels,
                    fft_seconds=0.02,
                    hop_seconds=0.005,
                ),
                timing=timing,
                memory=memory,
                evaluation=evaluate(),
                sensitivity=sensitivity,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
