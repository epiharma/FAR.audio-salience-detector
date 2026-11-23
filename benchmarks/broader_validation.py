"""Frozen broader acoustic validation, not a detector implementation.

Use --manifest before --evaluate. Labels are checked against the manifest digest.
All acoustic thresholds/defaults stay unchanged. --stress is accelerated audio time.
"""

import argparse
import gc
import hashlib
import json
import math
import runpy
import sys
import tempfile
import time
from collections import Counter, OrderedDict, deque
from dataclasses import asdict
from pathlib import Path

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
MANIFEST = Path(__file__).with_name(
    "acoustic-broader-validation-manifest-2026-11-23.json"
)


def config(channels=1):
    return DetectorConfig(
        ResourceLimits(4, max(8, channels), 1048576, 4194304, 128, 64, 2097152, 8, 16),
        0,
        0,
        ResourceReporting(10, 4096),
        acoustic=AcousticConfig(),
    )


def quantiles(values):
    if not values:
        return None
    v = sorted(values)
    return {
        name: v[min(len(v) - 1, math.ceil(q * len(v)) - 1)]
        for name, q in [("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("max", 1.0)]
    }


def match(predictions, labels, kind, channel):
    """Maximum-cardinality bipartite matching; nearest edges visited first."""
    ps = [p for p in predictions if p["kind"] == kind and p["channel"] == channel]
    gs = [g for g in labels if g["kind"] == kind and g["channel"] == channel]
    edges = [
        [
            j
            for j, g in sorted(
                enumerate(gs), key=lambda pair: abs(p["start"] - pair[1]["frame"])
            )
            if g["frame"] - g["early"] <= p["start"] <= g["frame"] + g["late"]
        ]
        for p in ps
    ]
    owner = {}

    def visit(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or visit(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(len(ps)):
        visit(i, set())
    assigned = set(owner.values())
    dupes = sum(bool(edges[i]) for i in range(len(ps)) if i not in assigned)
    return dict(
        positives=len(gs),
        tp=len(owner),
        fp=len(ps) - len(owner),
        fn=len(gs) - len(owner),
        duplicates=dupes,
        delay_frames=[ps[i]["confirmed"] - gs[j]["frame"] for j, i in owner.items()],
        delivery_delay_frames=[
            ps[i]["emitted_through_frame"] - gs[j]["frame"]
            for j, i in owner.items()
            if ps[i].get("emitted_through_frame") is not None
        ],
        start_error_frames=[ps[i]["start"] - gs[j]["frame"] for j, i in owner.items()],
    )


def fixtures():
    for split, seeds in [("development", (43,)), ("heldout", (107, 809))]:
        for seed in seeds:
            for rate in (8000, 16000, 48000):
                rng = np.random.default_rng(seed + rate)
                phase = 0.0 if split == "development" else 1.17
                freq = 800 if split == "development" else 1133

                def tone(n, a=0.2, f=freq):
                    return a * np.sin(2 * np.pi * f * np.arange(n) / rate + phase)

                def case(name, x, truth, *, scored=KINDS, tempo=None, notes=""):
                    channels = (
                        2
                        if split == "heldout"
                        and seed == 809
                        and name in ("sustained", "level_steps")
                        else 1
                    )
                    data = np.asarray(x, dtype=np.float32)
                    if channels == 2:
                        data = np.tile(data[:, None], (1, 2))
                    labels = []
                    for kind, t in truth:
                        for ch in range(channels):
                            labels.append(
                                dict(
                                    kind=kind,
                                    channel=ch,
                                    frame=round(t * rate),
                                    early=round(0.02 * rate),
                                    late=round(
                                        (
                                            5.0
                                            if name.startswith("gradual")
                                            and kind.startswith("tempo")
                                            else 2.0
                                            if kind.startswith("tempo")
                                            else 0.12
                                        )
                                        * rate
                                    ),
                                )
                            )
                    return dict(
                        name=f"{split}:{seed}:{rate}:{name}",
                        family=name,
                        split=split,
                        seed=seed,
                        rate=rate,
                        channels=channels,
                        samples=data,
                        labels=labels,
                        scored=list(scored),
                        tempo=tempo,
                        notes=notes,
                    )

                x = np.zeros(round(1.4 * rate))
                x[round(0.4 * rate) : round(0.9 * rate)] = tone(round(0.5 * rate))
                yield case(
                    "sustained",
                    x,
                    [
                        ("sound_onset", 0.4),
                        ("sound_offset", 0.9),
                        ("loudness_spike", 0.4),
                        ("loudness_drop", 0.9),
                    ],
                )
                x = np.zeros(rate)
                x[round(0.4 * rate) + 3] = 0.8
                yield case("click", x, [("sound_onset", (0.4 * rate + 3) / rate)])
                x = np.zeros(round(1.3 * rate))
                x[round(0.4 * rate) : round(0.9 * rate)] = tone(
                    round(0.5 * rate), 0.003
                )
                yield case("weak", x, [("sound_onset", 0.4), ("sound_offset", 0.9)])
                x = np.concatenate(
                    [tone(round(0.7 * rate), 0.2, 213), np.zeros(round(0.5 * rate))]
                )
                yield case(
                    "cold_hum_loss", x, [("sound_offset", 0.7), ("loudness_drop", 0.7)]
                )
                x = tone(round(1.5 * rate), 0.2, 213)
                x[round(0.5 * rate) : round(rate)] += tone(
                    round(0.5 * rate), 0.06, 1337
                )
                yield case(
                    "band_over_hum", x, [("sound_onset", 0.5), ("sound_offset", 1.0)]
                )
                n = round(1.5 * rate)
                f = np.fft.rfftfreq(n, 1 / rate)
                noise = rng.normal(size=n)
                spec = np.fft.rfft(noise)
                spec[(f < 650) | (f > 1550)] = 0
                noise = np.fft.irfft(spec, n=n)
                noise /= np.sqrt(np.mean(noise**2))
                t = np.arange(n) / rate
                envelope = 0.8 + 0.2 * np.sin(2 * np.pi * 6 * t)
                x = tone(n, 0.15, 213)
                x[round(0.5 * rate) : round(rate)] += (
                    0.07
                    * envelope[round(0.5 * rate) : round(rate)]
                    * noise[round(0.5 * rate) : round(rate)]
                )
                yield case(
                    "modulated_band_noise",
                    x,
                    [("sound_onset", 0.5), ("sound_offset", 1.0)],
                    notes="Speech-shaped noise surrogate, not speech or a source label.",
                )
                x = np.concatenate(
                    [
                        tone(round(0.5 * rate), 0.2, 213),
                        tone(round(0.7 * rate), 0.2, 1337),
                    ]
                )
                yield case(
                    "opposite_bands", x, [("sound_onset", 0.5), ("sound_offset", 0.5)]
                )
                x = np.concatenate(
                    [
                        tone(round(0.5 * rate), 0.03),
                        tone(round(0.5 * rate), 0.7),
                        tone(round(0.5 * rate), 0.03),
                    ]
                )
                yield case(
                    "level_steps", x, [("loudness_spike", 0.5), ("loudness_drop", 1.0)]
                )
                x = np.concatenate(
                    [
                        tone(round(0.5 * rate), 0.04),
                        tone(round(0.5 * rate), 0.08),
                        tone(round(0.5 * rate), 0.04),
                    ]
                )
                yield case("sub_extreme", x, [])
                n = round(1.5 * rate)
                carrier = rng.uniform(-1, 1, n)
                carrier /= np.sqrt(np.mean(carrier**2))
                gain = np.where(
                    np.arange(n) < round(0.5 * rate),
                    0.25,
                    np.where(np.arange(n) < round(rate), 0.025, 0.25),
                )
                yield case(
                    "noise_gain",
                    carrier * gain,
                    [("loudness_drop", 0.5), ("loudness_spike", 1.0)],
                )
                x = tone(round(1.5 * rate), 0.03, 213)
                t = np.arange(len(x)) / rate
                gain = np.interp(t, [0, 0.5, 1.0, 1.5], [1.0, 1.0, 5.0, 5.0])
                x *= gain
                yield case("slow_gain", x, [])
                for name, intervals, direction in [
                    ("stable_pulses", [0.3] * 24, None),
                    ("abrupt_faster", [0.4] * 10 + [0.2] * 12, "tempo_increase"),
                    ("abrupt_slower", [0.2] * 10 + [0.4] * 12, "tempo_decrease"),
                    (
                        "gradual_faster",
                        [0.5] * 10 + list(np.linspace(0.48, 0.22, 12)) + [0.22] * 6,
                        "tempo_increase",
                    ),
                    (
                        "gradual_slower",
                        [0.22] * 10 + list(np.linspace(0.24, 0.5, 12)) + [0.5] * 6,
                        "tempo_decrease",
                    ),
                    ("one_skipped", [0.3] * 10 + [0.6] + [0.3] * 12, None),
                ]:
                    times = np.cumsum([0.371] + intervals)
                    x = np.zeros(round((times[-1] + 0.15) * rate))
                    for at in times:
                        x[round(at * rate)] = 0.8
                    truth = [("sound_onset", float(at)) for at in times[:4]]
                    if direction:
                        truth.append((direction, float(times[11])))
                    timeline = [
                        (float(at), 1 / (intervals[max(0, i - 1)]))
                        for i, at in enumerate(times)
                    ]
                    if name == "one_skipped":
                        timeline = [(float(at), 1 / 0.3) for at in times]
                    yield case(name, x, truth, tempo=timeline)
                # Competing pulse trains have no changing individual rate. Only
                # temporal assertions are scored; activity grouping is undefined.
                aa = np.arange(0.4, 5.5, 0.2)
                bb = np.arange(0.47, 5.5, 0.313)
                x = np.zeros(round(5.8 * rate))
                for at in aa:
                    x[round(at * rate)] += 0.4
                for at in bb:
                    x[round(at * rate)] += 0.4
                yield case(
                    "competing_pulses",
                    x,
                    [],
                    scored=("tempo_increase", "tempo_decrease"),
                    notes="Activity labels undefined; all unscored predictions retained. Tempo should be unavailable once mixture is ambiguous.",
                )
                # Drift is intentionally ambiguous activity ground truth: record
                # diagnostics/availability without manufacturing positive labels.
                n = 3 * rate
                raw = rng.normal(0, 0.02, n)
                t = np.arange(n) / rate
                x = raw + 0.06 * np.sin(2 * np.pi * (400 * t + 100 * t * t))
                yield case(
                    "spectral_drift",
                    x,
                    [],
                    scored=(),
                    notes="Unscored exploratory changing spectrum; no claim of negative ground truth.",
                )


def description(case):
    d = {k: v for k, v in case.items() if k != "samples"}
    d["frames"] = len(case["samples"])
    d["sample_sha256"] = hashlib.sha256(case["samples"].tobytes()).hexdigest()
    return d


def freeze():
    data = dict(
        policy="Frozen before evaluation. Procedural split is not independent field data.",
        cases=[description(c) for c in fixtures()],
    )
    encoded = json.dumps(data, indent=2)
    if MANIFEST.exists() and MANIFEST.read_text(encoding="utf-8") != encoded:
        raise ValueError("Refusing to replace the frozen manifest")
    MANIFEST.write_text(encoded, encoding="utf-8")
    return dict(
        manifest=str(MANIFEST),
        sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        cases=len(data["cases"]),
    )


def run_case(case):
    rate = case["rate"]
    d = AudioSalienceDetector(config(case["channels"]))
    s = SourceSpec.microphone(
        label="validation fixture", sample_rate_hz=rate, channel_count=case["channels"]
    )
    d.start_source(s)
    preds = []
    revisions = []
    changes = {}
    sequence = 0
    violations = []
    confidence = []
    tempo_error = []
    tempo_valid = 0
    windows = 0
    traces = deque(maxlen=96)
    band_missing = Counter()
    tempo_reasons = Counter()
    sequence_changes = []
    last_sequence_by_channel = {}
    revision_lags = []
    first_confirmation = {}

    def consume(batch, through_frame):
        nonlocal sequence, windows, tempo_valid
        for event in batch.ordered:
            if event.header.emission_sequence != sequence + 1:
                violations.append("emission_sequence")
            sequence = event.header.emission_sequence
        for r in batch.salience:
            for w in r.windows:
                windows += 1
                tempo_reasons[w.tempo_reason or "available"] += 1
                channel = s.channel_ids.index(r.channels[0].channel_id)
                if (
                    w.sequence_id is not None
                    and last_sequence_by_channel.get(channel) != w.sequence_id
                ):
                    sequence_changes.append(
                        dict(
                            channel=channel,
                            frame=w.interval.stop_frame,
                            sequence_id=w.sequence_id,
                        )
                    )
                    last_sequence_by_channel[channel] = w.sequence_id
                confidence.append(w.confidence)
                for band in w.bands:
                    if band.power is None:
                        band_missing[band.reason] += 1
                if w.tempo_hz is not None:
                    tempo_valid += 1
                    if case["tempo"]:
                        truth = case["tempo"][0][1]
                        for at, hz in case["tempo"]:
                            if at * rate <= w.interval.stop_frame:
                                truth = hz
                        tempo_error.append(abs(w.tempo_hz - truth))
                near = any(
                    abs(w.interval.stop_frame - g["frame"]) <= round(0.15 * rate)
                    for g in case["labels"]
                )
                if near or r.changes:
                    traces.append(
                        dict(
                            frame=w.interval.stop_frame,
                            power=w.power,
                            confidence=w.confidence,
                            tempo=w.tempo_hz,
                            reason=w.tempo_reason,
                            bands=[asdict(v) for v in w.bands],
                        )
                    )
            for c in r.changes:
                revisions.append(
                    dict(
                        change_id=c.change_id,
                        revision=c.revision,
                        phase=c.phase,
                        kind=c.kind,
                        channel=s.channel_ids.index(c.channel.channel_id),
                        start_frame=c.start_frame,
                        confirmed_frame=c.confirmed_frame,
                        last_observed_frame=c.last_observed_frame,
                        end_frame=c.end_frame,
                        bands=[asdict(b) for b in c.bands],
                    )
                )
                old = changes.get(c.change_id)
                if c.revision != (old.revision + 1 if old else 1):
                    violations.append("change_revision")
                if old and c.kind != old.kind:
                    violations.append("change_kind_mutation")
                if c.phase == "interrupted" and (
                    c.end_frame is not None or c.duration_frames is not None
                ):
                    violations.append("interrupted_ending")
                if c.phase not in ("detected", "completed", "interrupted"):
                    violations.append("phase")
                if c.revision == 1:
                    first_confirmation[c.change_id] = c.confirmed_frame
                    preds.append(
                        dict(
                            kind=c.kind,
                            channel=s.channel_ids.index(c.channel.channel_id),
                            start=c.start_frame,
                            confirmed=c.confirmed_frame,
                            emitted_through_frame=through_frame,
                            change_id=c.change_id,
                            confidence=c.confidence,
                        )
                    )
                else:
                    revision_lags.append(
                        c.last_observed_frame - first_confirmation[c.change_id]
                    )
                changes[c.change_id] = c

    sizes = (480,) if case["split"] == "development" else (137, 480, 997)
    first = 0
    i = 0
    while first < len(case["samples"]):
        block = np.ascontiguousarray(
            case["samples"][first : first + sizes[i % len(sizes)]]
        )
        out = d.process(
            s.chunk(
                memoryview(block.ravel()),
                first_frame=first,
                measured_start_ns=first * 1000000000 // rate,
            )
        )
        if out.refusals:
            violations.append("unexpected_refusal")
        for result in out.salience:
            for w in result.windows:
                if (
                    result.timestamp is not None
                    and result.timestamp.clock_id != s.clock_id
                ):
                    violations.append("measurement_clock")
        consume(out, first + len(block))
        first += len(block)
        i += 1
    consume(d.close(), first)
    metrics = {}
    for kind in case["scored"]:
        combined = Counter()
        delays = []
        delivery_delays = []
        errors = []
        for ch in range(case["channels"]):
            m = match(preds, case["labels"], kind, ch)
            delays += m.pop("delay_frames")
            delivery_delays += m.pop("delivery_delay_frames")
            errors += m.pop("start_error_frames")
            combined.update(m)
        metrics[kind] = dict(
            combined,
            case_channel_negatives=sum(
                not any(
                    g["kind"] == kind and g["channel"] == ch for g in case["labels"]
                )
                for ch in range(case["channels"])
            ),
            delay_ms=[v * 1000 / rate for v in delays],
            delivery_delay_ms=[v * 1000 / rate for v in delivery_delays],
            start_error_ms=[v * 1000 / rate for v in errors],
            valid_channel_seconds=len(case["samples"]) * case["channels"] / rate,
        )
    errors = any(m["fp"] or m["fn"] for m in metrics.values()) or bool(violations)
    return dict(
        **description(case),
        predictions=preds,
        revisions=revisions,
        metrics=metrics,
        violations=violations,
        tempo_availability=tempo_valid / max(1, windows),
        tempo_mae_hz=sum(tempo_error) / len(tempo_error) if tempo_error else None,
        confidence=quantiles(confidence),
        revision_lag_ms=quantiles([v * 1000 / rate for v in revision_lags]),
        missing_bands=dict(band_missing),
        tempo_reasons=dict(tempo_reasons),
        sequence_changes=sequence_changes,
        trace=list(traces) if errors else [],
    )


def evaluate():
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = list(fixtures())
    assert json.loads(json.dumps([description(c) for c in cases])) == frozen["cases"], (
        "Frozen fixture/label manifest changed"
    )
    rows = []
    for i, case in enumerate(cases):
        rows.append(run_case(case))
        if i % 12 == 0:
            print(f"evaluated {i + 1}/{len(cases)}", file=sys.stderr, flush=True)
    summaries = {}
    for split in ("development", "heldout"):
        summary = {}
        for kind in KINDS:
            ms = [
                r["metrics"][kind]
                for r in rows
                if r["split"] == split and kind in r["metrics"]
            ]
            totals = {
                k: sum(m.get(k, 0) for m in ms)
                for k in (
                    "positives",
                    "case_channel_negatives",
                    "tp",
                    "fp",
                    "fn",
                    "duplicates",
                    "valid_channel_seconds",
                )
            }
            totals.update(
                precision=totals["tp"] / (totals["tp"] + totals["fp"])
                if totals["tp"] + totals["fp"]
                else None,
                recall=totals["tp"] / totals["positives"]
                if totals["positives"]
                else None,
                fp_per_valid_channel_hour=totals["fp"]
                * 3600
                / totals["valid_channel_seconds"]
                if totals["valid_channel_seconds"]
                else None,
                delay_ms=quantiles([v for m in ms for v in m["delay_ms"]]),
            )
            summary[kind] = totals
        summaries[split] = summary
    return dict(
        manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        summaries=summaries,
        cases=rows,
    )


def perf():
    rows = []
    for rate, block, ch in [
        (48000, 480, 1),
        (48000, 480, 2),
        (48000, 480, 8),
        (48000, 120, 1),
        (16000, 160, 1),
        (44100, 441, 2),
    ]:
        d = AudioSalienceDetector(config(ch))
        s = SourceSpec.microphone(
            label="performance", sample_rate_hz=rate, channel_count=ch
        )
        d.start_source(s)
        lat = []
        cpu = time.process_time()
        wall = time.perf_counter()
        for first in range(0, rate * 5, block):
            samples = np.tile(
                (
                    0.1 * np.sin(2 * np.pi * 213 * (first + np.arange(block)) / rate)
                ).astype(np.float32)[:, None],
                (1, ch),
            ).ravel()
            observation = s.chunk(memoryview(samples), first_frame=first)
            start = time.perf_counter_ns()
            out = d.process(observation)
            lat.append((time.perf_counter_ns() - start) / 1e6)
            assert not out.refusals
        elapsed = time.perf_counter() - wall
        rows.append(
            dict(
                rate=rate,
                block=block,
                channels=ch,
                process_ms=quantiles(lat),
                wall_seconds=elapsed,
                cpu_seconds=time.process_time() - cpu,
                audio_seconds=5,
                throughput=5 / elapsed,
            )
        )
        d.close()
    return rows


def stress(seconds):
    rss = runpy.run_path(str(Path(__file__).with_name("acoustic_audit.py")))["rss"]
    rate = 48000
    d = AudioSalienceDetector(config(2))
    s = SourceSpec.microphone(label="long stress", sample_rate_hz=rate, channel_count=2)
    epoch = 0
    last = 0
    run_id = d.run_id
    revisions = OrderedDict()
    ids = deque(maxlen=1024)
    samples = []
    counts = Counter()
    maxima = Counter()
    refusals = 0
    snapshots = 0
    restarts = 0
    prior_candidates = set()
    rng = np.random.default_rng(12026)

    def consume(out):
        nonlocal last, refusals
        refusals += len(out.refusals)
        for event in out.ordered:
            assert event.header.run_id == run_id
            assert event.header.emission_sequence == last + 1
            assert event.header.event_id not in ids
            ids.append(event.header.event_id)
            last = event.header.emission_sequence
        for r in out.salience:
            for c in r.changes:
                if c.change_id in revisions:
                    assert c.revision == revisions[c.change_id] + 1
                else:
                    assert c.revision == 1
                revisions[c.change_id] = c.revision
                revisions.move_to_end(c.change_id)
                if len(revisions) > 512:
                    revisions.popitem(last=False)
                counts[c.kind] += c.revision == 1

    consume(d.start_source(s))
    cpu = time.process_time()
    began = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="far-validation-") as td:
        path = Path(td) / "state.json"
        for first in range(0, seconds * rate, 480):
            epoch_frame = first - epoch
            mode = (first // rate // 60) % 5
            x = (0.03, 0.1, 0.2, 0.05, 0.1)[mode] * np.sin(
                2 * np.pi * 213 * (first + np.arange(480)) / rate
            )
            if mode == 0:
                x[:] = 0
            if mode == 2:
                x = 0.1 * rng.uniform(-1, 1, 480)
            if mode == 3:
                x[:] = 0
                if first % (rate // 2) == 0:
                    x[0] = 0.8
            if mode == 4:
                x += 0.03 * np.sin(2 * np.pi * 1133 * (first + np.arange(480)) / rate)
            both = (first // rate) % 60 == 0
            data = np.tile(x[:, None], (1, 2)).ravel() if both else x
            out = d.process(
                s.chunk(
                    memoryview(np.asarray(data, dtype=np.float32)),
                    first_frame=epoch_frame,
                    channel_ids=s.channel_ids if both else s.channel_ids[:1],
                )
            )
            consume(out)
            for st in d._acoustic.states.values():
                sizes = dict(
                    impulses=len(st.rhythm.times),
                    profiles=len(st.rhythm.profiles),
                    groups=len(st.group),
                    changes=len(st.changes),
                    history=len(st.history),
                )
                assert sizes["impulses"] <= d.config.acoustic.max_impulses
                assert sizes["profiles"] == sizes["impulses"] and sizes["groups"] <= 6
                assert sizes["changes"] <= d.config.acoustic.max_open_changes
                assert (
                    sizes["history"]
                    <= math.ceil(
                        max(d.config.acoustic.horizons_seconds) * rate / st.plan.hop
                    )
                    + 1
                )
                for k, v in sizes.items():
                    maxima[k] = max(maxima[k], v)
            if first % (rate * 30) == 0:
                assert prior_candidates <= set(d._baselines.records), "silent eviction"
                prior_candidates = set(d._baselines.records)
                maxima["candidates"] = max(maxima["candidates"], len(prior_candidates))
                before = rss()
                gc.collect()
                samples.append(
                    dict(
                        audio_second=first / rate,
                        rss_before_gc=before,
                        rss_after_gc=rss(),
                        candidates=len(prior_candidates),
                        parameters=sum(len(v) for v in d._acoustic.models.values())
                        + sum(len(v["stats"]) for v in d._acoustic.candidates.values()),
                    )
                )
                print(
                    f"stress audio seconds {first // rate}/{seconds}",
                    file=sys.stderr,
                    flush=True,
                )
            if first and first % (rate * 60) == 0:
                d.save_baselines(path)
                snapshots += 1
                if first % (rate * 300) == 0:
                    consume(d.end_source(s.ref))
                    d.save_baselines(path)
                    d = d.from_baselines(path, config=d.config)
                    assert not d._acoustic.states
                    run_id = d.run_id
                    last = 0
                    ids.clear()
                    revisions.clear()
                    s = s.new_session()
                    epoch = first + 480
                    consume(d.start_source(s))
                    restarts += 1
        consume(d.close())
    return dict(
        audio_seconds=seconds,
        calls=seconds * 100,
        wall_seconds=time.perf_counter() - began,
        cpu_seconds=time.process_time() - cpu,
        snapshots=snapshots,
        restarts=restarts,
        refusals=refusals,
        maxima=dict(maxima),
        changes=dict(counts),
        rss_samples=samples,
    )


def quality():
    results = []
    for rate in (8000, 16000, 44100, 48000):
        d = AudioSalienceDetector(config())
        source = SourceSpec.microphone(label="timestamp rounding", sample_rate_hz=rate)
        d.start_source(source)
        count = 0
        for first in range(0, rate, 137):
            n = min(137, rate - first)
            x = np.asarray(
                0.1 * np.sin(2 * np.pi * 800 * (first + np.arange(n)) / rate),
                dtype=np.float32,
            )
            anchor = 123456789 + first * 1000000000 // rate
            batch = d.process(
                source.chunk(memoryview(x), first_frame=first, measured_start_ns=anchor)
            )
            for result in batch.salience:
                for window in result.windows:
                    assert window.interval.source == source.ref
                    expected = (
                        anchor
                        + (window.interval.stop_frame - 1 - first) * 1000000000 // rate
                    )
                    assert result.timestamp.ns == expected
                    assert result.timestamp.clock_id == source.clock_id
                    count += 1
        d.close()
        results.append(
            dict(check="per_anchor_timestamp", rate=rate, windows=count, passed=True)
        )
    d = AudioSalienceDetector(config(2))
    sources = [
        SourceSpec.microphone(label="A", sample_rate_hz=48000),
        SourceSpec.microphone(label="B", sample_rate_hz=16000),
    ]
    for source in sources:
        d.start_source(source)
    frames = [0, 0]
    counts = [0, 0]
    for i in range(100):
        for j, source in enumerate(sources):
            if j == 1 and i % 3:
                continue
            n = source.sample_rate_hz // 100
            x = np.asarray(
                0.1
                * np.sin(
                    2 * np.pi * 800 * (frames[j] + np.arange(n)) / source.sample_rate_hz
                ),
                dtype=np.float32,
            )
            batch = d.process(
                source.chunk(
                    memoryview(x),
                    first_frame=frames[j],
                    measured_start_ns=frames[j] * 1000000000 // source.sample_rate_hz,
                )
            )
            assert not batch.health and not batch.refusals
            for result in batch.salience:
                assert all(
                    key.source_id == source.ref.source_id for key in result.channels
                )
                assert result.header.provenance.source_clocks == (
                    (source.ref, source.clock_id),
                )
            frames[j] += n
            counts[j] += len(batch.salience)
    before = d._sequence
    time.sleep(0.1)
    assert d._sequence == before
    d.close()
    results.append(
        dict(
            check="asynchronous_independent_sources_and_idle",
            clock_ids_distinct=sources[0].clock_id != sources[1].clock_id,
            frames=frames,
            results=counts,
            passed=True,
        )
    )
    # Reuse existing frozen tests for rich layout/reorder/overlap checks rather
    # than modifying their data or interpretation.
    return results


def independent_rhythm():
    cases = []
    for rate in (11025, 22050, 48000):
        for family, intervals, direction in [
            ("stable", [0.35] * 23, None),
            ("abrupt_faster", [0.35] * 10 + [0.175] * 12, "tempo_increase"),
            ("abrupt_slower", [0.18] * 10 + [0.36] * 12, "tempo_decrease"),
            (
                "gradual_faster",
                [0.46] * 10 + list(np.linspace(0.44, 0.21, 12)) + [0.21] * 6,
                "tempo_increase",
            ),
            (
                "gradual_slower",
                [0.21] * 10 + list(np.linspace(0.23, 0.46, 12)) + [0.46] * 6,
                "tempo_decrease",
            ),
            ("skipped", [0.35] * 10 + [0.7] + [0.35] * 12, None),
        ]:
            times = np.cumsum([0.443] + intervals)
            x = np.zeros(round((times[-1] + 0.17) * rate))
            for at in times:
                x[round(at * rate)] = 0.63
            truth = [
                dict(
                    kind="sound_onset",
                    channel=0,
                    frame=round(at * rate),
                    early=round(0.02 * rate),
                    late=round(0.12 * rate),
                )
                for at in times[:4]
            ]
            if direction:
                truth.append(
                    dict(
                        kind=direction,
                        channel=0,
                        frame=round(times[11] * rate),
                        early=round(0.02 * rate),
                        late=round((5 if family.startswith("gradual") else 2) * rate),
                    )
                )
            cases.append(
                dict(
                    name=f"independent:20261123:{rate}:{family}",
                    family=family,
                    split="heldout",
                    seed=20261123,
                    rate=rate,
                    channels=1,
                    samples=x.astype(np.float32),
                    labels=truth,
                    scored=list(KINDS),
                    tempo=[
                        (
                            float(at),
                            1
                            / (
                                0.35
                                if family == "skipped"
                                else intervals[max(0, i - 1)]
                            ),
                        )
                        for i, at in enumerate(times)
                    ],
                    notes="Different rhythm parameters/rates than first frozen set; same procedural family, not field data.",
                )
            )
    path = Path(__file__).with_name("acoustic-broader-rhythm-manifest-2026-11-23.json")
    serial = json.loads(json.dumps([description(c) for c in cases]))
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == serial
    else:
        path.write_text(json.dumps(serial, indent=2), encoding="utf-8")
    # All labels are frozen before the first detector evaluation in this set.
    original = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert not {c["sample_sha256"] for c in serial} & {
        c["sample_sha256"] for c in original["cases"]
    }
    return dict(
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cases=[run_case(c) for c in cases],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    group.add_argument("--perf", action="store_true")
    group.add_argument("--stress", type=int)
    group.add_argument("--quality", action="store_true")
    group.add_argument("--case", type=str)
    group.add_argument("--independent-rhythm", action="store_true")
    args = p.parse_args()
    if args.quality:
        print(json.dumps(quality(), indent=2))
        return
    if args.independent_rhythm:
        print(json.dumps(independent_rhythm(), indent=2))
        return
    if args.case:
        case = next(c for c in fixtures() if c["name"] == args.case)
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert json.loads(json.dumps(description(case))) == next(
            c for c in manifest["cases"] if c["name"] == args.case
        )
        result = run_case(case)
        print(json.dumps(result, indent=2))
        if result["violations"] or any(
            m["fp"] or m["fn"] for m in result["metrics"].values()
        ):
            raise SystemExit(1)
        return
    result = (
        freeze()
        if args.manifest
        else evaluate()
        if args.evaluate
        else perf()
        if args.perf
        else stress(args.stress)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
