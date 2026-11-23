"""Deterministic observable/state trace for comparing two source checkouts.

Run in separate processes with --source <checkout>/src. No baseline code is
vendored. UUIDs and diagnostic clocks are controlled, not discarded afterward.
"""

import argparse
import hashlib
import itertools
import json
import random
import sys
import tempfile
from array import array
from dataclasses import asdict, replace
from pathlib import Path
from uuid import UUID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--retention-ns", type=int, default=100_000_000)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import far_audio_salience as far
    import far_audio_salience.contracts as contracts
    import far_audio_salience.detector as implementation

    ids, ticks = itertools.count(1), itertools.count(1)
    contracts.uuid4 = lambda: UUID(int=next(ids))
    implementation.time.monotonic_ns = lambda: next(ticks) * 100

    def digest(value):
        data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        return hashlib.sha256(data).hexdigest()

    def evidence(index, rate):
        if isinstance(index, dict):
            return [
                (
                    f,
                    s.value,
                    asdict(s.measured) if s.measured else None,
                    asdict(s.received) if s.received else None,
                )
                for f, s in sorted(index.items())
            ]
        rows = []
        for span in index.spans:
            values = memoryview(span.backing).cast("f") if span.valid else None
            for f in range(span.start, span.stop):
                measured = span.timing.at(f, rate)
                rows.append(
                    (
                        f,
                        None
                        if values is None
                        else values[span.offset + f - span.start],
                        asdict(measured) if measured else None,
                        asdict(span.timing.received) if span.timing.received else None,
                    )
                )
        return rows

    def state(detector):
        lanes = []
        for key, lane in sorted(
            detector._lanes.items(), key=lambda item: repr(item[0])
        ):
            rate = detector._sources[key.source_id].sample_rate_hz
            lanes.append(
                (
                    asdict(key),
                    lane.watermark,
                    lane.high_water,
                    lane.unavailable,
                    lane.recovery_after,
                    evidence(lane.pending, rate),
                    evidence(lane.history, rate),
                )
            )
        return dict(
            lanes=lanes,
            count=detector.buffered_sample_count,
            sequence=detector._sequence,
            active=sorted(detector._active),
            sources=[asdict(s) for s in detector.sources],
            candidates=[asdict(c) for c in detector.candidates()],
            sync={k: asdict(v) for k, v in detector._sync.items()},
            windows={k: asdict(v) for k, v in detector._windows.items()},
            reference=detector._reference_times,
            feature_clocks=detector._feature_clocks,
        )

    trace = []

    def call(detector, name, *args, **kwargs):
        before = digest(state(detector))
        try:
            output = getattr(detector, name)(*args, **kwargs)
            result = asdict(output) if output is not None else None
        except (far.StructuralInputError, far.StateError) as error:
            result = dict(error=type(error).__name__, message=str(error))
            assert digest(state(detector)) == before
        trace.append(dict(operation=name, output=result, state=digest(state(detector))))
        return output if "output" in locals() else None

    for delay in (0, 1_000_000_000):
        config = far.DetectorConfig(
            far.ResourceLimits(4, 12, 512, 4096, 8, 4, 100000, 2, 8),
            delay,
            args.retention_ns,
            far.ResourceReporting(3, 8),
        )
        detector = far.AudioSalienceDetector(config)
        sources = [
            far.SourceSpec.microphone(
                label=str(i), sample_rate_hz=rate, channel_count=3
            )
            for i, rate in enumerate((44100, 48000))
        ]
        for source in sources:
            call(detector, "start_source", source)
        rng = random.Random(args.seed + delay)
        for step in range(180):
            spec = sources[step % 2]
            high = max(
                detector._lanes[far.ChannelKey(spec.ref.source_id, c)].high_water
                for c in spec.channel_ids
            )
            start = rng.randrange(max(1, high + 9))
            size = rng.randrange(1, 17)
            channels = list(spec.channel_ids)
            rng.shuffle(channels)
            channels = tuple(channels[: rng.randrange(1, 4)])
            planes = [
                array(
                    "f",
                    [
                        ((f * 17 + spec.channel_ids.index(c)) % 11 - 5) / 8
                        for f in range(start, start + size)
                    ],
                )
                for c in channels
            ]
            if step % 13 == 0:
                planes[0][-1] = float("nan")
            elif step % 17 == 0:
                planes[0][-1] = 0.875
            kwargs = dict(
                first_frame=start,
                measured_start_ns=123 + start * 10**9 // spec.sample_rate_hz,
                received_at=far.ClockStamp("receipt", step),
                channel_ids=channels,
            )
            if step % 2:
                chunk = spec.chunk(
                    tuple(memoryview(p) for p in planes), layout="planar", **kwargs
                )
            else:
                chunk = spec.chunk(
                    memoryview(array("f", [p[i] for i in range(size) for p in planes])),
                    **kwargs,
                )
            call(detector, "process", chunk)
            if step % 19 == 0:
                call(detector, "process", chunk)
            if step % 11 == 0:
                before = max(
                    detector._lanes[far.ChannelKey(spec.ref.source_id, c)].watermark
                    for c in channels
                )
                call(
                    detector,
                    "advance_watermark",
                    far.SourceWatermark(spec.ref, channels, max(before, high)),
                )
            if step % 23 == 0:
                call(
                    detector,
                    "process",
                    far.UnavailableObservation(spec.ref, channels, "missing"),
                )
                call(
                    detector,
                    "process",
                    far.UnavailableObservation(
                        spec.ref,
                        channels,
                        "missing",
                        far.SourceInterval(spec.ref, high, high + 2),
                    ),
                )
            if step % 29 == 0:
                call(
                    detector,
                    "process",
                    replace(chunk, interval=far.SourceInterval(spec.ref, 2, 1)),
                )
        scope = tuple(
            far.ChannelKey(s.ref.source_id, s.channel_ids[0]) for s in sources
        )
        policy = far.EligibilityPolicy(0.5)
        call(
            detector,
            "register_feature",
            far.FeatureSpec(
                "joint",
                scope,
                policy,
                policy,
                policy,
                True,
                far.AlignmentPolicy(10, 4, 0),
                20,
            ),
        )
        alignment = far.AlignmentEvidence(
            "alignment",
            scope,
            "ref",
            0,
            10,
            far.ClockStamp("ref", 0),
            1,
            tuple(far.ClockMapping(s.ref, s.clock_id, 0, 0, 1, 1) for s in sources),
        )
        call(detector, "update_alignment", alignment)
        for i in range(3):
            call(
                detector,
                "open_relationship_window",
                far.RelationshipWindow(f"w{i}", "joint", "ref", i, i + 2),
            )
        call(detector, "advance_reference_time", far.ClockStamp("ref", 30))
        created = call(
            detector,
            "register_candidate",
            scope,
            compatibility=(far.BaselineCompatibility("joint", "compatible"),),
        )
        record = created.baseline_lifecycle[0].after
        call(
            detector,
            "report_readiness",
            far.ReadinessEvidence(
                record.candidate_id,
                1,
                False,
                False,
                (far.BaselineCompatibility("joint", "incompatible"),),
            ),
        )
        command = far.CandidateCommand(
            far.new_id(), record.candidate_id, record.control_revision, "block"
        )
        call(detector, "command", command)
        call(detector, "command", command)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            detector.save_baselines(path)
            trace.append(dict(snapshot=json.loads(path.read_text())))
            restored = far.AudioSalienceDetector.from_baselines(path, config=config)
            call(restored, "command", command)
            call(restored, "start_source", restored.sources[0].new_session())
            call(restored, "close")
        for spec in sources:
            call(detector, "end_source", spec.ref)
        call(detector, "reset_learning")
        call(detector, "factory_reset")
        call(detector, "close")
    args.output.write_text(
        json.dumps(trace, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"{len(trace)} deterministic outputs/state checkpoints: {digest(trace)}")


if __name__ == "__main__":
    main()
