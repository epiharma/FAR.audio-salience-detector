"""Optional blocking microphone adapter; use --acoustic for experimental DSP."""

import argparse
import importlib
import json
import math
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field

from far_audio_salience import (
    AcousticConfig,
    AcquisitionQuality,
    AudioSalienceDetector,
    ClockStamp,
    DetectorConfig,
    ResourceLimits,
    ResourceReporting,
    SourceSpec,
    UnavailableObservation,
    new_id,
)


class AdapterError(RuntimeError):
    pass


@dataclass
class Statistics:
    captured: int = 0
    accepted: int = 0
    results: int = 0
    health: int = 0
    lifecycle: int = 0
    overflows: int = 0
    refusals: int = 0
    epochs: int = 0
    process_calls: int = 0
    last_sequence: int = 0
    run_id: str | None = None
    sequence_valid: bool = True
    worst_ns: int = 0
    latency: deque = field(default_factory=lambda: deque(maxlen=2048))
    device: dict | None = None
    actual: dict | None = None
    acoustic_changes: dict = field(default_factory=dict)

    def consume(self, batch):
        self.results += len(batch.salience)
        for result in batch.salience:
            for change in result.changes:
                if change.revision == 1:
                    self.acoustic_changes[change.kind] = (
                        self.acoustic_changes.get(change.kind, 0) + 1
                    )
        self.health += len(batch.health)
        self.lifecycle += len(batch.baseline_lifecycle)
        self.refusals += len(batch.refusals)
        self.accepted += sum(
            (span.interval.stop_frame - span.interval.start_frame)
            * len(span.channel_ids)
            for receipt in batch.input_receipts
            for span in receipt.accepted
        )
        if self.run_id is None:
            self.run_id = batch.run_id
        if batch.run_id != self.run_id:
            self.sequence_valid = False
        for event in batch.ordered:
            header = event.header
            if (
                header.run_id != self.run_id
                or header.emission_sequence != self.last_sequence + 1
            ):
                self.sequence_valid = False
            self.last_sequence = header.emission_sequence
        if batch.refusals:
            raise AdapterError(
                "FAR refusal: " + json.dumps([asdict(r) for r in batch.refusals])
            )
        if not self.sequence_valid:
            raise AdapterError(
                "Unexpected emission sequence gap, duplicate or run change"
            )

    def report(self):
        ordered = sorted(self.latency)

        def percentile(q):
            return (
                ordered[max(0, math.ceil(q * len(ordered)) - 1)] / 1e6
                if ordered
                else None
            )

        return dict(
            device=self.device,
            actual_stream=self.actual,
            captured_frames=self.captured,
            accepted_frames=self.accepted,
            salience_results=self.results,
            acoustic_changes=dict(self.acoustic_changes),
            health_events=self.health,
            baseline_events=self.lifecycle,
            overflows=self.overflows,
            refusals=self.refusals,
            epochs=self.epochs,
            process_calls=self.process_calls,
            sequence_order_valid=self.sequence_valid,
            last_sequence=self.last_sequence,
            latency_scope=f"most recent {len(ordered)} calls (capacity 2048)",
            process_p50_ms=percentile(0.5),
            process_p95_ms=percentile(0.95),
            process_p99_ms=percentile(0.99),
            process_worst_ms=self.worst_ns / 1e6 if self.process_calls else None,
        )


def configuration(rate, block_size, *, acoustic=False):
    """One source/channel, 250 ms comparison; explicit smoke-test resource sizes."""
    return DetectorConfig(
        ResourceLimits(
            max_sources=1,
            max_channels=1,
            max_chunk_bytes=4 * block_size,
            max_buffer_bytes=4 * (rate + block_size),
            max_baselines=1,
            max_receipts=4,
            max_snapshot_bytes=65536,
            max_relationship_windows=1,
            max_features=1,
        ),
        max_reorder_delay_ns=0,
        comparison_retention_ns=250_000_000,
        reporting=ResourceReporting(aggregate_every=10, materially_worse_by=4096),
        acoustic=AcousticConfig() if acoustic else None,
    )


def run(
    args,
    sd,
    *,
    detector_factory=AudioSalienceDetector,
    clock=time.monotonic_ns,
    timer=time.perf_counter_ns,
    emit=print,
    stats=None,
):
    """Injected capture/clock/reporting make normal tests independent of hardware."""
    stats = Statistics() if stats is None else stats
    detector = stream = source = None
    source_active = capture_active = False
    exit_code = 0
    try:
        if args.list_devices:
            emit(str(sd.query_devices()))
            return 0
        try:
            info = sd.query_devices(args.device, "input")
            sd.check_input_settings(
                device=args.device,
                channels=1,
                dtype="float32",
                samplerate=args.sample_rate,
            )
        except Exception as error:
            raise AdapterError(
                f"Input settings unsupported: {error}\nAvailable devices:\n{sd.query_devices()}"
            ) from error
        stats.device = dict(info)
        detector = detector_factory(
            configuration(
                args.sample_rate,
                args.block_size,
                acoustic=getattr(args, "acoustic", False),
            )
        )
        source = SourceSpec.microphone(
            label=str(info["name"]), sample_rate_hz=args.sample_rate
        )
        stats.consume(detector.start_source(source))
        source_active = True
        stats.epochs = 1
        stream = sd.RawInputStream(
            device=args.device,
            samplerate=args.sample_rate,
            blocksize=args.block_size,
            channels=1,
            dtype="float32",
        )
        stats.actual = dict(
            device=stream.device,
            sample_rate=stream.samplerate,
            channels=stream.channels,
            dtype=stream.dtype,
            block_size=stream.blocksize,
        )
        if (
            stream.samplerate != args.sample_rate
            or stream.channels != 1
            or stream.dtype != "float32"
            or stream.blocksize != args.block_size
        ):
            raise AdapterError(
                f"Actual stream settings differ from requested: {stats.actual}"
            )
        emit(
            "Experimental DSP smoke test; receipt-only timing, uncalibrated scores."
            if getattr(args, "acoustic", False)
            else "Microphone contract smoke test: acoustic extraction/scoring/learning are not implemented; None scores and not_implemented are expected."
        )
        emit(
            "Frames are delivered-buffer positions within each FAR epoch, not a verified hardware counter. Silence is valid audio; idle calls alone imply no loss."
        )
        emit(json.dumps(stats.report()))
        stream.start()
        capture_active = True
        began = clock()
        deadline = None if args.duration is None else began + int(args.duration * 1e9)
        next_report = began + 1_000_000_000
        receipt_clock = new_id()
        first_frame = 0
        while deadline is None or clock() < deadline:
            data, overflow = stream.read(args.block_size)
            received = ClockStamp(receipt_clock, clock())
            with memoryview(data) as raw:
                if (
                    raw.ndim != 1
                    or not raw.c_contiguous
                    or not raw.nbytes
                    or raw.nbytes % 4
                ):
                    raise AdapterError(
                        "Capture returned an empty, noncontiguous or non-float32-aligned buffer"
                    )
                with raw.cast("B") as byte_view, byte_view.cast("f") as samples:
                    frames = len(samples)
                    stats.captured += frames
                    if overflow:
                        stats.overflows += 1
                        stats.consume(
                            detector.process(
                                UnavailableObservation(
                                    source.ref,
                                    source.channel_ids,
                                    "missing",
                                    received_at=received,
                                )
                            )
                        )
                        ended = detector.end_source(source.ref)
                        source_active = False
                        stats.consume(ended)
                        source = source.new_session()
                        started = detector.start_source(source)
                        source_active = True
                        stats.consume(started)
                        stats.epochs += 1
                        first_frame = 0
                        emit(
                            "Input overflow: unknown loss reported; fresh FAR acquisition epoch, stable source/channel IDs. Missing counts and measurement time remain unknown."
                        )
                    observation = source.chunk(
                        samples,
                        first_frame=first_frame,
                        received_at=received,
                        quality=AcquisitionQuality(overflow=bool(overflow)),
                    )
                    first_frame += (
                        frames  # Count every delivered block, including refused audio.
                    )
                    tick = timer()
                    output = detector.process(observation)
                    elapsed = timer() - tick
                    stats.process_calls += 1
                    stats.latency.append(elapsed)
                    stats.worst_ns = max(stats.worst_ns, elapsed)
                    stats.consume(output)
                    del observation, output
            now = clock()
            if now >= next_report:
                emit(json.dumps(stats.report()))
                next_report = now + 1_000_000_000
    except KeyboardInterrupt:
        emit("Interrupted; closing capture.")
    except Exception as error:
        emit(f"Microphone test failed: {error}")
        exit_code = 1
    finally:
        if stream is not None:
            for action in ((stream.stop,) if capture_active else ()) + (stream.close,):
                try:
                    action()
                except Exception as error:
                    emit(f"Capture cleanup failed: {error}")
                    exit_code = 1
        if detector is not None:
            if source_active:
                try:
                    stats.consume(detector.end_source(source.ref))
                except Exception as error:
                    emit(f"FAR end_source failed: {error}")
                    exit_code = 1
            try:
                stats.consume(detector.close())
            except Exception as error:
                emit(f"FAR close failed: {error}")
                exit_code = 1
            emit("Final report: " + json.dumps(stats.report()))
    return exit_code


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_duration(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0 or number > (2**63 - 1) / 1e9:
        raise argparse.ArgumentTypeError(
            "must be a finite positive duration within timestamp bounds"
        )
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", help="input ID or device name substring")
    parser.add_argument("--sample-rate", type=positive_int, default=48000)
    parser.add_argument("--block-size", type=positive_int, default=480)
    parser.add_argument("--duration", type=positive_duration)
    parser.add_argument(
        "--acoustic",
        action="store_true",
        help="Enable experimental DSP (requires [dsp])",
    )
    args = parser.parse_args(argv)
    if args.device is not None and args.device.isdecimal():
        args.device = int(args.device)
    try:
        sd = importlib.import_module("sounddevice")
    except (ImportError, OSError) as error:
        print(
            f"Microphone support unavailable: {error}. Install with: python -m pip install -e '.[mic]'",
            file=sys.stderr,
        )
        return 1
    return run(args, sd)


if __name__ == "__main__":
    raise SystemExit(main())
