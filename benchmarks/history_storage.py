"""ADR-001 fresh-process profiling; no capture or runtime dependencies.

process() timing excludes adapter construction and consumers, with GC enabled.
Memory is traced separately, excluding imports/reused input buffers. RSS includes
interpreter, allocator and tracer. Compare --source trees in alternating processes.
"""

import argparse
import ctypes
import gc
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import tracemalloc
from array import array
from pathlib import Path

CASES = {
    # rate, channels, frames/chunk, retention seconds, reorder seconds
    "mono": (48000, 1, 480, 0.25, 0),
    "mono-long": (48000, 1, 480, 1, 0),
    "eight": (48000, 8, 480, 0.25, 0),
    "overlap": (48000, 8, 480, 0.25, 0),
    "tiny": (48000, 1, 1, 0.25, 1),
    "eight-tiny": (48000, 8, 1, 0.25, 1),
    "sparse": (48000, 1, 4, 1, 1),
    "eight-burst": (48000, 8, 8192, 0.25, 0),
}


class _Counters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
        (name, ctypes.c_size_t)
        for name in (
            "peak_rss",
            "rss",
            "peak_paged",
            "paged",
            "peak_nonpaged",
            "nonpaged",
            "pagefile",
            "peak_pagefile",
            "private",
        )
    ]


def rss():
    if sys.platform != "win32":
        return None
    data = _Counters()
    data.cb = ctypes.sizeof(data)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Counters),
        ctypes.c_ulong,
    ]
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(data), data.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(rss=data.rss, peak_rss=data.peak_rss, private=data.private)


def storage(detector):
    backings = {}
    descriptors = valid = old_records = 0
    for lane in detector._lanes.values():
        for index in (lane.pending, lane.history):
            if isinstance(index, dict):
                old_records += len(index)
                continue
            descriptors += len(index.spans)
            for span in index.spans:
                if span.valid:
                    valid += 4 * (span.stop - span.start)
                    backings[id(span.backing)] = len(span.backing)
    physical = sum(backings.values())
    return dict(
        logical_bytes=4 * detector.buffered_sample_count,
        sample_records=old_records,
        descriptors=descriptors,
        backing_blocks=len(backings),
        physical_backing_bytes=None if old_records else physical,
        backing_slack_bytes=None if old_records else physical - valid,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--mode", choices=("latency", "memory"), required=True)
    parser.add_argument("--calls", type=int, default=200)
    parser.add_argument("--block-bytes", type=int)
    parser.add_argument("--compaction-fraction", type=float)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import far_audio_salience as far

    try:
        import far_audio_salience._history as history
    except ImportError:
        history = None
    if history is not None:
        if args.block_bytes:
            assert args.block_bytes >= 4 and args.block_bytes % 4 == 0
            history.BLOCK_BYTES = args.block_bytes
            history.BLOCK_FRAMES = args.block_bytes // 4
        if args.compaction_fraction:
            assert 0 < args.compaction_fraction < 1
            history.COMPACTION_FRACTION = args.compaction_fraction
    rate, channels, frames, retention, delay = CASES[args.case]
    count = int(rate * retention)
    audio = memoryview(array("f", [0.125]) * (frames * channels))
    prefill = memoryview(array("f", [0.125]) * (count * channels))
    order = list(range(args.calls + 12))
    random.Random(73).shuffle(order)
    gc.collect()
    initial_rss = rss()
    if args.mode == "memory":
        tracemalloc.start()
    config = far.DetectorConfig(
        far.ResourceLimits(
            2,
            16,
            max(4 * count * channels, 4 * frames * channels),
            4 * channels * (count + int(rate * delay) + 2 * frames),
            8,
            8,
            1000000,
            16,
            16,
        ),
        int(delay * 1e9),
        int(retention * 1e9),
        far.ResourceReporting(10, 4096),
    )
    detector = far.AudioSalienceDetector(config)
    source = far.SourceSpec.microphone(
        label="synthetic", sample_rate_hz=rate, channel_count=channels
    )
    detector.start_source(source)
    if not delay:
        assert not detector.process(
            source.chunk(prefill, first_frame=0, measured_start_ns=0)
        ).refusals
    del prefill
    checkpoints = []

    def checkpoint(calls, detector):
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        return dict(
            calls=calls,
            current=current,
            peak=peak,
            storage=storage(detector),
            rss=rss(),
        )

    if args.mode == "memory":
        checkpoints.append(checkpoint(0, detector))
    times = []
    for i in range(args.calls + 12):
        if args.case in ("tiny", "eight-tiny"):
            first = order[i] * frames
        elif args.case == "sparse":
            first = 10**12 + order[i] * 29
        else:
            first = count + i * (frames // 2 if args.case == "overlap" else frames)
        observation = source.chunk(
            audio,
            first_frame=first,
            measured_start_ns=first * 10**9 // rate,
            received_at=far.ClockStamp("receipt", i),
        )
        tick = time.perf_counter_ns()
        output = detector.process(observation)
        elapsed = time.perf_counter_ns() - tick
        assert not output.refusals and all(
            r.salience_score is None for r in output.salience
        )
        del output, observation
        if i >= 12:
            times.append(elapsed)
        if args.mode == "memory" and i + 1 in (
            12,
            (args.calls + 12) // 2,
            args.calls + 12,
        ):
            checkpoints.append(checkpoint(i + 1, detector))
    footprint = storage(detector)
    finalization_ms = None
    if delay:
        end = max(item.high_water for item in detector._lanes.values())
        tick = time.perf_counter_ns()
        output = detector.advance_watermark(
            far.SourceWatermark(source.ref, source.channel_ids, end)
        )
        finalization_ms = (time.perf_counter_ns() - tick) / 1e6
        assert not output.refusals
        del output
    result = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        logical_cpus=os.cpu_count(),
        package=far.__file__,
        case=args.case,
        mode=args.mode,
        calls=args.calls,
        warmups=12,
        sample_rate=rate,
        channels=channels,
        chunk_frames=frames,
        retention_seconds=retention,
        reorder_seconds=delay,
        capacity_bytes=config.limits.max_buffer_bytes,
        storage=footprint,
        block_bytes=history.BLOCK_BYTES if history else None,
        compaction_fraction=history.COMPACTION_FRACTION if history else None,
        finalization_ms=finalization_ms,
        initial_rss=initial_rss,
    )
    if args.mode == "latency":
        ordered = sorted(times)
        step = frames // 2 if args.case == "overlap" else frames
        result.update(
            p50_ms=statistics.median(times) / 1e6,
            p95_ms=ordered[math.ceil(0.95 * len(times)) - 1] / 1e6,
            p99_ms=ordered[math.ceil(0.99 * len(times)) - 1] / 1e6,
            worst_ms=max(times) / 1e6,
            realtime_factor=None
            if delay
            else args.calls * step / rate / (sum(times) / 1e9),
        )
        mark = far.SourceWatermark(
            source.ref,
            source.channel_ids,
            max(item.watermark for item in detector._lanes.values()),
        )
        watermark_times = []
        for _ in range(31):
            tick = time.perf_counter_ns()
            output = detector.advance_watermark(mark)
            watermark_times.append(time.perf_counter_ns() - tick)
            assert not output.refusals
        del output
        result["unchanged_watermark_p50_ms"] = statistics.median(watermark_times) / 1e6
    else:
        result.update(checkpoints=checkpoints)
    detector.close()
    del detector
    if args.mode == "memory":
        gc.collect()
        result["traced_after_close"] = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
