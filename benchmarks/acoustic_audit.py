"""Bounded synthetic stress runner for the independent DESIGN-003 audit.

Run from repository root: python benchmarks/acoustic_audit.py --seconds 120
RSS is Windows process working set, sampled every ten audio seconds.
"""

import argparse
import ctypes
import gc
import json
import runpy
import time
from collections import Counter
from pathlib import Path

import numpy as np

from far_audio_salience import AudioSalienceDetector, SourceSpec, UnavailableObservation


def rss():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak",
                "working",
                "paged_peak",
                "paged",
                "nonpaged_peak",
                "nonpaged",
                "pagefile",
                "pagefile_peak",
            )
        ]

    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        api = ctypes.windll.psapi.GetProcessMemoryInfo
        api.argtypes = (ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong)
        handle = ctypes.windll.kernel32.GetCurrentProcess
        handle.restype = ctypes.c_void_p
        if api(handle(), ctypes.byref(counters), counters.cb):
            return counters.working
    except AttributeError:
        pass
    return None


def stress(seconds):
    b = runpy.run_path(str(Path(__file__).with_name("acoustic.py")))
    d = AudioSalienceDetector(b["configuration"]())
    s = SourceSpec.microphone(label="audit stress", sample_rate_hz=48000)
    d.start_source(s)
    counts = Counter()
    revisions = {}
    samples = []
    refusals = 0
    gaps = 0
    last_sequence = 0
    max_candidates = 0
    cpu = time.process_time()
    began = time.perf_counter()
    for first in range(0, round(seconds * 48000), 480):
        amplitude = (0.03, 0.1, 0.3)[int(first / 48000 / 10) % 3]
        x = np.asarray(
            amplitude * np.sin(2 * np.pi * 200 * (first + np.arange(480)) / 48000),
            dtype=np.float32,
        )
        batches = []
        if first == 60 * 48000:
            batches.append(
                d.process(UnavailableObservation(s.ref, s.channel_ids, "missing"))
            )
        if first == 30 * 48000:
            before = repr(d._acoustic.models)
            rejected = d.process(
                s.chunk(
                    memoryview(np.zeros(300000, dtype=np.float32)), first_frame=first
                )
            )
            assert rejected.refusals and repr(d._acoustic.models) == before
            batches.append(rejected)
        batches.append(d.process(s.chunk(memoryview(x), first_frame=first)))
        for out in batches:
            refusals += len(out.refusals)
            for event in out.ordered:
                gaps += event.header.emission_sequence != last_sequence + 1
                last_sequence = event.header.emission_sequence
            for result in out.salience:
                for change in result.changes:
                    assert change.revision == revisions.get(change.change_id, 0) + 1
                    revisions[change.change_id] = change.revision
                    counts[change.kind] += change.revision == 1
        max_candidates = max(max_candidates, len(d._baselines.records))
        if first % (48000 * 10) == 0:
            gc.collect()
            samples.append(
                dict(
                    audio_second=first / 48000,
                    rss_bytes=rss(),
                    candidates=len(d._baselines.records),
                    parameter_features=sum(len(v) for v in d._acoustic.models.values())
                    + sum(len(v["stats"]) for v in d._acoustic.candidates.values()),
                    history_rows=max(
                        len(st.history) for st in d._acoustic.states.values()
                    ),
                    open_changes=max(
                        len(st.changes) for st in d._acoustic.states.values()
                    ),
                )
            )
    print(
        json.dumps(
            dict(
                audio_seconds=seconds,
                frames=round(seconds * 48000),
                calls=round(seconds * 100),
                wall_seconds=time.perf_counter() - began,
                cpu_seconds=time.process_time() - cpu,
                refusals=refusals,
                sequence_gaps=gaps,
                logical_changes=dict(counts),
                max_revision=max(revisions.values(), default=0),
                max_candidates=max_candidates,
                rss_samples=samples,
            ),
            indent=2,
        )
    )
    d.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=int, default=120)
    args = p.parse_args()
    if not 70 <= args.seconds <= 600:
        p.error("require 70..600 seconds")
    stress(args.seconds)
