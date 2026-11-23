# Streaming contract examples

After the editable development installation, run from the repository root:

```sh
python examples/one_microphone.py
python examples/two_microphones.py
```

Both scripts are runnable **synthetic adapters**. They demonstrate the one-source
helper and asynchronous arrivals from two independent source clocks. They open no
devices, require no audio library, and produce no acoustic scores. Replace their
block producers with an external adapter for actual microphone or recorded input.
The configuration values are demonstration resource settings, not acoustic policy
defaults or measured resource budgets.

Live microphone streaming is the primary validation scenario (P0).
Recorded-audio replay and controlled synthetic audio are supporting scenarios.
Actual capture/decoding remains outside FAR; independent clocks require explicit
external alignment evidence for features that depend on synchronization.

## Local microphone smoke test

Install the optional capture dependency from the repository root:

```powershell
python -m pip install -e ".[mic]"
python examples/live_microphone.py --list-devices
python examples/live_microphone.py --device 3 --duration 30
```

Replace `3` with an input device ID or a quoted name substring. Omitting `--device`
selects the default input. On Windows, enable microphone access for desktop apps
under Settings > Privacy & security > Microphone. The example requires a working
PortAudio input device; it never saves or transmits captured audio.

Defaults are one channel, native float32, 48,000 Hz, and 480 frames per read.
Select supported alternatives explicitly:

```powershell
python examples/live_microphone.py --device "Microphone" --sample-rate 44100 --block-size 441
```

Unsupported settings fail with device information. The script does not resample
or silently change stream settings. `--duration` is an optional positive number
of seconds; otherwise use Ctrl+C. Blocking device reads mean duration is a soft
wall-time limit, checked between reads, not an interruptible read deadline.

The adapter stays in `examples/`; sounddevice is an optional `mic` extra, not a
core dependency. `RawInputStream` supplies raw buffers and requires no NumPy.
See [raw streams](https://python-sounddevice.readthedocs.io/en/latest/api/raw-streams.html)
and [device checking](https://python-sounddevice.readthedocs.io/en/latest/api/checking-hardware.html).

The test configuration uses one source/channel, no reorder delay, 250 ms of
comparison retention, a chunk limit of four times the requested block size, and
a logical buffer budget of four times (sample rate + block size). Baseline,
feature and relationship counts are minimal smoke-test bounds; no acoustic
policy or universal latency target is implied. Logical bytes are not an RSS cap.

Receipt times use a separate monotonic clock identity. Measurement timestamps
remain unset: blocking reads provide no reliable acquisition timestamps. Frame
coordinates count actual delivered samples within each FAR epoch and are **not a
verified hardware sample counter**. Captured totals include refused audio;
accepted totals mean FAR-admitted positions, including invalid markers, not
learned or scored audio.

On overflow the adapter reports unknown-duration unavailability in the old epoch,
ends that epoch, creates a new FAR acquisition session with stable source/channel
IDs, and processes the returned block at frame zero with `overflow=True`. The
device remains open; this is a FAR continuity boundary, not a device-clock
measurement. No exact loss interval or missing-frame/block count is invented.
The overflow notice is printed. Receipt-clock identity and lifetime totals remain
stable across epochs. Any FAR refusal stops capture without retrying or silently
continuing after a drop.

Periodic summaries (at most once per second between reads) and the final report
contain device/actual stream settings, captured and accepted frames, results,
health/lifecycle events, overflows, refusals and sequence validation. Latency
p50/p95/p99 are nearest-rank values over at most the most recent 2,048 **audio
process() calls**; the worst value covers all completed audio calls. Unknown-loss
report calls and lifecycle operations are not included in those latencies. No
audio/output history or unbounded latency list is retained for reporting.

Capture closes before final FAR source shutdown. Batches from `end_source()` and
`close()` are included in the final counts. Device errors or refusals return a
nonzero exit status; Ctrl+C performs normal cleanup.

This tests capture and streaming contracts. Acoustic feature extraction, salience
scoring and baseline learning remain unimplemented. `None` scores and
`not_implemented` results are expected even when speaking or clapping. Silence is
valid audio, not missing data; pausing calls alone does not require FAR to infer
an interruption. Mocked adapter tests do not establish live-hardware validation.
