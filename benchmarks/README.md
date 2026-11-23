# Segmented history validation (ADR-001)

These standard-library scripts exercise synthetic streaming contracts. They do
not capture audio, extract features, score salience, or learn a model. See the
[accepted ADR](../docs/adr/ADR-001-segmented-audio-history-storage.md).

Save a copy of `src/` before a change to compare source trees without altering Git
state. Run each command in a fresh process and alternate old/new invocations:

```powershell
.\.venv\Scripts\python.exe -B benchmarks/history_storage.py --source src --case eight --mode latency --calls 200 --output ../history-latency.json
.\.venv\Scripts\python.exe -B benchmarks/history_storage.py --source src --case eight --mode memory --calls 80 --output ../history-memory.json
.\.venv\Scripts\python.exe -B benchmarks/contract_trace.py --source src --output ../history-trace.json
```

Use the same arguments with `--source <saved-src>` for the reference version.
Compare parsed trace JSON exactly. Trace UUID generation and diagnostic time are
controlled; timestamps, sequence numbers, dispositions, reasons, logical evidence,
frontiers, continuity, candidates, and snapshot metadata are preserved. Additional
trace options are `--seed` and `--retention-ns`.

Latency includes only `process()`, with GC enabled, 12 warm-ups and nearest-rank
p95/p99. Setup, input construction and consumer work are excluded. Latency and
tracemalloc run separately. Real-time factor is represented audio duration divided
by summed processing time; overlap counts only new duration. Reorder/sparse cases
report no real-time factor and separately time explicit finalization. The unchanged
watermark measurement isolates descriptor/pruning overhead after streaming.

Memory mode excludes imports and reused input buffers. It reports current/peak
traced allocations, Windows working-set RSS when available, logical bytes,
descriptor counts, unique backing bytes and backing slack at checkpoints. RSS
also includes the interpreter, allocator and tracer; it is not the sample budget.
Physical slack equals unique backing bytes minus referenced valid sample bytes.
Invalid intervals still charge four logical bytes per position without allocating
fake audio. No new resource-refusal threshold or public telemetry API is added.

Implementation parameters are **16 KiB maximum backing blocks** and compaction
once **at least half of a boundary owner's prefix has expired**. Backing is immutable
native float32 bytes. Only surviving new ranges are copied after resource preflight.
Pending-to-comparison movement shares backing, and trimming preserves the original
timestamp anchor. Compaction copies a retained boundary span without rebasing time.

The initial 2026-11-23 sweep used eight channels, 48 kHz, 8,192-frame bursts and
250 ms retention, with 4/16/64 KiB blocks and 25/50/75% compaction thresholds.
For 50% compaction, the measured memory checkpoints were:

| Maximum block | Descriptors | Retained traced KiB | Peak traced KiB | Backing slack bytes |
|---|---:|---:|---:|---:|
| 4 KiB | 96 | 414.4 | 729.9 | 9,216 |
| 16 KiB | 24 | 400.7 | 719.1 | 9,216 |
| 64 KiB | 16 | 389.5 | 826.8 | 0 |

The 16 KiB choice reduces descriptor overhead versus 4 KiB and limits allocation
peaks versus 64 KiB. Burst latency medians were about 6.3–6.4 ms; their small
differences do not establish a speed advantage. At 16 KiB the sampled footprints
were equal for the three compaction thresholds; 50% avoids aggressive early
copies while bounding unused prefixes. The 64 KiB/75% case retained 140,288 bytes
of slack and peaked at 965.4 KiB. Reproduce with `--case eight-burst`,
`--block-bytes` and `--compaction-fraction`; these overrides affect only that
benchmark process and are not detector configuration options.

Matched 48 kHz, 480-frame, 250 ms eight-channel runs used three alternating pairs,
200 latency calls and 80 memory calls per process. Median-of-run p50 improved from
16.093 to 0.660 ms; retained traced memory fell from 22.365 to 0.416 MiB. p95/p99
were 23.009/27.394 ms before and 1.347/2.069 ms after. Worst observations across
the runs were 93.907/10.905 ms. These are shared-desktop observations on Python
3.14.2, Windows 11, Intel Core Ultra 7 155U (14 logical CPUs), not guarantees.

The sorted descriptor vectors still have O(S) staging/insertion costs. In the
524-position one-frame mono test, p50 regressed from 0.025 to 0.103 ms and retained
traced memory from 0.164 to 0.217 MiB. Eight-channel one-frame p50 regressed from
0.144 to 0.739 ms. Ordinary chunk targets do not justify hiding these regressions.
Future work may evaluate a paged index if such arrival patterns are required;
this backend introduces no tree, chunk-size restriction or silent eviction.

Accepted comparison expiry bounds transient evidence only. Future feature
history and learned baselines retain their separate ownership and persistence
contracts. Snapshot format/version, API, baseline controls and output families are
unchanged. Live microphone validation, Python 3.11 runtime and hosted CI require
their own actual execution.
