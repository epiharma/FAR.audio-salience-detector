# Controlled integration handoff

The project owner approved beginning controlled integration trials with the
experimental FAR implementation. The formal release gate remains unchanged.
This handoff verifies local package use and provides trial evidence. Integration
behavior and production readiness remain unverified.

## Install and run

The distribution is `far-audio-salience==0.1.0.dev0` and the import is
`far_audio_salience`. Use Python 3.11 or newer. The base install has no runtime
dependencies. Acoustic DSP requires the `dsp` extra (NumPy) and an explicit
`DetectorConfig(acoustic=AcousticConfig(), ...)`. The separate `mic` extra is
for the optional local smoke adapter. License selection is still open.

From this checkout, use a development environment for the smoke and suite.

```powershell
python -m pip install -e ".[dev,dsp]"
python -m pytest -q
```

Save the following as `smoke.py` and run `python smoke.py` in that environment.
It uses synthetic silence and produces three full DSP windows without opening
a device. These resource settings are for this example.

```python
from array import array
from far_audio_salience import (
    AcousticConfig, AudioSalienceDetector, DetectorConfig,
    ResourceLimits, ResourceReporting, SourceSpec,
)

config = DetectorConfig(
    limits=ResourceLimits(
        max_sources=1, max_channels=1, max_chunk_bytes=4096,
        max_buffer_bytes=65536, max_baselines=128, max_receipts=32,
        max_snapshot_bytes=1048576, max_relationship_windows=4, max_features=8,
    ),
    max_reorder_delay_ns=0,
    comparison_retention_ns=0,
    reporting=ResourceReporting(aggregate_every=10, materially_worse_by=4096),
    acoustic=AcousticConfig(),
)
detector = AudioSalienceDetector(config)
source = SourceSpec.microphone(label="trial-mic", sample_rate_hz=48000)
batches = [detector.start_source(source)]
samples = array("f", [0.0]) * 480
for first in (0, 480, 960):
    batch = detector.process(source.chunk(memoryview(samples), first_frame=first))
    assert batch.outcome == "accepted" and not batch.refusals
    assert batch.finalized[0].before_frame == first + len(samples)
    batches.append(batch)
batches.extend((detector.end_source(source.ref), detector.close()))
assert all(not batch.refusals for batch in batches)
events = [e for batch in batches for e in batch.ordered]
assert [e.header.emission_sequence for e in events] == list(range(1, len(events) + 1))
windows = [w for batch in batches for result in batch.salience for w in result.windows]
assert [(w.interval.start_frame, w.interval.stop_frame) for w in windows] == [
    (0, 960), (240, 1200), (480, 1440),
]
print(f"Accepted 1440 frames, {len(windows)} DSP windows, {len(events)} ordered emissions")
```

## Integration contract

- The application serializes calls to `AudioSalienceDetector`. Register a
  `SourceSpec` with `start_source()`. Submit each `AudioChunk` with `process()`
  or submit an `UnavailableObservation` for declared missing or invalid data.
  Buffers are native, normalized float32 memoryviews in `InterleavedAudio` or
  `PlanarAudio`. Borrowed buffers must stay valid through return. FAR copies
  retained samples. Silence is observed audio. An invalid channel block is
  isolated and reported, and structurally invalid input raises
  `StructuralInputError` without committing the input.
- Frame ranges are half-open and source-local. A `SourceRef` identifies source,
  session, and epoch. Track `ChannelKey(source_id, channel_id)` independently
  of buffer order and carry the correct identities across recovery and format
  changes. `SourceSpec.microphone()` supplies IDs for the simple one-mic path.
  Integrators managing durable identities must retain their own mapping.
  A subset of channels on one source does not imply loss on another source.
- Supply `measured_start` and `received_at` separately as `ClockStamp` values
  when available. An unknown acquisition time stays unavailable. Independent
  sources can arrive asynchronously while calls remain serialized. FAR does
  not align sample grids. The adapter may supply `AlignmentEvidence` with
  `ClockMapping` through `update_alignment()` for registered feature eligibility.
  Alignment support does not enable joint-channel acoustic inference.
- Check `OutputBatch.outcome`, `input_receipts`, `refusals` and `finalized`.
  Bounded reordering finalizes coverage per source/channel. Use
  `advance_watermark(SourceWatermark(...))` or `end_source(ref)` to finalize
  pending tails. Handle ignored and partial spans from receipts. Never count a
  refused transaction as accepted coverage or as permission to replay old
  finalized frames. Use `reconfigure_source()` for an explicit replacement
  stream specification and observe the resulting lifecycle outputs.
- Consume one view of the three families, `SalienceResult`, `SensorHealthEvent`
  and `SensorBaselineLifecycleEvent`. `batch.ordered` is the same events sorted
  by emission sequence. Track `event_id` and `emission_sequence` with `run_id`.
  A sequence gap can reveal lost delivery. A sequence alone cannot confirm
  delivery of the last emission. For acoustic changes, `change_id` groups the
  full-state `detected`, `completed` or `interrupted` snapshots and
  `revision` orders them. Carry source-frame and clock provenance. Keep unknown
  values unavailable, and distinguish start, confirmation and delivery timing.
- Health events describe local evidence. Inspect reason codes, resource details,
  episode counts, and provenance. At baseline capacity
  `capacity_exhausted` / `candidate_allocation_skipped` reports lost learning
  opportunity while valid audio, salience and eligible existing updates commit.
  The skipped candidate consumes no ID and there is no eviction or pending queue.
  Intake and processing resource refusals still roll back input. Treat resource
  ceilings as logical bounds, with RSS and backpressure measured separately.
- Inspect `candidates()` and use `command(CandidateCommand(...))` with a
  stable command ID, candidate ID and expected control revision. The supported
  actions are `approve`, `reject`, `block`, `unblock`, `archive` and
  `reactivate`. A block persists for that ID. A reject dismisses the proposal.
  Retained command receipts support bounded idempotency. The default promotion
  mode is `external_approval`, with `automatic` available explicitly.
  `save_baselines()` and `AudioSalienceDetector.from_baselines()` preserve
  compatible learned state and controls, while sources, pending audio and
  timing continuity start fresh. `reset_learning()` retains administrative
  identities. `factory_reset()` clears the retained catalog. Consume the
  batches from teardown and check refusals before treating `close()` as final.

See [DESIGN-002 §§18–19](design/DESIGN-002-streaming-audio-api.md),
[DESIGN-003](design/DESIGN-003-acoustic-change-detection-APPROVED.md),
and [ADR-002](adr/ADR-002-non-blocking-baseline-candidate-capacity.md).
The base architecture is documented in
[DESIGN-001](design/DESIGN-001-local-audio-salience.md).

## Ownership and decisions for review

FAR consumes audio and emits local evidence. The integration owns acquisition,
input conversion, identity mapping, clock evidence, watermark policy, upstream
buffering and drop decisions, event delivery and monitoring. Thalamus routing
and downstream interpretation are outside FAR. This handoff defines no
Thalamus implementation or interface.

Owner review is still needed for trial sources and adapter/clock choice, queue
budgets and backpressure, sink delivery and gap reconciliation, candidate
approver and snapshot custody, and trial acceptance and stop criteria. These
are integration choices. New integration policy or architecture requires
separate approval.

## Validation status

### Verified in Task 20

Tested HEAD `5e92f6b9d91cf5928bb3676dc1b6d56a339bd4ed` from a clean
worktree on Windows x64. Python 3.14.2 imported the editable checkout, with
NumPy 2.5.5 and pytest 9.2.0. The single `python -m pytest -q` run passed
**334 tests in 62.51 s**. The example above accepted 1,440 contiguous
frames in three 480-frame chunks, emitted three DSP windows and four ordered
emissions, then closed without refusals. Configuration is explicit in the
example. Only the handoff and documentation links were added afterward.

### Historical verification

The full historical results are in the companion evidence index. The
historical gate also built wheel and sdist and verified clean-wheel base
import and ingest without NumPy. No fresh wheel, synthetic matrix, soak, or
real-audio rerun was part of Task 20.

| Kind | Frozen synthetic TP / FP / FN | Adversarial TP / FP / FN |
|---|---:|---:|
| sound_onset | 377 / 0 / 0 | 10 / 0 / 0 |
| sound_offset | 65 / 0 / 0 | 8 / 0 / 0 |
| tempo_increase | 24 / 0 / 0 | 0 / 0 / 0 |
| tempo_decrease | 24 / 0 / 0 | 0 / 0 / 0 |
| loudness_spike | 45 / 1 / 0 | 1 / 0 / 0 |
| loudness_drop | 54 / 1 / 0 | 1 / 0 / 0 |

These are Task 06 results for 213 records (147 distinct PCM hashes) and 21
adversarial records. The historical gate counted zero duplicates. The original
18 amplitude controls and seed-785 step/ramp had zero onset FP or FN. Task 06
removed the exact 213 Hz gain-only false onset/offset pair. A mixed-signal
probe with a 4 kHz component spanning both gain edges still exhibited an extra
offset. The Task 07 signal review found support for both unmatched brown-noise
loudness events as unlabelled transitions. Their frozen scores remain as shown.
The older gate statements in the acoustic guide describe their respective
milestones. DESIGN-002 §19's metadata-only persistence and unavailable acoustic
scoring describe the original skeleton. Opt-in DSP and version-2 snapshots
extend that implementation as documented in the current acoustic guide.

The historical 48 kHz synthetic soak covered 1,800 source seconds and 180,000
calls in 1,434.04 s wall time, with zero refusals and no bounds or ordering
assertion failures. It included 29 periodic snapshots and five restart/restores.
Maximum sampled Windows working set after GC was 42.47 MiB. The soak applies
to the historical revision/configuration. It supplies no target-host deadline
or field-performance evidence.

Partition/layout regressions are exercised by the test suite. Historical
recorded-pilot equivalence covers acoustic windows and logical revisions,
with source coordinates preserved. Call-return publication frames may vary
by partition. Task 13A demonstrated non-blocking candidate allocation at
128 retained records. Tasks 14, 16 and 18 subsequently observed aggregated
skipped-allocation diagnostics with complete ingest. Task 18 observed 87
skips for clip 2376 and 2,460 for clip 0184 in each partition, at the 128-record
ceiling. Durable candidate creation pressure remains unresolved.

| Recorded cohort | Unique channel-seconds | Per-kind exposure and TP / FP / FN | Unscored detections |
|---|---:|---|---:|
| Task 12A, quiet negative | 15 | spike and drop, 15 s each, each 0 / 0 / 0 | 0 |
| Task 14, positive re-evaluation | 0.45 | onset, 0.45 s, 2 / 0 / 0. Spike, 0.18 s, 1 / 0 / 0 | 44 per partition |
| Task 16, threshold-defined negatives | 46 | spike and drop, 46 s each, each 0 / 0 / 0 | 12 per partition |
| Task 18, targeted reviewed | 4.55 | drop, 4.55 s, 1 / 0 / 0. Spike, 4 s, 0 / 0 / 0 | 21 per partition |

The positive rows have precision and recall of 1/1 or 2/2, with the
denominators shown by the TP counts. Negative-only precision and recall
are undefined with these counts. Shared masks count once in unique exposure.
Unscored detections remain diagnostics. Kinds omitted from a cohort were
unassessed. The recorded pilots provide zero approved positive offset or tempo
coverage. Task 19 withheld the 0721 offset and both 407450 tempo masks because
component disappearance and pulse membership, respectively, remained ambiguous.
Other unresolved Task 15 and 17 candidates remain outside scoring.

### Unchanged release criteria and unavailable evidence

Retain the DESIGN-003 §11 verification requirements, including annotated
synthetic and lifecycle checks, per-kind accuracy and timing, tempo
availability/error, process latency, throughput, CPU, memory and refusal
measurements. The controlled live protocol retains its acquisition/DAC timing,
integrity and all three independent marker checks before labelled scoring.
No approved numerical release budgets or target integration validation
artifacts were located. The last consolidated formal gate reported
**NEEDS REMEDIATION**. Later targeted checks resolved the specific gain-only
probe, and no superseding formal gate result was located. The earlier
PortAudio timing gate failed. A later
capture-only native WASAPI probe had one startup discontinuity flag and
contiguous observed adjacent positions. That probe supplied no duplex
marker or physical latency evidence.

**Labelled live accuracy and physical acquisition-to-decision latency are
NOT MEASURED.** Small prerecorded masks and current unit/smoke verification
support this handoff only. Formal gate clearance remains unverified.
