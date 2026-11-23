# ADR-001: Segmented Audio History Storage

- **Status:** Accepted
- **Date:** 2026-11-23
- **Project:** FAR Local Audio Salience Detector
- **Scope:** Internal acquisition and comparison history representation; no public API changes
- **Related designs:** `docs/design/DESIGN-001-local-audio-salience.md`, `docs/design/DESIGN-002-streaming-audio-api.md`

## Context

FAR v1's streaming contract skeleton passes 84 tests, but profiling identified substantial Python-object overhead and history-dependent processing costs. The current per-channel representation uses dictionaries keyed by frame position and individual `_Sample` objects, with per-sample values and timestamps. Transaction staging copies dictionaries, finalization traverses/sorts pending positions, and retention pruning rebuilds comparison dictionaries even when no entries expire.

In the read-only storage review, 4,096 timestamped samples (16,384 bytes of logical float32 payload) occupied approximately 1,049,077 reachable bytes. In earlier 48 kHz, 10 ms chunk benchmarks, eight channels with 250 ms comparison retention achieved approximately 19.85 ms median `process()` latency and 0.46× real-time throughput. These measurements are environment-specific; they are not performance guarantees or evidence of an unbounded leak.

We need to preserve the already-approved independent-source timelines, stable channel IDs, layout support, bounded reordering, exact overlap/duplicate/conflict semantics, health and baseline lifecycle events, and resource refusal behavior while reducing per-sample Python allocations and full-history traversals.

## Decision

1. **Replace per-sample Python-object history with owned, immutable native float32 sample blocks and interval descriptors.** Store pending acquisition evidence and finalized comparison evidence separately, with per-channel indexes keyed by exact source-epoch frame intervals. Invalid observations use interval markers without fabricated audio samples. Missing observations remain gaps, not silence.
2. **Use a hybrid segmented representation with an initially sorted interval-descriptor index based on the Python standard library (`bisect`).** This is a deliberately simple first backend, not a claim of logarithmic insertion or optimal behavior under extreme fragmentation. Measure descriptor-count scaling and revisit a paged or balanced index only if evidence justifies it.
3. **Use bounded backing blocks and measured compaction.** The implementation must bound block sizes and prevent tiny retained slices from pinning arbitrarily large allocations. Codex may select and document initial block-size/compaction parameters based on reproducible experiments; changes to public resource semantics require separate approval.
4. **Permit representation-neutral changes to tests that assert private `_Sample`/dictionary identities.** Preserve all 84 existing behavioral cases and their intent; replace representation-specific assertions with equivalent evidence-preservation, ownership, atomicity, rollback, and sharing assertions. Never weaken tests merely to obtain a pass.
5. **Require matched before/after benchmarks.** Experimental targets for the previously measured heavy eight-channel/250 ms retention case are at least 4× lower retained traced memory and at least 2× lower median `process()` latency, with p95/p99 reported. These are evaluation goals, not acceptance guarantees or universal real-time promises. Report misses honestly and do not alter contracts to reach targets.

## Storage responsibility boundaries

| Layer | Responsibility | Persistence |
|---|---|---|
| Pending acquisition history | Validated, FAR-owned raw audio and invalid-position markers awaiting finalization; per-source/channel reorder state | Transient |
| Finalized comparison history | Bounded evidence for duplicate/conflict checks, with `late_unverifiable` when evidence has expired | Transient |
| Feature history | Future derived acoustic features with independent multiscale horizons; **not implemented by this ADR** | To be designed |
| Learned baselines | Future statistical models plus existing candidate/control metadata; independent of raw comparison retention | Existing snapshot contract; learned algorithms remain open |

A future extractor's raw lookback requirement must be explicit. Neither comparison retention nor feature-history duration silently dictates raw audio retention.

## Non-negotiable behavioral invariants

- No changes to the public API, source/channel identity, snapshot schema, or three output families.
- Independent source clocks and timelines; no inferred cross-source synchronization.
- Planar and interleaved input; caller owns borrowed buffers only until `process()` returns; FAR retains owned copies of admitted data.
- Exact source-frame interval boundaries, bounded reordering, channel-scoped watermarks, and no retroactive published-result changes.
- Precise trimming of ordinary overlap; no double admission or learning; proven conflicting overlap quarantines the entire affected incoming channel block under the current provisional policy.
- Validate original channel blocks before overlap trimming. Invalid values invalidate the affected block; invalid markers are not silently backfilled.
- `late_unverifiable` beyond bounded comparison evidence; never assert a conflict without retained proof.
- Correct loss/recovery acquisition boundaries (F1), ignored-invalid duplicate continuity (F2), compatibility lifecycle reporting (F3), historical alignment eligibility (F4), and resource-episode closure (F5).
- Atomic admission/refusal and rollback: expected refusals must not mutate acoustic histories, continuity, baselines, or finalized frontiers; diagnostic episode state may advance as already approved.
- Existing logical resource accounting remains four bytes per stored valid sample or invalid marker; physical backing/slack/metadata overhead must be separately measured. No silent eviction or new forgetting policy.
- Preserve timing anchors and existing rounding behavior; for example, `+0.0` and `-0.0` compare numerically equal even if their bytes differ. Storage boundaries must not alter logical validity-run grouping, result cardinality, or event ordering.

## Implementation guidance (not additional approvals)

Use private immutable span descriptors with half-open frame ranges, an owned float32 backing reference plus offset, validity status, and original timing provenance. Reuse immutable backing blocks when moving finalized spans from acquisition to comparison history. Perform overlap preflight and resource accounting before committing changes; copy only admitted ranges. Expire whole descriptors cheaply and compact boundary slices only when measured policy calls for it.

A sorted descriptor vector has `O(S)` insertion/staging worst cases for `S` segments, even if overlap lookup begins with `bisect`; pathological one-frame fragmentation can make `S` approach sample count. The implementation must benchmark such cases explicitly and must not claim per-chunk complexity independent of retained descriptor count.

## Alternatives considered

- **Per-channel ring buffers:** efficient for dense chronological streams, but sparse frame coordinates, out-of-order insertion, overwrite safety, and rollback complicate v1.
- **Unindexed chunk segments:** compact and append-friendly, but overlap discovery can scan all segments.
- **Interval index alone:** efficient lookup without solving per-sample payload overhead; a custom balanced tree adds implementation risk.
- **Hybrid segments + sorted index (chosen):** compact data and straightforward review, with an acknowledged fragmentation worst case.

## Migration and verification

1. Capture deterministic traces from the current implementation as a differential reference.
2. Introduce internal interval operations while retaining existing behavior.
3. Migrate comparison history, then pending acquisition history; keep the change private.
4. Add tests for layout/ownership, signed zero, non-divisible sample-rate timing, nested overlaps, sparse gaps, conflicting prefixes/suffixes, invalid markers, exact retention cutoffs, rollback, F1–F5, epoch changes, and one-frame fragmentation.
5. Compare all observable dispositions, event ordering, timestamps, continuity, resource counters, and snapshot behavior; normalize only genuinely nondeterministic IDs or durations.
6. Benchmark old/new builds on identical hardware/interpreter/input traces in alternating fresh processes. Report median, p95, p99, worst latency, traced retained/peak memory, RSS, and real-time throughput; include long-history and pathological fragmentation cases.
7. If goals are missed, report measured reasons and a bounded follow-up proposal; do not silently broaden scope.

## Consequences and remaining open work

Expected benefits are substantially fewer Python objects and less work proportional to retained *samples*. These benefits are inferred until measured after implementation. Complexity may still scale with the number of descriptors, especially under fragmentation. Block-size, compaction, and operational memory/latency budgets must be chosen and documented from measurements. Python 3.11, hosted CI, and live-microphone validation remain separate verification tasks.

This ADR does **not** authorize acoustic feature extraction, novelty scoring, baseline-learning algorithms, AGI/Subconsciousness integration, new dependencies, or changes to approved public contracts.
