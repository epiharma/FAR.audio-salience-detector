# DESIGN-002 — FAR Streaming Audio API and Multichannel Sensor Contract

**Status:** Approved behavioral requirements through 2026-11-22, including final contract clarifications in Section 18; concrete signatures and acoustic algorithms remain subject to implementation review  
**Scope:** FAR Local Audio Salience Detector v1  
**Related:** DESIGN-001 — Local Audio Salience Detector; Codex streaming API proposal (reviewed, not approved wholesale)

## 1. Purpose and boundary

FAR is an independent, stateful streaming audio salience package. An external sensory layer owns audio capture and supplies observations and metadata. FAR owns feature extraction, local acoustic baselines, temporal analysis, salience scoring, and observation of stream/channel continuity. FAR must not import AGI PoC modules or depend on a specific microphone SDK.

The **Global Salience Detector**, operating in Subconsciousness, interprets local observations in the broader agent context. FAR does not infer global causes or decide whether an agent-wide anomaly merits Consciousness escalation.

## 2. APPROVED — Multichannel input

- Support an **arbitrary positive number of channels**, subject to externally configurable technical resource limits; not mono-only or stereo-only.
- Accept **both interleaved and planar** audio layouts with explicit layout metadata. Convert internally to a canonical representation without discarding channel information.
- Preserve **stable channel IDs**, independent of their positions in the supplied buffer. Preserve mapping of channel IDs to channels in learned baselines.
- A channel reordering alone must not be interpreted as an acoustic anomaly when stable identities are unchanged.
- Acoustic features are extracted per channel, with additional cross-channel features and learned relationships where appropriate.
- **Microphone geometry is optional**. When supplied, it may inform spatial features; when absent, FAR must not assume physical microphone positions.
- Changing known geometry must be reflected in baseline compatibility and diagnostics.

**OPEN (superseded in part by Sections 14–16):** Geometry schema and units, supported geometry transformations, sample-rate-specific feature compatibility, numerical resource limits, and specific spatial algorithms. Sample dtype and buffer ownership are approved in Section 14.

## 3. APPROVED — Channel loss, recovery, and geometry changes

### Channel loss

- Continue processing available channels if one or more channels are lost.
- Exclude features requiring unavailable channels rather than inventing missing audio.
- Preserve the lost channel's learned baseline for potential recovery.
- Surface the channel loss through a **separate `SensorHealthEvent`**.
- Global significance of the channel loss is evaluated by Global Salience, not FAR.

### Channel recovery

- Reuse the recovered channel's previous baseline **provisionally**.
- Temporarily lower confidence for affected assessments, and validate the restored observations against learned acoustic and cross-channel patterns.
- Restore confidence as supporting observations accumulate.
- Emit a distinct recovery `SensorHealthEvent`.

### Explicit geometry change

- Preserve individual channel acoustic baselines.
- Re-evaluate spatial and cross-channel baselines that depend on the changed geometry.
- Preserve the stable channel identity when a microphone moves.
- Report the explicit configuration change through `SensorHealthEvent` and reduce confidence for affected spatial assessments during adaptation.

### Unannounced geometry change

- FAR detects **changes in observed interchannel relationships**, but does **not** independently diagnose `suspected_geometry_change` from acoustic evidence.
- Inferring that a microphone physically moved from multiple observations or wider context belongs to **Global Salience / Subconsciousness**.

**OPEN:** Exact recovery confidence policy, handling of channel-count changes within a session, and criteria for promoting/revalidating affected baselines.

## 4. APPROVED — Sample-rate change

- A change in sample rate does **not automatically discard** the learned baseline.
- Feature extraction must account for sample rate.
- Temporarily exclude features incompatible with the new rate until they can be safely evaluated or adapted.
- Emit a `SensorHealthEvent` for the change.
- Validate baseline compatibility per affected feature; do not assume all parameters transfer unchanged.

**OPEN:** Whether a sample-rate change requires a new acquisition session, resampling policy, and exact compatibility tests.

## 5. APPROVED — Separate output channels

### Acoustic salience results

`SalienceResult` reports the unusualness of observed audio and temporal/interchannel patterns. Prior v1 decisions require continuous assessments with an overall `salience_score`, `confidence`, `is_salient`, `time_scale`, `baseline_status`, and diagnostic decomposition (`acoustic_deviation`, `temporal_deviation`, `pattern_novelty`). Diagnostic features may be optional.

### Sensor-health events

`SensorHealthEvent` is a **separate event type**, not merely an acoustic anomaly result. It reports observed stream/channel conditions such as loss, interruption, invalid data, recovery, explicit geometry/configuration changes, and sample-rate changes.

- Health events must preserve source/channel identity and available timing/context.
- Sensor health and acoustic salience may both be emitted for a related occurrence, without conflating their meanings.
- The wider agent's Global Salience Detector consumes health information and determines global significance.

### Discontinuity semantics

- If the external sensory layer provides no observations, FAR is not invoked; FAR need not infer a missing stream solely from the absence of calls.
- If the active stream contains observable interruptions or gaps, FAR must notice them and report them.
- Missing or invalid audio must not be fabricated as silence.
- A separately reported stream-discontinuity diagnostic may contribute to overall salience where appropriate, but health-event identity remains separate.

**OPEN (partially narrowed by Sections 14–16):** Precise event/result schemas, discontinuity-score naming, numeric ranges, composite aggregation, unknown-duration gap policy, and session/epoch transitions. Event transport responsibility is clarified in Section 16.

## 6. Previously approved v1 principles retained

- Streaming, stateful online processing of raw audio supplied externally.
- Feature extraction within FAR, not the sensory adapter.
- Configurable multi-scale temporal windows; automatic window-size adaptation is a future enhancement.
- Selective online baseline adaptation.
- Provisional baseline at cold start, with appropriately qualified confidence.
- Candidate baselines for sustained changes in the acoustic environment; previous baselines may be retained for context return.
- Persistent learned baselines across restarts with explicit reset.
- **Live microphone streaming is the primary validation scenario**; synthetic and recorded replay remain supporting tests.
- No semantic sound-source classification required for v1.

## 7. Design implications and proposed API shape — NOT YET APPROVED

The original Codex API proposal assumed mono float32 and one active session. It must be revised to accommodate multichannel identity, both layouts, optional geometry, channel health, and independent acoustic/health outputs.

Candidate contracts to review before implementation:

```python
@dataclass(frozen=True)
class ChannelSpec:
    channel_id: str
    # optional geometry and other acquisition metadata


@dataclass(frozen=True)
class StreamSpec:
    stream_id: str
    session_id: str
    sample_rate_hz: int
    channels: tuple[ChannelSpec, ...]
    layout: Literal["interleaved", "planar"]
    # timing/clock contract to be finalized


@dataclass(frozen=True)
class AudioChunk:
    session_id: str
    first_frame: int
    samples: object  # exact typed buffer contract OPEN
    # measurement time, receipt time, quality metadata OPEN


@dataclass(frozen=True)
class SensorHealthEvent:
    event_type: str  # replace with approved typed enum/Literal
    stream_id: str
    session_id: str
    channel_ids: tuple[str, ...]
    # evidence, timing and diagnostics OPEN
```

The types above are **illustrative only**, not approved public API signatures. Preserve the distinction between acquisition timestamps and receipt timestamps. Ensure output does not silently discard unavailable components or misstate confidence.

## 8. Questions to resolve before implementation

1. What are the exact input sample dtype, normalization, buffer shape, ownership, and conversion rules for both layouts?
2. How are stable channel IDs, optional geometry, and changing channel membership represented across sessions?
3. What are the exact `SalienceResult` and `SensorHealthEvent` schemas, score semantics, and event ordering guarantees?
4. What should happen to temporal history and stream session identity after known and unknown-duration gaps?
5. How are sample-rate changes handled while preserving valid baselines and avoiding invalid feature comparisons?
6. What confidence and candidate-baseline policies apply during recovery and geometry changes?
7. How is real-time processing bounded, and how are latency, chunk partition invariance, and overload tested?
8. What is the persistence compatibility/versioning policy for learned multichannel baselines?

## 9. Scope guardrails

Do not implement ML algorithms, audio capture, Global Salience, Subconsciousness, Consciousness, or AGI PoC integration as part of **API contract design**. Do not silently promote the original mono-only Codex proposal to an approved contract. Explicitly distinguish approved decisions from implementation proposals and unresolved questions.


## 10. APPROVED — Independent acquisition clocks and synchronization boundaries

This section records decisions made after the original DESIGN-002 draft and the Codex API review. Where the Codex proposal assumes a single common sampling grid, **these newer approved decisions take precedence**.

- FAR accepts both synchronized multichannel streams and observations originating from independent microphones with distinct acquisition clocks.
- An external **Audio Adapter** owns clock synchronization or alignment estimates, acquisition timestamps, and synchronization-quality metadata. FAR consumes and validates those claims but does **not** synchronize clocks itself.
- Each cross-channel feature declares its own alignment accuracy requirements. Features requiring precise alignment are unavailable if the reported accuracy is insufficient **or unknown**. Cross-channel features not requiring that alignment may continue when their own prerequisites are satisfied.
- Missing or uncertain synchronization is not, by itself, an acoustic anomaly; independent per-channel assessment remains available and its confidence is not reduced solely because another channel is unsynchronized.
- Synchronization quality can change during a session. FAR disables only dependent features when their requirements cease to be met; their learned baselines remain intact.
- Each alignment-dependent feature uses configurable **hysteresis**: a disable threshold, a stricter recovery threshold, and a confirmation duration. Relevant state transitions produce separate `SensorHealthEvent` records, not an event for every small accuracy fluctuation.
- On synchronization loss, temporal continuity for affected cross-channel features is broken. On recovery, begin a fresh continuous observation sequence; do not join samples across the gap. Retain learned baselines, revalidate compatibility, and restore confidence only as valid evidence accumulates.
- Independent per-channel histories remain continuous if their own data streams were not interrupted.

**OPEN (partially narrowed by Section 15):** Exact clock-domain and alignment metadata schema, clock mapping uncertainty representation, allowable clock drift and update semantics, per-feature numerical hysteresis defaults, and how independent timelines are reconciled for shared-window results. Asynchronous per-source arrivals via synchronous `process()` are approved in Section 15. No implicit alignment or resampling may be assumed.

## 11. APPROVED — Partial windows and feature-specific data eligibility

- Preserve valid observation segments on either side of a gap; **never** interpolate missing samples or synthesize silence by default.
- Each feature declares whether it requires a continuous sequence or supports multiple disjoint valid segments within one analytical window. Continuous-only features restart their observation window after a relevant break; segment-tolerant features may evaluate eligible separated segments.
- Track nominal analytical-window duration separately from the duration of valid observations.
- Each feature has configurable `min_valid_coverage` in `[0, 1]`, measured from **feature-eligible** observed durations divided by nominal window duration; overlapping segments count only once.
- If the threshold is not met, that feature is unavailable with reason `insufficient_valid_coverage`; this does not erase its baseline or automatically create an acoustic anomaly. Sensor/stream loss remains observable via `SensorHealthEvent`.
- Each feature can independently specify an optional `min_contiguous_duration`. Segments shorter than that requirement are excluded for that feature. If insufficient eligible segments remain, return unavailability reason `insufficient_contiguous_duration`.
- Assessment eligibility and baseline-learning eligibility are **separate** decisions. A partial window may be adequate for a salience assessment but inadequate for learning; in that case emit the assessment and leave the learned baseline unchanged.
- Feature-specific learning criteria may be stricter than assessment criteria, including coverage, continuity, and data quality. Reasons for declining a learning update must be inspectable in diagnostics.
- Anomalous observations are **not automatically incorporated into an established baseline**.

**OPEN:** Exact per-feature thresholds and duration defaults, partial-window score/confidence calculations, interval and segment representations, handling of very short fragments, and event/result schema extensions for unavailable components and learning diagnostics.

## 12. APPROVED — Parallel candidate baselines and lifecycle

- FAR may learn a **candidate baseline** from observations anomalous relative to the current established baseline, provided candidate-specific quality and eligibility requirements are met.
- Candidate learning is isolated from established baseline learning. Anomalous evidence can support a new candidate without silently changing an established baseline.
- FAR supports **multiple concurrent candidates**, associates observations with candidates based on acoustic-pattern similarity, and may create a new candidate when none is suitable.
- Temporal proximity alone does not justify merging candidates.
- Candidate counts and resource use are bounded by configuration. Reaching a limit must not silently delete learned experience or invent an unapproved forgetting policy.
- Each candidate independently accumulates evidence of stability. Promotion requires a distinct readiness decision; old established baselines remain preserved after promotion.
- Candidate lifecycle includes `active`, `archived`, and `reactivating` states. Archiving preserves learned parameters and may unload them from active memory; reactivation requires a compatibility check before normal learning resumes. Physical deletion is governed by a **separate, not-yet-approved forgetting policy**.
- Candidate promotion mode is configurable: `automatic` (FAR promotes upon approved readiness criteria) or `external_approval` (FAR reports readiness and awaits an external decision). FAR remains independent of AGI PoC; Subconsciousness may serve as the external authority when integrated.
- Introduce a **third, separate output type named exactly `SensorBaselineLifecycleEvent`**, distinct from `SalienceResult` and `SensorHealthEvent`. It reports candidate creation, readiness, promotion, archiving, reactivation, compatibility changes, rejection, and promotion blocking/unblocking as appropriate.

### External rejection and explicit promotion block

- A normal promotion **rejection** retains the candidate, records the decision through `SensorBaselineLifecycleEvent`, and does not prevent further candidate learning. FAR must not repeat the approval request on every window.
- FAR may propose promotion again after materially changed stability evidence or an explicit external request. An earlier rejection is **not** an indefinite prohibition; a new observation period can support reassessment.
- An explicit `promotion_blocked` state is distinct from rejection. A block prevents automatic or externally initiated promotion until **explicitly lifted by the external authority**, but does not prevent observation or candidate learning.
- **Block scope is exactly the identified `candidate_id` by default.** A different candidate does **not** inherit a block merely because it is acoustically similar. No implicit block propagation or candidate-wide acoustic-family ban is approved.
- Promotion blocks and their candidate-ID associations persist across process restarts; candidate identity must remain stable across persistence and restoration.
- Rejection and block/unblock transitions must be inspectable through `SensorBaselineLifecycleEvent`.

**OPEN:** Candidate similarity/matching and creation criteria, stability/readiness metrics, evidence-change threshold for re-proposal, candidate archival triggers, resource-limit behavior, candidate-ID generation and collision rules, exact lifecycle event schema, external approval command protocol, block authorization, and persistence format/migrations.

## 13. Contract revision required before implementation

The Codex API review predates Sections 10–12. Its illustrative single-grid `StreamSpec`, `AudioChunk`, two-output `OutputBatch`, and baseline state machine must be revised rather than accepted verbatim. In particular:

1. Represent independently clocked microphone sources and externally supplied synchronization-quality evidence without pretending they share a sampling grid.
2. Express per-feature synchronization eligibility, hysteresis, temporal continuity, partial-window coverage, and separate assessment-versus-learning eligibility.
3. Extend outputs with **`SensorBaselineLifecycleEvent`** as a separate event category, without conflating learning with sensor health or acoustic salience.
4. Support parallel, persistent candidate baselines and their lifecycle, including candidate-scoped persistent promotion blocks and nonpermanent rejections.
5. Preserve the original separation of local observations from Global Salience interpretation. FAR must not infer physical microphone relocation solely from changed acoustic relationships.

These are **approved behavioral requirements**, not approval of any particular Python signatures, algorithms, persistence representation, or numerical defaults. Earlier OPEN items remain OPEN unless explicitly resolved above.


## 14. APPROVED — Float32 input, borrowed buffers, and invalid-data handling

The decisions in this section supersede the earlier OPEN input-format and buffer-ownership questions.

- Public input samples are **normalized native-endian `float32`**, with permitted finite values in **`[-1.0, 1.0]`**. Other device formats (`int16`, `int24`, `float64`, etc.) are converted by the external Audio Adapter, not FAR.
- Both approved layouts, **planar** and **interleaved**, use this same sample dtype. Stable `channel_id` metadata identifies each channel regardless of buffer position.
- FAR's synchronous `process()` uses a **borrowed-buffer contract**. The caller must not mutate the buffer during the call. FAR must not retain references to caller-owned buffers after `process()` returns. Any samples needed for future windows are copied into bounded internal storage.
- A layout switch does not reset baseline state.
- **Structural contract errors** (wrong buffer length, duplicate IDs, unsupported dtype, invalid shape) reject the call explicitly **without mutating detector state**.
- **Invalid sample values** (`NaN`, `Inf`, out-of-range values) in an otherwise structurally valid buffer are handled as invalid observations: exclude the **affected channel block** from acoustic assessment and baseline learning, continue processing other valid channels, and emit `SensorHealthEvent(event_type="invalid_audio")`.
- Per-sample validity masks and fine-grained salvage of valid subspans are deferred beyond v1. An invalid channel block must not be interpreted as observed silence.

**OPEN:** Exact Python buffer protocol/type declarations, endianness validation and copy strategy, exception/result type for structural errors, and how channel-block invalidity affects time-window coverage. These are implementation contract details, not permission to accept extra dtypes.

## 15. APPROVED — Independent per-source ingestion, reordering, and overlap

This section **supersedes any earlier suggestion that each `AudioChunk` must account for every configured channel or share a single sampling grid**.

- Independent microphone sources may produce chunks at different times and rates. The public `process()` remains synchronous, but each call can accept one source's observation; asynchronous arrival across calls is supported.
- Each observation unambiguously identifies its source, channels, and acquisition-time coordinates. **A channel omitted from another source's chunk is not unavailable** and must not trigger a channel-loss event.
- Source loss or discontinuity is detected from that source's own known timeline or an explicit Audio Adapter report. Interchannel features remain conditional on valid alignment evidence and available observations (Section 10).
- Maintain a **bounded reorder buffer per source**, with configurable `max_reorder_delay`. Order by acquisition coordinates, not arrival time. Do not wait indefinitely for late chunks, exceed resource limits, or retroactively revise emitted salience results.
- Once an interval is finalized, an entirely late chunk is excluded from acoustic windows and **both established and candidate baseline learning**. Record lateness in diagnostics; emit a `SensorHealthEvent` when delivery behavior is itself reportable as a sensor-stream health condition. Lateness alone is not acoustic novelty.
- Duplicate observations must not be counted or learned twice.
- For a chunk partially overlapping a finalized interval, **trim the precisely identifiable finalized portion and accept only the new portion** after normal validation. If the overlap boundary cannot be established reliably, do not guess or silently double-count; report the condition diagnostically.
- Published results are immutable with respect to subsequent late, duplicate, or conflicting chunks.
- **Conflicting observations** (same source and acquisition coordinates, incompatible sample values) do not replace accepted data or update baselines. Emit `SensorHealthEvent(event_type="conflicting_observation")`, retain diagnostic evidence, and continue processing other valid data.
- The Global Salience Detector may consume the conflict event and assess its broader significance; FAR does not diagnose tampering, hardware failure, or a physical cause.

**OPEN:** Exact per-source timestamp/clock schema; watermark/finalization rules; how to compare samples for conflict without retaining unbounded history; duplicate/conflict detection window; exact event metadata; overlap trimming with independently clocked sources; and numeric reordering limits.

## 16. APPROVED — Event delivery, overload, and local resource diagnostics

### Output delivery boundary

- FAR produces `SalienceResult`, `SensorHealthEvent`, and `SensorBaselineLifecycleEvent` in a synchronous output batch, without waiting for consumer acknowledgements.
- **At-most-once** delivery is the default integration contract. FAR does **not** implement a message broker, durable retry queue, consumer ACK protocol, or Subconsciousness delivery logic.
- Events have stable `event_id` and monotonic `emission_sequence` within an identified `run_id`. Sequence tracking allows a consumer to notice detectable gaps, but **cannot prove delivery of the final event or guarantee end-to-end receipt**.
- Integration Layer is responsible for forwarding outputs. An external watchdog/integrity mechanism may monitor liveness and delivery failures; this mechanism is **not part of the FAR package**.
- A delivery failure is a system-integrity concern, not automatically an acoustic anomaly.

### Overload and hard resource limits

- FAR enforces bounded internal memory and processing limits. It must not silently evict accepted data or learned baselines under an unapproved forgetting policy.
- **Audio Adapter / Integration Layer** owns flow control and decides whether to hold or drop upstream chunks. FAR reports inability to accept data; FAR does not silently choose which upstream audio to discard.
- Temporary capacity exhaustion should be distinguishable from a hard limit violation. Exact backpressure/result types remain OPEN.
- A **hard resource-limit violation** (e.g., a chunk exceeding `max_chunk_bytes`) is rejected safely and reported as a structured failure containing `SensorHealthEvent(event_type="resource_limit_violation")`, not merely an uncaught exception. Include the resource/limit identity, observed value, threshold, and exceedance magnitude when known. Avoid mutating acoustic/baseline state for rejected audio.
- The first occurrence of a repeated resource violation is reported immediately. Subsequent occurrences may be **aggregated** into periodic health updates containing count, episode duration, peak exceedance, and affected resource. A materially worse violation is reported promptly rather than waiting for the next interval; emit an episode-end update when the series ceases.
- Rejected or dropped data are never used to train baselines. Known missing intervals remain observable through source continuity/health information, without fabricating silence.
- The health-event reporting path must remain available for expected audio rejection, subject to bounded resources; the exact reserved-capacity/failure contract remains OPEN.

### Explicit AGI PoC integration boundary — NOT FAR IMPLEMENTATION

- In the wider AGI PoC, Subconsciousness may interpret resource-limit violations as **potential pain**, using configured severity thresholds and learned experience. FAR **does not compute pain**.
- The Global Salience Detector evaluates the wider significance of sensor-health events. FAR **does not compute global salience**.
- Cognitive Scheduler priorities, Learned Response Model policies, protective actions, emergency delivery paths, and Consciousness escalation are **outside FAR**. If the diagnostic channel fails, an independent integrity mechanism in the wider system may need to report that failure; FAR cannot guarantee delivery through a failed channel.
- Whether a protective action occurs depends on **learned responses and external system policy**, not hard-coded FAR behavior. No direct dependency on AGI PoC, Subconsciousness, or Consciousness is permitted.

**OPEN:** Exact backpressure schema; detailed event ordering across source streams; numerical aggregation cadence and worsening thresholds; diagnostics under simultaneous resource exhaustion; and the Integration Layer's independent integrity-monitor contract (outside this repository). Resource-refusal event cadence is APPROVED in Section 18.

## 17. Consolidated implementation handoff and unresolved decisions

**Approved and ready to be reflected in the API design:** normalized float32, both buffer layouts, borrowed buffers, structural-error atomicity, per-channel invalid sample handling, stable channel identities, independent acquisition clocks, externally supplied alignment quality, per-source asynchronous arrivals through synchronous calls, bounded reordering, late/duplicate/conflict handling, per-feature partial-window eligibility, independent candidate baseline lifecycle, three output categories, at-most-once integration delivery, and bounded resource rejection/health reporting.

**Still OPEN for Codex to propose rather than silently decide:** concrete Python types; detailed clock/source/session/epoch and watermark signatures; feature-specific scoring and availability representation; exact event schemas; composite formula; candidate similarity/readiness/archival algorithms; snapshot serialization/versioning; numerical defaults and overload behavior. The discontinuity component name is APPROVED as `stream_discontinuity_score` in Section 18.

**Scope rule:** Design FAR as an independently installable audio salience library. Do not implement Global Salience, pain, scheduler, protective actions, Consciousness escalation, delivery retries, external watchdog, audio capture, or AGI PoC integration. Keep unresolved algorithmic questions visibly `OPEN` rather than making their provisional choices sound approved.


## 18. APPROVED — Final API contract clarifications (2026-11-22)

This section supersedes conflicting earlier OPEN labels and recommendations, including the Codex API proposal. All items below are behavioral requirements; illustrative Python signatures in Codex output remain proposals until reviewed in code.

### 18.1 Discontinuity evidence and acoustic salience

- Name the component **`stream_discontinuity_score`**, replacing ambiguous `stream_integrity_score`. It measures local unusualness of an observed discontinuity, **not** agent-wide harm or physical severity. When scored, use the convention 0 = no detected discontinuity and 1 = maximal unusualness under the local model; detailed calibration/composition remain OPEN.
- An observed loss of audio generates a separate `SensorHealthEvent` even if no acoustic salience assessment can be made.
- Never synthesize silence or acoustic observations for missing data. When sufficient continuity history exists, FAR **may** evaluate the unusualness of the interruption as a distinct local component; otherwise its score is `None`, never a fabricated zero.
- The role of the discontinuity component in a defensible composite assessment must be explicit and configurable; **do not silently invent** a composite when no acoustic assessment is available. Global interpretation remains outside FAR.

### 18.2 Bounded comparison history and late unverifiable input

- Retain accepted samples/evidence for duplicate and conflicting-observation comparison only within configurable **`comparison_retention`** bounds. This retention is distinct from learned baselines and must not silently delete them.
- A finalized late chunk beyond retained comparison evidence is excluded from acoustic analysis and both established/candidate learning; return diagnostic **`late_unverifiable`**.
- Do not claim equality or emit `conflicting_observation` when original evidence is no longer available. Never revise published assessments.
- When exact source-frame overlap boundaries are known, discard previously finalized overlap and admit a valid new suffix once; if not resolvable, reject ambiguous data without guessing. The Codex proposal to preserve a precisely identifiable nonconflicting tail after a conflicting overlap is **provisional** and should be called out for review, not silently expanded into a new approval.

### 18.3 Explicit reset operations

Replace ambiguous `reset()` with **`reset_learning()`** and **`factory_reset()`**.

| State | `reset_learning()` | `factory_reset()` |
|---|---|---|
| Learned established baselines | Clear | Clear |
| Candidate baselines, including archived | Clear | Clear |
| Stable source/channel identities | Preserve | Clear |
| Promotion blocks and administrative control records | Preserve | Clear |
| History of control decisions / revision protections | Preserve | Clear |
| Previously saved snapshot files | Do not delete | Do not delete |

- Both resets are explicit and require that acquisition sources are stopped.
- After `reset_learning()`, preserved candidate-ID-specific blocks remain administrative records; they do **not** attach to unrelated newly created candidate IDs.
- Reset behavior must not silently erase or bypass explicit promotion blocks.

### 18.4 Idempotent candidate commands

- Each candidate-control command carries unique **`command_id`** and **`expected_control_revision`**.
- Repeating an identical command while its receipt is retained returns the prior result without repeating effects/events.
- Reusing the same ID with different content is rejected as **`command_id_conflict`**.
- A command with stale control revision is rejected as **`stale_control_revision`**.
- Persist a bounded command-receipt history with snapshots. Even after receipt expiration, revision checks must prevent stale commands from reapplying effects.
- Apply a control transition and its corresponding `SensorBaselineLifecycleEvent` atomically.
- Promotion blocks are specific to `candidate_id`, persist across restart, and are removed only through explicit unblocking or `factory_reset()`; similarity does not propagate blocks.
- The application is responsible for authorizing callers; optional actor metadata does not itself authenticate anyone.

### 18.5 Resource refusal event semantics

**APPROVED — Non-blocking acoustic candidate allocation ([ADR-002](../adr/ADR-002-non-blocking-baseline-candidate-capacity.md)).** If an
otherwise eligible window cannot allocate a new candidate solely because
`max_baselines` is full, admit otherwise valid audio and continue assessment
and eligible updates to existing models. Skip that allocation without consuming
a candidate ID or changing retention or promotion rules. A required learned
reference that is unavailable still makes the dependent assessment unavailable.

- Publish `SensorHealthEvent(event_type="capacity_exhausted")` with reason
  `candidate_allocation_skipped` and resource `max_baselines`. The first committed
  skip starts an episode, subsequent committed skips count toward the configured
  aggregation cadence, and the episode ends when an authorized operation clears
  the capacity condition or the detector closes.
- Accepted coverage, acoustic state, learned updates and outputs commit
  together. Other admission or processing failures retain transactional refusal,
  including rollback of any candidate skips staged by the failed call. Explicit
  administrative candidate registration can still be refused at capacity.

- First hard-limit violation in an episode: emit an immediate **`SensorHealthEvent`** and a structured refusal.
- Every subsequent rejected call returns a structured **`Refusal`** referring to its **`resource_episode_id`**, even when no new health event is emitted.
- Emit periodic aggregate health updates with violation count, episode duration, affected resource and peak exceedance; materially worse violations trigger an immediate update.
- Emit an episode-end health event when the series is known to have ceased. Exact numeric reporting cadence and material-worsening thresholds are configuration/design OPEN.
- Rejected audio does not mutate acoustic windows or learned baselines; diagnostic episode/sequence state may advance.
- Reserve bounded capacity for expected diagnostic reporting; do not claim delivery guarantees beyond the synchronous output contract. Upstream drop/hold decisions belong to Audio Adapter / Integration Layer.
- FAR reports objective resource facts; pain evaluation, protective actions and escalation belong to AGI PoC and are **out of scope**.

### 18.6 Bounded cross-source analytical-window finalization

- Each relationship/feature window has a configurable, finite **cross-source wait horizon** and depends only on its participating sources.
- Consider source-specific finalized data, clock alignment evidence/uncertainty and feature-specific coverage/continuity eligibility.
- When required observations are not available by the finalization boundary, emit an explicit **unavailable** feature decision with reason; never wait indefinitely or fabricate missing audio.
- Independent channel assessments must proceed and publish without being held back by unrelated source arrivals.
- Late arrivals never rewrite already emitted relationship results. Missing a joint window does not itself prove a sensor malfunction.
- The exact reference-clock/window anchoring and how the horizon advances without incoming source activity require a concrete, bounded proposal. **No implicit wall-clock timer, background worker, or synchronization service is approved inside FAR.**

### 18.7 Implementation authorization and remaining OPEN items

The architectural contract is sufficiently agreed to implement the **API skeleton, deterministic input validation, state/lifecycle transitions, event plumbing, bounded buffering, control commands, and contract tests**. Do not claim a working acoustic novelty detector until its feature extraction, scoring, learning and promotion algorithms are separately specified and tested.

Codex must distinguish **APPROVED behavior**, **proposed implementation detail**, and **algorithmic OPEN**. In particular, no invented defaults for acoustic features, score aggregation, candidate matching/readiness, synchronization accuracy, or numerical thresholds. No integration with AGI PoC, Global Salience, Subconsciousness, pain, watchdog, scheduler, or Consciousness.

## 19. Implemented contract skeleton — implementation details, not new approvals

This section documents the first contract implementation. Sections 1–18 are
preserved. The source of exact record fields is
[`contracts.py`](../../src/far_audio_salience/contracts.py), eligibility records
are in [`eligibility.py`](../../src/far_audio_salience/eligibility.py), and the
public exports are in [`__init__.py`](../../src/far_audio_salience/__init__.py).

### 19.1 Implemented public entry points

```python
SourceSpec.microphone(*, label: str, sample_rate_hz: int,
                      channel_count: int = 1) -> SourceSpec
SourceSpec.new_session(*, sample_rate_hz: int | None = None) -> SourceSpec
SourceSpec.chunk(samples: memoryview | tuple[memoryview, ...], *, first_frame: int,
                 measured_start_ns: int | None = None,
                 received_at: ClockStamp | None = None,
                 layout: Literal["interleaved", "planar"] = "interleaved",
                 channel_ids: tuple[str, ...] | None = None,
                 quality: AcquisitionQuality | None = None) -> AudioChunk

AudioSalienceDetector(config: DetectorConfig)
start_source(spec: SourceSpec) -> OutputBatch
process(observation: AudioChunk | UnavailableObservation) -> OutputBatch
advance_watermark(mark: SourceWatermark) -> OutputBatch
end_source(ref: SourceRef) -> OutputBatch
reconfigure_source(replacement: SourceSpec) -> OutputBatch
update_geometry(geometry: MicrophoneGeometry) -> OutputBatch
register_feature(spec: FeatureSpec) -> OutputBatch
update_alignment(evidence: AlignmentEvidence) -> OutputBatch
alignment_available(feature_id: str, at: ClockStamp) -> bool
feature_status(feature_id: str, at: ClockStamp) -> FeatureStatus
open_relationship_window(window: RelationshipWindow) -> OutputBatch
advance_reference_time(at: ClockStamp) -> OutputBatch
register_candidate(channels: tuple[ChannelKey, ...], *,
                   compatibility: tuple[BaselineCompatibility, ...] = ()) -> OutputBatch
report_readiness(evidence: ReadinessEvidence) -> OutputBatch
candidates() -> tuple[BaselineRecord, ...]
command(command: CandidateCommand) -> OutputBatch
save_baselines(path: Path) -> None
AudioSalienceDetector.from_baselines(path: Path, *,
                                    config: DetectorConfig) -> AudioSalienceDetector
reset_learning() -> OutputBatch
factory_reset() -> OutputBatch
close() -> OutputBatch
```

`sources` is a read-only tuple of the stable source catalog; `buffered_sample_count`
counts pending and retained comparison entries. Calls require a single serialized
owner. The library creates no acquisition devices, tasks, threads, or timers.
`SourceSpec.microphone()` generates UUID identities and metadata only. Keep/reuse
the saved source/channel catalog when reopening physical sources; labels are not
hardware discovery or identity matching. `new_session()` preserves source/channel
IDs and generates new acquisition/clock identities.

### 19.2 Input validation, buffering, and finalization

`SourceRef(source_id, session_id, epoch_id)` identifies one sampling epoch.
`SourceInterval` uses exact integer half-open frame coordinates. Each epoch's
known origin is frame zero. Adapters
normalize an existing device counter to that origin rather than implying prior loss.
A chunk may contain any nonempty subset of that source's current channels. Omitted channels
do not imply loss. Interleaved memoryviews are flat, frame-major; planar input is
a tuple of flat memoryviews. Each view must be one-dimensional, contiguous,
format `f`, itemsize 4, with the exact declared length. Other buffer formats,
including non-native float formats, are rejected; external adapters convert them.

Structural errors raise `StructuralInputError` before detector state or emission
sequence changes. Invalid numeric values exclude the entire original channel
block, including when the bad value is in an overlapping prefix; v1 does not
salvage valid subspans of such a block. Other lanes continue. Borrowed views are
never stored: retained values/timing are copied into bounded internal entries.

Watermarks are source-local and channel-scoped. Automatic progress is the highest
admitted channel frame minus the configured `max_reorder_delay_ns` converted to
frames using ceiling division. A monotonic explicit watermark finalizes all
positions before it; `end_source()` finalizes the remaining admitted extent.
This is an acquisition-distance bound, not a background wall-clock deadline.
The adapter must explicitly watermark/end an inactive source's pending tail.

Finalization emits unavailable `SalienceResult` records for contiguous valid
observation segments. These are **not implemented multi-scale model windows**:
`time_scale`, scores, confidence, `is_salient`, and diagnostic features are `None`;
`unavailable_reasons` contains `not_implemented`. Gaps produce health evidence,
not synthesized samples. Known and unknown adapter unavailability reports are
health evidence; they never replace already admitted audio. A known report does
not implicitly finalize other pending observations; use an explicit watermark.

Comparison history expires by `comparison_retention_ns` relative to each channel's
finalized frontier and is independent of baseline metadata. Equal retained/pending
float32 values compare exactly (signed zero compares equal). Finalized overlap is
ignored, and a precisely located new suffix is admitted once. Beyond retained
evidence, the receipt says `late_unverifiable`; no equality/conflict is asserted.
Invalid markers cannot later be backfilled as valid acoustic evidence.
Well-formed packets from an older source epoch are ignored as `late_unverifiable`,
including their old clock metadata and channels retained in the identity catalog;
they are never reinterpreted using the current epoch's rate or clock.

**PROVISIONAL, conservative conflicting-overlap policy:** a provable conflict
quarantines the affected incoming channel block, including its new suffix. Other
channels continue. Nonconflicting-tail salvage remains deferred for review; no
accepted values or published assessments are replaced.

### 19.3 Alignment and partial-window interfaces

`AlignmentEvidence` contains explicit source-epoch mappings into a named reference
clock, a validity interval, observation time, and pairwise error bound. FAR does
not fit mappings, infer clock equality, or resample. Unknown/expired accuracy is
unavailable. `AlignmentPolicy` requires caller-selected disable/recovery bounds
and confirmation duration; recovery must be stricter. Confirmation uses explicit
reference-clock evidence, not idle wall time. Loss/recovery affects only dependent
feature continuity. New source epochs require fresh mapping evidence.

`evaluate_eligibility()` unions/clamps already feature-eligible segments, filters
short segments, measures coverage against nominal duration, and returns explicit
coverage/contiguity reasons. Continuous-only eligibility uses the latest segment
after a break. `feature_eligibility()` evaluates separate assessment, established-
learning, and candidate-learning policies. These are deterministic eligibility
interfaces, not feature computation or permission to learn anomalous observations.
The caller/extension supplies segments in a single declared analysis coordinate
system; FAR does not implicitly align independently clocked segment lists.

**PROVISIONAL opt-in relationship finalization:** callers register a `FeatureSpec`
with a finite wait horizon, explicitly open a reference-clock `RelationshipWindow`,
and call `advance_reference_time()`. Each registered feature uses one explicitly
chosen reference clock. At `stop_ns + wait_horizon`, FAR removes the pending window
and emits an unavailable relationship assessment, including `not_implemented` and
any alignment-unavailability reason. No relationship coverage/extractor is faked.
Late arrivals cannot reopen a finalized reference interval. Source-local results
continue independently. Without explicit progress calls there is no clock advance
or background service. `close()` aborts unfinished relationships with an explicit
unavailable reason and no invented measurement timestamp. Automatic cross-source
anchoring/advancement remains OPEN; this opt-in interface does not settle it.

### 19.4 Output identity, refusals, and bounds

`OutputBatch` contains `salience`, `health`, `baseline_lifecycle`, `input_receipts`,
`refusals`, `command_receipt`, and `finalized`, plus `run_id` and an outcome.
Its `ordered` property merges the same immutable records by one cross-family
`emission_sequence`; it is not another delivery. Header UUID `event_id` values
remain stable for those records. Sequence numbers increase within the detector
run, including after resets. Restoration creates a new run and does not replay
previously returned events. Sources, clocks, measurement/receipt times, and
alignment evidence are included where available; missing provenance is not guessed.

Input acceptance means admission, possibly pending reordering, not a learning
update. Structural rejection changes nothing. Expected resource rejection changes
no admitted audio, frontier, or learned state; separate diagnostic episode and
sequence state may advance. Resource facts never imply acoustic novelty or global
severity. Valid and invalid lanes may have different input dispositions.

All `ResourceLimits` and `ResourceReporting` numeric values are caller supplied.
`max_buffer_bytes` is a **sample-payload accounting budget**, four bytes per stored
sample/invalid marker, not an operating-system resident-memory measurement. Python
bookkeeping, staged copies, and emissions have additional bounded overhead tied to
the bounded samples, identifiers, sources, channels, features, and baseline counts.
Identifiers and frame/time integers also have implementation bounds (256
characters and nonnegative signed-64-bit coordinates). No real-time CPU deadline
or measured resource budget is claimed.

Diagnostic episode storage uses a fixed set of package resource-limit names and
is separate from the sample budget, reserving logical reporting capacity for
expected admission refusals. This cannot guarantee reporting after a Python
allocator or operating-system failure. Temporary sample/window capacity and hard
limits have distinct structured refusal codes. Upstream hold/drop is external.

First violation emits a health event plus refusal; repeated refusals reference
the same episode ID. Aggregates occur every configured number of additional
violations, or promptly when peak exceedance increases by the configured absolute
amount in that resource's units. Episode duration uses a local monotonic diagnostic
clock. A compliant audio admission ends its audio-limit episodes; shutdown ends
remaining episodes. No quiet-period timer is implied.

### 19.5 Candidate control, reset, and safe metadata snapshots

`register_candidate()` explicitly creates an **administrative metadata record**;
there are no learned acoustic parameters. `ReadinessEvidence` is an explicit
extension/test decision, not a FAR-calculated similarity/stability metric.
`materially_changed` with a newer evidence revision, or `explicit_review=True`,
allows another proposal after rejection. Automatic mode acts only on such explicit
readiness/compatibility evidence; no automatic readiness algorithm is supplied.
Multiple records, archived/reactivating activity, readiness, role, and promotion
block are independent. Existing established records are retained after promotion.

`CandidateCommand` carries `command_id`, `candidate_id`,
`expected_control_revision`, action (`approve`, `reject`, `block`, `unblock`,
`archive`, `reactivate`), optional proposal ID and actor metadata. A retained
identical command returns the prior receipt without another event. Different
content with a retained ID is refused. New accepted commands advance the control
guard, including administrative no-ops. Old revisions remain stale after bounded
receipt expiration and restart. Actor metadata is audit information only.

`reset_learning()` requires idle acquisition, clears all baseline/candidate and
transient observation state, but preserves the stable identity catalog, blocked-ID
administration, receipts and revision guards. New candidates inherit no old block.
`factory_reset()` also clears identities/control records. Neither deletes snapshot
files. Retained administrative identities count against the baseline/control bound;
there is no silent deletion to recover capacity.

Snapshots use strict UTF-8 JSON, format `far-contract-metadata`, version `1`.
Unknown keys/versions, duplicate keys/identities, invalid lifecycle fields,
nonfinite values, inconsistent revisions, and configured size/count violations
fail explicitly with `SnapshotError`. File reads are byte bounded; writes use a
same-directory temporary file, flush/fsync, then atomic replacement. An oversized
save does not replace an existing file. Pickle, executable payloads, and migrations
are not supported. Stored metadata includes current source specs, the retained
channel catalog, records, administrative blocks/revisions, bounded command receipts,
and geometry. It does not contain audio, alignment validity, active acquisition,
output queues, or learned acoustic parameters. Restore starts idle. New acquisition
and rate/geometry changes require per-feature compatibility declarations; unrelated
retained state is preserved.

### 19.6 Validation and deferred capabilities

Contract tests cover buffers/layouts/identities, structural atomicity, invalid
lanes, independent sources, ordering/overlap/history retention, alignment and
eligibility, all output families, candidate commands and restoration, resets,
resources, and explicitly bounded relationship windows. Synthetic examples are
runnable in `examples/`; neither captures a microphone. **P0 live microphone
validation has not been performed.**

Feature extraction, acoustic/temporal/novelty/discontinuity scoring, composite
formation, baseline learning, candidate matching/readiness/archival heuristics,
spatial features, and compatibility algorithms remain **not implemented / OPEN**.
No global interpretation, audio SDK, AGI adapter, watchdog, message broker,
background synchronizer, or automatic protective action is included.

### 19.7 Audit remediation F1–F5 — implemented behavior

These corrections change implementation behavior without changing Sections 1–18,
the public method signatures, or the deferred acoustic capabilities.

- **F1:** A known loss interval records a channel-local recovery boundary at its
  exclusive end frame. Earlier/reordered reports do not lower that boundary.
  Finalizing pre-loss buffered audio does not recover the channel; recovery cites
  only accepted valid frames at or beyond the boundary. Unknown acquisition-time
  boundaries are conservative: valid audio continues processing, but same-epoch
  recovery is not asserted until the adapter supplies an explicit known boundary.
  A fresh acquisition epoch preserves the unavailable marker for matching channel
  identities and allows its first finalized valid observations to establish
  recovery. No receipt time is substituted for acquisition evidence.
- **F2:** Numeric validation still quarantines the entire affected input channel
  block. An `invalid_audio` transition is emitted only when that block contributes
  newly admitted acquisition positions, and its provenance cites those positions.
  Entirely ignored late/duplicate invalid input has receipt diagnostics but creates
  no new continuity break. Rejected audio changes no acquisition state.
- **F3:** `report_readiness()` emits `baseline_compatibility_changed` whenever the
  compatibility tuple changes, independently of readiness/activity. Combined
  transitions are ordered compatibility, reactivation, readiness, then promotion
  when applicable, once per transition. Unchanged evidence returns `no_change`;
  an evidence-revision-only update is accepted without a fictitious lifecycle event.
- **F4:** Historical relationship-window alignment uses continuous eligibility
  periods starting only after hysteresis qualification and ending at actual loss
  or evidence expiry. The complete window must fit a supporting period for this
  diagnostic; actual feature coverage remains unevaluated/not implemented. Later
  expiry cannot invalidate an earlier supported window, and later recovery cannot
  qualify a window crossing the earlier break. Bounded state consists of one latest
  period per registered feature and at most one protected supporting period per
  pending window, bounded by `max_features` and `max_relationship_windows`. Each
  period holds only its boundary evidence, not an unbounded report ledger. Protected
  evidence is released on finalization/reset/close. Late-opened older windows with
  no retained supporting evidence remain unavailable rather than inferred aligned.
  Result provenance retains supporting evidence IDs and source/clock identities.
- **F5:** Finalizing relationship windows ends the corresponding window-capacity
  episode when slots are released. Successful window admission and explicit reset
  also close that episode if still open. The end is emitted once; unrelated
  resource episodes remain independent. First/worse/aggregate refusal reporting
  and per-refusal episode references are unchanged.

Caller-driven reference-clock advancement and whole affected-channel-block
conflict quarantine remain PROVISIONAL and unchanged. Command-ID conflict detection
is guaranteed for retained receipts; after expiration, revision checks reject old
commands, but reuse of an expired ID with a fresh revision is not universally
detectable. No unbounded ledger or token protocol is introduced.
