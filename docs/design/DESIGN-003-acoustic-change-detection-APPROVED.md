# DESIGN-003 — Local Acoustic Change Detection

**Project:** FAR Local Audio Salience Detector  
**Status:** Behavioral and architectural principles APPROVED; numerical calibration and implementation details delegated to measured implementation  
**Scope:** FAR v1, independent installable Python package  
**Related:** DESIGN-001, DESIGN-002, ADR-001; Codex acoustic model design review (124 tests observed at review time)

## 1. Purpose and boundaries

FAR detects local acoustic appearance/disappearance, changes in rhythmic tempo, and extreme abrupt loudness changes. FAR **does not** classify sources, recognize speech, infer physical causes, compute pain, decide global salience/escalation, or interpret cross-modal anomalies. Global Salience Detector and Consciousness are separate downstream responsibilities. Do not implement unrelated DESIGN-001 joint-pattern objectives as part of this scope.

Preserve existing ingestion, per-channel source epochs and clocks, timing/eligibility rules, channel identities, health and baseline lifecycle semantics, resource refusals, baseline controls, and ADR-001 segmented history behavior. No assumption of alignment across independent microphones. No borrowed input buffers retained after `process()`.

## 2. Exactly six acoustic change kinds

| Kind | Meaning |
|---|---|
| `sound_onset` | Newly observed acoustic activity, potentially confined to one or more frequency bands, relative to established background |
| `sound_offset` | Previously established acoustic component ceases to be observed, including a learned steady background component |
| `tempo_increase` | Confirmed acceleration of a stable, measurable impulse sequence |
| `tempo_decrease` | Confirmed deceleration of a stable, measurable impulse sequence |
| `loudness_spike` | Extreme rapid increase in acoustic level |
| `loudness_drop` | Extreme rapid decrease in acoustic level, not necessarily silence |

Simultaneous distinct kinds are permitted if independently supported. Changes are transitions, not per-window repeated labels. Preserve continuous assessments separately from change annotations.

## 3. Approved output contract

- Extend `SalienceResult` additively with a typed `changes` collection, default empty; **no fourth output family**. Keep `SalienceResult`, `SensorHealthEvent`, and `SensorBaselineLifecycleEvent` as the three families.
- Each logical acoustic change has a stable `change_id` and monotonically increasing `revision`. A revision is a **complete current representation**, not a patch. Earlier emitted results are immutable.
- Phases: **`detected`**, **`completed`**, **`interrupted`**. Do not use `censored`.
- **Early publication is mandatory:** emit a confirmed change promptly, without waiting for the full multiband grouping horizon. Subsequent matching band observations inside a bounded, anchored grouping window revise the same `change_id`. After that window closes, band membership is fixed.
- Group only compatible observations by channel, continuity episode, kind, temporal proximity and direction; never combine simultaneous opposite-direction onset/offset. Preserve per-band timing and magnitudes. Grouping never implies a shared physical source.
- Duration may be unknown on first detection. Later `completed` revisions can report observed termination and duration. On missing/invalid data or interrupted observation, emit `interrupted` with last supported boundary and **unknown actual ending**, without fabricated duration or `sound_offset`.
- Distinguish emission identity/ordering from logical `change_id` and revision. Preserve at-most-once delivery integration and cross-family ordering. Multiple analysis windows arriving in one chunk must remain inspectable; choose an additive typed per-window diagnostic representation that does not discard windows.
- Implementation must define and test exactly when a `completed` revision applies, including sustained episodes, and how pending revisions are handled at shutdown; do not assume `completed` means that a physical source was identified.

## 4. Activity, presence and cold-start behavior

- Per-channel broadband and configurable multiband acoustic energy; absolute sensitivity floor, adaptive background statistics and hysteresis.
- **Separate background-relative new activity from expected acoustic presence.** A long-lived hum can be normal background yet its disappearance can generate `sound_offset`. Promotion/adaptation must not reset a continuing activity episode.
- Cold start: already-present audio is `observed_active`, not a fabricated `sound_onset`. Baseline/presence confidence is provisional until supported by sufficient observations. Once stable presence is learned, a subsequently observed disappearance may produce `sound_offset`.
- Recovery after gaps: do not fabricate onset, offset, silence or duration. Re-establish observed activity from new evidence; never splice rhythmic or activity continuity across an unobserved interval.
- **Asymmetric confirmation:** strong onset/short impulses can confirm rapidly with intensity-dependent duration; offset requires a separately configurable, sustained confirmation interval. A brief attenuation is not necessarily an offset; an independently confirmed extreme attenuation may be `loudness_drop`.
- A short isolated impulse produces one `sound_onset` and later a same-ID completion with known duration; no separate `sound_offset`. Sustained activity gets onset and, when observed ending is confirmed, offset. Avoid repeated onsets on adjacent windows and automatic assimilation of impulses into established baseline.

## 5. Loudness changes

- `loudness_spike` and `loudness_drop` use **independent configurations**.
- Each requires both a relative change gate and an absolute magnitude gate, within a bounded short temporal horizon; pre/post activity guards prevent pathological near-silence ratios.
- Detect extreme **transitions**, not persistent high/low levels. Use hysteresis/latching and stable rearm; a large drop can occur without offset.
- dBFS/power changes are digital signal measurements, not calibrated physical sound pressure; do not infer danger or pain.

## 6. Impulses and rhythm

- Extract distinct acoustic impulses with source-frame timestamps; merge coincident band detections of the same local impulse without asserting a common source.
- Track sequences internally without emitting `sound_onset` for every pulse in an already-established ongoing sequence. Preserve individual impulse timing for rhythm analysis.
- Tempo is `unavailable`, **not zero**, unless a sufficiently stable, unambiguous sequence exists. If incompatible pulse trains cannot be separated, expose `ambiguous_impulse_sequence`; do not invent an averaged rhythm. Parallel rhythm tracks are out of v1 scope.
- Estimate tempo using actual observed inter-onset intervals and robust statistics. Evaluate abrupt and gradual changes with multiple horizons; support increasing/decreasing trends and direction-specific hysteresis.
- **Hybrid absolute + relative change metric:** absolute tolerance matters at low rates, relative tolerance at high rates. The Codex-proposed normalized difference is an implementation candidate, not a numerically calibrated standard. Expose configurable increase/decrease thresholds; validate experimentally.
- Adapt expected next-impulse deadline to the **observed** interval trend, with configurable lower/upper bounds. Freeze a computed deadline until new evidence arrives; time passing alone must not extend it. Missing expected pulses/timeout alone do not prove deceleration.
- Short pauses do not automatically end a sequence. When distinguishable impulses merge into continuous audio, tempo becomes unavailable **without artificial acoustic offset/onset**. If distinguishable pulses reappear, reacquire a **new** sequence; never bridge the unmeasurable interval with an invented IOI.

## 7. DSP and dependency policy

- Recommended causal pipeline: rolling power/RMS/peak; per-band spectral power; impulse candidate extraction; rhythm statistics; signed short-horizon loudness changes; selective background and presence learning.
- NumPy is an **optional DSP dependency**, e.g. a package extra `[dsp]` using the actual distribution name. The base infrastructure stays dependency-free. If full DSP is enabled without NumPy, report a clear configuration error; do not silently use a slow pure-Python DFT. SciPy is not required for v1.
- Causal real FFT with correctly normalized window and one-sided power, and analysis windows anchored to source-frame coordinates rather than incoming chunk boundaries. Never pad missing evidence with fabricated zeros or span a gap with an FFT/IOI.
- Frequency-band count and boundaries configurable and validated against Nyquist. Preserve compatible baseline statistics on sample-rate changes, mark unavailable bands `unavailable` (not silence), reset incompatible transient DSP state, and do not emit offsets due to inaccessible bands. Reuse restored compatible baselines provisionally; re-confirm current activity. Changing band definitions changes feature compatibility identity.
- Exact FFT/window/hop defaults, band edges, effective integer frame rounding, intensity/duration curves, gates and thresholds are implementation proposals to benchmark and document, **not already approved numerical values**.

## 8. Baseline learning and persistence

- Maintain separate channel/band background location and variability, expected acoustic presence, maturity/exposure, and compatible rhythm reference statistics. Evaluate against pre-update baseline before any adaptation.
- Distinguish assessment eligibility, established-learning eligibility and candidate-learning eligibility. Transients, transition evidence, gaps, recovery and strong anomalies must not immediately contaminate established models. Count each observed exposure once despite overlapping windows.
- Preserve multiple candidate baselines, compatibility, promotion, archival/reactivation, idempotent revision-checked commands and candidate-ID-specific promotion blocks. Do not change their administrative meaning.
- Extend snapshots **with explicit versioning and legacy compatibility** to save acoustic parameters and candidate statistics. Existing metadata-only snapshots remain readable. Do not restore active episodes, pending annotations, impulse queues or rhythm continuity after restart.
- Document feature algorithm revisions, units, sample-rate/band compatibility and migration rules; do not silently reinterpret old parameters.

## 9. Scoring, confidence and availability

- Compute score/confidence only from actually available, valid and eligible features. Missing features are **not zero**, and missing weights must not be silently redistributed.
- Confidence reflects observation coverage, baseline maturity and recovery quality. A detected change is not automatically globally salient; do not equate `changes` being nonempty with `is_salient`.
- Preserve `pattern_novelty` or unrelated score fields as unavailable until separately implemented. Propose transparent feature-to-score mapping, confidence interpretation and optional overall threshold with tests; do not claim calibration without measurement.

## 10. Atomicity, resources and failures

- Existing detector finalization can precede input admission/materialization: **do not blindly insert DSP into `_finalize_lane()`** or rely solely on comparison retention. Stage borrowed input plus owned pending evidence, preflight resources, compute private draft state, and commit atomically only after valid admission.
- Bound DSP lookback, spectral buffers, impulse history, rhythm state, grouping/revision queues, candidate parameters and learned-state memory independently of ADR-001 comparison history. Preserve `max_buffer_bytes` meaning unless explicitly versioned.
- Refuse resource-exhausting transactions structurally with appropriate health evidence or use a documented unavailable assessment; never silently discard state, corrupt baseline, or pretend a missing feature is silence.
- **APPROVED ([ADR-002](../adr/ADR-002-non-blocking-baseline-candidate-capacity.md)).**
  Exhausting slots for new baseline candidate IDs is a
  learning limitation. Skip that allocation, report it via `SensorHealthEvent`,
  and commit otherwise valid audio, salience and eligible existing-model
  learning. Preserve all retained identities, parameters and controls. A
  missing required reference leaves the dependent assessment unavailable.
  Processing and admission limits still refuse atomically. Candidate creation
  criteria and the rate of persistent ID creation are separate from this policy.
- Invalid samples, conflicting overlaps, gaps and clock uncertainty preserve existing quarantine/eligibility rules. Do not cross independent source clocks for per-channel features; cross-channel aggregation is out of scope absent explicitly aligned policy.

## 11. Verification and acceptance

- Preserve all existing tests (Codex observed **124 passing** at its design review; verify current count rather than treating this as an eternal invariant).
- Add deterministic frame-annotated synthetic fixtures for: cold start; learned hum disappearance; band onset over steady hum; simultaneous opposite band changes; strong short click/weak sustained onset; completion/revision/interruption; rapid level spike/drop without silence; stable/accelerating/decelerating and ambiguous rhythms; pauses/timeouts; merge/reacquisition; gaps/invalid input; sample-rate/Nyquist changes; snapshot migration; candidate promotion/block; chunk partition and layout invariance; resource exhaustion and transaction rollback.
- Test **early detection then same-ID increasing revisions**, full-state revisions, immutable past results, stable event identity and no duplicates.
- Measure precision/recall by kind, timing tolerance and decision latency, false events per valid observed hour, tempo error/availability, p50/p95/p99 process latency, throughput, CPU, memory and refusals. Run controlled live microphone trials after synthetic correctness; record gain/AGC and distinguish receipt timestamps from actual acquisition timing.
- Do not claim unmeasured latency, accuracy or real-time performance. Establish numerical acceptance budgets from pilot results and report them explicitly.

## 12. Implementation governance

Behavioral/API decisions in this document are approved. Codex may select reasonable **provisional** numerical parameters and DSP algorithms, clearly label them as experimental, make them configurable, justify trade-offs and measure them. Do not silently expand the six-event vocabulary or FAR's responsibility. Implement in staged, independently tested commits, maintaining backward compatibility and providing an audit of remaining unimplemented fields.
