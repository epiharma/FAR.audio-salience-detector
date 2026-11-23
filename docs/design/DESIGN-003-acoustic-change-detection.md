# DESIGN-003 — Local Acoustic Change Detection

**Project:** FAR Local Audio Salience Detector  
**Status:** Approved behavioral scope; algorithmic design OPEN  
**Scope:** FAR v1 acoustic feature extraction and change detection  
**Dependencies:** DESIGN-001, DESIGN-002, ADR-001

## 1. Purpose and responsibility boundary

FAR detects **local changes in acoustic activity and temporal dynamics**. It does not classify sound sources, identify speech, infer physical causes, recognize complex acoustic situations, or determine global significance. The Global Salience Detector combines FAR observations with other sensors and experience; Consciousness may investigate possible causes.

FAR continues to own local audio ingestion, feature extraction, channel-specific acoustic baselines, local change detection, confidence/availability reporting, sensor-health reporting, and sensor-baseline lifecycle reporting. It remains an independent installable library, without dependencies on AGI PoC, Subconsciousness, or Consciousness.

## 2. Approved event vocabulary

| Event | Meaning |
| --- | --- |
| `sound_onset` | Significant acoustic activity appears relative to the learned background, including activity in one or more frequency bands. |
| `sound_offset` | Previously established acoustic activity disappears or returns to the background level, possibly only in some bands. |
| `tempo_increase` | A measurable sequence of acoustic impulses/events accelerates. |
| `tempo_decrease` | A measurable sequence of acoustic impulses/events decelerates. |
| `loudness_spike` | A sufficiently abrupt, extreme increase in acoustic energy. |
| `loudness_drop` | A sufficiently abrupt, extreme decrease in acoustic energy, even if sound continues. |

These are **change descriptions, not source labels**. Simultaneous event types are allowed when independently supported (for example, `sound_onset` plus `loudness_spike`). The output must preserve event timing, magnitude, confidence, and applicable frequency-band diagnostics.

The existing continuous `SalienceResult` contract remains in force; event-like transitions must not be repeatedly reported on every subsequent analysis window. How these six change types are encoded in the existing public output contract is **OPEN** and must be designed without silently breaking DESIGN-002.

## 3. Change sensitivity and temporal scales

- Detect both **abrupt** and **gradual** changes, using externally configurable short, medium, and long analysis windows.
- Sensitivity thresholds are configurable and feature-specific. The exact numerical thresholds and score composition are **OPEN**.
- Detection eligibility and baseline-learning eligibility are distinct; an anomalous observation does not automatically modify an established baseline.
- An observed change should be assessed against learned local behavior, not against a universal assumption that loud or fast sounds are inherently anomalous.
- Use hysteresis to avoid repeated onset/offset or loudness events caused by fluctuations near a threshold.
- An event is emitted **once per meaningful transition**; continued observation produces ongoing results but not duplicate transition events.
- A sustained new acoustic condition may train a separate candidate baseline subject to existing selective adaptation, readiness, archival, promotion, and blocking policies. Existing established baselines remain preserved.

## 4. Acoustic activity and adaptive background

- Learn a **channel-specific background acoustic activity baseline**. Silence means absence of significant activity relative to that background, not necessarily zero-amplitude samples.
- Determine `sound_onset` and `sound_offset` using an adaptive background-relative activity threshold with a configurable minimum absolute threshold.
- During cold start, use provisional background estimates and reduced confidence until sufficient experience is available.
- Do not immediately absorb abrupt background changes into the established baseline.
- Extract activity features from raw audio within FAR; external adapters only supply correctly formatted audio and timing metadata.

### 4.1 Multiband activity

- Frequency-band boundaries and band count are configurable, subject to sample-rate and Nyquist constraints.
- Maintain acoustic activity statistics and baseline compatibility **per frequency band**, as applicable.
- Detect additional activity over an existing background even when total energy changes only slightly.
- Detect the disappearance of activity in some bands even when other bands remain active.
- Group temporally close changes **by event type and direction** into a single event, retaining the contributing bands, magnitudes, and time intervals.
- Opposite transitions (`sound_onset` versus `sound_offset`) remain distinct even within the same grouping interval.
- Temporal grouping does **not** assert that changes share a physical source.
- Grouping duration is configurable; independent changes separated in time remain separate events.

## 5. Extreme loudness changes

- Detect both `loudness_spike` and `loudness_drop` using **separate, high, configurable thresholds**.
- Consider the absolute acoustic energy and the magnitude of change relative to the recent background/baseline.
- Require sufficiently rapid change over a configurable short observation horizon; ordinary fluctuations do not trigger these events.
- A drop need not reach silence.
- These changes may co-occur with onset/offset; do not force mutual exclusion.
- After the transition, continue monitoring the new level without repeatedly re-emitting the same event. A stable new level may become a candidate baseline, not an automatic replacement.
- Extreme loudness is not itself a pain judgment; global pain evaluation and escalation belong outside FAR.

## 6. Impulses, sequences, and tempo

### 6.1 Tempo eligibility

- Estimate tempo from distinguishable acoustic impulses/events or a sufficiently stable sequence, **not** from loudness alone.
- If there is no defensible rhythm or the impulses are no longer distinguishable, `tempo_increase` and `tempo_decrease` are **unavailable**, not zero.
- Rhythm stability criteria and sensitivity thresholds are configurable.
- Preserve individual impulse timing for temporal analysis without requiring each impulse to become a public `sound_onset`.
- Detect abrupt and gradual tempo changes on multiple time scales.

### 6.2 Hybrid change measure

- Consider both **absolute change** in event rate (events per second) and **relative change** against the previously observed rate.
- Both sensitivities are configurable; the precise combination formula remains **OPEN**.
- Do not confuse an increase in signal amplitude with an increase in event rate.

### 6.3 Short impulses versus sustained activity

- A short, sufficiently intense impulse may trigger `sound_onset` without a long confirmation interval; weaker activity requires longer confirmation.
- The minimum confirmation duration depends on intensity relative to background; criteria are configurable.
- A short isolated impulse yields **one `sound_onset`**, without an additional `sound_offset` solely for its brief termination.
- Record its start, duration once known, intensity, affected bands, and confidence. Do not assert a final duration while the impulse is still ongoing.
- Sustained activity uses `sound_onset` at its beginning and `sound_offset` when it ends.
- A short impulse is not automatically learned as a new stable background.
- Repeated impulses may form a rhythm sequence. Once an ongoing sequence is recognized, FAR need not issue a separate public onset for every impulse; preserve their timing internally.

### 6.4 Rhythm sequence continuity

- Brief pauses do not automatically end a rhythmic sequence.
- The allowable pause is adaptive relative to the current estimated inter-impulse interval, with configurable minimum and maximum bounds.
- Track trends in inter-impulse intervals. Consistent lengthening can support `tempo_decrease` and temporarily extend the expected next-impulse deadline within a bounded maximum.
- A missing expected impulse **alone** is not evidence of deceleration.
- End the sequence if no impulse arrives by the adaptive timeout; preserve its learned baseline.
- If impulses accelerate until they merge into continuous sound, stop tempo assessment when individual events cannot be distinguished. Continue acoustic activity tracking **without artificial `sound_offset`/`sound_onset`**.
- If distinguishable impulses reappear, start a **new sequence**, verify its stability, and optionally compare with retained learned rhythm baselines. Do not bridge the unmeasurable interval or claim continuous tempo history.

## 7. Availability, confidence, and stream integrity

- Unavailable observations are never synthesized as silence.
- Missing, invalid, or unsynchronized data must respect the established DESIGN-002 per-feature eligibility, minimum valid coverage, minimum contiguous duration, and continuity rules.
- Unavailable feature values are represented as unavailable (`None` plus reason), not as zero anomaly.
- Feature confidence reflects observation quality, baseline maturity, and continuity/recovery limitations; its numerical calibration remains **OPEN**.
- `SensorHealthEvent` reports stream/device issues; `SensorBaselineLifecycleEvent` reports baseline/candidate changes. Neither is a source-classification event.
- A stream discontinuity may have its own local novelty assessment when evidence supports one, but FAR must not invent acoustic evidence during missing data.
- Independent channels continue to be analyzed when other channels are missing or unsynchronized.

## 8. Multichannel and geometry boundary

- Use stable channel identities, planar/interleaved float32 input, independent source clocks, and optional geometry as already specified in DESIGN-002.
- Local change detection is available per channel. Combining channel evidence requires an explicit policy and defensible alignment where needed; no physical source identity is inferred.
- No sound-source separation, localization, classification, speech recognition, or global multimodal anomaly inference is in scope for this design.

## 9. Explicit non-goals

FAR v1 does **not** attempt:

- Sound or speaker classification, transcription, or semantic recognition.
- Complex spectral-pattern anomaly recognition or causal explanation.
- Global cross-sensor anomaly detection or Consciousness escalation.
- Autonomous physical responses or pain evaluation.
- Inventing a tempo when no stable sequence is observable.
- Training established baselines indiscriminately on anomalous observations.

## 10. OPEN implementation decisions for design review

1. Exact feature-extraction methods for energy, configurable spectral bands, impulse candidates, and temporal activity; DSP dependencies and numeric stability.
2. Event representation and mapping into the existing `SalienceResult`/output families **without silently adding an incompatible fourth output family**.
3. Score scaling, confidence calculation, per-feature sensitivity semantics, and composition of absolute/relative tempo change.
4. Threshold and hysteresis defaults, cold-start maturity, impulse-versus-sustained classification, and event-grouping intervals.
5. Multi-scale scheduling, event timestamps and duration finalization, and deduplication across overlapping windows.
6. Candidate-baseline feature schemas, selective update rules, and compatibility across sample-rate/band changes.
7. Multi-channel aggregation policy, if any, without source classification.
8. Live-microphone acceptance scenarios and measurement targets (detection latency, false events/hour, recovery and adaptation).

## 11. Suggested validation scenarios

- Quiet background → sustained activity → background, including low-level steady hum.
- New band-limited activity over an existing steady background with little total RMS change.
- Simultaneous band-specific onset and offset, kept as separate events.
- Short intense click: one onset with duration when known; no duplicate offset.
- Weak onset requiring temporal confirmation; hysteresis under fluctuating noise.
- Extreme energy spike and extreme energy drop without full silence.
- Repeated stable impulses → faster → slower, including gradual changes.
- Trend-aware slowing versus sequence timeout; short pauses versus termination.
- Impulses becoming indistinguishable continuous sound, then becoming distinguishable again.
- Unknown rhythm, missing samples, invalid audio, channel loss and recovery.
- Candidate baseline adaptation without overwriting established acoustic experience.
- Cross-layout equivalence and sample-rate/band compatibility.

**Implementation gate:** DESIGN-003 records approved *behavior*, not approval of a specific DSP algorithm or numerical policy. Propose those details for review before implementing the acoustic model.
