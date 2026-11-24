# FAR — Local Audio Salience Detector v1

**Project:** FAR
**Document type:** Architecture / Research Design  
**Status:** Approved high-level requirements; implementation details remain open  
**Target:** Independent installable package, suitable for integration with the AGI PoC Sensory Layer

## 1. Purpose and Scope

Build a **stateful, streaming local audio salience detector** that learns expected acoustic patterns from incoming audio and detects meaningful deviations. The first version targets **acoustic novelty, not semantic sound classification**. It must be demonstrable independently of the AGI PoC and later usable within its sensory/cognitive architecture.

The detector should be able to notice:

- Appearance of a previously unfamiliar sound.
- Disappearance of an expected background sound.
- Changes in rhythm or temporal structure.
- Changes in acoustic characteristics even when loudness is stable.
- Unusual **combinations of features** even when each individual feature is within its usual range.
- Significant interruptions or irregularities in an otherwise active audio stream.

No ASR, sound-source identification, or semantic interpretation is required for v1.

## 2. Package Boundary — Approved

The **Sensory Layer** is responsible for audio capture and delivery of raw chunks and acquisition metadata, including timestamps, sample rate, and available stream-quality / dropped-block information.

The **Audio Salience package** owns:

- Audio feature extraction (internal, replaceable component).
- Temporal context across successive chunks.
- Local learned acoustic baseline(s).
- Online adaptation and candidate baseline handling.
- Salience scoring, diagnostics, and stream discontinuity assessment.
- Persistence and restoration of learned baseline state.

The package must **not depend on AGI PoC `core`, `brain`, or `sense_organs` imports**. It should accept equivalent input from a live microphone, recorded audio, or synthetic stream through the same neutral contract.

**Subconsciousness** is a future consumer of detector results; implementing it is not a prerequisite for the independent detector.

## 3. Streaming Input and Temporal Analysis — Approved

- Primary input is a **sequence of raw audio chunks**, as produced by a microphone stream.
- Processing is **stateful**: relevant context is retained between chunks.
- Feature extraction happens **inside** the detector package.
- Use **multi-scale temporal analysis** with short, medium, and long observation windows.
- Window durations are **externally configurable** for v1.
- Automatic adaptation of window sizes is a **future enhancement**, not a v1 requirement.
- Preserve measurement-time information distinctly from delivery/receipt timing where available.

**OPEN:** Exact input schema, supported audio formats/channels, chunk-size constraints, window durations, overlap policy, and buffering limits.

## 4. Baseline Learning — Approved

### 4.1 Selective online adaptation

- Learn acoustic expectations progressively from incoming valid observations.
- Evaluate an observation against the existing baseline **before** deciding whether it should update the baseline.
- Strong anomalies must **not automatically become normal** through immediate training.
- Exact admission/update rules remain open.

### 4.2 Provisional baseline / cold start

- Begin evaluating incoming audio immediately rather than waiting for a fixed calibration period.
- With insufficient relevant history, mark estimates as **low-confidence / provisional**.
- Confidence should evolve as relevant evidence accumulates; insufficient history must not be presented as a high-confidence anomaly.

**OPEN:** Confidence definition/calibration and minimum-history criteria.

### 4.3 Candidate baseline

- Observations that do not fit the current baseline may accumulate in a **candidate baseline**.
- A stable new acoustic pattern may eventually be accepted as a new normal context.
- Preserve previous baseline information so returning to a former environment remains recognizable.

**OPEN:** Stability criteria, candidate promotion, baseline selection/switching, number of retained contexts, and forgetting/eviction policy.

## 5. Salience Scoring — Approved

Produce a **single composite `salience_score`** with diagnostic contributions from:

1. `acoustic_deviation` — divergence from expected acoustic characteristics.
2. `temporal_deviation` — divergence from expected temporal sequences.
3. `pattern_novelty` — unfamiliar joint feature configurations.

Component weights are **externally configurable** in v1. Automatic weight learning is a future enhancement.

**OPEN:** Component normalization, score range, weighting equation, thresholds, calibration, and behavior when component estimates are unavailable.

## 6. Stream Discontinuity — Approved

**Important: Stream interruption is a detectable anomaly, not something the model must automatically ignore.**

- If the sensor is inactive and does not call the detector, the detector is not required to infer a missing stream merely from the absence of calls.
- If an **active incoming stream** contains missing blocks, gaps, or irregular timing, the detector should detect meaningful discontinuities.
- Missing samples must **not be fabricated as silence**.
- Invalid or missing data must not be treated as valid acoustic observations for baseline learning.
- Report discontinuity via a separate diagnostic `stream_integrity_score` that **contributes to the overall `salience_score`**.
- Distinguish acoustic novelty from stream-integrity anomalies in diagnostics.

**OPEN:** Discontinuity detection thresholds, score direction/range, timing tolerance, how explicit loss metadata combines with inferred gaps, and handling of state across long interruptions.

## 7. Output Contract — Approved at Field Level

Return a result for **each processed observation/window**, not only when an anomaly crosses a threshold.

| Field | Meaning |
|---|---|
| `timestamp` | Observation timestamp; precise timestamp semantics TBD |
| `salience_score` | Composite degree of unusualness |
| `confidence` | Confidence in the current assessment, including provisional cold-start behavior |
| `is_salient` | Whether the configured attention threshold is exceeded |
| `time_scale` | Relevant temporal scale(s) for the result |
| `baseline_status` | State/maturity of the learned baseline |
| `features` | Optional diagnostic acoustic features |
| `acoustic_deviation` | Acoustic score contribution |
| `temporal_deviation` | Temporal score contribution |
| `pattern_novelty` | Joint-pattern novelty contribution |
| `stream_integrity_score` | Separate discontinuity diagnostic, included in composite salience |

The field list expresses **approved semantics**, not a finalized Python type or serialization schema.

**OPEN:** Types, score ranges, optionality, per-window vs per-chunk cardinality, event IDs, and thresholding semantics.

## 8. Persistence — Approved

- Learned baseline state **persists across detector restarts**.
- On startup, the detector can restore previously learned baseline state.
- Provide an explicit way to reset learned state.
- Persistence is **independent of central AGI Memory** for v1; later integration must remain possible.

**OPEN:** Serialization format, compatibility/versioning, atomic writes, state migration, and persistence frequency.

## 9. Validation Strategy — Approved

**P0 — Live microphone stream is the primary development and validation scenario.**

Validate continuous operation, detection latency, false alerts per hour, baseline stability/adaptation, and real stream interruptions in live operation.

**P1 — Synthetic audio:** controlled changes with known event timing, including novel sounds, background disappearance, rhythmic changes, joint-feature novelty, and stream gaps.

**P1 — Recorded audio:** reproducible playback, comparisons, and regression tests across different acoustic environments.

Where labeled ground truth exists, evaluate precision/recall and detection latency. Live recordings may require annotation or controlled event injection to support objective metrics.

**OPEN:** Quantitative acceptance thresholds, evaluation protocol, benchmark datasets, and performance/resource budgets.

## 10. Implementation Strategy and Non-Goals

### Intended initial delivery

1. Specify neutral streaming input/output types and configuration.
2. Implement an initial replaceable feature extractor and simple baseline/detector candidate.
3. Validate on a **live microphone** early, not only after offline development.
4. Add synthetic and recorded regression scenarios.
5. Demonstrate online adaptation, candidate baselines, persistence, and discontinuity scoring.
6. Integrate with the existing AGI PoC Sensory Layer through an adapter without coupling package imports to AGI internals.

The exact learning algorithm is **not yet approved**. Statistical online methods and small trainable models can be evaluated comparatively; do not assume a neural network is required.

### Explicitly out of scope for v1

- Semantic sound recognition, ASR, or identification of sound sources.
- Autonomous window-size optimization or score-weight learning.
- Global cross-sensor salience and Global Baseline revision.
- Full Subconsciousness, Consciousness, or hybrid cognitive scheduler implementation.
- Repairing legacy reasoning components unrelated to this integration.

## 11. Decision Summary

| ID | Decision | Status |
|---|---|---|
| D1 | Acoustic novelty, including joint-feature and temporal novelty | **APPROVED** |
| D2 | Stateful streaming raw audio; feature extraction internal | **APPROVED** |
| D3 | Multi-scale windows externally configurable | **APPROVED** |
| D4 | Selective online baseline adaptation | **APPROVED** |
| D5 | Provisional cold-start baseline with confidence | **APPROVED** |
| D6 | Candidate baseline for persistent new patterns | **APPROVED** |
| D7 | Composite salience score with configurable component weights | **APPROVED** |
| D8 | Active-stream discontinuities are salience; separate integrity score contributes to total | **APPROVED** |
| D9 | Persistent baselines and explicit reset | **APPROVED** |
| D10 | Live microphone is primary validation; synthetic/recorded are supporting tests | **APPROVED** |

**Next design checkpoint:** Finalize the input/output data types and the minimum live-microphone acceptance scenario before selecting a baseline algorithm. Do not silently resolve the OPEN items during implementation.
