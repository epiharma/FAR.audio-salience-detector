# Experimental DESIGN-003 model

Enable explicitly with `pip install -e ".[dev,dsp]"`, then add
`acoustic=AcousticConfig()` to the existing `DetectorConfig`. The default remains
infrastructure-only. Enabling without NumPy raises a configuration error with
installation guidance. No SciPy, audio device SDK, classifier or global interpreter
is supplied. Calls remain synchronous and serialized by the consumer.

```python
from far_audio_salience import AcousticConfig, AudioSalienceDetector
from dataclasses import replace

# config is the application's existing explicit resource/timing configuration.
detector = AudioSalienceDetector(replace(config, acoustic=AcousticConfig()))
detector.start_source(source)
# samples is a borrowed normalized native-float32 memoryview from the adapter.
batch = detector.process(source.chunk(samples, first_frame=first_frame))
for result in batch.salience:
    for change in result.changes:
        print(change.change_id, change.revision, change.kind, change.phase)
```

## Cadence, time and identity

Each eligible full FFT frame produces one `SalienceResult` with one typed
`AcousticWindow`; every frame in a larger call remains inspectable. Before the
first full contiguous frame, an unavailable assessment may be returned. No FFT
padding or IOI crosses missing/invalid evidence. Frame size/hop are ceil(seconds
× sample rate), with at least 3 FFT samples and 1 hop sample. Windows are anchored
to source frame coordinates, independent of chunk boundaries. Frequency bands
use half-open edges except inclusion of Nyquist when it is the upper edge.

Window intervals are `[start_frame, stop_frame)`. Result measurement timestamps
refer to the last observed sample, using the adapter's measurement clock; they
remain `None` for receipt-only input. Receipt timestamps and source/clock provenance
stay distinct. No cross-source clock conversion occurs. Change start/confirmation
frames are detector evidence boundaries, not claims about physical source time.
Confirmation includes causal analysis/confirmation delay. An interrupted change
has no actual end or duration. `last_observed_frame` is an exclusive supported
boundary; an estimated ending can precede confirmation.

`change_id` identifies a logical change; revision increases for every complete,
immutable snapshot. Emission headers identify delivery in the existing three
output families. Matching band membership may expand only through the anchored
`first confirmed_frame + grouping_seconds` boundary; it never postpones initial
publication. Directions/kinds and channels remain separate. Band members retain
individual start/end/power/dB measurements. Completion of an onset means its
observed activity ended; short impulses have no separate offset. Confirmed offset,
level and tempo transitions complete when their transition settles. Those
transition durations describe the detector's transition interval, not an inferred
physical object's lifetime. End/close interrupts open observations; it cannot
establish an offset. Closing with an admission refusal leaves the detector open
so pending data can be finalized in smaller transactions.

Cold/recovered active audio is `observed_active`. Compatible learned expected
presence is retained but current activity is re-observed. Band definitions,
window/hop duration, Hann normalization, units and algorithm revision determine
feature schema IDs. Sample rate is deliberately excluded: compatible mean-square
powers can be provisionally reused; inaccessible bands are individually unavailable.
Integer FFT/bin changes can still change leakage; this needs calibration.

## Features, learning and scores

Broadband power is raw mean-square PCM; RMS is its square root; peak is digital
absolute amplitude. Band power is the Hann-windowed one-sided FFT energy divided
by N × sum(window²), including correct DC/Nyquist weights. The numerical floor
only stabilizes ratios. It is distinct from the acoustic sensitivity floor.

Established and candidate eligibility are separate from assessment eligibility.
Established EW location/variance and expected-presence exposure learn only from
non-transition, non-extreme evidence. Exposure counts the new hop once, never the
full overlapping FFT window. Initial lookback is conservatively not extra exposure.
Long horizon power uses only observed hop evidence and configured coverage and
contiguous-segment gates; unavailable horizons remain `None`. Raw FFTs require a
full contiguous frame even when aggregate coverage policies permit partial windows.

Persistent changed conditions can create multiple candidate IDs under existing
administrative bounds. Candidate statistics are separate; readiness requires
stable independent exposure. Reject dismisses the current proposal; a fresh full
readiness exposure can support a new proposal. A candidate-ID block continues to
prevent automatic promotion. Existing approve/reject/block/unblock/archive/reactivate
commands retain revision/idempotency semantics. Active established candidate models
can serve as compatible per-feature references without resetting current activity.

Acoustic deviation is u/(1+u), where u is the maximum available band power relative
deviation from pre-update learned means, with absolute acoustic floor denominator;
confirmed transition magnitude also contributes. Temporal deviation compares a
measurable impulse rate to pre-update learned rate using a hybrid absolute/relative
denominator. A positive-weight missing component makes the combined score missing;
weights are not redistributed. Thresholding the score alone sets `is_salient`;
change existence does not. Confidence is the minimum of contiguous observation and
baseline exposure maturity, in [0,1], **not a calibrated probability**. Initial
unlearned assessment has unavailable score. `pattern_novelty`, stream discontinuity
scoring, classification and aligned cross-channel acoustic inference stay unavailable.

Rhythm requires observed distinguishable impulses and robust IOIs, not elapsed
silence. Short and long trend fits reject irregular residuals; incompatible band
profiles/IOIs expose `ambiguous_impulse_sequence`. Timeout is bounded and frozen
until another impulse. Merged continuous sound loses tempo; reacquisition starts a
new sequence. This single-track heuristic cannot identify every pair of overlapping
pulse trains (commensurate mixtures may look regular).

## Bounds and snapshots

DSP ring/workspace, impulse history, diagnostic history, open changes, per-call
windows/revisions, and learned feature count have explicit bounds independent of
comparison retention. Zero comparison retention works. Input is copied into owned
ring storage; no borrowed array/view survives return. A private engine/store draft
is evaluated before ingestion commits. Resource exhaustion returns a structured
refusal, with aggregated health reporting, without learning or advancing watermarks.
The byte reservation is a conservative logical allowance, not a hard RSS/allocator
cap. Resource handling does not silently evict learned models. Upstream drop and
backpressure decisions remain outside FAR.

Version-2 snapshots add `far-acoustic-parameters-v1` scalar models to version-1
metadata. Strict validation checks identity, schema, bounds and finite numbers.
Version-1 snapshots load with no acoustic parameters. Loading version 2 with DSP
disabled preserves validated parameters without importing NumPy. No active episode,
ring, lookback, sequence, pending annotation or alignment continuity is restored.
Explicit learning/factory resets clear acoustic parameters as well as transient state.

## Experimental numerical policy

All values below are provisional implementation choices, not approved calibration.
Durations are seconds, power is normalized PCM², dB is digital power ratio, and
rhythm rate is events/second. All numeric values must be finite/nonnegative;
positive-only and coupled constraints are enforced by `AcousticConfig` validation.
Hop is at least 1 microsecond, frame at most 10 seconds, horizon at most one hour;
resource bounds may reject much smaller but impractical configurations.

| Parameter | Default | Units / role |
|---|---|---|
| `bands` | low 40–400; mid 400–2000; high 2000–12000 Hz | fraction / dimensionless policy |
| `frame_seconds` | 0.02 | seconds |
| `hop_seconds` | 0.005 | seconds |
| `horizons_seconds` | (0.05, 0.25, 1.0) | seconds |
| `absolute_power_floor` | 1e-06 | PCM² |
| `numerical_power_floor` | 1e-20 | PCM² |
| `onset_ratio` | 4.0 | fraction / dimensionless policy |
| `band_onset_decline_ratio` | 0.9 | fraction / dimensionless policy |
| `stable_level_db` | 1.0 | digital power ratio (dB) |
| `offset_ratio` | 1.8 | fraction / dimensionless policy |
| `strong_peak` | 0.1 | fraction / dimensionless policy |
| `strong_confirmation_seconds` | 0.0 | seconds |
| `weak_confirmation_seconds` | 0.03 | seconds |
| `offset_confirmation_seconds` | 0.04 | seconds |
| `sustained_seconds` | 0.12 | seconds |
| `grouping_seconds` | 0.04 | seconds |
| `maturity_seconds` | 0.3 | seconds |
| `adaptation_seconds` | 10.0 | seconds |
| `learning_max_relative_error` | 0.3 | fraction / dimensionless policy |
| `spike_relative_db` | 12.0 | digital power ratio (dB) |
| `drop_relative_db` | 12.0 | digital power ratio (dB) |
| `spike_absolute_power` | 0.01 | PCM² |
| `drop_absolute_power` | 0.01 | PCM² |
| `loudness_activity_floor` | 1e-05 | fraction / dimensionless policy |
| `loudness_horizon_seconds` | 0.04 | seconds |
| `rearm_seconds` | 0.05 | seconds |
| `impulse_ratio` | 4.0 | fraction / dimensionless policy |
| `impulse_min_peak` | 0.01 | fraction / dimensionless policy |
| `impulse_separation_seconds` | 0.025 | seconds |
| `impulse_max_seconds` | 0.2 | seconds |
| `rhythm_min_intervals` | 4 | positive IOI count |
| `rhythm_horizons` | (3, 8) | IOI counts, at least 2 and within impulse bound |
| `rhythm_relative_jitter` | 0.15 | fraction / dimensionless policy |
| `rhythm_absolute_jitter_seconds` | 0.005 | seconds |
| `rhythm_band_distance` | 0.65 | fraction / dimensionless policy |
| `tempo_increase_absolute` | 0.5 | positive events/second |
| `tempo_increase_relative` | 0.15 | fraction / dimensionless policy |
| `tempo_decrease_absolute` | 0.5 | positive events/second |
| `tempo_decrease_relative` | 0.15 | fraction / dimensionless policy |
| `tempo_rearm_fraction` | 0.5 | fraction / dimensionless policy |
| `pause_multiplier` | 2.5 | fraction / dimensionless policy |
| `pause_min_seconds` | 0.15 | seconds |
| `pause_max_seconds` | 3.0 | seconds |
| `candidate_seconds` | 1.0 | seconds |
| `candidate_ready_seconds` | 2.0 | seconds |
| `candidate_relative_spread` | 0.15 | fraction / dimensionless policy |
| `score_threshold` | 0.7 | fraction / dimensionless policy |
| `acoustic_weight` | 1.0 | fraction / dimensionless policy |
| `temporal_weight` | 0.0 | fraction / dimensionless policy |
| `max_impulses` | 64 | positive integer bound |
| `max_open_changes` | 64 | positive integer bound |
| `max_windows_per_call` | 4096 | positive integer bound |
| `max_revisions_per_call` | 4096 | positive integer bound |
| `max_state_bytes` | 16777216 | positive byte reservation bound |
| `max_parameter_features` | 4096 | positive integer bound |
| `assessment_min_coverage` | 1.0 | fraction / dimensionless policy |
| `established_min_coverage` | 1.0 | fraction / dimensionless policy |
| `candidate_min_coverage` | 1.0 | fraction / dimensionless policy |
| `min_contiguous_seconds` | 0.0 | seconds |

Defaults favor a 20 ms spectral lookback with 5 ms cadence, fast strong onset,
40 ms separately confirmed offset, and restrained 12 dB AND 0.01 PCM² level gates.
Low floor permits weak signals but increases leakage/noise sensitivity. Slow
10-second adaptation and anomaly exclusion protect established experience at the
cost of delayed candidate maturation. Rhythm needs at least four observed IOIs;
this intentionally delays tempo availability. Bounds are engineering allowances,
not learned-model budgets guaranteed suitable for every microphone.

Measured offset-confirmation sensitivity on the sustained fixture: 20/40/80 ms
confirmation produced 35/55/95 ms decision delay. Other defaults have fixture
coverage but no broad population calibration or systematic parameter search.
Increasing confirmation adds latency; absolute power gates are sensitive to device
gain and AGC. Changing bands/frame/hop changes model schema compatibility.

## Validation and reproducibility

Run `python -m pytest`, `python benchmarks/acoustic.py --seconds 5 --channels 1`,
and the optional live adapter:

```sh
pip install -e ".[dev,dsp]"
pip install sounddevice
python examples/live_microphone.py --acoustic --device 1 --sample-rate 48000 --block-size 480 --duration 3
```

Tests never require a microphone. The live script prints local change counts,
sequence integrity, refusals and process percentiles; it does not record audio.
The benchmark reports separately instrumented memory and uninstrumented timing,
CPU seconds, throughput, deterministic labels and matching tolerance. See
[the recorded pilot](../benchmarks/acoustic-results.md). These small synthetic and
ambient trials are regression evidence, not field accuracy or safety guarantees.
Controlled labelled live sound trials, measured AGC/gain, calibrated confidence,
broader noise/room/tempo mixtures, long-run memory/RSS and additional Python/platform
measurements remain follow-up validation.

## Independent audit correction (2026-11-23)

Band onset intensity-dependent confirmation now uses that band's sinusoidal-equivalent
amplitude `sqrt(2 * band_power)`; broadband continues to use observed peak. A strong
component elsewhere cannot give weak spectral leakage instantaneous confirmation.
This is an amplitude proxy, not an isolated-band waveform peak or source classifier.
No thresholds, band definitions or learned-feature schemas were changed. Genuine
new-band activity and simultaneous loudness changes remain independently eligible.
The original level-step false onset is corrected, but five of eighteen additional
phase/gain step controls still produce false onsets. This is an OPEN P0 limitation,
not a claim that amplitude-step discrimination is solved. A more discriminating
transient evidence rule needs validation against short genuine band impulses.

Rhythm expiry is checked before a newly arriving impulse can extend the frozen
deadline. Admission refusal episodes end only after materialization and DSP state
commit; a failed draft must not consume their end diagnostics. See
[the independent audit](../benchmarks/acoustic-audit-2026-11-23.md).

## Targeted gain-edge remediation (2026-11-23)

The five audited tone failures now pass unchanged. For established audible
background, band onset confirmation also checks a counterfactual spectrum after
hop RMS-envelope equalization inside the current contiguous window. Raw calibrated
powers, loudness decisions, learning feature units and snapshot schemas are unchanged.
Below-floor subblocks make this hypothesis unavailable; it must not veto newly
audible activity. Raw band evidence lacking independent confirmation is labelled
`common_mode_onset_ambiguous` through the existing optional band reason field.
Still-audible bands following common attenuation within the existing offset
hysteresis are provisionally retained (`common_mode_presence_ambiguous`).

This is a causal competing hypothesis, not source identification. Its extra FFT
increases compute and workspace (logical per-frame allowance 48→80 bytes/sample).
No acoustic thresholds were retuned. One new abrupt stochastic-noise gain control remains
a strict expected failure; the integration gate remains NEEDS REMEDIATION. See
[the full before/after report](../benchmarks/onset-remediation-2026-11-23.md).

For common attenuation, the overall pattern must remain audible; an individual
band may temporarily be below its floor and is then provisionally retained with
an ambiguity reason, rather than confidently declared absent. This resolves the
AGC-like ramp control; the abrupt noise-step negative control still fails.

## Directional stochastic-presence correction (2026-11-23)

The last audited stationary-noise gain-recovery false onset is corrected. During
common attenuation, band power above the existing gain-adjusted lower hysteresis
bound supports continued provisional presence. Excess power is no longer rejected
by an upper exact-gain-fit bound: that is not evidence of disappearance. The lower
bound, audible broadband requirement, all raw measurements and existing ambiguity
reason remain unchanged. No estimator, thresholds, state size, public record or
learned/snapshot schema changed. Genuine silence/negative residuals retain the
ordinary offset path; new inactive bands retain independent onset confirmation.

The unchanged seed-785 case is now a normal regression with no xfail. The prior
NEEDS REMEDIATION gate is superseded for this tested defect by READY FOR BROADER
VALIDATION, not production readiness. See [the report](../benchmarks/stochastic-onset-remediation-2026-11-23.md)
for seeded noise controls, positive-band resolution/preconditions, measurements and
remaining uncertainty/calibration limits.

## Broader validation remediation (2026-11-23)

During continuous observation, the activity detector keeps a bounded scalar
average of each cold-observed band's share of broadband power. For an occupied
band, a new event requires both raw power above the existing band reference and
a sustained increase in share. Uniform gain across an analysis frame leaves
the share unchanged. For an
occupied band without separate hop-level impulse evidence, contrast must persist
for at least one causal analysis frame. This adds confirmation time for the
occupied-band path. The absolute sensitivity floor and intensity-based
confirmation remain in use.

A confirmed relative contribution ends when both smoothed contrast and its own
band power fall, or when band power falls below the absolute floor. An
unrelated rise elsewhere cannot end the contribution by dilution alone.
The independent offset-confirmation interval still applies. The background can
remain present after the contribution ends. These transient scalars are cleared
on continuity breaks and are absent from snapshots.

Rhythm rearming requires a short plateau and consistency with the slope of each
available accepted fit. An observed reversal can use the supported extreme of an
open transition as its reference. A short trend cannot resolve a conflicting
longer history unless the short intervals support a plateau.
