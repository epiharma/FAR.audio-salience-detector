# FAR DESIGN-003 broader acoustic validation — 2026-11-23

## Executive gate

**NEEDS REMEDIATION.** OBSERVED reproducible false onsets in continuous noise,
fragmentation of a single gradual acceleration into two logical tempo changes,
and missed onset/offset pairs over a colored background prevent a readiness claim
for controlled integration trials. This is a validation finding, not a runtime fix.
The additional live-accuracy gate is **BLOCKED / INSUFFICIENT EVIDENCE** because
the actual capture's independent timing markers failed. Passing unit tests does
not override these findings. No production-readiness claim is made.

Runtime source, AcousticConfig defaults, existing tests, approved design documents,
and all prior reports have identical SHA-256 hashes to task start. No commits or
pushes were made. Only validation artifacts, a metric-harness test, and narrowly
justified sdist manifest entries were added/changed in this task.

## Environment, scope and provenance

- OBSERVED: Windows 11 build 26200, CPython 3.14.2 AMD64, NumPy 2.5.5;
  `Intel64 Family 6 Model 170 Stepping 4, GenuineIntel`, 14 logical CPUs.
  A marketing CPU model was not obtained. Shared host; CPU/power scheduling was
  not controlled. pytest 9.2.0, Ruff 0.17.6, build 1.6.1, setuptools 84.2.1.
- Git HEAD `ad2ea735cf72a5b2f62915a9640b989bd2374baf` with extensive pre-existing uncommitted
  implementation. Exact initial status and source/test/doc hashes are in the JSON
  evidence. HEAD alone does not identify the measured implementation.
- Read DESIGN-001, DESIGN-002, **DESIGN-003-acoustic-change-detection-APPROVED.md**,
  ADR-001, public contracts, acoustic model guide, and all four previous reports.
  DESIGN-002's old implemented-skeleton description and chronological model-guide
  addenda are historical where superseded by approved DESIGN-003 and remediation.
  The older unapproved DESIGN-003 file was already deleted at task start; that
  deletion was preserved. No approved specification was rewritten.
- Defaults: `AcousticConfig()` without threshold/band/horizon overrides. Harness
  `config()` explicitly supplies bounded ResourceLimits (4 sources, at least 8
  channels, 128 baseline records); this is fixture capacity, not acoustic tuning.
- 213 synthetic case runs: initial 171, additional rhythm 18, lifecycle 4, colored
  noise 20. Samples and exact source-frame labels were frozen in each set's JSON
  manifest **before that set's first detector evaluation**. Later delivery-timing
  instrumentation reproduced identical initial confusion counts. All inputs are
  synthesized locally; no field dataset or reviewed recorded-audio corpus exists.
- Initial development seed 43; nominal heldout seeds 107/809; 8/16/48 kHz.
  Development chunks 480 frames; heldout chunks cycle 137/480/997, including final
  partial chunks. Two heldout families at seed 809 mirror the same signal into two
  channels; these are correlated channels, not independent microphones.
- Initial set: 105 unique sample hashes across 171 cases. Development has 57 unique
  hashes, nominal heldout 72 across 114 cases, **24 hashes overlap splits** (clicks,
  rhythm controls and other deterministic signals). No statistical independence
  or generalization claim is supported by this nominal split.
- The additional rhythm set has 18 unique hashes, **zero exact input overlap**
  with the initial set, new intervals/amplitudes and 11.025/22.05/48 kHz. It remains
  the same procedural family, not independent field evidence. Colored-noise seeds
  16031/99277 at 16/48 kHz give 20 unique hashes. Lifecycle adds 4 unique waveforms.
- White/pink/brown spectra are procedural FFT-shaped noise, not physical source
  identities. Gain controls use continuous realizations; mixed-background probes
  use .03-RMS band-limited pink noise plus .06-peak 213-Hz hum, adding .02/.06-peak
  1337-Hz activity. This defines digital component/background power, not measured
  acoustic SNR, SPL, human speech or real fan noise.

Frozen manifest SHA-256:

- `acoustic-broader-lifecycle-manifest-2026-11-23.json`: `2bf0f06c0a8a2d066f16f5185fa216433361af6f6fef13d0a996a944e57dc4a4`
- `acoustic-broader-noise-manifest-2026-11-23.json`: `a25564828b38f00b457073fcc5c07a2fa9e6372012000148e055270ad8146df6`
- `acoustic-broader-rhythm-manifest-2026-11-23.json`: `2544e0648a41346de3df527ae8a97977f81d8b8c03fffc4df4cbfbff5b7a016c`
- `acoustic-broader-validation-manifest-2026-11-23.json`: `38889dc501fb79ada6bf7412423392bf828af383688f8e62bf714dad2a1a85ca`

## Matching policy and interpretation

One logical prediction is revision 1 of each change_id. Later full-state revisions
do not create additional events. Matching is maximum-cardinality bipartite,
same kind/channel, visiting nearest start-frame candidates first. One ground-truth
event can match at most one prediction. Remaining nearby predictions are duplicate
FPs; unmatched labels are FNs. Three harness tests check non-greedy matching,
duplicates, channel/kind isolation, undefined-positive subsets and distinct
confirmation/delivery delays.

Frozen start-frame match windows are 20 ms early / 120 ms late for activity and
loudness; tempo allows 20 ms early / 2 s late for abrupt changes or 5 s for gradual
changes, measured from the first changed IOI. These are **declared evaluation
tolerances, not approved latency budgets**. A gradual monotonic episode has one
label. Pulse-train activity labels cover the first four acquisition pulses, not
every pulse after a sequence is established. A skipped pulse is not deceleration.

Competing-train activity grouping has no reliable label, so those activity kinds
are explicitly unscored; temporal negatives test unavailability for incompatible
trains. Spectral drift is exploratory and entirely unscored. All predictions are
retained in evidence, including unscored ones; their empty confusion cells are not
successful negative tests. Brown-noise loudness/tempo negative assumptions need
independent signal-level review (see B5); frozen counts remain unchanged.

`Neg` below counts **case/channel contexts with zero labels for that kind**, not
true-negative frames/events. Precision is TP/(TP+FP); recall TP/positives, undefined
when no positives. No accuracy/TN statistic is fabricated. Valid exposure is
observed **channel-seconds**, not wall time or independent recording duration.
Kind-specific exposures exclude unscored kinds. FP/hour is descriptive scaling
of very short synthetic exposures and repetitions, not a field-rate estimate,
confidence bound or authoritative zero-error claim. The long stress run is not
labelled and is never added to the accuracy denominator.

## Per-kind results (frozen expectations)

| Set | Kind | GT+ | Neg | TP/FP/FN | Dup FP | Precision/recall | Valid channel-s | FP/channel-hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial_development | sound_onset | 90 | 15 | 90/2/0 | 0 | 0.978/1.000 | 197.958 | 36.371 |
| initial_development | sound_offset | 18 | 33 | 18/0/0 | 0 | 1.000/1.000 | 197.958 | 0.000 |
| initial_development | tempo_increase | 6 | 48 | 6/6/0 | 3 | 0.500/1.000 | 215.358 | 100.298 |
| initial_development | tempo_decrease | 6 | 48 | 6/0/0 | 0 | 1.000/1.000 | 215.358 | 0.000 |
| initial_development | loudness_spike | 9 | 42 | 9/0/0 | 0 | 1.000/1.000 | 197.958 | 0.000 |
| initial_development | loudness_drop | 12 | 39 | 12/0/0 | 0 | 1.000/1.000 | 197.958 | 0.000 |
| initial_heldout | sound_onset | 183 | 33 | 183/0/0 | 0 | 1.000/1.000 | 404.616 | 0.000 |
| initial_heldout | sound_offset | 39 | 69 | 39/0/0 | 0 | 1.000/1.000 | 404.616 | 0.000 |
| initial_heldout | tempo_increase | 12 | 102 | 12/12/0 | 6 | 0.500/1.000 | 439.416 | 98.312 |
| initial_heldout | tempo_decrease | 12 | 102 | 12/0/0 | 0 | 1.000/1.000 | 439.416 | 0.000 |
| initial_heldout | loudness_spike | 24 | 84 | 24/0/0 | 0 | 1.000/1.000 | 404.616 | 0.000 |
| initial_heldout | loudness_drop | 30 | 78 | 30/0/0 | 0 | 1.000/1.000 | 404.616 | 0.000 |
| rhythm | sound_onset | 72 | 0 | 72/0/0 | 0 | 1.000/1.000 | 151.824 | 0.000 |
| rhythm | sound_offset | 0 | 18 | 0/0/0 | 0 | undefined/undefined | 151.824 | 0.000 |
| rhythm | tempo_increase | 6 | 12 | 6/3/0 | 3 | 0.667/1.000 | 151.824 | 71.135 |
| rhythm | tempo_decrease | 6 | 12 | 6/0/0 | 0 | 1.000/1.000 | 151.824 | 0.000 |
| rhythm | loudness_spike | 0 | 18 | 0/0/0 | 0 | undefined/undefined | 151.824 | 0.000 |
| rhythm | loudness_drop | 0 | 18 | 0/0/0 | 0 | undefined/undefined | 151.824 | 0.000 |
| lifecycle | sound_onset | 24 | 0 | 24/0/0 | 0 | 1.000/1.000 | 31.200 | 0.000 |
| lifecycle | sound_offset | 0 | 4 | 0/0/0 | 0 | undefined/undefined | 31.200 | 0.000 |
| lifecycle | tempo_increase | 0 | 4 | 0/0/0 | 0 | undefined/undefined | 31.200 | 0.000 |
| lifecycle | tempo_decrease | 0 | 4 | 0/0/0 | 0 | undefined/undefined | 31.200 | 0.000 |
| lifecycle | loudness_spike | 0 | 4 | 0/0/0 | 0 | undefined/undefined | 31.200 | 0.000 |
| lifecycle | loudness_drop | 0 | 4 | 0/0/0 | 0 | undefined/undefined | 31.200 | 0.000 |
| noise | sound_onset | 8 | 12 | 0/8/8 | 0 | 0.000/0.000 | 34.000 | 847.059 |
| noise | sound_offset | 8 | 12 | 0/2/8 | 0 | 0.000/0.000 | 34.000 | 211.765 |
| noise | tempo_increase | 0 | 20 | 0/0/0 | 0 | undefined/undefined | 34.000 | 0.000 |
| noise | tempo_decrease | 0 | 20 | 0/1/0 | 0 | 0.000/undefined | 34.000 | 105.882 |
| noise | loudness_spike | 12 | 8 | 11/1/1 | 0 | 0.917/0.917 | 34.000 | 105.882 |
| noise | loudness_drop | 12 | 8 | 12/0/0 | 0 | 1.000/1.000 | 34.000 | 0.000 |

The colored-noise table includes 8/8 missed onset labels and 8/8 missed offset
labels, with early/unrelated predictions unmatched; it must not be hidden behind
the initial set's high recall. The one brown-noise spike FP/FN pair is a label/
latch ambiguity under the frozen policy, not a confirmed scoring bug.

### Timing, tempo and confidence

All values below are p50/p95/p99/max milliseconds with explicit matched-event
denominators. Confirmation is `confirmed_frame - ground_truth_frame`.
Delivery adds chunk granularity: the final submitted source frame of the first
`process()` call returning that change minus ground truth. It is source-audio
availability latency under accelerated replay, not wall-time acquisition-to-
decision latency. Microphone/OS capture and DSP CPU time are not included in these
frame differences. Unmatched events/FNs have no successful timing measurement.

| Set | Kind | Matched n | Confirmation ms | First-delivery source ms |
| --- | --- | --- | --- | --- |
| initial | sound_onset | 273 | 4.000/15.000/30.000/30.000 | 19.000/49.000/105.250/105.250 |
| initial | sound_offset | 57 | 55.000/55.000/55.000/55.000 | 80.000/109.625/109.625/109.625 |
| initial | tempo_increase | 18 | 404.000/2699.000/2699.000/2699.000 | 472.750/2729.000/2729.000/2729.000 |
| initial | tempo_decrease | 18 | 804.000/1199.000/1199.000/1199.000 | 860.500/1224.000/1224.000/1224.000 |
| initial | loudness_spike | 33 | 5.000/15.000/15.000/15.000 | 12.938/105.250/105.250/105.250 |
| initial | loudness_drop | 42 | 20.000/20.000/20.000/20.000 | 39.750/108.750/108.750/108.750 |
| rhythm | sound_onset | 72 | 2.086/5.034/5.034/5.034 | 15.625/80.091/85.351/85.351 |
| rhythm | tempo_increase | 6 | 352.200/2497.000/2497.000/2497.000 | 420.227/2568.707/2568.707/2568.707 |
| rhythm | tempo_decrease | 6 | 724.490/1133.379/1133.379/1133.379 | 764.082/1179.683/1179.683/1179.683 |
| lifecycle | sound_onset | 24 | 10.000/10.000/10.000/10.000 | 22.500/60.500/60.500/60.500 |
| noise | sound_onset | 0 | undefined | undefined |
| noise | sound_offset | 0 | undefined | undefined |
| noise | loudness_spike | 11 | 5.000/15.000/15.000/15.000 | 9.625/18.188/18.188/18.188 |
| noise | loudness_drop | 12 | 20.000/20.000/20.000/20.000 | 38.875/43.812/43.812/43.812 |

Offset confirmation reaches 55 ms in the initial cases. A dedicated grouping
probe publishes onset at source 405 ms for a 400-ms start (before the configured
40-ms grouping interval closes), adds low/mid-band membership at +5/+10 ms from
first confirmation, keeps one ID across 6 revisions and preserves the original
full-state object unchanged. General `revision_lag_ms` in raw case records also
includes completion/interruption lifetime and **is not grouping latency**.

Tempo availability is nonmissing tempo windows / all observed feature windows,
including startup and post-sequence tails. Error is MAE against current labelled
IOI rate only where tempo is available. The table reports p50/p95/p99/max of
**per-case** availability and MAE, not a falsely pooled independent estimate.
Unavailable estimates are excluded from MAE, so error must be read with coverage.
Ambiguous mixtures have no correct single tempo target and MAE is undefined.

| Family | Runs | Availability fractions | MAE Hz among available |
| --- | --- | --- | --- |
| initial/abrupt_faster | 9 | 0.659/0.659/0.659/0.659 | 0.000/0.000/0.000/0.000 |
| initial/abrupt_slower | 9 | 0.732/0.732/0.732/0.732 | 0.000/0.000/0.000/0.000 |
| initial/competing_pulses | 9 | 0.050/0.050/0.050/0.050 | undefined |
| initial/gradual_faster | 9 | 0.786/0.786/0.786/0.786 | 0.086/0.086/0.086/0.086 |
| initial/gradual_slower | 9 | 0.878/0.878/0.878/0.878 | 0.102/0.102/0.102/0.102 |
| initial/one_skipped | 9 | 0.681/0.681/0.681/0.681 | 0.000/0.000/0.000/0.000 |
| initial/stable_pulses | 9 | 0.798/0.798/0.798/0.798 | 0.000/0.000/0.000/0.000 |
| rhythm/abrupt_faster | 3 | 0.649/0.649/0.649/0.649 | 0.000/0.001/0.001/0.001 |
| rhythm/abrupt_slower | 3 | 0.722/0.722/0.722/0.722 | 0.000/0.000/0.000/0.000 |
| rhythm/gradual_faster | 3 | 0.781/0.781/0.781/0.781 | 0.088/0.088/0.088/0.088 |
| rhythm/gradual_slower | 3 | 0.868/0.868/0.868/0.868 | 0.104/0.104/0.104/0.104 |
| rhythm/skipped | 3 | 0.680/0.681/0.681/0.681 | 0.000/0.000/0.000/0.000 |
| rhythm/stable | 3 | 0.789/0.789/0.789/0.789 | 0.000/0.000/0.000/0.000 |
| lifecycle/merge_reacquire | 2 | 0.523/0.523/0.523/0.523 | undefined |
| lifecycle/pause_timeout | 2 | 0.482/0.482/0.482/0.482 | undefined |

Predicted-event confidence across all synthetic outputs has p50/p95/p99/max
`1.000/1.000/1.000/1.000`; this includes unscored outputs.
Per-case window-confidence quantiles are retained in JSON (not pooled quantiles).
False events can carry high confidence; the noise-gain false onsets have .200 and
.283 confidence. **Confidence is uncalibrated**, not a probability. There was no
independent calibration experiment or legitimate Brier/ECE claim.

### Per-condition counts

Counts combine scored kinds within a condition; spectral_drift's zeros mean
unscored, not pass. Full per-kind/per-case labels, predictions, missing reasons,
delays, sequence transitions and failures are in the JSON evidence.

| Condition | Runs | Channel-s | TP/FP/FN | Duplicate FP |
| --- | --- | --- | --- | --- |
| initial/abrupt_faster | 9 | 62.289 | 45/0/0 | 0 |
| initial/abrupt_slower | 9 | 65.889 | 45/0/0 | 0 |
| initial/band_over_hum | 9 | 13.500 | 18/0/0 | 0 |
| initial/click | 9 | 9.000 | 9/0/0 | 0 |
| initial/cold_hum_loss | 9 | 10.800 | 18/0/0 | 0 |
| initial/competing_pulses | 9 | 52.200 | 0/9/0 | 0 |
| initial/gradual_faster | 9 | 99.369 | 45/9/0 | 9 |
| initial/gradual_slower | 9 | 91.449 | 45/0/0 | 0 |
| initial/level_steps | 9 | 18.000 | 24/0/0 | 0 |
| initial/modulated_band_noise | 9 | 13.500 | 18/0/0 | 0 |
| initial/noise_gain | 9 | 13.500 | 18/2/0 | 0 |
| initial/one_skipped | 9 | 69.489 | 36/0/0 | 0 |
| initial/opposite_bands | 9 | 10.800 | 18/0/0 | 0 |
| initial/slow_gain | 9 | 13.500 | 0/0/0 | 0 |
| initial/spectral_drift | 9 | 27.000 | 0/0/0 | 0 |
| initial/stable_pulses | 9 | 69.489 | 36/0/0 | 0 |
| initial/sub_extreme | 9 | 13.500 | 0/0/0 | 0 |
| initial/sustained | 9 | 16.800 | 48/0/0 | 0 |
| initial/weak | 9 | 11.700 | 18/0/0 | 0 |
| rhythm/abrupt_faster | 3 | 18.639 | 15/0/0 | 0 |
| rhythm/abrupt_slower | 3 | 20.199 | 15/0/0 | 0 |
| rhythm/gradual_faster | 3 | 31.119 | 15/3/0 | 3 |
| rhythm/gradual_slower | 3 | 28.839 | 15/0/0 | 0 |
| rhythm/skipped | 3 | 27.039 | 12/0/0 | 0 |
| rhythm/stable | 3 | 25.989 | 12/0/0 | 0 |
| lifecycle/merge_reacquire | 2 | 15.600 | 8/0/0 | 0 |
| lifecycle/pause_timeout | 2 | 15.600 | 16/0/0 | 0 |
| noise/brown_gain | 4 | 6.800 | 7/5/1 | 0 |
| noise/colored_hum_component_0.02 | 4 | 6.800 | 0/3/8 | 0 |
| noise/colored_hum_component_0.06 | 4 | 6.800 | 0/3/8 | 0 |
| noise/pink_gain | 4 | 6.800 | 8/0/0 | 0 |
| noise/white_gain | 4 | 6.800 | 8/1/0 | 0 |

## Defect register and bounded trace analysis

**B1 — High, confirmed false onset / presence tracking defect.** Reproducer:
`python benchmarks/broader_validation.py --case development:43:16000:noise_gain`
returns exit 1. White-noise activity is present throughout; first gain drop is
frame 8000. OBSERVED false low-band onsets start at 6080 (confirm 6160) and 16080
(confirm 16400). Before the first, state changes active→inactive at frame 1200
while broadband power remains .05945 and low-band power .0013496; provisional
reference mean .0050038, exposure .02 s. It later rises above the retained absence
reference with no new component. INFERRED cause: a short stochastic band excursion
crosses relative absence hysteresis against an immature/low-variance reference;
common-mode guard does not cover this non-common fluctuation. This broadens beyond
the previous seed-785 fix; it does not establish that the earlier targeted fixture
regressed. Additional white/brown controls also produce false recovery onsets.
No fix applied; retain exact repro as future failing regression, alongside positives.

**B2 — High, confirmed logical-event fragmentation.** Reproducer:
`python benchmarks/broader_validation.py --case development:43:48000:gradual_faster`
returns exit 1. OBSERVED one monotonic acceleration generates two distinct
tempo_increase IDs in all 9 initial runs (only 3 unique input waveforms) and all 3
new-parameter runs. Trace at 48 kHz: detection 410400; completion/reference reset
437280 while the IOIs still decrease; another detection 470160 and completion
491280. INFERRED cause: nonzero gradual slope qualifies as stable under jitter ×
tempo_rearm_fraction; three short-horizon intervals rearm inside an ongoing trend.
This violates the frozen one-episode/no-duplicate expectation. End-of-trend
confirmation policy needs explicit review; thresholds were not tuned in validation.
No fix applied; new-parameter evidence prevents explaining this by one exact fixture.

**B3 — High, ambiguity/unavailability limitation with interpretation caveat.**
`python benchmarks/broader_validation.py --case development:43:48000:competing_pulses`
returns exit 1. Two fixed constituent periods (.2/.313 s) produce a temporal FP
under the predeclared ambiguous-mixture policy. OBSERVED at frame 86640: unavailable
ambiguous history becomes 8.1967 Hz with retained reference 5.4645 Hz and emits
acceleration. INFERRED: a locally coherent short fit overrides incompatible longer
history; availability is about .05013 for this condition. The union may locally
look like acceleration; this is an ambiguity-policy limitation, not evidence of
physical-source identification. Needs adjudication against DESIGN-003's required
unavailability, not automatic relabeling or inventing parallel rhythm tracks.

**B4 — High, background-relative sensitivity gap.** `python benchmarks/broader_noise.py`
retains all counts and traces; filter `colored:*:colored_hum_component_*` in JSON.
OBSERVED 8/8 onsets and 8/8 offsets missed under exact injected-component labels.
At 16 kHz/seed16031/.06 peak, initial mid-band leakage power 2.5819e-6 is above
the floor and initializes observed_active. Reference remains at that mean with
.005-s exposure; by frame 9920, injected band power is .0018215 but state remains
active, so no relative-new-activity event. After cessation, residual power also
stays above the low reference and no offset confirms. INFERRED: binary presence
and cold/provisional background selection obscure background-relative component
changes when leakage is already above floor. No source semantics are required to
observe the injected spectral change. This does not justify detecting every level
change as a new onset. No fix applied; protect gain-only negative controls in any fix.

**B5 — Medium, stochastic negative-label uncertainty / latch risk.** Seed99277
brown noise at 48 kHz emits spike at .565 s (confirm .585), before the scheduled
drop, then no separate recovery spike at 1.1 s; frozen scoring records FP1/FN1.
INFERRED: natural brown-noise envelope may itself meet the extreme gates and hold
the transition latch. Therefore this is **not yet a confirmed false loudness
event**; independently compute qualifying raw transitions before assigning blame.
At 16 kHz a brown-noise run emits tempo_decrease .390/.395 s without a designed
pulse train: a policy FP, but stochastic local regularity also needs adjudication.
Labels/counts are preserved, not silently repaired. Activity recovery FPs in these
runs share B1's band absence/recovery mechanism. Public failed-case traces and
three additional private-state representative traces are retained.

Bounded traces: failed-case deques keep at most 96 nearby public windows; they can
drop early context and are not full recordings. `broader_error_traces.py` reruns
six representative frozen failures using observation-only wrappers that delegate
to original methods and restore them in finally; it records bounded pre/post
activity/reference or rhythm states without changing algorithms. Per-case failure
list maps every scored error to these families. No failing expectation was xfailed,
removed, or weakened, and no runtime remediation was attempted in this task.

## Requirement-to-test traceability (executed full suite)

| Approved behavior | Executed evidence |
| --- | --- |
| Six kinds, cold observed_active, band onset/offset and opposite directions | Frozen initial + colored-noise matrix; test_sustained_onset_offset_revisions, test_cold_hum_and_disappearance, test_band_onset_over_hum_and_opposite_directions; B1/B4 remain |
| Strong click / weak onset / no offset for isolated impulse | click/weak families; test_click_has_completion_without_offset, test_ringing_impulse_is_one_onset_without_offset, test_weak_confirmation_and_shutdown_interrupt |
| Gain-only steps, dual loudness gates, onset + independent spike | level_steps/sub_extreme/noise_gain and colored controls; test_loudness_step_and_dual_gate, test_genuine_band_onset_and_spike_remain_independent, frozen 18 amplitude controls; B1/B5 remain |
| Stable/abrupt/gradual tempo, no slowing from skipped pulse | Both frozen rhythm sets; test_stable_and_missing_beat, test_abrupt_acceleration_and_deceleration, test_late_impulse_cannot_reopen_expired_sequence; B2/B3 remain |
| Merge/pauses/timeouts/new sequence on reacquisition | 4 lifecycle probes: two distinct sequence IDs each, with reacquisition after merge/pause; test_merge_reacquires_new_sequence, test_rhythm_pipeline_tempo_and_merge |
| Full-state immutable revisions, interrupted end unknown, three families/order | All 213 case-run invariants no violations; dedicated grouping probe; test_no_events_can_mutate_previous_batches, test_gap_interrupts_not_offset, test_revision_and_state_resource_refusals |
| Chunk/layout/reorder/channel identity invariance | test_partition_layout_and_zero_history, test_frame_domain_layout_and_partition_equivalence, test_layout_channel_reorder_and_borrowed_memory, test_out_of_order_watermark_late_overlap_and_conflict |
| Independent clocks, exact source timestamps, no implicit alignment | Additional per-anchor tests at 8/16/44.1/48 kHz and asynchronously interleaved 48/16-kHz sources; test_independent_clocks_and_reorder, test_measurement_time_and_full_window_cadence; alignment/hysteresis suite |
| Invalid/quarantined channels, gaps and no fabricated silence | test_invalid_channel_isolation, test_already_active_noise_after_gap_and_invalid_block, test_gap_interrupts_not_offset, F1/F2 recovery/invalid-tail tests |
| Sample-rate changes preserve compatible baseline / Nyquist unavailable | test_sample_rate_nyquist_keeps_compatible_models, test_geometry_and_rate_preserve_individual_compatibility; 8/16-kHz high band reports band_above_nyquist, not zero power |
| Partial valid windows / assessment separate from learning | test_partial_horizon_eligibility_and_no_missing_score_renormalization, test_no_assessment_transition_when_contiguity_policy_fails, test_union_coverage_nominal_duration_and_separate_learning |
| Candidate block/reject/promote, archival/reactivation, command idempotency | test_parallel_candidate_block_reject_and_promotion, test_candidate_rejection_fresh_readiness_and_promotion, test_archive_reactivate_and_automatic_extension_readiness, test_command_idempotency_conflict_stale_and_restart |
| V2/legacy persistence and cold temporal restart | test_snapshot_roundtrip_legacy_and_transient_reset, corruption/schema tests; long stress periodic saves and 5 restore/new-session cycles |
| Refusal transaction rollback, aggregation/end, no eviction | test_resource_rollback_and_borrowed_copy, test_capacity_refusal_rolls_back_all_staged_channels_and_pruning, test_hard_refusal_first_repeat_worse_aggregate_end, F5 tests, test_materialization_failure_does_not_end_refusal_episode, test_candidate_limit_does_not_evict_experience |
| No retained borrowed buffers / bounded history | test_owned_buffers_and_original_timing_survive_caller_release, test_closed_model_releases_transient_audio, segmented/staging history suites; long stress internal caps |

Test coverage is evidence for those fixtures; it is not a claim that all combinations
or unlabelled real-world conditions are correct.

## Actual microphone trial and live blocker

OBSERVED actual sounddevice RawInputStream capture: input index 1 (API-reported
`Microphone Array (Intel® Smart `, truncated by device API), output index 3
`Speakers (Realtek(R) Audio)`, MME; 48 kHz native float32 mono, 480-frame reads.
1,361,280 frames captured, 0 reported overflows, 0 event sequence gaps. Device API
default rate is 44.1 kHz; the measured stream was opened successfully at 48 kHz.
Gain, OS volume, AGC, echo cancellation and physical sound levels: **UNKNOWN**.
sounddevice is optional external validation only; no new product capture dependency.

Process timing p50/p95/p99/max = 2.992/5.523/7.383/11.186 ms. These timings include
only synchronous `process()`, not acquisition-to-decision latency. Low digital
speaker peak .019999 does not establish actual acoustic audibility at the mic.

Predeclared independent normalized-correlation marker criterion (both ≥.1, fitted
ratio within .005 of 1, no overflow; **unapproved pilot data-quality criterion**)
failed: coefficients .05766/.05613. A chance fit produced .822854; **this is an
invalid alignment fit, not measured clock drift**. Planned warped regions and
their derived power ratio in raw diagnostic output are invalid as labels/SNR.
No predictions from this capture are scored TP/FP/FN. No audio was retained or
uploaded. Quiet human speech-like activity, physical taps/fan, independently
annotated cessation/level changes and a second room/device remain **NOT VERIFIED**.

**BLOCKED: independently labelled live validation was not performed successfully.**
Executable rerun/operator protocol: `controlled-real-audio-protocol.md`;
`controlled_audio_validation.py`; `real-audio-label-template.json` is explicitly
unreviewed with null measurements, not invented data. Do not lower the marker
gate after observing the failure or infer timestamps from receipt time.

## Longer resource run

OBSERVED 1800 s generated and accepted primary-channel audio,
180,000 process calls, 48 kHz/480 frames, 0 refusals. Channel 2 is present
for 1 s per minute (30 s total), omitted otherwise; this is **not** a continuous
two-channel or independent-device stress. Total valid channel exposure 1830 s.
Five repeating one-minute modes: valid zeros, steady hum, stochastic noise,
impulses, hum + second tone. Zero samples are valid silence, not idle; a separate
0.1-s wall-idle probe observed no new emissions. Stress is accelerated/replayed
source time, not a 30-minute hardware recording or a multi-hour soak.

Wall 1930.99 s, process CPU 1810.14 s.
Whole-driver audio/wall throughput = 0.932×,
including snapshots, GC and restarts; this measured run did not sustain real time.
29 periodic snapshots, 5 end/restore/new-session
cycles. Restored DSP transient states asserted empty; baseline record IDs checked
for retention without silent eviction. Driver does not exercise approve/block/
archive commands in the long run; those are covered by executed unit tests.
Emission sequences were contiguous within each run with explicit run reset;
recent IDs checked in a bounded 1024-entry deque and change revisions in a bounded
512-ID LRU. This is not an unbounded lifetime unique-ID proof.

Observed maxima: `{"candidates": 82, "changes": 2, "groups": 2, "history": 200, "impulses": 64, "profiles": 64}`.
The harness field `candidates` counts **all baseline catalog records**, including
established records, not exclusively candidate-status records. The last sample
has 336 model/candidate feature-stat entries.
Every call checked impulse/profile/group/open-change/history caps. These are
Python/DSP structure bounds for this workload, not an OS allocator ceiling.
Transactional refusals and episode aggregation were deliberately tested in the
unit suite; the long run had no refusals and therefore did not exercise them.

Windows working-set RSS sampled every 30 audio-seconds, before/after explicit GC:
after-GC first/last/median/max 39.10/41.21/
40.95/41.21 MiB.
After 300-s warmup, min/median/max = 40.48/
41.03/41.21 MiB.
First-to-last sampled change = 2.11 MiB; post-warmup
first-to-last change = 0.73 MiB. Last sample is at
1770 audio-seconds, not after teardown.
All sample coordinates and pre-GC values are retained; short-run working-set
behavior cannot establish long-term RSS stability or absence of every leak.

## Process performance

Final benchmark after stress completed, no other FAR validation process intentionally
running; host otherwise uncontrolled. Each cell is 5 audio-seconds of cold-start
steady 213-Hz hum, correlated channels, no microphone and no refusal. Percentiles
time only `process()`; wall/CPU/throughput include fixture/chunk creation and
consumption, but exclude detector/source initialization and final close.
Small sample count, no isolated-host repeated confidence interval or real-time
guarantee. Initial concurrent benchmark is retained separately, not used for a
causal pre/post speedup claim.

| Rate Hz | Block frames | Channels | Process p50/p95/p99/max ms | Audio/wall × | Wall s | CPU s |
| --- | --- | --- | --- | --- | --- | --- |
| 48000 | 480 | 1 | 1.041/2.322/3.571/6.919 | 7.945 | 0.629 | 0.562 |
| 48000 | 480 | 2 | 2.046/4.245/6.437/13.944 | 4.074 | 1.227 | 1.078 |
| 48000 | 480 | 8 | 8.610/14.768/18.128/25.826 | 1.082 | 4.620 | 4.312 |
| 48000 | 120 | 1 | 0.566/1.350/2.242/5.264 | 3.950 | 1.266 | 1.125 |
| 16000 | 160 | 1 | 0.992/2.314/3.475/5.319 | 8.414 | 0.594 | 0.562 |
| 44100 | 441 | 2 | 2.145/3.927/6.604/17.437 | 4.031 | 1.241 | 1.094 |

Compare each measured service time/throughput with input cadence; the 8-channel
48-kHz final workload has only 1.082× average throughput and p95 exceeds its 10-ms
input cadence. It does not establish sustained real-time behavior, especially as
learned catalogs grow; the initial concurrent run was much slower. No runtime
change separates these timings, so their difference is not a remediation speedup.
There is no approved numerical performance acceptance budget. FAR does not own
upstream backpressure/drop policy, and this report adds no capture queue/broker.

## Validation commands and actual results

From repository root, using `.venv/Scripts/python.exe` (abbreviated `python`):

```sh
python -m pip install -e . --no-deps --no-build-isolation
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python benchmarks/acoustic.py --seconds 5 --channels 1
python benchmarks/broader_validation.py --manifest
python benchmarks/broader_validation.py --evaluate
python benchmarks/broader_validation.py --independent-rhythm
python benchmarks/broader_lifecycle.py
python benchmarks/broader_noise.py
python benchmarks/broader_error_traces.py
python benchmarks/broader_validation.py --quality
python benchmarks/broader_validation.py --stress 1800
python benchmarks/broader_validation.py --perf
python benchmarks/controlled_audio_validation.py --input 1 --output 3
python -m build --no-isolation --outdir <scratch>/dist
<clean-env>/python -m pip install --no-index --no-deps --force-reinstall <wheel>
<clean-env>/python -I <clean-wheel-probe.py>
```

- Initial full baseline: 270 passed, zero xfail. Final full suite after 3 metric
  tests: **273 passed**, zero xfail; final checks in evidence. Ruff lint and
  formatting passed. The three known strict acoustic repro commands return 1;
  aggregate evaluation commands exit 0 after collecting JSON **with errors**, not
  a pass gate. No assertions were weakened to make CI green.
- Selected existing 25-s labelled benchmark still has TP/FP/FN onset10/0/0,
  offset2/0/0, tempo increase1/0/0, decrease1/0/0, spike2/0/0, drop3/0/0.
  Prior expanded 70-run report remains historical evidence, not independent new
  measurements or a substitute for this broader matrix.
- Editable install succeeded offline with preinstalled backend after setting child
  TEMP/TMP to a writable scratch directory. First attempt hit setuptools path
  interpolation on Codex sandbox temporary directory braces; it failed before
  installation. No project packaging change was made for that environment issue.
- Wheel + sdist built with preinstalled backend (`--no-isolation`, no dependency
  network fetch). Initial sdist inspection found frozen manifest JSON omitted;
  MANIFEST.in now includes the new manifests, label template and this evidence.
  Rebuilt archive entries were compared byte-for-byte. Clean wheel installed in
  a venv **without NumPy**; isolated import/default ingestion and explicit missing
  `[dsp]` ValueError passed. No runtime dependencies were added.
- Only Windows/CPython 3.14.2/NumPy 2.5.5 executed. Repository CI declares Ubuntu
  Python 3.11/3.14, but remote CI, other Python/NumPy versions, Linux/macOS and
  other hardware were **NOT RUN / NOT VERIFIED** in this task. No OS matrix pass
  is implied. See JSON packaging details for actual commands/probes.

## Changes and next actions

New files: `broader_validation.py`, `broader_lifecycle.py`, `broader_noise.py`,
`broader_error_traces.py`, four `acoustic-broader-*-manifest-2026-11-23.json`
files, this Markdown report and JSON evidence, `controlled_audio_validation.py`,
`controlled-real-audio-protocol.md`, `real-audio-label-template.json`, and
`tests/test_broader_validation_metrics.py`. Modified existing file: MANIFEST.in
only, to keep the validation assets reproducible from sdist. Pre-existing source,
config, documentation changes and deletion are preserved; no unrelated refactor.
No commit or push occurred.

1. Remediate B1/B4 together with exact failing noise/mixed-background regressions
   and retained gain-only/weak/real-new-band positive controls. Avoid a suppression
   rule that hides new components. Do not change frozen labels to pass.
2. Review logical tempo episode completion/rearm and ambiguity criteria, then
   remediate B2/B3 with both original and parameter-held-out controls. Independently
   adjudicate B5's stochastic loudness/tempo labels without altering this evidence.
3. Obtain independently reviewed live source-frame annotations using the supplied
   protocol, known or explicitly unknown gain/AGC, with a second session/device.
4. Establish intended supported workloads, run longer/more diverse RSS soaks and
   actual Python/OS CI matrix; profile 8-channel cost before promising cadence.

No arbitrary numerical integration acceptance thresholds were approved or invented.
Any future pilot budgets must be separately reviewed proposals. Until remediation
and the labelled-live gate succeed, the evidence does not support controlled
integration readiness. FAR's scope remains the six local changes and local health;
no AGI-global salience, capture SDK, scheduler, escalation or protective action was
implemented.
