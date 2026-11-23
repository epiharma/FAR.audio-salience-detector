# Targeted false-onset remediation — 2026-11-23

**Gate: NEEDS REMEDIATION.** Original 18 frozen controls now pass; one new stochastic-noise control still fails and remains a strict expected failure. No approved document, public schema, dependency or numerical acoustic default changed. No commit/push.

## Baseline and cause

Starting full suite: **165 passed, 5 xfailed**. `test_open_amplitude_edge_false_onset` contains the five original strict failures. Full source-frame traces were captured before runtime edits; JSON retains per-band raw power, RMS, peak, baseline mean/exposure, expected presence, before/after activity, confirmation, and full grouped revisions. Waveforms are one-second constant carriers at 48 kHz, with gain high on frames [19200,33600), phase continuous. Counterfactual evidence is diagnostic only.

| Failure (Hz, low→high, phase) | Publication frame / band | Raw power / counterfactual power PCM² | Confirmation seconds / baseline exposure seconds |
|---|---|---|---|
| 200, 0.03→0.7, 1.3 | 19680 / mid | 0.00823468 / 9.34601e-10 | 0 / 0.385 |
| 200, 0.01→0.99, 0.0 | 19680 / mid | 0.00261856 / 1.07453e-09 | 0.0082896 / 0.385 |
| 200, 0.01→0.99, 1.3 | 19440 / mid | 0.00450594 / 9.33271e-10 | 0.00152074 / 0.385 |
| 200, 0.01→0.99, 1.3 | 19680 / high | 0.00264268 / 3.62885e-12 | 0.00818987 / 0.385 |
| 1000, 0.01→0.99, 0.0 | 19680 / low | 0.00253308 / 9.23975e-12 | 0.00864692 / 0.385 |
| 1000, 0.01→0.99, 1.3 | 19680 / high | 0.00323423 / 1.03693e-11 | 0.00587197 / 0.385 |

Common chain: an established audible carrier is amplified; its discontinuous envelope produces real FFT sidebands; previously inactive bands cross the activity floor; band-intensity confirmation becomes fast enough for the finite-window sideband transient; `_new()` publishes immediately and sometimes groups later band contributions into the same ID. Grouping is not the cause and was not delayed. FFT energy normalization is correct; established models remain largely frozen by anomaly gating. The defect is treating gain-envelope spectral energy as independent presence evidence. The labels are valid for these explicitly constructed constant-carrier controls, not inferred physical source labels.

## Targeted internal correction

- `_acoustic_features.py:72`: an additional competing gain hypothesis equalizes source-frame-anchored hop RMS inside the existing full contiguous FFT frame. Any subblock below the acoustic floor makes that hypothesis unavailable, preserving below-floor emergence. Raw FFT powers/RMS/peak remain untouched.
- `_acoustic.py:499`: for established audible background, inactive-band confirmation requires both raw and envelope-equalized evidence. The existing floor/ratio/intensity thresholds apply; there is no fixture-specific frequency, phase or loudness-event veto. `common_mode_onset_ambiguous` explains above-floor raw energy lacking independent support.
- `_acoustic.py:850`: activity following whole-pattern attenuation within existing offset hysteresis is not ended solely by reduced level. This fixes multiband attenuation/recovery and AGC-like ramp false onsets discovered in the new negative controls. `common_mode_presence_ambiguous` exposes this tentative interpretation. Genuine spectral losses/silence continue through ordinary offset confirmation.
- DSP logical workspace reservation increased from 48 to 80 bytes per frame sample to account for the additional scratch/FFT. This is resource accounting, not a new acoustic threshold. All operations stay on the private transaction draft; no extra retained audio/learned model or snapshot migration is introduced.

The rejected alternative was stronger/longer universal onset confirmation, which can erase genuine short components and merely tunes the control sweep. The new hypothesis is causal and frame-anchored; it is evidence about local level modulation, not source classification or proof of a physical cause. RMS equalization can distort short subblocks and is less reliable for low-frequency/nonstationary content; the minimum evidence rule is conservative and may defer or miss ambiguous short band energy. Confidence remains an uncalibrated baseline/observation heuristic.

## Frozen holdout results

Only after all five tests produced strict XPASS were their markers removed. Input parameters, waveform construction and assertions were not changed. The permanent 18-case test uses the original frequencies, levels and phases. Baseline replay disables only the two new evidence guards; its event lists match the actual pre-edit capture for all 18 cases.

| Frozen case | Before FP | After FP |
|---|---|---|
| holdout:200:0.01:0.3:0.0 | 0 | 0 |
| holdout:200:0.01:0.3:1.3 | 0 | 0 |
| holdout:200:0.03:0.7:0.0 | 0 | 0 |
| holdout:200:0.03:0.7:1.3 | 1 | 0 |
| holdout:200:0.01:0.99:0.0 | 1 | 0 |
| holdout:200:0.01:0.99:1.3 | 1 | 0 |
| holdout:1000:0.01:0.3:0.0 | 0 | 0 |
| holdout:1000:0.01:0.3:1.3 | 0 | 0 |
| holdout:1000:0.03:0.7:0.0 | 0 | 0 |
| holdout:1000:0.03:0.7:1.3 | 0 | 0 |
| holdout:1000:0.01:0.99:0.0 | 1 | 0 |
| holdout:1000:0.01:0.99:1.3 | 1 | 0 |
| holdout:4000:0.01:0.3:0.0 | 0 | 0 |
| holdout:4000:0.01:0.3:1.3 | 0 | 0 |
| holdout:4000:0.03:0.7:0.0 | 0 | 0 |
| holdout:4000:0.03:0.7:1.3 | 0 | 0 |
| holdout:4000:0.01:0.99:0.0 | 0 | 0 |
| holdout:4000:0.01:0.99:1.3 | 0 | 0 |

Each case is 48,000 samples / one second. Total: 864,000 samples, 18 seconds, zero expected onsets, FP **5→0**. Precision/recall is undefined for this negative-only set; zero false events here is not a field rate estimate. Loudness gates remain independent and the positive simultaneous onset+spike regression passes.

## Adversarial matrix

| Case | Before TP / FP / FN | After TP / FP / FN | After onset decision delays ms |
|---|---|---|---|
| noise_step | 0 / 1 / 0 | 0 / 1 / 0 | [] |
| noise_ramp | 0 / 1 / 0 | 0 / 0 / 0 | [] |
| multiband | 0 / 1 / 0 | 0 / 0 / 0 | [] |
| dip | 0 / 0 / 0 | 0 / 0 / 0 | [] |
| slow_ramp | 0 / 0 / 0 | 0 / 0 / 0 | [] |
| emergence:0.0001 | 1 / 0 / 0 | 1 / 0 / 0 | [5.000000000000004] |
| emergence:0.001 | 1 / 0 / 0 | 1 / 0 / 0 | [5.000000000000004] |
| emergence:0.002 | 0 / 0 / 0 | 0 / 0 / 0 | [] |
| short_impulse | 1 / 0 / 0 | 1 / 0 / 0 | [9.999999999999954] |
| new_broadband | 1 / 0 / 0 | 1 / 0 / 0 | [29.99999999999997] |

Adversarial waveform exposure: 720,000 samples / 15 seconds. After: 4 TP, 1 FP, 0 FN (onset precision 0.800, recall 1.0 on only FOUR labelled positives). Start matching tolerance ±40 ms; expected positive onset at frame 19200. The gain-only controls have unchanged negative labels. Full event lists with starts/confirmations, including loudness/offset, are in JSON. Added layout regression compares full frame-domain change revisions for two channels, planar/interleaved inputs and 480 vs 71/401/960-frame partitions. Existing suite covers opposite-direction band changes, cold/recovery/invalid/Nyquist handling, snapshots, early/full revisions, click completion, independent offset confirmation, rhythm timeout and refusal atomicity. These tests were not weakened.

**Unresolved:** stationary-noise gain step still emits a low-band onset during gain recovery. Broadband remains observable, but finite low-band power fluctuates around sensitivity/presence thresholds. Whether a stochastic band becoming measurable again represents a new observable component or continued stationary background cannot be decided from scalar per-band episodes alone. The abrupt-step control remains negatively labelled and a strict expected failure; the ramp now passes unchanged and was not relabelled. A more robust continuity/uncertainty rule for stochastic spectral patterns needs separate evidence; no new persistent pattern model or universal suppression was silently introduced. In-phase same-frequency additive audio is also observationally identical to gain modulation: source/component attribution cannot be guaranteed.

Final full suite: **198 passed, 1 xfailed**. All original five expected failures are now ordinary passing tests. The remaining new strict failure remains visible. Final Ruff lint and formatting, wheel/sdist and clean-wheel checks are recorded below.

## Performance and resource validation

Windows 11 build 26200, CPython 3.14.2, NumPy 2.5.5, Intel64 Family 6 Model 170 Stepping 4, 14 logical CPUs. Same 48 kHz / 480-frame, 20 ms Hann / 5 ms hop configuration as prior audit. Raw steady-tone work uses five audio seconds; separate tracemalloc runs measure peak Python-visible allocations, not total RSS.

| Workload | p50 / p95 / p99 / worst ms | Wall / CPU s | Audio s / wall s | Traced peak bytes |
|---|---|---|---|---|
| mono | 1.305 / 3.050 / 3.895 / 5.426 | 0.798 / 0.750 | 6.27 | 129607 |
| stereo | 2.846 / 4.612 / 5.463 / 8.138 | 1.529 / 1.469 | 3.27 | 195601 |

Prior audit mono p50/p95/p99 1.192/2.622/3.423 ms; stereo 2.284/4.106/5.638 ms. Additional FFT increases median work and scratch memory; uncontrolled host jitter makes tails variable. No hard-real-time guarantee or calibrated latency acceptance budget is inferred.

Stress rerun: 120 audio seconds / 5,760,000 frames / 12,000 ordinary calls, 27.691 wall s / 26.406 CPU s; one deliberately oversized rejection, no sequence gaps, max revision 2, max candidates 8/16, learned features plateau 36 and horizon rows 200. Sampled Windows working set 37,740,544→38,060,032 bytes; final samples plateau (see raw JSON); prior audit plateau 37,347,328 bytes. This is a short accelerated scenario, not a long-run RSS/heap proof.

Live configured five-second ambient smoke: 499 calls / 239520 frames, no overflow/refusal/sequence gap, no acoustic changes; p50/p95/p99/worst 3.041/4.583/5.303/9.716 ms. Receipt-based timing is not physical acquisition latency. AGC/gain unknown; no labelled live accuracy or controlled audible stimulus claim. Other OS/Python versions and remote CI were not run.

## Reproduction and files

```sh
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python benchmarks/onset_remediation.py
python benchmarks/acoustic.py --seconds 5 --channels 1
python benchmarks/acoustic.py --seconds 5 --channels 2
python benchmarks/acoustic_audit.py --seconds 120
python examples/live_microphone.py --acoustic --device 1 --duration 5
python -m build --no-isolation
```

Changed for this task: `_acoustic.py`, `_acoustic_features.py`, original five xfail markers in `tests/test_acoustic_audit.py`, new `tests/test_onset_adversarial.py`, new `benchmarks/onset_remediation.py`, this report/JSON evidence, and model-guide addendum. The pre-existing uncommitted repository work is preserved. Existing public contracts and default infrastructure-only behavior are unchanged. Scalar learned feature units/schemas and legacy/v2 snapshot behavior remain unchanged; added evidence and flags are transient only.

**NEEDS REMEDIATION** remains the gate because the abrupt noise-step control is not safely resolved. A narrow follow-up must decide/test continuity evidence for stochastic bands near the sensitivity floor, preserving true below-floor emergence and short genuine activity. No production or general field-accuracy claim.

Packaging verification completed successfully: wheel and sdist built; the wheel
installed offline into a separate environment without NumPy; isolated default
import/ingestion passed and DSP enablement raised the expected missing-extra error.
Ruff lint/format and git diff whitespace checks passed. No additional OS/Python
runtime or remote CI was exercised.

Final presence guard clarification: the whole broadband observation must remain
above the acoustic floor, while each retained band's attenuation must still agree
with common gain within existing hysteresis. Individual bands may temporarily
become unmeasurable; they are retained provisionally with an explicit ambiguity
reason, not labelled silent. The unchanged AGC-like ramp now passes. The abrupt
noise-step case remains unresolved; no variance-tolerance widening was retained.
