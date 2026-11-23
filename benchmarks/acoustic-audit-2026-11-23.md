# FAR DESIGN-003 independent audit — 2026-11-23

**Gate: NEEDS REMEDIATION.** The original P0 reproduction is corrected, but an independent amplitude-edge sweep still has five false-onset controls. No production readiness claim. Existing uncommitted implementation work was preserved; no commit/push was performed. Approved designs and numerical defaults were not modified.

## Findings and dispositions

| Priority / status | Exact reference | Reproduction and root cause | Correction / test |
|---|---|---|---|
| P0 OBSERVED → FIXED for original control | `src/far_audio_salience/_acoustic.py:788` | 48 kHz 1 kHz tone, amplitude .03→.7 at frame 19200, back at 33600. False low-band onset: 0.00029218 PCM² spectral edge, accelerated by .7 broadband peak despite weak band evidence. | Band confirmation uses sqrt(2*band power); broadband retains raw peak. No threshold change or suppression tied to a loudness event. `tests/test_acoustic_audit.py:55` fails before/passes after, at 80/480/960-frame partitions. Positive independent onset+spike test at line 85. |
| P1 OBSERVED → FIXED | `src/far_audio_salience/_rhythm.py:42` | New impulse arrives after frozen deadline before any idle call; old code appended it and extended deadline before checking expiry, preserving old sequence identity/IOI. | Expire before accepting a rising impulse; ongoing high merge path preserved. Regression `tests/test_acoustic_audit.py:14` fails before/passes after. |
| P1 OBSERVED → FIXED | `src/far_audio_salience/detector.py:177` | First establish acoustic_windows_per_call refusal, then inject materialization failure on next smaller call. Draft evaluation cleared the refusal episode even though ingestion never committed. | End episodes in commit helper after successful materialization, not draft evaluation. Regression `tests/test_acoustic_audit.py` (`test_materialization_failure_does_not_end_refusal_episode`) fails before/passes after; no model/watermark commit on exception. |
| P0 OPEN | `_acoustic.py` band activity/confirmation | Wider phase/gain sweep: 5/18 constant-carrier amplitude step cases still create false band onsets. Four are high-contrast edges, one is a 200 Hz .03→.7 step at phase 1.3. | Five strict expected-failure regressions retain unchanged ground truth. Do not label them passed. A tested amplitude-modulation/transient discrimination rule is still needed. |
| P2 OBSERVED / OPEN risk | selective candidate matching | 120-second stress with three levels generated eight candidate IDs, below bound 16, then stabilized. | No confirmed contract violation or arbitrary eviction introduced. Candidate fragmentation merits broader testing. |
| NOT VERIFIED | other platforms / controlled acoustic live labels | Only installed CPython 3.14.2 / Windows ran. Existing CI covers Ubuntu 3.11/3.14, but no remote run was dispatched. | No commit/push permitted; no cross-platform success claimed. Ambient microphone capture is not labelled live accuracy. |

P0 semantics: constant carrier phase/frequency and only amplitude changes provide the negative control; sustained energy in a genuinely new band provides the positive control. A Fourier amplitude edge has real transient energy in other bands, so power evidence alone does not identify its cause. The narrow remaining decision is an evidence rule that distinguishes such transients while preserving genuine short band impulses; arbitrary long confirmation or blanket suppression during loudness changes would violate the positive requirements. This audit leaves that rule open and blocks the final gate.

## Starting and final checks

Before runtime edits: `python -m pytest -q`: **158 passed**; Ruff check and format passed; `python -m build --no-isolation` built wheel + sdist. The first new regression run yielded **5 failed, 1 passed** (three partition cases reproduce P0; two additional defects). After targeted fixes and ringing control: **165 passed**. Final run including the five newly documented unresolved controls: **165 passed, 5 xfailed**. No original assertion/test was weakened or removed. The additional module skips when optional NumPy is absent.

Commands (using `.venv/Scripts/python.exe` locally):

```sh
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m build --no-isolation
python benchmarks/acoustic.py --seconds 5 --channels 1
python benchmarks/acoustic.py --seconds 5 --channels 2
python benchmarks/acoustic_audit.py --seconds 120
python examples/live_microphone.py --acoustic --device 1 --sample-rate 48000 --block-size 480 --duration 5
```

Local TEMP/TMP were directed to a plain workspace scratch directory to avoid the known sandbox-brace/setuptools issue. Live capture used sounddevice from the existing microphone virtual environment; no dependency files changed. Raw numeric evidence is retained in [acoustic-audit-2026-11-23.json](acoustic-audit-2026-11-23.json).

## Audit coverage

Inspected approved DESIGN-001/002/003, ADR-001 and contracts, ingestion, segmented history, FFT/ring, rhythm, eligibility, baseline administration, snapshots and benchmarks. Re-ran all original tests covering frame anchoring, planar/interleaved layout, independent clocks, cold start, observed-active recovery, gaps, unavailable Nyquist bands, expected hum loss, click completion, weak onset, opposite-direction band swap, grouping/full revisions, no double exposure on overlap, candidate block/reject/promotion, snapshot migration and malformed data, missing score components, zero retention and resource rollback. New ringing fixture is at frame 14400 (0.3 seconds): one initial onset, no offset. Original source-clock/layout and snapshot tests are regression evidence, not proof of every possible arrival schedule.

Output vocabulary and three-family union are unchanged. Logical revisions are full immutable records; completion is not counted as a second onset. Positive tests preserve early confirmation, anchored grouping, separate opposite directions and simultaneous onset+level change. No source classification, cross-microphone association, global interpretation, new public contracts or dependencies were added. FFT one-sided normalization and DC/Nyquist handling remain unchanged; existing numerical tests pass. Confidence remains uncalibrated. A full allocator/OOM fault campaign and exhaustive schedule fuzzing were not performed.

## Labelled pilot re-evaluation

Identical original fixtures, labels, tolerances and defaults: 25 seconds × 48,000 = **1,200,000 valid frames** across eight mono fixtures, repeated in both benchmark runs. No relabelling to improve scores. Match tolerance remains ±120 ms activity/level and ±1.5 seconds tempo; one-to-one matching counts duplicate first revisions as unmatched predictions (FP). The original labels are tiny development controls, not held-out accuracy.

| Kind | TP / FP / FN | Precision / recall | FP per valid hour |
|---|---|---|---|
| sound_onset | 10 / 0 / 0 | 1.000 / 1.000 | 0.0 |
| sound_offset | 2 / 0 / 0 | 1.000 / 1.000 | 0.0 |
| tempo_increase | 1 / 0 / 0 | 1.000 / 1.000 | 0.0 |
| tempo_decrease | 1 / 0 / 0 | 1.000 / 1.000 | 0.0 |
| loudness_spike | 2 / 0 / 0 | 1.000 / 1.000 | 0.0 |
| loudness_drop | 3 / 0 / 0 | 1.000 / 1.000 | 0.0 |

Original pilot onset FP decreased from one to zero. Broader amplitude controls are reported separately, not hidden in this favorable pilot: 18 seconds / 864,000 frames, **5 false onsets** and zero true onsets (precision 0 for those predictions; recall undefined). This is not a field false-alert estimate. Canonical revision counters in the stress runner detected no repeated/out-of-order revision. Duplicate-specific detection beyond one-to-one pilot matching is bounded to these fixtures.

| Pilot fixture / kind | Start error ms | Decision delay ms | Tempo availability / MAE Hz |
|---|---|---|---|
| sustained / sound_onset | 0.0 | 5.0 | unavailable / n/a |
| sustained / loudness_spike | -10.0 | 10.0 | unavailable / n/a |
| sustained / loudness_drop | 0.0 | 20.0 | unavailable / n/a |
| sustained / sound_offset | 15.0 | 55.0 | unavailable / n/a |
| click / sound_onset | 0.0 | 5.0 | unavailable / n/a |
| cold_hum_loss / loudness_drop | 0.0 | 20.0 | unavailable / n/a |
| cold_hum_loss / sound_offset | 5.0 | 45.0 | unavailable / n/a |
| level_steps / loudness_spike | -15.0 | 5.0 | unavailable / n/a |
| level_steps / loudness_drop | 0.0 | 20.0 | unavailable / n/a |
| tempo_increase / sound_onset | 0.0 | 5.0 | 0.662 / 0.0048 |
| tempo_increase / sound_onset | 0.0 | 5.0 | 0.662 / 0.0048 |
| tempo_increase / sound_onset | 0.0 | 5.0 | 0.662 / 0.0048 |
| tempo_increase / sound_onset | 0.0 | 5.0 | 0.662 / 0.0048 |
| tempo_increase / tempo_increase | 300.0 | 305.0 | 0.662 / 0.0048 |
| tempo_decrease / sound_onset | 0.0 | 5.0 | 0.734 / 0.0041 |
| tempo_decrease / sound_onset | 0.0 | 5.0 | 0.734 / 0.0041 |
| tempo_decrease / sound_onset | 0.0 | 5.0 | 0.734 / 0.0041 |
| tempo_decrease / sound_onset | 0.0 | 5.0 | 0.734 / 0.0041 |
| tempo_decrease / tempo_decrease | 600.0 | 605.0 | 0.734 / 0.0041 |

## Timing and memory

Windows 11 build 26200, CPython 3.14.2, NumPy 2.5.5, Intel64 Family 6 Model 170 Stepping 4, 14 logical CPUs. 48 kHz, 480 input frames, 20 ms FFT / 5 ms hop, zero comparison retention, five-second steady-tone workloads, same configuration as prior pilot. Timing includes wrapper+process. Memory is measured in a separate instrumented run, not subtracted from timing; Python-visible tracing does not equal total RSS.

| Workload | p50 / p95 / p99 / worst ms | CPU / wall s | Throughput × realtime | Traced peak bytes |
|---|---|---|---|---|
| mono | 1.192 / 2.622 / 3.422 / 6.141 | 0.703 / 0.727 | 6.88 | 121336 |
| stereo | 2.284 / 4.106 / 5.638 / 6.156 | 1.234 / 1.268 | 3.94 | 186209 |

Prior mono p50/p95/p99: 1.15/2.71/3.14 ms; stereo: 2.19/5.17/5.93 ms. Current values vary in both directions; this small uncontrolled benchmark does not establish a speedup or regression budget, nor hard realtime. Tracing overhead also varied significantly between runs.

## Bounded stress and live checks

Stress: 120 seconds synthetic mono audio / 5,760,000 frames / 12,000 ordinary process calls; wall 26.048 s, CPU 24.766 s. Three ten-second alternating levels, explicit unavailable observation at 60 audio seconds, and one deliberate oversized chunk refusal at 30 seconds. Rejected audio left model parameters unchanged. Zero sequence gaps; max revision 2; maximum candidates 8/16; learned feature count plateaued at 36; horizon history stayed at 200 rows. Sampled Windows working set rose from 37,072,896 to 37,347,328 bytes, then remained at that value from 40 through 110 audio seconds. These sampled working-set observations are not a multi-hour retained-heap proof or allocator cap. Stress did not saturate every candidate bound.

Live: configured five-second ambient smoke trial, Intel Smart Sound microphone array / MME, actual 48 kHz float32 mono / 480 frames. 499 calls, 239520 accepted frames; zero overflow/refusal/sequence gaps; zero acoustic changes. Process p50/p95/p99/worst 2.814/4.622/5.616/6.634 ms. Receipt-based process timing excludes physical acquisition latency. Gain/AGC unknown; no audio retained, no controlled audible stimulus or live labels. Controlled labelled live accuracy remains NOT VERIFIED.

## Changed files and residual gate

Audit-only changes: three runtime files (`_acoustic.py`, `_rhythm.py`, `detector.py`), new `tests/test_acoustic_audit.py`, new `benchmarks/acoustic_audit.py`, this dated report and JSON evidence, and targeted additions to `docs/acoustic-model.md`. Existing pilot report is preserved as historical evidence. No feature/threshold/configuration/public schema/dependency changes.

**NEEDS REMEDIATION** due to the five reproducible P0 amplitude-edge false onsets. Other residual risks: commensurate overlapping rhythms can appear regular; candidate fragmentation; calibration, gain/AGC and labelled live validation; only two minutes of accelerated stress; no additional OS/Python execution; no exhaustive memory-failure injection. Existing CI was inspected but not run remotely. A narrow evidence rule and positive tests for genuine short-band impulses are needed before clearing P0.

Final packaging checks also passed: wheel/sdist rebuild, offline clean-wheel
installation, isolated default ingestion with no NumPy, and the explicit missing-DSP
configuration error. `py -0p` exposed only Python 3.14; additional Python versions
were unavailable locally. Positive new-band control additionally preserves the hum
and keeps broadband mean-square energy constant (.2² = .19² + new-band amplitude²).
