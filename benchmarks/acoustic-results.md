# DESIGN-003 experimental pilot — 2026-11-23

Measured on Windows 11 build 26200; CPython 3.14.2, NumPy 2.5.5, Intel64 Family 6 Model 170 Stepping 4, 14 logical CPUs. Exact CPU marketing name was not available (CIM query denied). Python 3.11/Linux/macOS were not run locally; CI includes 3.11/3.14 and DSP.

## Repeatable commands

```sh
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python benchmarks/acoustic.py --seconds 5 --channels 1
python benchmarks/acoustic.py --seconds 5 --channels 2
python examples/live_microphone.py --acoustic --device 1 --sample-rate 48000 --block-size 480 --duration 3
python -m build --no-isolation
```

Timings below use 48 kHz, 480-frame input blocks, 960-sample Hann FFT and 240-sample hop; finalized per-channel DSP is enabled, comparison retention is zero for synthetic work. Timing and tracemalloc are separate runs; tracing is very intrusive and its timings must not be used as realtime evidence. Five-second steady-tone workloads have no tempo; tempo availability there is zero by design.

| Synthetic workload | p50 / p95 / p99 / worst ms | Wall / CPU seconds | Audio seconds per wall second | Separate traced peak | Refusals |
|---|---|---|---|---|---|
| 1 channel(s), 5 s | 1.149 / 2.714 / 3.143 / 5.308 | 0.704 / 0.703 | 7.10 | 120,998 bytes | 0 |
| 2 channel(s), 5 s | 2.185 / 5.173 / 5.926 / 7.507 | 1.332 / 1.328 | 3.75 | 187,325 bytes | 0 |

Traced peaks cover Python-visible allocations during ingestion, not total process RSS, initial NumPy import or a measured long-run maximum. Logical DSP reservation is separately bounded; no OS-level RSS cap is claimed. Synthetic timing includes chunk wrapping and processing; the live timing isolates `process()`.

## Frame-labelled synthetic accuracy

Eight deliberately small deterministic fixtures provide 25 seconds of valid observations: sustained activity, click, cold hum disappearance, non-silent level steps, abrupt acceleration, abrupt deceleration, quiet and steady tone. Tempo fixtures disable level gates to isolate tempo evaluation. Matching is one-to-one on first revision only: ±120 ms start-frame tolerance for activity/level, ±1.5 s for tempo (multiple IOIs required). These are declared pilot matching tolerances, not approved acceptance budgets.

| Kind | TP / FP / FN | Precision | Recall | FP per valid observed hour |
|---|---|---|---|---|
| `sound_onset` | 10 / 1 / 0 | 0.909 | 1.000 | 144.0 |
| `sound_offset` | 2 / 0 / 0 | 1.000 | 1.000 | 0.0 |
| `tempo_increase` | 1 / 0 / 0 | 1.000 | 1.000 | 0.0 |
| `tempo_decrease` | 1 / 0 / 0 | 1.000 | 1.000 | 0.0 |
| `loudness_spike` | 2 / 0 / 0 | 1.000 | 1.000 | 0.0 |
| `loudness_drop` | 3 / 0 / 0 | 1.000 | 1.000 | 0.0 |

The extra onset occurs during a level step, from spectral transient energy. The 144/hour figure extrapolates ONE false event in 25 seconds, with substantial uncertainty; it is not a field-rate estimate. Quiet + steady controls (10 seconds) emitted no changes. Fixtures are also used during development: these results are regression accuracy, not held-out evaluation.

| Fixture / kind | Start error ms | Confirmation delay ms |
|---|---|---|
| sustained / sound_onset | 0.0 | 5.0 |
| sustained / loudness_spike | -10.0 | 10.0 |
| sustained / loudness_drop | 0.0 | 20.0 |
| sustained / sound_offset | 15.0 | 55.0 |
| click / sound_onset | 0.0 | 5.0 |
| cold_hum_loss / loudness_drop | 0.0 | 20.0 |
| cold_hum_loss / sound_offset | 5.0 | 45.0 |
| level_steps / loudness_spike | -15.0 | 5.0 |
| level_steps / loudness_drop | 0.0 | 20.0 |
| tempo_increase / sound_onset | 0.0 | 5.0 |
| tempo_increase / sound_onset | 0.0 | 5.0 |
| tempo_increase / sound_onset | 0.0 | 5.0 |
| tempo_increase / sound_onset | 0.0 | 5.0 |
| tempo_increase / tempo_increase | 300.0 | 305.0 |
| tempo_decrease / sound_onset | 0.0 | 5.0 |
| tempo_decrease / sound_onset | 0.0 | 5.0 |
| tempo_decrease / sound_onset | 0.0 | 5.0 |
| tempo_decrease / sound_onset | 0.0 | 5.0 |
| tempo_decrease / tempo_decrease | 600.0 | 605.0 |

Stable/gradual rhythm, alternating ambiguous IOIs, missing beats, merge/reacquisition, weak sustained activity, band swaps and interruption are additional assertions in tests. The pilot benchmark does not provide general rhythmic mixture accuracy or full systematic sensitivity.

## Live microphone trials

Two ambient, unlabelled smoke trials used device 1: Intel Smart Sound microphone array, MME, native float32, one channel, requested/actual 48 kHz and 480 frames. Default device rate was 44.1 kHz; PortAudio delivered 48 kHz. Device gain/AGC and hardware acquisition timing are unknown. No audio was retained. These trials validate integration, not controlled acoustic accuracy.

| Trial | Calls / accepted frames | p50 / p95 / p99 / worst ms | Overflow / refusal | Sequence gaps |
|---|---|---|---|---|
| First | 298 / 143040 | 2.325 / 3.644 / 4.067 / 11.744 | 0 / 0 | 0 |
| Second | 297 / 142560 | 2.608 / 4.384 / 5.469 / 6.679 | 0 / 0 | 0 |

Second trial emitted 592 continuous results and zero acoustic changes/health/lifecycle events. No labelled live precision/recall, false-event rate or acquisition latency is claimed. One call in the first trial exceeded the 10 ms block duration; long-run jitter and callback backpressure remain unmeasured.

## Parameters, stages and remaining validation

Offset confirmation sensitivity (20 / 40 / 80 ms) produced 35 / 55 / 95 ms observed decision delays on the sustained fixture. The [configuration guide](../docs/acoustic-model.md) lists every experimental default, units, validation constraints and trade-offs. Broader threshold/gain/room sensitivity is not calibrated.

Initial repository baseline: **124 tests passed**. Contract-only stage preserved 124. The first DSP numerical/activity stage had **12 passing** tests; selective learning/persistence added coverage; rhythm and integration subsequently reached **147**, then **158 total passing** tests. No historical test was removed. The final total includes 30 NumPy-dependent acoustic cases and four dependency-free rhythm tests; without NumPy the acoustic module is skipped and base infrastructure remains importable.

| Approved requirement | Code | Contract evidence |
|---|---|---|
| Typed additive changes, immutable full revisions, three families | `contracts.py`, `_acoustic.py` | default serialization, IDs/revision timeline, original output tests |
| Causal calibrated per-band features; source/layout/chunk independence | `_acoustic_features.py`, detector transaction | three-rate tone/noise calibration, planar/interleaved/partition, reorder, zero retention |
| Activity/presence/click/weak onset and independent offset; early grouping | `_acoustic.py` activity machine | sustained, click, cold hum, band swap, weak confirmation |
| Signed level transitions with absolute AND relative gates | `_acoustic.py` loudness machine | extreme non-silent steps, near-silence gate |
| Distinguishable impulses, robust IOIs, trends, timeout, merge/reacquisition | `_rhythm.py`, acoustic impulse extraction | stable/missed beat, abrupt both directions, gradual, ambiguous, merge/new ID, end-to-end tempo |
| Selective exposure, candidates, command guards, compatible persistence | `_acoustic.py`, `_acoustic_snapshot.py`, existing baseline store | block, reject/new proposal/approve, legacy/v2, disabled-model preservation, Nyquist |
| Available-evidence scoring/confidence; explicit missingness | `_acoustic.py` window assessment | partial coverage/horizons, temporal missingness without weight redistribution |
| Atomic resource refusal and owned buffers | detector draft integration, engine reservations | byte/window/revision refusal, rollback, invalid isolation, borrowed mutation, duplicate/overlap no double learning |

All six event kinds are implemented. Deliberately unimplemented/out of scope: pattern novelty, learned score weights, calibrated confidence/SPL, semantic classification, joint-channel acoustic inference, capture inside the package, and global interpretation. Known model limits: spectral transients can create extra band onsets; heuristic single-track rhythm cannot reject every commensurate overlapping pulse train; compatibility across rates is provisional, with bin/leakage changes; recovery confidence ramps conservatively. Controlled labelled live fixtures, long-duration memory/RSS/CPU profiling, broader noise/room/gain mixtures, confidence calibration, held-out accuracy, additional platforms and installed Python versions remain validation work.

## Final installation and tempo checks

Editable installation (`python -m pip install --no-deps --no-build-isolation -e .`), Ruff lint, Ruff format, `python -m pytest -q` (**158 passed**), wheel/sdist build (`python -m build --no-isolation`), wheel module-byte comparison, and sdist content checks succeeded. A separate clean virtual environment installed the wheel with `pip install --no-index --no-deps`; isolated import/default ingestion worked without NumPy and explicit DSP enablement raised the expected missing-extra error. No Python 3.11 run or remote CI result is claimed.

The first local no-isolation attempt lacked setuptools. After installing the existing declared build backend, Windows sandbox temporary paths containing braces prevented editable building; redirecting TEMP/TMP to an ordinary task scratch directory resolved it. Final builds and editable installation succeeded. These were environment issues; no build-backend requirement was changed.

Tempo error below compares available estimates against the generator's instantaneous known rate, including causal estimator lag during the step. Availability is the fraction of full analysis windows with a nonmissing rate, including acquisition/settling. It is not an error measurement during unavailable periods.

| Fixture | Tempo availability | Mean absolute error (events/s) |
|---|---|---|
| tempo_increase | 0.662 | 0.0048 |
| tempo_decrease | 0.734 | 0.0041 |
