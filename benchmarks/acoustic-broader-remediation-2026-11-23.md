# FAR DESIGN-003 broader validation remediation

**Gate: NEEDS REMEDIATION.** All onset, offset, and tempo errors in the 213 frozen
cases cleared. The separate 21-case run still exposes two missed isolated
impulses over hum and one tempo event in smooth brown noise. The original
brown-noise loudness mismatch also remains. Independent live accuracy is
**NOT MEASURED**.

## Causal evidence and changes

| Issue | Before | Remediation and verification |
| --- | --- | --- |
| Continuous noise | In the frozen 16 kHz seed 43 fixture, the low band fell inactive at frame 1200 while broadband power was .05945. The .0050 reference represented only .02 s of exposure. Subsequent variation crossed the trough-derived reference and produced onsets at 6080 and 16080. | [The activity decision](../src/far_audio_salience/_acoustic.py) retains the prior power reference through relative troughs and compares a band's current share to a causal average. Sustained contrast is required in an occupied band when the hop detector has no independent impulse evidence. Both frozen false onsets clear. The 12 new continuous-noise cases have zero onset and offset FPs. |
| Gradual tempo | At 48 kHz the acceleration opened at frame 410400, rearmed at 437280 while the IOIs still shrank, then opened a second ID at 470160. The shortest fitted horizon could also qualify a conflicting mixture. | [Rhythm](../src/far_audio_salience/_rhythm.py) requires plateau evidence before rearm and checks slope across accepted horizons. It uses the supported extreme of an open transition for an observed reversal and completes the preceding opposite-direction event. The 12 duplicate tempo IDs clear across the two rhythm sets. The nine ambiguous-mixture FPs clear under the original policy. |
| New activity over colored background | At 16 kHz, seed 16031, mid-band leakage around 2.58e-6 initialized the band as present. Added 1337 Hz power reached .00182 at frame 9920. The old active branch checked only for falling energy. | [The activity decision](../src/far_audio_salience/_acoustic.py) can open a relative contribution while the band background remains present. Smoothed contrast and the existing sustained confirmation govern the corresponding offset. All eight labelled pairs now match at both amplitudes and rates. Gaps still interrupt the episode and reset transient evidence. |
| Live timing | The earlier harness replayed identical full-band noise markers through a separate playback stream, searched wide time ranges, and fit two unverified peaks. It had no independent residual check. | [The duplex adapter](controlled_audio_validation.py) searches for three different chirps using ADC/DAC callback metadata. Two captured peaks fit the frame map and the middle peak validates it. The synthetic harness test rejects invalid ADC timestamps and a displaced middle marker. The hardware trials did not pass all gates. |

No acoustic configuration values, public schemas, feature or snapshot schema
IDs, dependencies, approved specs, or frozen labels changed. The new activity
and rhythm evidence lives in the existing transactional per-channel draft
state. The detector still emits early revisions under the same ID once
confirmation completes. A relative contribution over an occupied band can
take longer to confirm than a rise from an empty band.

## Measured accuracy

Each cell is TP / FP / FN. The exact kind/channel match policy and timing
tolerances from the earlier report apply. The original labels, manifests,
sample hashes and scoring masks matched byte or value for value across the two
runs. The new cases were scored separately at 32 kHz with two seeds and with
one 48 kHz overlap control. Counts are
procedural fixture results. The nominal held-out split repeats some exact
waveforms, and the extra rhythm set uses the same procedural family.

| Kind | Frozen 213 before | Frozen 213 after | New 21 before | New 21 after |
| --- | --- | --- | --- | --- |
| sound_onset | 369 / 10 / 8 | 377 / 0 / 0 | 7 / 9 / 3 | 8 / 0 / 2 |
| sound_offset | 57 / 2 / 8 | 65 / 0 / 0 | 7 / 3 / 1 | 8 / 0 / 0 |
| tempo_increase | 24 / 21 / 0 | 24 / 0 / 0 | 0 / 2 / 0 | 0 / 1 / 0 |
| tempo_decrease | 24 / 1 / 0 | 24 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| loudness_spike | 44 / 1 / 1 | 44 / 1 / 1 | 1 / 0 / 0 | 1 / 0 / 0 |
| loudness_drop | 54 / 0 / 0 | 54 / 0 / 0 | 1 / 0 / 0 | 1 / 0 / 0 |

The [gzip-compressed JSON evidence](acoustic-broader-remediation-2026-11-23.json.gz)
contains counts by kind and condition, samples' hashes, first predictions,
band membership and boundaries for every emitted revision, confirmation and
first-delivery frame delays, IDs and phases, and bounded representative private
state traces. It records zero harness ordering, phase or revision violations
in both frozen runs. Confidence remains uncalibrated. This experiment supplies
no real-world false-alarm rate.

For the eight recovered colored-background pairs, source-frame confirmation
delay was p50/p95/max 30/65/65 ms for onset and 85/125/125 ms for offset.
In the initial matrix, median tempo-decrease confirmation stayed at 804 ms.
The slowest of the 18 matched cases rose from 1199 to 1559 ms after the
multi-scale consistency change. These are source-frame delays. Per-case
delivery delays, which add the submitting chunk boundary, are in the JSON.

In the overlapping-band control, a louder 4500 Hz tone enters at .82 s.
A candidate that relied only on relative share closed the mid-band event
at .85 s, while its 1337 Hz component continued until 1.10 s. Requiring
each contribution's own band power to fall fixes that regression. The final
run matches both onsets, both offsets and the independent level changes.

The remaining mismatch in the frozen loudness schedule needs separate
adjudication. At frame 28080, before any planned gain step, the raw signal rose
by .20149 power and 13.23 dB within the configured horizon. That rise meets both
spike gates. The frozen negative at that frame is uncertain. An observed drop
then met its independent gates. At frame 53040 the recovery again met both
spike gates (+.01606 power, +15.85 dB), but the spike latch remained set because
no stable rearm interval had elapsed. The evidence preserves the original
FP/FN pair and the latch trace.

The new impulse case injects a single .7 sample over a continuous low-frequency
noise/hum mixture. In the seed 62013 trace the .697 measured peak raised
broadband frame power from about .0018 to .0024. The hop gate did not qualify
it. The high band had contrast for three analysis hops, but its power-based
confirmation ran longer than the evidence. Both baseline and candidate miss
this impulse at both seeds. Sustained narrowband, broadband, and near-floor
new signals all matched in the new controls. In seed 62013 smooth brown noise,
the energy-based tracker still reported a tempo increase at confirmation frame
56640. The fixture contains no prescribed pulse train. These cases limit the
readiness claim and remain visible in the regression data.

## Live observations

Windows 11 build 26200, CPython 3.14.2, NumPy 2.5.5 and sounddevice 0.5.6.
Intel Smart Sound microphone array and Realtek speakers. All four trials captured
1,392,000 delivered frames in 29 seconds at 48 kHz, mono float32, 480 frames per
callback, with zero reported input or output buffer faults, queue drops,
refusals and sequence gaps. Gain, OS volume, AGC and echo processing are unknown.
The capture stayed in memory and was discarded.

| Host API | p50 / p95 / p99 / worst process time (ms) | Alignment evidence |
| --- | --- | --- |
| MME | 2.822 / 5.138 / 7.316 / 50.550 | ADC started at 0 and ended at .01 s across the 29 s capture. DAC advanced about 29 s. Invalid timestamp evidence. |
| WASAPI | 1.344 / 2.384 / 3.423 / 10.443 | Correlations .0495 / .0202 / .1377. All failed the combined threshold and peak isolation check. No regions mapped. |
| WASAPI, pre-final runtime | 1.660 / 2.695 / 4.065 / 17.080 | Callback clock spacing failed. No marker search or region mapping. |
| WASAPI, final runtime | 1.280 / 2.142 / 3.133 / 8.838 | ADC time moved backward by the equivalent of about 472 frames while capture advanced 480. No marker search or region mapping. |

These timings cover the synchronous detector call on a shared host. PortAudio callback times are
driver-provided stream coordinates. Host receipt stamps and elapsed process time
use separate clocks. No calibrated relationship to physical sound arrival was
established. Driver buffering and physical acquisition-to-decision latency
remain unmeasured. The earlier and current trials supply no independently
annotated input events for precision or recall.

The [operator protocol](controlled-real-audio-protocol.md) records how to select
a matched host API pair, check marker evidence against captured frames, and
independently annotate observed events before scoring. The known equipment
still needs a verified audible path or a synchronized reference.

## Commands and compatibility

The baseline runtime came from a copy of the starting working tree at
`ad2ea735cf72a5b2f62915a9640b989bd2374baf`. For the before rerun, `PYTHONPATH`
pointed to that copy. Its two runtime modules were captured before any edits.
The same observation-only harness recorded revisions for both versions.
Commands below abbreviate the installed Windows venv Python as `python`.

```sh
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python benchmarks/broader_validation.py --evaluate
python benchmarks/broader_validation.py --independent-rhythm
python benchmarks/broader_lifecycle.py
python benchmarks/broader_noise.py
python benchmarks/broader_remediation_cases.py
python benchmarks/broader_error_traces.py
python benchmarks/broader_validation.py --quality
python benchmarks/controlled_audio_validation.py --input 1 --output 3
python benchmarks/controlled_audio_validation.py --input 9 --output 8
python -m pip install -e . --no-deps --no-build-isolation
python -m build --no-isolation --outdir <scratch>/dist
<clean-python> -m pip install --no-index --no-deps --force-reinstall <wheel>
```

273 baseline tests passed. 285 final tests passed. Ruff lint and format passed.
The source-local timestamp probe passed at 8, 16, 44.1 and 48 kHz. The probe
also interleaved unaligned source clocks. The full suite exercised partition
and layout invariance, gaps, invalid input, rate changes, candidate/persistence
semantics, transactional refusals and early immutable revisions. The editable
install, wheel and sdist build passed using the installed backend and writable
build temp paths. The offline clean wheel probe imported the package, ingested
base audio without NumPy, and verified the informative missing-`[dsp]` error.
Other Python versions, Linux, macOS and remote CI were outside the local run.

Changes are confined to the acoustic and rhythm modules, the validation and
duplex harnesses, the new procedural regressions, the model guide, the operator
protocol, this report and its evidence, and the sdist manifest entry for the
evidence. No commits or pushes.


## Resource and performance measurements

`python benchmarks/broader_validation.py --stress 1800` completed 180,000
calls over 1,800 seconds of generated 48 kHz audio in 480-frame blocks.
Source time includes zeros, hum, noise, impulses and mixed tones. The
primary channel was supplied throughout. A second correlated channel was
supplied for one second per minute. It was omitted in other blocks.
There were 0 resource refusals, 29 periodic snapshots and
5 end/restore/new-session cycles. The runner verified run-scoped
event ordering, bounded recent-revision identity, DSP history limits,
initially empty restored transient state, and no silent catalog eviction.
The resource suite separately exercises intentional refusals, rollback and
episode aggregation.

Wall time 1100.63 s and process CPU 1032.50 s.
Whole-driver throughput was 1.635× source time,
including snapshots, GC and restarts. Max retained scalar history 200,
impulses 64, corresponding profiles 64, group entries
2, retained change records 2, and sampled baseline catalog records
82. Catalog records include established and candidate records.

Windows working set sampled every 30 source seconds, before/after an explicit
GC. After-GC first/last/median/max were
39.86/42.16/41.95/42.18 MiB.
After a 300 source-second warmup the working set ranged from 41.44 to
42.18 MiB with median 41.95 MiB. The last sample
was at source second 1770. This is one bounded synthetic run on Windows.
Exact samples are in the evidence.

Baseline and final `python benchmarks/broader_validation.py --perf` ran
sequentially after the soak. Host CPU is Intel64 Family 6 Model 170 Stepping 4
with 14 logical CPUs. Each row uses five seconds of synthetic steady hum.
Multichannel rows use correlated copies. Percentiles time only `process()`.
Wall/CPU and audio per wall second also include fixture construction
and batch consumption. The host was shared and each row is a single short run.

| Runtime | Rate Hz | Frames/block | Channels | p50/p95/p99/worst ms | Audio/wall × | Wall s | CPU s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| before | 48000 | 480 | 1 | 1.069/2.130/3.532/4.325 | 8.311 | 0.602 | 0.594 |
| before | 48000 | 480 | 2 | 2.022/5.574/6.372/9.437 | 4.178 | 1.197 | 1.188 |
| before | 48000 | 480 | 8 | 8.433/15.258/21.046/88.034 | 1.049 | 4.767 | 4.406 |
| before | 48000 | 120 | 1 | 0.569/1.295/2.437/59.568 | 3.633 | 1.376 | 1.219 |
| before | 16000 | 160 | 1 | 0.971/2.264/6.829/51.971 | 7.604 | 0.658 | 0.531 |
| before | 44100 | 441 | 2 | 2.115/4.748/6.914/14.108 | 4.056 | 1.233 | 1.109 |
| final | 48000 | 480 | 1 | 1.065/2.560/3.519/7.868 | 8.024 | 0.623 | 0.609 |
| final | 48000 | 480 | 2 | 2.080/5.035/6.463/6.855 | 4.160 | 1.202 | 1.156 |
| final | 48000 | 480 | 8 | 8.166/16.826/19.384/21.635 | 1.117 | 4.478 | 4.375 |
| final | 48000 | 120 | 1 | 0.533/1.225/2.325/6.247 | 4.052 | 1.234 | 1.234 |
| final | 16000 | 160 | 1 | 1.010/1.427/3.070/4.373 | 9.391 | 0.532 | 0.516 |
| final | 44100 | 441 | 2 | 2.054/5.243/6.578/25.651 | 4.080 | 1.226 | 1.141 |

The five-second traces and soak do not establish sustained microphone cadence
for every supported rate or channel count.
