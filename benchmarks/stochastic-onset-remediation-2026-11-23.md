# Stochastic-noise onset remediation — 2026-11-23

**Gate: READY FOR BROADER VALIDATION in the tested scope.** The unchanged remaining strict expected failure now passes normally. No production readiness or general stochastic-noise accuracy is claimed. Approved documents, public contracts, feature/snapshot schemas, acoustic configuration defaults, dependency policy and input/state admission semantics were not changed. No commit/push.

## Exact reproduction and causal evidence

Commands used before runtime edits:

```sh
python -m pytest -q
python -m pytest "tests/test_onset_adversarial.py::test_same_pattern_gain_has_no_new_onset[noise_step]" --runxfail -q
```

Observed baseline: **198 passed, 1 xfailed**. Explicit execution with `--runxfail` produced **1 failed**: false `sound_onset` start 1.205 s, confirmation 1.230 s. The original input/seed/label/assertion remain unchanged: 48 kHz, RNG 785 uniform stationary white noise, two seconds, gain .2 initially, .01 on [28800,57600), then .2. The gain timeline, not new content, accounts for the changes.

The first erroneous decision is absence confirmation during attenuation, not onset confirmation at recovery. A two-sided exact common-gain fit was being used to decide whether a band could retain presence. Positive stochastic power residuals above the fit upper bound were treated as evidence against presence. They accumulated a fall interval even though the broadband observation stayed audible.

| Frame (exclusive evidence boundary) | Low-band PCM² | Raw band / gain-scaled reference | Before → after (old code) | After correction |
|---|---|---|---|---|
| 54000 | 4.70377668e-07 | 1.739231 | active True→True; fall=None | active=True; fall=None |
| 54240 | 5.96833593e-07 | 2.236588 | active True→True; fall=54000 | active=True; fall=None |
| 54720 | 7.23738383e-07 | 2.783323 | active True→True; fall=54000 | active=True; fall=None |
| 55200 | 9.29862505e-07 | 3.548899 | active True→True; fall=54000 | active=True; fall=None |
| 55680 | 1.01501769e-06 | 3.955534 | active True→True; fall=54000 | active=True; fall=None |
| 55920 | 1.51091907e-06 | 5.945166 | active True→False; fall=None | active=True; fall=None |
| 58080 | 1.80237095e-05 | 0.338879 | active False→False; fall=None | active=True; fall=57600 |
| 59040 | 0.00016410831 | 1.543801 | active False→True; fall=None | active=True; fall=None |

At frame 54240 (1.130 s), the fit residual is +2.237×, exceeding the old upper bound 1.8; fall starts at frame 54000. At frame 55920 (1.165 s), it is +5.945×, but the 40 ms fall confirmation wrongly sets the band inactive. `absent_power` becomes 1.510919e-6 PCM². At gain recovery, the unchanged process crosses the newly created absent reference and publishes the false onset at frame 59040, start frame 57840, revision 1. No true component disappeared or appeared. Grouping/publication merely reported the upstream state error.

Relevant pre-update low-band mean is 1.0715655e-4 PCM², variance 3.1943271e-12, admitted exposure .105 s; event confidence .35, expected-presence flag false. Selective learning leaves this noise estimate provisional, but its uncertainty does not justify interpreting **excess** energy as absence. The correction does not retune learning or variance. All observations are valid and source/frame coordinates continuous. Source-clock timestamps are unset in these synthetic adapters; seconds here are frame/rate conversions, not acquisition measurements.

Compatibility assessment: raw current and reference powers have the same algorithm, Hann normalization, band edges, units, sample rate and clock domain. They estimate different random windows; exact proportional shape agreement is not guaranteed even under a constant noise process. Gain transitions may add finite-window spectral residuals, but the false fall in this case begins long after attenuation settles. The envelope-equalized onset hypothesis is **unavailable/inactive** here because the background has not matured; trace `onset_power` is None. Thus neither FFT energy normalization nor a new gain-edge artifact is the direct trigger. The raw power comparison was dimensionally compatible but interpreted with the wrong direction.

Full pre/post low-band activity traces and selected broadband/envelope diagnostics, with complete emitted change snapshots/IDs, are retained in [JSON evidence](stochastic-onset-remediation-2026-11-23.json). No microphone audio is retained.

## Minimal correction and files

`src/far_audio_salience/_acoustic.py:847–859`, `_activity()`: keep the existing lower gain-adjusted hysteresis bound, remove the upper fit-bound rejection. While the whole observation remains audible during common attenuation, band energy at or above that lower bound cannot establish disappearance just because it is larger than the prediction. Substantially negative residuals, genuine silence, and unavailable observations retain their previous paths. `common_mode_presence_ambiguous` remains the existing provisional diagnostic.

Only that condition and explanatory comments changed at runtime (the saved prior wheel was compared after line-ending normalization). The common-gain estimator, raw/normalized features, presence floor, onset/offset confirmation curves, loudness gates, score/confidence mappings, learning, candidate commands, FFT, buffers and snapshot schemas were unchanged. No additional state, allocation, delay or lookahead was added. Alternative variance estimation/gain fitting changes were unnecessary once this directional error was identified.

Changed files in this task:
- `_acoustic.py`: one directional condition plus comments.
- `tests/test_onset_adversarial.py`: remove only the strict-xfail marker after unchanged assertion yielded strict XPASS.
- New `tests/test_stochastic_onset.py`: independent seeded controls, directional evidence tests and positive/gap/invalid controls.
- New `benchmarks/stochastic_onset.py`: reproducible labelled matrix and instrumented existing stress runner (requires dev + DSP extras).
- This report, JSON evidence, and a targeted model-guide addendum.

## Regression outcomes

| Frozen controls | Before | After |
|---|---|---|
| Original 18 tone gain controls (864,000 frames / 18 s) | 18 pass, 0 false onsets | 18 pass, 0 false onsets |
| Remaining unchanged seed-785 stationary-noise case (96,000 frames / 2 s) | 1 false onset, strict xfail | 0 false onsets, ordinary pass |
| Existing noise-ramp/multiband/short impulse/weak emergence controls | passing | passing |

After the correction the target emits only the independently supported loudness drop (start .600, confirmation .620 s) and spike (start 1.195, confirmation 1.215 s). Their raw power basis is unchanged. The false onset is absent. `tests/test_stochastic_onset.py:65` also exercises positive residual factors 2/6/10 for 100 ms and requires continuing activity with no fabricated offset.

Final full suite: **270 passed**, no expected failures. This includes all original rhythm timeout, resource-refusal atomicity, zero comparison retention, overlap/no-double-learning, candidate controls, legacy/v2 snapshots, cold start, Nyquist availability and immutable revision checks. Ruff lint and format checks pass.

## Expanded matrix and labels

Inputs and expected labels were declared before evaluating the extended matrix; none of the original 18 controls or seed-785 fixture was changed. Colored generation is offline fixture construction, not a causal detector operation: white uniform noise; pink-like spectral weighting 1/sqrt(max(f,40)); band-limited noise 500–6000 Hz. Digital gains are peak scale factors, not calibrated SPL.

| Scenario | Seeds / settings | Run records | Expected onset TP / observed FP / FN |
|---|---|---|---|
| Gain-only white/pink-like/band-limited noise | 13,79,2026; initial gains .1/.2/.3; low gains .01/.04; drop/recovery frames 19217/43271 or 33599/62401 | 54 | 0 / 0 / 0 |
| Gradual recovery, chunk/layout equivalence | seed 321, all three spectra; 480 frames vs planar 71/1001/480; same evidence coordinates | 6 (3 replay pairs) | 0 / 0 / 0 |
| New spectrally distinct band during/after recovery | seed 84; new 8 kHz tone amplitudes .003/.08; onset frames 57600/58080/67200 | 6 | 6 / 0 / 0 |
| Strong 10 ms band bursts during attenuation/recovery | seed 167; onset frames 43200/57600; same-ID completed revision, no offset | 2 | 2 / 0 / 0 |
| New band plus qualifying loudness spike | seed 91, onset frame 57600 | 1 | 1 / 0 / 0, plus one independent spike |
| Established noisy sound disappears | seed 85; 10 s valid warmup, verified established status, silence from frame 480000 | 1 | 0 / 0 / 0, plus one true offset |

The 70 signal-run records contain 7,128,000 input frames / 148.5 seconds, including 6 seconds of layout replay; unique fixture exposure is 142.5 seconds. Total onset counts: **9 TP / 0 FP / 0 FN**. Point precision/recall 1.0 on NINE positives is only a tiny deterministic regression result, not held-out real-world accuracy. Gain-only controls also require no false `sound_offset`. The additional known-gap/invalid test uses seed 548, explicit missing interval [28800,29280), invalid block [57600,58080), and already-active recovery; no invented onset/offset and expected health events. It covers 2 seconds nominal timeline, 1.98 seconds valid input. Three direct directional tests have no waveform exposure.

Positive timing: strong new bands confirm after 720 frames (15 ms); weak bands after 1680–1920 frames (35–40 ms); short bursts after 480 frames (10 ms). Matrix matching is ±1920 frames (40 ms) on onset starts, while individual test confirmation bounds are tighter and unchanged. Full per-case start errors, delays, frames, labels and events are in JSON. Offset timing is asserted separately.

Fixture preconditions were checked explicitly during construction: an early prototype used only 1.2 s of noise and incorrectly assumed an established baseline; the final disappearance control first asserts established status after a 10 s warmup. Another prototype put a new 8 kHz tone into an already-active default 2–12 kHz band and attenuated the entire background below the sensitivity floor, making recovery itself a legitimate onset. The final positive controls instead keep background audible and explicitly configure 500–6000 Hz background and 7000–9000 Hz new bands. Their one-onset/timing assertions are unchanged; this tests band-resolved presence, not source separation inside an already-active coarse band. These prototype corrections did not modify the original failing fixture or its negative label. A known-gap interval is used when asserting `channels_recovered`; unknown-duration loss alone does not supply a recovery boundary.

## Performance, stress and microphone

Windows 11 build 26200, CPython 3.14.2, NumPy 2.5.5, Intel64 Family 6 Model 170 Stepping 4, 14 logical CPUs. Same 48 kHz / 480-frame input, 20 ms FFT / 5 ms hop as the prior run. Synthetic timing includes chunk wrapping + processing; microphone timing isolates `process()`. Memory tracing runs are separate and intrusive.

| Synthetic 5 s workload | p50 / p95 / p99 / worst ms | Wall / CPU s | Audio s per wall s | Separate traced peak bytes |
|---|---|---|---|---|
| mono | 1.407 / 3.131 / 3.846 / 9.079 | 0.862 / 0.797 | 5.80 | 129671 |
| stereo | 2.743 / 4.729 / 5.782 / 8.428 | 1.519 / 1.391 | 3.29 | 195827 |

Prior mono p50/p95/p99 1.305/3.050/3.895 ms; stereo 2.847/4.612/5.463 ms; live 3.04/4.58/5.30 ms. Values vary with uncontrolled host scheduling/driver state; no hard-real-time guarantee, controlled equivalence or performance improvement is inferred from fewer comparisons.

Stress: 120 audio seconds / 5,760,000 frames / 12,000 ordinary calls, 28.690 wall s / 27.734 CPU s, one deliberate oversized-chunk refusal, zero sequence gaps. Maximum candidates 8/16, maximum revision 2; checked maxima {'impulses': 1, 'profiles': 1, 'groups': 1, 'open_changes': 1, 'history': 200}. Learned feature count plateaued at 36. Sampled process working set 36,823,040→37,167,104 bytes; warmup samples and maxima are in JSON. State assertions cover impulses/profiles, groups (≤6), open changes and prior configured bounds on every draft. This is not a multi-hour leak/RSS proof or OS allocator cap.

Live five-second configured ambient smoke: Intel Smart Sound microphone array, MME; actual 48 kHz native float32 mono / 480 frames; 499 calls / 239520 accepted frames, zero overflow/refusal/sequence gaps, zero acoustic changes. Process p50/p95/p99/worst 1.548/2.894/3.837/8.122 ms. Receipt-only process timing is not physical acquisition latency. AGC/gain and labelled acoustic live accuracy are NOT VERIFIED; no audio retained.

## Commands, packaging and gate

```sh
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python benchmarks/stochastic_onset.py
python benchmarks/stochastic_onset.py --stress
python benchmarks/acoustic.py --seconds 5 --channels 1
python benchmarks/acoustic.py --seconds 5 --channels 2
python examples/live_microphone.py --acoustic --device 1 --sample-rate 48000 --block-size 480 --duration 5
python -m build --no-isolation
```

Commands used the local `.venv/Scripts/python.exe`; capture imported sounddevice from the existing microphone environment without changing dependencies. Build TEMP/TMP uses an ordinary task scratch path to avoid sandbox-brace/setuptools issues. Clean-wheel default ingestion/import without NumPy and clear missing-DSP configuration failure are checked separately. Python launcher exposes only 3.14 locally; other Python versions/OS and remote CI remain NOT VERIFIED.

**READY FOR BROADER VALIDATION**, scoped to the corrected directional-presence defect and tested matrix. No confirmed regression remains in these tests. Confidence is not calibrated; noise/gain/AGC/room coverage is small, model maturity is selective, and results do not guarantee identifying arbitrary components within one already-active coarse band or observationally identical same-frequency additions. More diverse recordings and controlled live trials remain necessary. No new policy decision was needed to recognize that positive residual power cannot establish absence. No production-readiness claim.

Final packaging result: wheel and sdist built successfully; the wheel installed
with `pip install --no-index --no-deps --force-reinstall` in a separate environment
without NumPy. Isolated default import/ingestion and the explicit missing-DSP error
both passed. Built wheel runtime source bytes match the final checkout. Ruff lint,
format and whitespace checks passed. Approved DESIGN-001, DESIGN-002, approved
DESIGN-003 and ADR-001 hashes match the earlier baseline. The tracked older
unapproved DESIGN-003 draft is absent in the working-tree inventory; this task
performed no writes/deletions to design paths and left that deletion untouched.
