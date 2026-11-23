# Controlled real-audio validation protocol

Status for the 2026-11-23 remediation runs: duplex trials captured real
microphone frames. MME supplied unusable ADC timestamps. WASAPI supplied usable
search timestamps but failed the independent correlation and uniqueness checks.
**Labelled live accuracy: NOT MEASURED.** The adapter discarded all recordings.

## Reproduce the adapter

Use a local environment with `pip install -e ".[dev,dsp,mic]"`. Tests remain
hardware-independent. From the repository root:

```sh
python examples/live_microphone.py --list-devices
python benchmarks/controlled_audio_validation.py --input 9 --output 8
```

Choose input and output devices on the same PortAudio host API. The indices above
identified the WASAPI pair on the measured machine. The duplex adapter uses 48 kHz,
native float32, one input/output channel, and 480-frame callbacks. It plays low
digital peaks (at most .02) with three distinct swept-sine markers in quiet slots,
modulated band noise, isolated burst surrogates, changing repetition rates and
steady-tone level changes. These are **synthetic acoustic playback surrogates**,
not verified human speech, a physical fan, or physical knocks. OS volume and AGC
are unknown unless independently recorded. Do not increase levels automatically
or assume the output is audible at the microphone. Capture is processed in memory;
there is no recording/upload side effect.

The callback records a delivered frame index, an ADC time for the input buffer,
and a DAC time for the output buffer. These PortAudio stream timestamps predict
where to search for each marker in captured frames. The prediction is driver
metadata and must pass monotonicity and block-spacing checks. Independently, each
chirp must produce a unique correlation peak within ±250 ms of its predicted
position, with absolute normalized correlation ≥.1 and at least 1.5× the strongest
separate arrival. The first and last peaks fit a source-frame map whose slope
must be within .005 of one. The middle peak independently checks the fit within
480 frames. Any reported buffer fault, queue drop, missing block or unusable map
invalidates automatic region mapping. These are pilot measurement criteria.
Passing them locates the playback schedule and still requires independent review
of each stimulus in observed input frames.

The callback's `monotonic_ns` receipt stamp and the consumer's `perf_counter_ns`
duration have separate roles. FAR receives the former as receipt time, with its
measurement stamp unset. The latter measures `process()` time only. The experiment
does not calibrate an ADC-to-host clock transform, separately identify driver
buffering from acoustic propagation, or measure physical acquisition-to-decision
latency.

## Operator-labelled trial procedure

1. Use a quiet, nonconversational environment. Independently verify device identity,
   audibility, microphone gain, OS volume, echo cancellation and AGC where accessible.
   Record unknown settings as UNKNOWN; do not infer them from FAR output.
2. Run marker calibration first. Confirm marker correspondence on captured-frame
   evidence and measure uncertainty. Receipt timestamps alone are not acquisition
   timestamps and must not be used to claim physical latency or shared clocks.
3. Execute separate short trials: quiet→modulated acoustic activity→cessation;
   isolated clicks/knocks; repeating taps with stable, accelerating and decelerating
   sections; steady ambient/fan-like sound; and level changes with unchanged content.
   Include unchanged negative controls, gaps/invalid observations and repeated runs.
4. Annotate **observed input source frames**, with uncertainty, using an independent
   waveform/spectrogram review or external synchronized instrumentation. Do not let
   FAR's score/events determine labels. Below-floor/overlapped/unobservable events
   must be marked ambiguous/unscored before evaluating. Extreme loudness labels
   require independent digital pre/post power evidence meeting both configured gates;
   a planned speaker-volume change alone does not establish those input conditions.
5. Use `real-audio-label-template.json` as the manifest. `annotations` entries require
   the fields listed in `annotation_fields`, one of six approved local change kinds,
   channel identity, frame coordinates and explicit review status. Missing/invalid
   intervals are excluded from valid-audio exposure, never filled with zeros.
6. If an operator elects to retain audio for review, retain only the shortest needed
   nonconversational clips under local privacy policy. That explicit recording step
   is outside this default no-retention adapter. Do not upload automatically.
7. Freeze label/clip digests before examining detector results. Evaluate with the
   same one-to-one frame-domain matching policy as the synthetic harness. Report
   denominators, per-condition failures and unscored outputs; do not combine synthetic
   and real exposures into a single apparent field accuracy estimate.

No reviewed labelled recordings or operator annotations were supplied in this task.
A second independent session/room/device is required before making generalization
claims. Cross-platform/device timing remains NOT VERIFIED.
