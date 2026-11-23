"""Optional duplex capture and independent source-frame marker validation."""

import argparse
import json
import math
import time
from queue import Empty, Full, Queue

import numpy as np
from broader_validation import config, quantiles

from far_audio_salience import AudioSalienceDetector, ClockStamp, SourceSpec, new_id

RATE = 48000
BLOCK = 480
MARKER_SECONDS = (2, 17, 26)
MARKER_CORRELATION = 0.1  # Retain the prior correlation and slope gates.
SLOPE_ERROR = 0.005


def stimuli():
    rate = RATE
    rng = np.random.default_rng(66421)
    output = np.zeros(27 * rate, dtype=np.float32)
    markers = []
    t = np.arange(7200) / rate
    for second, (lo, hi) in zip(
        MARKER_SECONDS, ((350, 1700), (2400, 700), (550, 2900))
    ):
        phase = 2 * np.pi * (lo * t + (hi - lo) * t**2 / (2 * len(t) / rate))
        marker = (0.02 * np.sin(phase) * np.hanning(len(t))).astype(np.float32)
        first = second * rate
        output[first : first + len(marker)] = marker
        markers.append((first, marker))
    n = 3 * rate
    f = np.fft.rfftfreq(n, 1 / rate)
    spec = np.fft.rfft(rng.uniform(-1, 1, n))
    spec[(f < 500) | (f > 2000)] = 0
    sound = np.fft.irfft(spec, n=n)
    sound *= 0.015 / np.max(np.abs(sound))
    sound *= 0.8 + 0.2 * np.sin(2 * np.pi * 6 * np.arange(n) / rate)
    output[4 * rate : 7 * rate] = sound
    tap = 0.015 * np.sin(2 * np.pi * 1200 * np.arange(BLOCK) / rate) * np.hanning(BLOCK)
    rhythm = np.cumsum([12.0] + [0.3] * 5 + [0.15] * 7 + [0.3] * 5)
    for at in (9.0, 9.7, 10.4, *rhythm):
        first = round(at * rate)
        output[first : first + len(tap)] += tap
    first = np.arange(18 * rate, 23 * rate)
    amp = np.where((first >= 19.5 * rate) & (first < 21 * rate), 0.018, 0.004)
    output[first] = amp * np.sin(2 * np.pi * 213 * first / rate)
    regions = [
        ("band_noise", 4, 7),
        ("isolated_bursts", 9, 10.5),
        ("changing_rhythm", 12, float(rhythm[-1] + 0.2)),
        ("steady_level_changes", 18, 23),
    ]
    return output, markers, regions


def locate(signal, template, expected_frame):
    """Find an isolated normalized correlation peak near a clock prediction."""
    radius = RATE // 4
    start = max(0, expected_frame - radius)
    stop = min(len(signal), expected_frame + radius + len(template))
    segment = signal[start:stop].astype(np.float64)
    if len(segment) < len(template):
        return None
    template = template.astype(np.float64)
    n = 1 << (len(segment) + len(template) - 2).bit_length()
    spectrum = np.fft.rfft(template, n)
    corr = np.fft.irfft(np.fft.rfft(segment, n) * spectrum.conj(), n)
    squared = np.r_[0.0, np.cumsum(segment * segment)]
    energy = squared[len(template) :] - squared[: -len(template)]
    template_energy = np.dot(template, template)
    coefficients = np.abs(corr[: len(energy)]) / np.sqrt(
        np.maximum(energy * template_energy, 1e-30)
    )
    index = int(np.argmax(coefficients))
    peak = float(coefficients[index])
    # Mask only the offsets where this chirp correlates with itself at the
    # marker threshold. A fixed quarter-marker mask hid nearby duplicates.
    autocorrelation = np.fft.irfft(spectrum.real**2 + spectrum.imag**2, n)
    exclusion = int(
        np.flatnonzero(
            np.abs(autocorrelation[: len(template) // 4])
            >= MARKER_CORRELATION * template_energy
        )[-1]
    )
    alternate = max(
        float(np.max(coefficients[: max(0, index - exclusion)], initial=0)),
        float(np.max(coefficients[index + exclusion + 1 :], initial=0)),
    )
    return dict(
        expected_frame=expected_frame,
        matched_frame=start + index,
        correlation=peak,
        alternate_correlation=alternate,
        self_match_radius_frames=exclusion,
        unique=peak >= MARKER_CORRELATION and peak >= 1.5 * alternate,
    )


def align(audio, markers, anchors):
    """Use duplex ADC/DAC timestamps for search, then validate a frame map."""
    if not anchors:
        return dict(valid=False, reason="no_duplex_timestamp_evidence", markers=[])
    frames, adc, dac = np.asarray(anchors, dtype=float).T
    if (
        frames[0] != 0
        or len(audio) != len(frames) * BLOCK
        or np.any(np.diff(frames) != BLOCK)
    ):
        return dict(valid=False, reason="invalid_delivered_frame_evidence", markers=[])
    frame_steps = np.diff(frames)
    invalid_anchors = []
    for name, stamps in (("adc", adc), ("dac", dac)):
        if not np.all(np.isfinite(stamps)):
            invalid_anchors.append(dict(clock=name, error="nonfinite_timestamp"))
            continue
        steps = np.diff(stamps) * RATE
        non_increasing = steps <= 0
        spaced_incorrectly = np.abs(steps - frame_steps) > BLOCK
        bad = np.flatnonzero(non_increasing | spaced_incorrectly)
        if len(bad):
            i = int(bad[0])
            failed_checks = [
                check
                for check, failures in (
                    ("strict_increase", non_increasing),
                    ("block_spacing", spaced_incorrectly),
                )
                if failures[i]
            ]
            invalid_anchors.append(
                dict(
                    clock=name,
                    errors=len(bad),
                    first_interval_start_frame=int(frames[i]),
                    timestamp_start_seconds=float(stamps[i]),
                    timestamp_stop_seconds=float(stamps[i + 1]),
                    timestamp_step_frames=float(steps[i]),
                    frame_step=int(frame_steps[i]),
                    failed_checks=failed_checks,
                )
            )
    if invalid_anchors:
        return dict(
            valid=False,
            reason="invalid_duplex_timestamp_evidence",
            markers=[],
            invalid_anchors=invalid_anchors,
        )
    matches = []
    for output_frame, template in markers:
        i = int(np.searchsorted(frames, output_frame, side="right")) - 1
        if i < 0 or output_frame >= frames[-1] + BLOCK:
            return dict(valid=False, reason="marker_outside_capture", markers=matches)
        dac_time = dac[i] + (output_frame - frames[i]) / RATE
        j = int(np.clip(np.searchsorted(adc, dac_time) - 1, 0, len(frames) - 1))
        expected_frame = round(frames[j] + (dac_time - adc[j]) * RATE)
        match = locate(audio, template, expected_frame)
        if match is None:
            return dict(valid=False, reason="marker_outside_capture", markers=matches)
        matches.append(
            dict(output_frame=output_frame, dac_time=float(dac_time), **match)
        )
    if not all(m["unique"] for m in matches):
        return dict(valid=False, reason="missing_or_ambiguous_marker", markers=matches)
    a, middle, b = matches
    slope = (b["matched_frame"] - a["matched_frame"]) / (
        b["output_frame"] - a["output_frame"]
    )
    intercept = a["matched_frame"] - slope * a["output_frame"]
    residual = middle["matched_frame"] - (intercept + slope * middle["output_frame"])
    if abs(slope - 1) > SLOPE_ERROR or abs(residual) > BLOCK:
        return dict(
            valid=False,
            reason="inconsistent_marker_frame_map",
            markers=matches,
            candidate_slope=slope,
            middle_residual_frames=residual,
        )
    return dict(
        valid=True,
        markers=matches,
        slope=slope,
        intercept_frames=intercept,
        middle_residual_frames=residual,
    )


def _json_timestamp(stamp):
    return stamp if math.isfinite(stamp) else str(stamp)


def trial(sd, input_device, output_device, *, trace_callback_timing=False):
    output, markers, regions = stimuli()
    # One duplex callback provides ADC and DAC time in this stream's timebase.
    # Host receipt stamps and elapsed process timings stay out of alignment.
    pending = Queue(maxsize=64)
    total = 29 * RATE
    adc_evidence = []
    callback_timing = [] if trace_callback_timing else None
    received_clock = new_id()
    overflows = underflows = dropped = 0
    cursor = 0
    peak = float(np.max(np.abs(output)))
    if peak > 0.02:
        raise ValueError("Stimulus exceeds pilot digital output limit")

    def callback(indata, outdata, frames, timing, status):
        nonlocal cursor, overflows, underflows, dropped
        received = time.monotonic_ns()
        target = np.frombuffer(outdata, dtype=np.float32)
        target.fill(0)
        n = min(frames, max(0, len(output) - cursor))
        target[:n] = output[cursor : cursor + n]
        if cursor >= total:
            raise sd.CallbackStop
        overflows += bool(status.input_overflow or status.input_underflow)
        underflows += bool(status.output_underflow or status.output_overflow)
        adc, dac = timing.inputBufferAdcTime, timing.outputBufferDacTime
        try:
            pending.put_nowait((cursor, bytes(indata), adc, dac, received))
            queued = True
        except Full:
            dropped += 1
            queued = False
        if callback_timing is not None:
            callback_timing.append(
                dict(
                    first_frame=cursor,
                    frame_count=frames,
                    adc_seconds=_json_timestamp(adc),
                    dac_seconds=_json_timestamp(dac),
                    current_seconds=_json_timestamp(timing.currentTime),
                    receipt_monotonic_ns=received,
                    status=str(status) if status else "",
                    queued=queued,
                )
            )
        cursor += frames

    d = AudioSalienceDetector(config())
    s = SourceSpec.microphone(label="controlled real capture", sample_rate_hz=RATE)
    started = d.start_source(s)
    latency = []
    predictions = []
    audio = np.zeros(total + BLOCK, dtype=np.float32)
    actual_frames = last_sequence = gaps = refusals = 0
    prior_stop = 0
    discontinuity = False

    def consume(batch):
        nonlocal last_sequence, gaps, refusals
        refusals += len(batch.refusals)
        for event in batch.ordered:
            gaps += event.header.emission_sequence != last_sequence + 1
            last_sequence = event.header.emission_sequence
        predictions.extend(
            dict(
                kind=c.kind,
                start=c.start_frame,
                confirmed=c.confirmed_frame,
                confidence=c.confidence,
            )
            for result in batch.salience
            for c in result.changes
            if c.revision == 1
        )

    consume(started)
    try:
        with sd.RawStream(
            device=(input_device, output_device),
            samplerate=RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK,
            callback=callback,
        ) as stream:
            if stream.samplerate != RATE:
                raise ValueError(f"Unexpected duplex sample rate: {stream.samplerate}")
            deadline = time.monotonic() + total / RATE + 10
            while stream.active or not pending.empty():
                if time.monotonic() > deadline:
                    raise TimeoutError("Duplex capture did not finish")
                try:
                    first, raw, adc, dac, received = pending.get(timeout=0.1)
                except Empty:
                    continue
                samples = memoryview(raw).cast("f")
                stop = first + len(samples)
                actual_frames += len(samples)
                audio[first:stop] = samples
                adc_evidence.append((first, adc, dac))
                discontinuity |= first != prior_stop
                prior_stop = stop
                if discontinuity or overflows or underflows or dropped or refusals:
                    continue
                observation = s.chunk(
                    samples,
                    first_frame=first,
                    received_at=ClockStamp(received_clock, received),
                )
                tick = time.perf_counter_ns()
                batch = d.process(observation)
                latency.append((time.perf_counter_ns() - tick) / 1e6)
                consume(batch)
    finally:
        consume(d.close())
    complete = (
        actual_frames >= total
        and not discontinuity
        and not overflows
        and not underflows
        and not dropped
    )
    alignment = (
        align(audio[:prior_stop], markers, adc_evidence)
        if complete
        else dict(valid=False, reason="incomplete_or_discontinuous_capture")
    )
    valid = alignment["valid"] and not refusals and not gaps
    mapped = []
    if valid:
        for condition, first, last in regions:
            start = round(
                alignment["intercept_frames"] + first * RATE * alignment["slope"]
            )
            stop = round(
                alignment["intercept_frames"] + last * RATE * alignment["slope"]
            )
            if not 0 <= start < stop <= prior_stop:
                valid = False
                mapped.clear()
                break
            mapped.append(dict(condition=condition, start_frame=start, stop_frame=stop))
    result = dict(
        status="OBSERVED_MARKER_ALIGNED" if valid else "BLOCKED_LABEL_ALIGNMENT",
        labelled_accuracy="NOT MEASURED",
        rate=RATE,
        channels=1,
        block=BLOCK,
        input_device=dict(sd.query_devices(input_device)),
        output_device=dict(sd.query_devices(output_device)),
        host_api=sd.query_hostapis(sd.query_devices(input_device)["hostapi"])["name"],
        gain_agc="UNKNOWN",
        output_peak=peak,
        captured_frames=actual_frames,
        processed_calls=len(latency),
        process_ms=quantiles(latency),
        input_overflows_or_underflows=overflows,
        output_underflows_or_overflows=underflows,
        queue_drops=dropped,
        sequence_gaps=gaps,
        resource_refusals=refusals,
        stream_time_examples=adc_evidence[:1] + adc_evidence[-1:],
        measurement_time="ADC/DAC stream metadata used only for marker search. FAR receipt is monotonic_ns; FAR measurement stamps unset.",
        alignment=alignment,
        mapped_stimulus_regions=mapped,
        predictions=predictions,
        audio_retained=False,
        limitation="Marker mapping does not certify per-stimulus observability. Independent capture-frame annotation is required for acoustic accuracy.",
    )
    if callback_timing is not None:
        result.update(
            callback_timing=callback_timing,
            sounddevice_version=sd.__version__,
            portaudio_version=sd.get_portaudio_version()[1],
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=int, default=1)
    parser.add_argument("--output", type=int, default=3)
    parser.add_argument(
        "--callback-timing",
        action="store_true",
        help="Include raw callback timing and status metadata in the JSON report.",
    )
    args = parser.parse_args()
    try:
        import sounddevice as sd

        result = trial(
            sd, args.input, args.output, trace_callback_timing=args.callback_timing
        )
    except (ImportError, RuntimeError, ValueError, OSError) as error:
        result = dict(
            status="BLOCKED_LIVE",
            labelled_accuracy="NOT MEASURED",
            reason=f"{type(error).__name__}: {error}",
            audio_retained=False,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
