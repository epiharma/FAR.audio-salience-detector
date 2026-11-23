"""Causal FFT plans and owned bounded ring state; no import-time DSP dependency."""

import hashlib
import importlib
import json
import math
from dataclasses import dataclass


def numpy_module():
    try:
        return importlib.import_module("numpy")
    except ImportError as error:
        raise ValueError(
            "Acoustic DSP requires NumPy: pip install 'far-audio-salience[dsp]'"
        ) from error


def schema_id(config, band):
    value = (
        "far-power-v1",
        "normalized-PCM-power",
        config.frame_seconds,
        config.hop_seconds,
        "hann",
        band,
    )
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


@dataclass
class SpectralPlan:
    np: object
    size: int
    hop: int
    window: object
    weights: object
    masks: dict
    schemas: dict

    @classmethod
    def create(cls, np, config, rate):
        size = max(3, math.ceil(config.frame_seconds * rate))
        hop = max(1, math.ceil(config.hop_seconds * rate))
        window = np.hanning(size)
        window.flags.writeable = False
        weights = np.full(size // 2 + 1, 2.0 / (size * float(np.dot(window, window))))
        weights[0] /= 2
        if size % 2 == 0:
            weights[-1] /= 2
        weights.flags.writeable = False
        frequencies = np.fft.rfftfreq(size, 1 / rate)
        masks, schemas = {}, {"broadband": schema_id(config, "broadband")}
        for band in config.bands:
            schemas[band.band_id] = schema_id(
                config, (band.band_id, band.low_hz, band.high_hz)
            )
            if band.high_hz > rate / 2:
                masks[band.band_id] = (None, "band_above_nyquist")
            else:
                mask = (frequencies >= band.low_hz) & (frequencies < band.high_hz)
                if band.high_hz == rate / 2:
                    mask[-1] = True
                mask.flags.writeable = False
                masks[band.band_id] = (
                    (mask, None)
                    if mask.any()
                    else (None, "insufficient_frequency_resolution")
                )
        return cls(np, size, hop, window, weights, masks, schemas)

    def level_invariant_bands(self, frame, floor, gain_power_ratio):
        """Counterfactual band evidence after removing the local gain envelope.

        Only observed contiguous samples participate. Quiet subblocks make the
        hypothesis unavailable: gaining an inaudible component can be an onset.
        This does not replace the calibrated raw feature measurements.
        """
        values = frame.astype(self.np.float64)
        total = float(self.np.mean(values * values))
        levels = []
        for first in range(0, self.size, self.hop):
            block = values[first : first + self.hop]
            level = float(self.np.mean(block * block))
            if level < floor:
                return None
            levels.append((first, level))
        if max(level for _, level in levels) > gain_power_ratio * min(
            level for _, level in levels
        ):
            corrected = self._undo_coherent_gain_edge(values, floor, gain_power_ratio)
            if corrected is not None:
                return self.extract(corrected)[2]
        for first, level in levels:
            values[first : first + self.hop] *= math.sqrt(total / level)
        return self.extract(values)[2]

    def _undo_coherent_gain_edge(self, values, floor, gain_power_ratio):
        """Test a gain step on a phase-continuous, locally periodic waveform."""
        np = self.np
        center = values[1:-1]
        audible = center * center >= floor
        if not audible.any():
            return None
        neighbors = values[:-2] + values[2:]
        # A sinusoidal recurrence is independent of amplitude except at a
        # discontinuity. Its largest residual brackets the inferred edge.
        coefficient = float(np.median(neighbors[audible] / center[audible]))
        if abs(coefficient) > 2:
            return None
        residual = neighbors - coefficient * center
        worst = int(np.argmax(np.abs(residual))) + 1
        for edge in (worst, worst + 1):
            if not 2 <= edge <= len(values) - 2:
                continue
            predicted = coefficient * values[edge - 1] - values[edge - 2]
            backcast = coefficient * values[edge] - values[edge + 1]
            denominator = predicted**2 + values[edge - 1] ** 2
            if not denominator:
                continue
            gain = (
                predicted * values[edge] + values[edge - 1] * backcast
            ) / denominator
            if gain <= 0 or 1 / gain_power_ratio <= gain**2 <= gain_power_ratio:
                continue
            if min(np.mean(values[:edge] ** 2), np.mean(values[edge:] ** 2)) < floor:
                continue
            corrected = values.copy()
            if gain > 1:
                corrected[:edge] *= gain
            else:
                corrected[edge:] /= gain
            predicted_center = coefficient * corrected[1:-1]
            error = corrected[:-2] + corrected[2:] - predicted_center
            # Competing band evidence may be weak but real. Require continuity
            # to float32 precision rather than allowing an acoustic residual.
            rounding = np.finfo(np.float32).eps * (
                np.abs(corrected[:-2])
                + np.abs(corrected[2:])
                + np.abs(predicted_center)
            )
            if np.all(np.abs(error) <= rounding):
                return corrected
        return None

    def isolated_peak_offset(self, frame, strong_peak, prominence_ratio):
        """Find a sharp peak with a full hop of quieter audio on either side."""
        hop = self.hop
        if len(frame) < 3 * hop:
            return None
        # Examine each candidate hop once, after its right flank is observed.
        first = len(frame) - 2 * hop
        offset = first + int(self.np.argmax(self.np.abs(frame[first : first + hop])))
        peak = abs(float(frame[offset]))
        if peak < strong_peak:
            return None
        flank = max(
            float(self.np.max(self.np.abs(frame[offset - hop : offset]))),
            float(self.np.max(self.np.abs(frame[offset + 1 : offset + hop + 1]))),
        )
        return offset if peak * peak > prominence_ratio * flank * flank else None

    def extract(self, frame):
        np = self.np
        values = frame.astype(np.float64)
        power = float(np.dot(values, values) / self.size)
        peak = float(np.max(np.abs(values)))
        spectrum = np.fft.rfft(values * self.window)
        powers = (spectrum.real**2 + spectrum.imag**2) * self.weights
        bands = {"broadband": (power, None)}
        bands.update(
            {
                name: (None, reason)
                if mask is None
                else (float(np.sum(powers[mask])), None)
                for name, (mask, reason) in self.masks.items()
            }
        )
        recent = values[-self.hop :]
        return (
            power,
            peak,
            bands,
            float(np.dot(recent, recent) / len(recent)),
            int(np.argmax(np.abs(recent))),
        )


@dataclass
class Ring:
    data: object
    filled: int = 0
    write: int = 0

    def append(self, values):
        size = len(self.data)
        if len(values) >= size:
            self.data[:] = values[-size:]
            self.write = 0
            self.filled = size
            return
        first = min(len(values), size - self.write)
        self.data[self.write : self.write + first] = values[:first]
        self.data[: len(values) - first] = values[first:]
        self.write = (self.write + len(values)) % size
        self.filled = min(size, self.filled + len(values))

    def ordered(self, np):
        return np.concatenate((self.data[self.write :], self.data[: self.write]))
