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

    def level_invariant_bands(self, frame, floor):
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
        for first, level in levels:
            values[first : first + self.hop] *= math.sqrt(total / level)
        return self.extract(values)[2]

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
