"""Experimental, explicit DSP policies. Importing configuration needs no NumPy."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class FrequencyBand:
    band_id: str
    low_hz: float
    high_hz: float

    def __post_init__(self):
        if (
            not isinstance(self.band_id, str)
            or not 1 <= len(self.band_id) <= 128
            or self.band_id == "broadband"
        ):
            raise ValueError(
                "band_id must be unique, nonempty, bounded and not broadband"
            )
        if (
            not all(
                type(x) in (int, float) and math.isfinite(x)
                for x in (self.low_hz, self.high_hz)
            )
            or not 0 <= self.low_hz < self.high_hz
        ):
            raise ValueError("band edges must be finite, nonnegative and increasing")


@dataclass(frozen=True, slots=True)
class AcousticConfig:
    """Uncalibrated pilot defaults, documented in docs/acoustic-model.md.

    Durations use seconds; powers are normalized PCM mean-square, not SPL.
    Bands outside a particular source's Nyquist become unavailable independently.
    """

    bands: tuple[FrequencyBand, ...] = (
        FrequencyBand("low", 40.0, 400.0),
        FrequencyBand("mid", 400.0, 2000.0),
        FrequencyBand("high", 2000.0, 12000.0),
    )
    frame_seconds: float = 0.02
    hop_seconds: float = 0.005
    horizons_seconds: tuple[float, ...] = (0.05, 0.25, 1.0)
    absolute_power_floor: float = 1e-6
    numerical_power_floor: float = 1e-20
    onset_ratio: float = 4.0
    band_onset_decline_ratio: float = 0.9
    stable_level_db: float = 1.0
    offset_ratio: float = 1.8
    strong_peak: float = 0.1
    strong_confirmation_seconds: float = 0.0
    weak_confirmation_seconds: float = 0.03
    offset_confirmation_seconds: float = 0.04
    sustained_seconds: float = 0.12
    grouping_seconds: float = 0.04
    maturity_seconds: float = 0.3
    adaptation_seconds: float = 10.0
    learning_max_relative_error: float = 0.3
    spike_relative_db: float = 12.0
    drop_relative_db: float = 12.0
    spike_absolute_power: float = 0.01
    drop_absolute_power: float = 0.01
    loudness_activity_floor: float = 1e-5
    loudness_horizon_seconds: float = 0.04
    rearm_seconds: float = 0.05
    impulse_ratio: float = 4.0
    impulse_min_peak: float = 0.01
    impulse_separation_seconds: float = 0.025
    impulse_max_seconds: float = 0.2
    rhythm_min_intervals: int = 4
    rhythm_horizons: tuple[int, ...] = (3, 8)
    rhythm_relative_jitter: float = 0.15
    rhythm_absolute_jitter_seconds: float = 0.005
    rhythm_band_distance: float = 0.65
    tempo_increase_absolute: float = 0.5
    tempo_increase_relative: float = 0.15
    tempo_decrease_absolute: float = 0.5
    tempo_decrease_relative: float = 0.15
    tempo_rearm_fraction: float = 0.5
    pause_multiplier: float = 2.5
    pause_min_seconds: float = 0.15
    pause_max_seconds: float = 3.0
    candidate_seconds: float = 1.0
    candidate_ready_seconds: float = 2.0
    candidate_relative_spread: float = 0.15
    score_threshold: float = 0.7
    acoustic_weight: float = 1.0
    temporal_weight: float = 0.0
    max_impulses: int = 64
    max_open_changes: int = 64
    max_windows_per_call: int = 4096
    max_revisions_per_call: int = 4096
    max_state_bytes: int = 16_777_216
    max_parameter_features: int = 4096
    assessment_min_coverage: float = 1.0
    established_min_coverage: float = 1.0
    candidate_min_coverage: float = 1.0
    min_contiguous_seconds: float = 0.0

    def __post_init__(self):
        if (
            not isinstance(self.bands, tuple)
            or len(self.bands) > 128
            or any(not isinstance(b, FrequencyBand) for b in self.bands)
        ):
            raise ValueError("bands must be a bounded tuple of FrequencyBand")
        if len({b.band_id for b in self.bands}) != len(self.bands):
            raise ValueError("duplicate band IDs")
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, tuple):
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{f.name} must be finite and nonnegative")
            if f.name.startswith("max_") or f.name == "rhythm_min_intervals":
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{f.name} must be a positive integer")
        if (
            not 1e-6 <= self.hop_seconds <= self.frame_seconds
            or self.frame_seconds > 10
        ):
            raise ValueError("require 1us <= hop <= frame <= 10 seconds")
        for name in ("horizons_seconds", "rhythm_horizons"):
            values = getattr(self, name)
            if (
                not values
                or len(values) > 8
                or any(
                    type(v) not in (int, float)
                    or not math.isfinite(v)
                    or not 0 < v <= 3600
                    for v in values
                )
            ):
                raise ValueError(f"{name} requires bounded positive horizons")
        if (
            any(
                type(v) is not int or v < 2 or v > self.max_impulses
                for v in self.rhythm_horizons
            )
            or self.rhythm_min_intervals < 2
            or self.rhythm_min_intervals >= self.max_impulses
        ):
            raise ValueError("rhythm history must fit max_impulses")
        if (
            not 0 < self.numerical_power_floor < self.absolute_power_floor
            or not 1 < self.offset_ratio < self.onset_ratio
        ):
            raise ValueError(
                "require numerical < acoustic floor and 1 < offset < onset"
            )
        if not 0 <= self.strong_confirmation_seconds <= self.weak_confirmation_seconds:
            raise ValueError("strong confirmation must not exceed weak confirmation")
        if (
            not 0 < self.maturity_seconds <= self.adaptation_seconds
            or not self.pause_min_seconds <= self.pause_max_seconds
        ):
            raise ValueError("invalid maturity/adaptation or pause bounds")
        if not self.candidate_ready_seconds >= self.candidate_seconds > 0:
            raise ValueError("candidate readiness must follow candidate admission")
        if (
            not 0 < self.score_threshold <= 1
            or not 0 < self.tempo_rearm_fraction < 1
            or self.strong_peak > 1
            or self.impulse_min_peak > 1
        ):
            raise ValueError("invalid normalized threshold")
        if (
            not math.isfinite(self.acoustic_weight + self.temporal_weight)
            or not self.acoustic_weight + self.temporal_weight > 0
        ):
            raise ValueError("at least one score component must have positive weight")
        if self.loudness_horizon_seconds > 3600:
            raise ValueError("loudness horizon must not exceed one hour")
        if any(
            not 0 <= getattr(self, name) <= 1
            for name in (
                "band_onset_decline_ratio",
                "assessment_min_coverage",
                "established_min_coverage",
                "candidate_min_coverage",
            )
        ):
            raise ValueError("coverage must be in [0, 1]")
        if any(
            getattr(self, name) <= 0
            for name in (
                "onset_ratio",
                "offset_confirmation_seconds",
                "sustained_seconds",
                "loudness_horizon_seconds",
                "rearm_seconds",
                "tempo_increase_absolute",
                "tempo_decrease_absolute",
                "pause_multiplier",
            )
        ):
            raise ValueError(
                "confirmation, time horizons and absolute tempo tolerances must be positive"
            )
