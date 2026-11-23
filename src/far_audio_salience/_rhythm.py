"""Bounded single-sequence rhythm evidence. IDs never identify physical sources."""

from collections import deque
from dataclasses import dataclass, field
from statistics import median

from .contracts import new_id


@dataclass
class Rhythm:
    times: deque = field(default_factory=deque)
    profiles: deque = field(default_factory=deque)
    sequence_id: str | None = None
    rate: float | None = None
    reference: float | None = None
    reason: str | None = "insufficient_impulses"
    deadline: int | None = None
    high_since: int | None = None
    pulse_high: bool = False
    last_pulse: int | None = None
    ever_tracked: bool = False
    direction: int = 0
    stable_count: int = 0
    extreme: float | None = None

    def clear(self, reason):
        self.times.clear()
        self.profiles.clear()
        self.sequence_id = None
        self.rate = self.reference = None
        self.deadline = None
        self.direction = 0
        self.stable_count = 0
        self.extreme = None
        self.reason = reason
        if reason == "sequence_timeout":
            self.ever_tracked = False
            self.last_pulse = None

    def observe(self, *, high, pulse_frame, end, rate, profile, config):
        """Return (direction, magnitude, finish_previous) for observed changes."""
        change = None
        # Expire against the frozen deadline before a newly arriving impulse
        # can extend it. Continuous high activity follows the merge path below.
        if (
            self.deadline is not None
            and pulse_frame > self.deadline
            and high
            and not self.pulse_high
        ):
            self.clear("sequence_timeout")
        if high:
            if self.high_since is None:
                self.high_since = end
            if (
                self.ever_tracked
                and (end - self.high_since) / rate >= config.impulse_max_seconds
            ):
                self.clear("impulses_indistinguishable")
        else:
            self.high_since = None
        rising = high and not self.pulse_high
        self.pulse_high = high
        if rising and (
            self.last_pulse is None
            or (pulse_frame - self.last_pulse) / rate
            >= config.impulse_separation_seconds
        ):
            self.last_pulse = pulse_frame
            if self.sequence_id is None:
                self.sequence_id = new_id()
            self.times.append(pulse_frame)
            self.profiles.append(profile)
            if self.deadline is None:
                self.deadline = pulse_frame + round(config.pause_max_seconds * rate)
            while len(self.times) > config.max_impulses:
                self.times.popleft()
                self.profiles.popleft()
            if len(self.times) >= 2:
                intervals = [
                    (b - a) / rate
                    for a, b in zip(list(self.times)[:-1], list(self.times)[1:])
                ]
                recent = intervals[-max(config.rhythm_horizons) :]
                slope = (
                    median([b - a for a, b in zip(recent[:-1], recent[1:])])
                    if len(recent) > 1
                    else 0.0
                )
                predicted = max(
                    config.impulse_separation_seconds, recent[-1] + max(0.0, slope)
                )
                pause = min(
                    config.pause_max_seconds,
                    max(config.pause_min_seconds, config.pause_multiplier * predicted),
                )
                self.deadline = pulse_frame + round(pause * rate)
            if len(self.times) - 1 >= config.rhythm_min_intervals:
                intervals = [
                    (b - a) / rate
                    for a, b in zip(list(self.times)[:-1], list(self.times)[1:])
                ]
                windows = []
                for horizon in sorted(config.rhythm_horizons):
                    values = intervals[-horizon:]
                    if len(values) < horizon:
                        continue
                    slope = median([b - a for a, b in zip(values[:-1], values[1:])])
                    intercept = median([v - slope * i for i, v in enumerate(values)])
                    residual = max(
                        [abs(v - (intercept + slope * i)) for i, v in enumerate(values)]
                    )
                    tolerance = (
                        config.rhythm_absolute_jitter_seconds
                        + config.rhythm_relative_jitter * median(values)
                    )
                    if residual <= tolerance and max(values) - min(
                        values
                    ) <= tolerance * 2 + abs(slope) * (len(values) - 1):
                        windows.append((horizon, 1 / median(values), slope))
                profiles = list(self.profiles)[-config.rhythm_min_intervals :]
                coherent = all(
                    sum(abs(a - b) for a, b in zip(profiles[0], p))
                    <= config.rhythm_band_distance
                    for p in profiles
                )
                short = intervals[-min(config.rhythm_horizons) :]
                # A short plateau can confirm a step while the longer fit still
                # straddles it. A short trend alone cannot explain a conflicting
                # longer history (e.g. two interleaved pulse trains).
                plateau = (
                    max(short) - min(short) <= config.rhythm_absolute_jitter_seconds
                )
                long_fit = bool(
                    windows and windows[-1][0] == max(config.rhythm_horizons)
                )
                enough_long_history = len(intervals) >= max(config.rhythm_horizons)
                if (
                    not windows
                    or not coherent
                    or (enough_long_history and not long_fit and not plateau)
                ):
                    self.rate = None
                    self.reason = "ambiguous_impulse_sequence"
                else:
                    _, current, trend = windows[0]
                    self.rate, self.reason, self.ever_tracked = current, None, True
                    if self.reference is None:
                        self.reference = current
                    # An observed reversal is measured from the supported
                    # extreme of the open transition.
                    reference = self.reference if self.extreme is None else self.extreme
                    direction = 1 if current > reference else -1
                    absolute = (
                        config.tempo_increase_absolute
                        if direction > 0
                        else config.tempo_decrease_absolute
                    )
                    relative = (
                        config.tempo_increase_relative
                        if direction > 0
                        else config.tempo_decrease_relative
                    )
                    excess = (current - reference) / (absolute + relative * reference)
                    # A single long observed interval cannot establish slowing:
                    # require the short horizon to support the new level/trend.
                    short_spread = max(short) - min(short)
                    support = (
                        short_spread
                        <= config.rhythm_absolute_jitter_seconds * 2
                        + config.rhythm_relative_jitter * median(short)
                        + abs(trend) * (len(short) - 1)
                    )
                    if abs(excess) >= 1 and support and direction != self.direction:
                        if self.direction:
                            self.reference = reference
                        change = (direction, current - reference, False)
                        self.direction = direction
                        self.stable_count = 0
                        self.extreme = current
                    if self.direction:
                        if (current - self.extreme) * self.direction > 0:
                            self.extreme = current
                        stable = (
                            plateau
                            and short_spread
                            <= (
                                config.rhythm_absolute_jitter_seconds
                                + config.rhythm_relative_jitter * median(short)
                            )
                            * config.tempo_rearm_fraction
                            and all(
                                abs(slope) * (horizon - 1)
                                <= config.rhythm_absolute_jitter_seconds
                                for horizon, _, slope in windows
                            )
                        )
                        self.stable_count = self.stable_count + 1 if stable else 0
                        if self.stable_count >= min(config.rhythm_horizons):
                            if change is None:
                                change = (
                                    self.direction,
                                    current - self.reference,
                                    True,
                                )
                            self.reference = current
                            self.direction = 0
                            self.stable_count = 0
                            self.extreme = None
        if self.deadline is not None and end > self.deadline and not high:
            self.clear("sequence_timeout")
        return change
