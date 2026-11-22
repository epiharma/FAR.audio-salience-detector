"""Deterministic eligibility contracts, not acoustic extraction or learning."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .contracts import ChannelKey, ClockStamp, SourceRef, StructuralInputError


@dataclass(frozen=True, slots=True)
class EligibilityPolicy:
    min_valid_coverage: float
    min_contiguous_duration_ns: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.min_valid_coverage, (int, float))
            or not isfinite(self.min_valid_coverage)
            or not 0 <= self.min_valid_coverage <= 1
        ):
            raise ValueError("min_valid_coverage must be finite and in [0, 1]")
        value = self.min_contiguous_duration_ns
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError("min_contiguous_duration_ns must be positive or None")


@dataclass(frozen=True, slots=True)
class AlignmentPolicy:
    disable_error_ns: int
    recover_error_ns: int
    confirmation_duration_ns: int

    def __post_init__(self) -> None:
        values = (
            self.disable_error_ns,
            self.recover_error_ns,
            self.confirmation_duration_ns,
        )
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError("alignment policy requires nonnegative integer durations")
        if self.recover_error_ns >= self.disable_error_ns:
            raise ValueError("recovery error bound must be stricter than disable bound")


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    feature_id: str
    channels: tuple[ChannelKey, ...]
    assessment: EligibilityPolicy
    established_learning: EligibilityPolicy
    candidate_learning: EligibilityPolicy
    continuous_only: bool
    alignment: AlignmentPolicy | None = None
    cross_source_wait_horizon_ns: int | None = None


@dataclass(frozen=True, slots=True)
class ClockMapping:
    source: SourceRef
    source_clock_id: str
    source_anchor_ns: int
    reference_anchor_ns: int
    rate_numerator: int
    rate_denominator: int


@dataclass(frozen=True, slots=True)
class AlignmentEvidence:
    evidence_id: str
    channels: tuple[ChannelKey, ...]
    reference_clock_id: str
    valid_start_ns: int
    valid_stop_ns: int
    observed_at: ClockStamp
    max_pairwise_error_ns: int | None
    mappings: tuple[ClockMapping, ...]


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    eligible: bool
    nominal_duration_ns: int
    valid_duration_ns: int
    valid_coverage: float
    segments: tuple[tuple[int, int], ...]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FeatureEligibility:
    assessment: EligibilityDecision
    established_learning: EligibilityDecision
    candidate_learning: EligibilityDecision


@dataclass(frozen=True, slots=True)
class RelationshipWindow:
    window_id: str
    feature_id: str
    reference_clock_id: str
    start_ns: int
    stop_ns: int


@dataclass(frozen=True, slots=True)
class FeatureStatus:
    feature_id: str
    alignment_available: bool
    continuity_revision: int
    evidence_id: str | None


def evaluate_eligibility(
    *,
    start_ns: int,
    stop_ns: int,
    segments: tuple[tuple[int, int], ...],
    policy: EligibilityPolicy,
    continuous_only: bool,
) -> EligibilityDecision:
    """Union/clamp eligible segments, then apply the caller's declared policy.

    Input segments must already exclude invalid/unavailable/unaligned data.
    Continuous-only eligibility uses only the latest segment after a break.
    """
    if type(start_ns) is not int or type(stop_ns) is not int or stop_ns <= start_ns:
        raise StructuralInputError("nominal window must have positive integer duration")
    merged: list[tuple[int, int]] = []
    for a, b in sorted(segments):
        if type(a) is not int or type(b) is not int or a >= b:
            raise StructuralInputError("invalid eligible segment")
        a, b = max(a, start_ns), min(b, stop_ns)
        if a >= b:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    if continuous_only:
        merged = merged[-1:]
    minimum = policy.min_contiguous_duration_ns
    filtered = [s for s in merged if minimum is None or s[1] - s[0] >= minimum]
    valid = sum(b - a for a, b in filtered)
    nominal = stop_ns - start_ns
    coverage = valid / nominal
    reasons: list[str] = []
    if minimum is not None and not filtered:
        reasons.append("insufficient_contiguous_duration")
    if coverage < policy.min_valid_coverage or not filtered:
        reasons.append("insufficient_valid_coverage")
    return EligibilityDecision(
        not reasons, nominal, valid, coverage, tuple(filtered), tuple(reasons)
    )


def feature_eligibility(
    spec: FeatureSpec,
    *,
    start_ns: int,
    stop_ns: int,
    segments: tuple[tuple[int, int], ...],
) -> FeatureEligibility:
    """Evaluate distinct assessment/learning policies without updating a model."""

    def decide(policy: EligibilityPolicy) -> EligibilityDecision:
        return evaluate_eligibility(
            start_ns=start_ns,
            stop_ns=stop_ns,
            segments=segments,
            policy=policy,
            continuous_only=spec.continuous_only,
        )

    return FeatureEligibility(
        decide(spec.assessment),
        decide(spec.established_learning),
        decide(spec.candidate_learning),
    )
