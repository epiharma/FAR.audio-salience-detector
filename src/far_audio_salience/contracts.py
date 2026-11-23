"""Public streaming, lifecycle and opt-in acoustic result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from .acoustic_config import AcousticConfig


def new_id() -> str:
    """Generate an opaque identity, never derived from an acoustic pattern."""
    return str(uuid4())


class StructuralInputError(ValueError):
    """A malformed call was rejected without changing detector state."""


class StateError(RuntimeError):
    """An operation is not legal in the current acquisition state."""


class SnapshotError(ValueError):
    """A snapshot is unsupported, corrupt, or exceeds configured bounds."""


@dataclass(frozen=True, slots=True)
class ClockStamp:
    clock_id: str
    ns: int


@dataclass(frozen=True, slots=True)
class ChannelKey:
    source_id: str
    channel_id: str


@dataclass(frozen=True, slots=True)
class SourceRef:
    source_id: str
    session_id: str
    epoch_id: str


@dataclass(frozen=True, slots=True)
class SourceSpec:
    ref: SourceRef
    label: str
    sample_rate_hz: int
    clock_id: str
    channel_ids: tuple[str, ...]

    @classmethod
    def microphone(
        cls, *, label: str, sample_rate_hz: int, channel_count: int = 1
    ) -> SourceSpec:
        """Create metadata only; this helper never opens a device."""
        if type(channel_count) is not int or channel_count <= 0:
            raise StructuralInputError("channel_count must be a positive integer")
        return cls(
            SourceRef(new_id(), new_id(), new_id()),
            label,
            sample_rate_hz,
            new_id(),
            tuple(new_id() for _ in range(channel_count)),
        )

    def new_session(self, *, sample_rate_hz: int | None = None) -> SourceSpec:
        """Keep physical identities; create a fresh acquisition/clock epoch."""
        return SourceSpec(
            SourceRef(self.ref.source_id, new_id(), new_id()),
            self.label,
            self.sample_rate_hz if sample_rate_hz is None else sample_rate_hz,
            new_id(),
            self.channel_ids,
        )

    def chunk(
        self,
        samples: memoryview | tuple[memoryview, ...],
        *,
        first_frame: int,
        measured_start_ns: int | None = None,
        received_at: ClockStamp | None = None,
        layout: Literal["interleaved", "planar"] = "interleaved",
        channel_ids: tuple[str, ...] | None = None,
        quality: AcquisitionQuality | None = None,
    ) -> AudioChunk:
        """Wrap an external block; generate IDs, never acquisition time."""
        channels = self.channel_ids if channel_ids is None else channel_ids
        if not channels:
            raise StructuralInputError("chunk requires at least one channel")
        if layout == "interleaved" and isinstance(samples, memoryview):
            if len(samples) % len(channels):
                raise StructuralInputError(
                    "interleaved length is not a whole frame count"
                )
            audio = InterleavedAudio(channels, samples)
            count = len(samples) // len(channels)
        elif layout == "planar" and isinstance(samples, tuple) and samples:
            audio = PlanarAudio(channels, samples)
            count = len(samples[0])
        else:
            raise StructuralInputError(
                "sample container does not match explicit layout"
            )
        return AudioChunk(
            SourceInterval(self.ref, first_frame, first_frame + count),
            audio,
            new_id(),
            None
            if measured_start_ns is None
            else ClockStamp(self.clock_id, measured_start_ns),
            received_at,
            quality,
        )


@dataclass(frozen=True, slots=True)
class SourceInterval:
    source: SourceRef
    start_frame: int
    stop_frame: int


@dataclass(frozen=True, slots=True)
class DataSpan:
    interval: SourceInterval
    channel_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InterleavedAudio:
    channel_ids: tuple[str, ...]
    samples: memoryview
    layout: Literal["interleaved"] = "interleaved"


@dataclass(frozen=True, slots=True)
class PlanarAudio:
    channel_ids: tuple[str, ...]
    planes: tuple[memoryview, ...]
    layout: Literal["planar"] = "planar"


@dataclass(frozen=True, slots=True)
class AcquisitionQuality:
    overflow: bool | None = None
    reported_dropped_frames: int | None = None
    reported_dropped_blocks: int | None = None


@dataclass(frozen=True, slots=True)
class AudioChunk:
    interval: SourceInterval
    audio: InterleavedAudio | PlanarAudio
    observation_id: str = ""
    measured_start: ClockStamp | None = None
    received_at: ClockStamp | None = None
    quality: AcquisitionQuality | None = None


@dataclass(frozen=True, slots=True)
class UnavailableObservation:
    source: SourceRef
    channel_ids: tuple[str, ...]
    reason: Literal["missing", "invalid"]
    interval: SourceInterval | None = None
    observation_id: str = ""
    measured_start: ClockStamp | None = None
    received_at: ClockStamp | None = None


@dataclass(frozen=True, slots=True)
class SourceWatermark:
    source: SourceRef
    channel_ids: tuple[str, ...]
    before_frame: int


@dataclass(frozen=True, slots=True)
class MicrophonePosition:
    channel: ChannelKey
    position_m: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class MicrophoneGeometry:
    geometry_id: str
    reference_frame_id: str
    positions: tuple[MicrophonePosition, ...]


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    max_sources: int
    max_channels: int
    max_chunk_bytes: int
    max_buffer_bytes: int
    max_baselines: int
    max_receipts: int
    max_snapshot_bytes: int
    max_relationship_windows: int
    max_features: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ResourceReporting:
    aggregate_every: int
    materially_worse_by: int

    def __post_init__(self) -> None:
        if any(
            type(v) is not int or v <= 0
            for v in (
                self.aggregate_every,
                self.materially_worse_by,
            )
        ):
            raise ValueError("resource reporting values must be positive integers")


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    limits: ResourceLimits
    max_reorder_delay_ns: int
    comparison_retention_ns: int
    reporting: ResourceReporting
    promotion_mode: Literal["automatic", "external_approval"] = "external_approval"
    acoustic: AcousticConfig | None = None

    def __post_init__(self) -> None:
        if self.acoustic is not None and not isinstance(self.acoustic, AcousticConfig):
            raise ValueError("acoustic must be AcousticConfig or None")
        if not isinstance(self.limits, ResourceLimits) or not isinstance(
            self.reporting, ResourceReporting
        ):
            raise ValueError("typed ResourceLimits and ResourceReporting are required")
        for value in (self.max_reorder_delay_ns, self.comparison_retention_ns):
            if type(value) is not int or value < 0:
                raise ValueError(
                    "retention/reorder durations must be nonnegative integers"
                )
        if self.promotion_mode not in ("automatic", "external_approval"):
            raise ValueError("unknown promotion mode")


@dataclass(frozen=True, slots=True)
class Provenance:
    spans: tuple[DataSpan, ...] = ()
    measurement_times: tuple[ClockStamp, ...] = ()
    receipt_times: tuple[ClockStamp, ...] = ()
    alignment_evidence_ids: tuple[str, ...] = ()
    geometry_id: str | None = None
    source_clocks: tuple[tuple[SourceRef, str], ...] = ()


@dataclass(frozen=True, slots=True)
class EmissionHeader:
    event_id: str
    run_id: str
    emission_sequence: int
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class AcousticQuantity:
    name: str
    value: float
    unit: str


@dataclass(frozen=True, slots=True)
class AcousticBandChange:
    band_id: str
    start_frame: int
    last_observed_frame: int
    end_frame: int | None
    power: float
    delta_db: float | None


ChangeKind = Literal[
    "sound_onset",
    "sound_offset",
    "tempo_increase",
    "tempo_decrease",
    "loudness_spike",
    "loudness_drop",
]


@dataclass(frozen=True, slots=True)
class AcousticChange:
    change_id: str
    revision: int
    kind: ChangeKind
    phase: Literal["detected", "completed", "interrupted"]
    channel: ChannelKey
    source: SourceRef
    start_frame: int
    confirmed_frame: int
    last_observed_frame: int
    end_frame: int | None
    duration_frames: int | None
    bands: tuple[AcousticBandChange, ...]
    quantities: tuple[AcousticQuantity, ...]
    confidence: float | None
    provenance: Provenance
    sequence_id: str | None = None


@dataclass(frozen=True, slots=True)
class AcousticBandPower:
    band_id: str
    power: float | None
    state: str
    schema_id: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class AcousticWindow:
    interval: SourceInterval
    rms: float
    power: float
    peak: float
    bands: tuple[AcousticBandPower, ...]
    tempo_hz: float | None
    tempo_reason: str | None
    sequence_id: str | None
    confidence: float
    assessment_eligible: bool
    established_learning_eligible: bool
    candidate_learning_eligible: bool
    continuity_revision: int
    horizons: tuple[tuple[float, float | None], ...] = ()


@dataclass(frozen=True, slots=True)
class SalienceResult:
    header: EmissionHeader
    channels: tuple[ChannelKey, ...]
    timestamp: ClockStamp | None = None
    time_scale: str | None = None
    salience_score: float | None = None
    confidence: float | None = None
    is_salient: bool | None = None
    baseline_status: str = "unavailable"
    acoustic_deviation: float | None = None
    temporal_deviation: float | None = None
    pattern_novelty: float | None = None
    stream_discontinuity_score: float | None = None
    features: tuple[tuple[str, float], ...] | None = None
    unavailable_reasons: tuple[str, ...] = ("not_implemented",)
    changes: tuple[AcousticChange, ...] = ()
    windows: tuple[AcousticWindow, ...] = ()


@dataclass(frozen=True, slots=True)
class ResourceHit:
    resource_id: str
    observed: int
    limit: int
    exceedance: int
    unit: str


@dataclass(frozen=True, slots=True)
class ResourceEpisode:
    episode_id: str
    phase: Literal["first", "update", "worse", "end"]
    hit: ResourceHit
    count: int
    duration_ns: int
    peak_exceedance: int


HealthKind = Literal[
    "invalid_audio",
    "channels_unavailable",
    "channels_recovered",
    "stream_discontinuity",
    "conflicting_observation",
    "alignment_lost",
    "alignment_recovered",
    "sample_rate_changed",
    "geometry_changed",
    "channel_membership_changed",
    "resource_limit_violation",
    "capacity_exhausted",
]


@dataclass(frozen=True, slots=True)
class SensorHealthEvent:
    header: EmissionHeader
    event_type: HealthKind
    channels: tuple[ChannelKey, ...] = ()
    reason_codes: tuple[str, ...] = ()
    feature_ids: tuple[str, ...] = ()
    resource: ResourceEpisode | None = None


@dataclass(frozen=True, slots=True)
class BaselineCompatibility:
    feature_id: str
    status: Literal[
        "compatible", "revalidate", "unavailable", "incompatible", "unknown"
    ]
    depends_on_geometry: bool = False
    reason: str = "explicit extension metadata"


@dataclass(frozen=True, slots=True)
class BaselineRecord:
    """Administrative lifecycle metadata, with no acoustic model parameters."""

    baseline_id: str
    candidate_id: str
    channels: tuple[ChannelKey, ...]
    role: Literal["candidate", "established"] = "candidate"
    activity: Literal["active", "archived", "reactivating"] = "active"
    control_revision: int = 0
    evidence_revision: int = 0
    ready: bool = False
    proposal_id: str | None = None
    rejected_proposal_id: str | None = None
    promotion_blocked: bool = False
    compatibility: tuple[BaselineCompatibility, ...] = ()


LifecycleKind = Literal[
    "candidate_created",
    "candidate_ready",
    "candidate_promoted",
    "baseline_archived",
    "baseline_reactivation_started",
    "baseline_reactivated",
    "baseline_compatibility_changed",
    "promotion_rejected",
    "promotion_blocked",
    "promotion_unblocked",
    "learning_reset",
    "factory_reset",
]


@dataclass(frozen=True, slots=True)
class SensorBaselineLifecycleEvent:
    header: EmissionHeader
    event_type: LifecycleKind
    before: BaselineRecord | None = None
    after: BaselineRecord | None = None
    command_id: str | None = None
    candidate_id: str | None = None
    control_revision: int | None = None


@dataclass(frozen=True, slots=True)
class CandidateCommand:
    command_id: str
    candidate_id: str
    expected_control_revision: int
    action: Literal["approve", "reject", "block", "unblock", "archive", "reactivate"]
    proposal_id: str | None = None
    actor_id: str | None = None


@dataclass(frozen=True, slots=True)
class CommandReceipt:
    command_id: str
    candidate_id: str
    outcome: Literal["applied", "no_change", "refused"]
    control_revision: int
    event_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Refusal:
    operation: str
    code: str
    retryable: bool = False
    resource: ResourceHit | None = None
    resource_episode_id: str | None = None


@dataclass(frozen=True, slots=True)
class InputReceipt:
    observation_id: str
    accepted: tuple[DataSpan, ...] = ()
    ignored: tuple[DataSpan, ...] = ()
    reason_codes: tuple[str, ...] = ()


Emission = SalienceResult | SensorHealthEvent | SensorBaselineLifecycleEvent


@dataclass(frozen=True, slots=True)
class OutputBatch:
    run_id: str
    outcome: Literal[
        "accepted", "partially_accepted", "ignored", "refused", "no_change"
    ]
    salience: tuple[SalienceResult, ...] = ()
    health: tuple[SensorHealthEvent, ...] = ()
    baseline_lifecycle: tuple[SensorBaselineLifecycleEvent, ...] = ()
    input_receipts: tuple[InputReceipt, ...] = ()
    refusals: tuple[Refusal, ...] = ()
    command_receipt: CommandReceipt | None = None
    finalized: tuple[SourceWatermark, ...] = ()

    @property
    def ordered(self) -> tuple[Emission, ...]:
        """One cross-family view of the same emissions, not another delivery."""
        return tuple(
            sorted(
                (*self.salience, *self.health, *self.baseline_lifecycle),
                key=lambda item: item.header.emission_sequence,
            )
        )
