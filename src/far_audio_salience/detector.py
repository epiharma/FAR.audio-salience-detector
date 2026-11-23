"""Bounded streaming infrastructure. Acoustic capabilities are unavailable."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from ._history import _Index, _planned, _Timing
from .baselines import BaselineStore, ReadinessEvidence
from .contracts import (
    AudioChunk,
    BaselineCompatibility,
    BaselineRecord,
    CandidateCommand,
    ChannelKey,
    ClockStamp,
    DataSpan,
    DetectorConfig,
    EmissionHeader,
    InputReceipt,
    InterleavedAudio,
    MicrophoneGeometry,
    OutputBatch,
    PlanarAudio,
    Provenance,
    Refusal,
    ResourceEpisode,
    ResourceHit,
    SalienceResult,
    SensorBaselineLifecycleEvent,
    SensorHealthEvent,
    SourceInterval,
    SourceRef,
    SourceSpec,
    SourceWatermark,
    StateError,
    StructuralInputError,
    UnavailableObservation,
    new_id,
)
from .eligibility import (
    AlignmentEvidence,
    AlignmentPolicy,
    ClockMapping,
    EligibilityPolicy,
    FeatureSpec,
    FeatureStatus,
    RelationshipWindow,
)


def _integer(value: object, name: str, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise StructuralInputError(f"{name} must be a bounded integer >= {minimum}")


def _identity(value: object) -> None:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise StructuralInputError("identifiers must contain 1..256 characters")


def _stamp(value: ClockStamp | None) -> None:
    if value is not None:
        if not isinstance(value, ClockStamp):
            raise StructuralInputError("expected ClockStamp")
        _identity(value.clock_id)
        _integer(value.ns, "timestamp")


@dataclass(slots=True)
class _Lane:
    watermark: int = 0
    high_water: int = 0
    pending: _Index = field(default_factory=_Index)
    history: _Index = field(default_factory=_Index)
    unavailable: bool = False
    recovery_after: int | None = None


@dataclass(slots=True)
class _Sync:
    enabled: bool = False
    confirmation_start: int | None = None
    evidence: AlignmentEvidence | None = None
    continuity_revision: int = 0
    period: _AlignmentPeriod | None = None


@dataclass(frozen=True, slots=True)
class _AlignmentPeriod:
    """A compact continuous hysteresis-qualified interval, not a report ledger."""

    start_ns: int
    stop_ns: int
    initial_evidence: AlignmentEvidence
    evidence: AlignmentEvidence

    def supports(self, window: RelationshipWindow) -> bool:
        return (
            window.reference_clock_id == self.evidence.reference_clock_id
            and self.start_ns <= window.start_ns
            and window.stop_ns <= self.stop_ns
        )


@dataclass(slots=True)
class _Episode:
    episode_id: str
    first_ns: int
    count: int
    last_report_count: int
    peak: int
    reported_peak: int
    hit: ResourceHit
    temporary: bool


# Drafts have no identity until the enclosing expected operation can commit.
Draft = tuple[str, dict]


class AudioSalienceDetector:
    """One synchronous owner, many independently clocked acquisition sources.

    ``process`` emits honest unavailable assessments for finalized valid audio.
    It never extracts acoustic features, scores sound, or trains a baseline.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.run_id = new_id()
        self._sequence = 0
        self._closed = False
        self._sources: dict[str, SourceSpec] = {}
        self._identities: dict[str, tuple[str, ...]] = {}
        self._active: set[str] = set()
        self._lanes: dict[ChannelKey, _Lane] = {}
        self._baselines = BaselineStore(config.limits.max_receipts)
        self._episodes: dict[str, _Episode] = {}
        self._features: dict[str, FeatureSpec] = {}
        self._sync: dict[str, _Sync] = {}
        self._windows: dict[str, RelationshipWindow] = {}
        self._window_alignment: dict[str, _AlignmentPeriod] = {}
        self._reference_times: dict[str, int] = {}
        self._feature_clocks: dict[str, str] = {}
        self._geometry: MicrophoneGeometry | None = None

    def _check_open(self) -> None:
        if self._closed:
            raise StateError("detector is closed")

    def _batch(
        self,
        drafts: list[Draft] | None = None,
        *,
        outcome: str = "accepted",
        receipts: tuple[InputReceipt, ...] = (),
        refusals: tuple[Refusal, ...] = (),
        finalized: tuple[SourceWatermark, ...] = (),
        command_receipt=None,
    ) -> OutputBatch:
        salience, health, lifecycle = [], [], []
        for family, data in drafts or []:
            data = dict(data)
            provenance = data.pop("provenance", Provenance())
            channels = data.get("channels", ())
            record = data.get("after") or data.get("before")
            if record is not None:
                channels = record.channels
            source_ids = {c.source_id for c in channels}
            source_ids.update(s.interval.source.source_id for s in provenance.spans)
            if not provenance.source_clocks:
                provenance = replace(
                    provenance,
                    source_clocks=tuple(
                        (self._sources[k].ref, self._sources[k].clock_id)
                        for k in sorted(source_ids)
                        if k in self._sources
                    ),
                )
            if family == "health" and data.get("event_type") in (
                "invalid_audio",
                "channels_unavailable",
                "channels_recovered",
            ):
                affected = set(channels)
                for feature_id, feature in self._features.items():
                    if affected.intersection(feature.channels):
                        self._sync[feature_id].continuity_revision += 1
            self._sequence += 1
            header = EmissionHeader(new_id(), self.run_id, self._sequence, provenance)
            if family == "salience":
                salience.append(SalienceResult(header=header, **data))
            elif family == "health":
                health.append(SensorHealthEvent(header=header, **data))
            else:
                lifecycle.append(SensorBaselineLifecycleEvent(header=header, **data))
        return OutputBatch(
            self.run_id,
            outcome,
            tuple(salience),
            tuple(health),
            tuple(lifecycle),
            receipts,
            refusals,
            command_receipt,
            finalized,
        )

    def _refuse_resource(
        self,
        resource: str,
        observed: int,
        limit: int,
        *,
        temporary: bool = False,
        operation: str = "ingest",
        unit: str = "count",
        provenance: Provenance | None = None,
    ) -> OutputBatch:
        # Keys are package-owned limit names, never attacker-supplied IDs.
        hit = ResourceHit(resource, observed, limit, max(0, observed - limit), unit)
        now = time.monotonic_ns()
        episode = self._episodes.get(resource)
        phase = None
        if episode is None:
            episode = _Episode(
                new_id(), now, 0, 0, hit.exceedance, hit.exceedance, hit, temporary
            )
            self._episodes[resource] = episode
            phase = "first"
        episode.count += 1
        episode.hit = hit
        episode.peak = max(episode.peak, hit.exceedance)
        reporting = self.config.reporting
        if phase is None:
            if episode.peak - episode.reported_peak >= reporting.materially_worse_by:
                phase = "worse"
            elif episode.count - episode.last_report_count >= reporting.aggregate_every:
                phase = "update"
        drafts: list[Draft] = []
        if phase is not None:
            episode.reported_peak = episode.peak
            episode.last_report_count = episode.count
            drafts.append(
                (
                    "health",
                    {
                        "event_type": "capacity_exhausted"
                        if temporary
                        else "resource_limit_violation",
                        "resource": ResourceEpisode(
                            episode.episode_id,
                            phase,
                            hit,
                            episode.count,
                            now - episode.first_ns,
                            episode.peak,
                        ),
                        "provenance": provenance or Provenance(),
                    },
                )
            )
        return self._batch(
            drafts,
            outcome="refused",
            refusals=(
                Refusal(
                    operation,
                    "temporary_capacity" if temporary else "hard_limit",
                    temporary,
                    hit,
                    episode.episode_id,
                ),
            ),
        )

    def _end_episodes(self, resources: tuple[str, ...] | None = None) -> list[Draft]:
        now = time.monotonic_ns()
        drafts: list[Draft] = []
        for key in tuple(self._episodes):
            if resources is not None and key not in resources:
                continue
            episode = self._episodes.pop(key)
            drafts.append(
                (
                    "health",
                    {
                        "event_type": "capacity_exhausted"
                        if episode.temporary
                        else "resource_limit_violation",
                        "resource": ResourceEpisode(
                            episode.episode_id,
                            "end",
                            episode.hit,
                            episode.count,
                            now - episode.first_ns,
                            episode.peak,
                        ),
                    },
                )
            )
        return drafts

    def _validate_spec(self, spec: SourceSpec) -> None:
        if not isinstance(spec, SourceSpec) or not isinstance(spec.ref, SourceRef):
            raise StructuralInputError("expected SourceSpec/SourceRef")
        for value in (
            spec.ref.source_id,
            spec.ref.session_id,
            spec.ref.epoch_id,
            spec.clock_id,
            spec.label,
        ):
            _identity(value)
        _integer(spec.sample_rate_hz, "sample rate", 1)
        self._validate_channels(spec.channel_ids, spec.channel_ids)

    @staticmethod
    def _validate_channels(channels: tuple[str, ...], allowed: tuple[str, ...]) -> None:
        if not isinstance(channels, tuple) or not channels:
            raise StructuralInputError("channels must be a nonempty tuple")
        for channel in channels:
            _identity(channel)
        if len(set(channels)) != len(channels) or not set(channels) <= set(allowed):
            raise StructuralInputError("duplicate or unknown channel identity")

    @property
    def sources(self) -> tuple[SourceSpec, ...]:
        """Stable identity catalog, including stopped sources."""
        return tuple(self._sources.values())

    @property
    def buffered_sample_count(self) -> int:
        """Pending plus comparison samples, including invalid-block markers."""
        return sum(
            len(lane.pending) + len(lane.history) for lane in self._lanes.values()
        )

    def start_source(self, spec: SourceSpec) -> OutputBatch:
        self._check_open()
        self._validate_spec(spec)
        source_id = spec.ref.source_id
        if source_id in self._active:
            raise StateError("stop or reconfigure the active source first")
        old = self._sources.get(source_id)
        if old is not None and old.ref == spec.ref:
            raise StructuralInputError("reopening requires a fresh acquisition epoch")
        count = len(self._sources) + (old is None)
        if count > self.config.limits.max_sources:
            return self._refuse_resource(
                "max_sources",
                count,
                self.config.limits.max_sources,
                operation="start_source",
            )
        known = tuple(
            dict.fromkeys((*self._identities.get(source_id, ()), *spec.channel_ids))
        )
        channels = sum(
            len(v) for k, v in self._identities.items() if k != source_id
        ) + len(known)
        if channels > self.config.limits.max_channels:
            return self._refuse_resource(
                "max_channels",
                channels,
                self.config.limits.max_channels,
                operation="start_source",
            )
        self._sources[source_id] = spec
        self._identities[source_id] = known
        self._active.add(source_id)
        unavailable = {
            key.channel_id
            for key, lane in self._lanes.items()
            if key.source_id == source_id and lane.unavailable
        }
        for key in tuple(self._lanes):
            if key.source_id == source_id:
                del self._lanes[key]
        for channel in spec.channel_ids:
            self._lanes[ChannelKey(source_id, channel)] = _Lane(
                unavailable=channel in unavailable,
                recovery_after=0 if channel in unavailable else None,
            )
        drafts = self._configuration_change(old, spec)
        drafts.extend(self._end_episodes(("max_sources", "max_channels")))
        return self._batch(drafts)

    def _source(self, ref: SourceRef) -> SourceSpec:
        if not isinstance(ref, SourceRef):
            raise StructuralInputError("expected SourceRef")
        for item in (ref.source_id, ref.session_id, ref.epoch_id):
            _identity(item)
        spec = self._sources.get(ref.source_id)
        if spec is None:
            raise StructuralInputError("unknown source")
        return spec

    def _validate_interval(self, interval: SourceInterval) -> SourceSpec:
        if not isinstance(interval, SourceInterval):
            raise StructuralInputError("expected SourceInterval")
        spec = self._source(interval.source)
        _integer(interval.start_frame, "start_frame")
        _integer(interval.stop_frame, "stop_frame", 1)
        if interval.start_frame >= interval.stop_frame:
            raise StructuralInputError("interval must have positive duration")
        return spec

    @staticmethod
    def _view(view: memoryview, length: int) -> None:
        if not isinstance(view, memoryview):
            raise StructuralInputError("samples must be a memoryview")
        try:
            view.format
        except ValueError as error:
            raise StructuralInputError("sample buffer has been released") from error
        if (
            view.format != "f"
            or view.itemsize != 4
            or view.ndim != 1
            or not view.c_contiguous
            or len(view) != length
        ):
            raise StructuralInputError(
                "expected contiguous native float32 with exact shape"
            )

    def _validate_audio(self, chunk: AudioChunk) -> tuple[SourceSpec, int]:
        spec = self._validate_interval(chunk.interval)
        if not isinstance(chunk.audio, (InterleavedAudio, PlanarAudio)):
            raise StructuralInputError("explicit planar/interleaved audio required")
        current_epoch = chunk.interval.source == spec.ref
        self._validate_channels(
            chunk.audio.channel_ids,
            spec.channel_ids if current_epoch else self._identities[spec.ref.source_id],
        )
        if chunk.observation_id:
            _identity(chunk.observation_id)
        _stamp(chunk.measured_start)
        _stamp(chunk.received_at)
        if (
            current_epoch
            and chunk.measured_start
            and chunk.measured_start.clock_id != spec.clock_id
        ):
            raise StructuralInputError("measurement clock does not match source epoch")
        n = chunk.interval.stop_frame - chunk.interval.start_frame
        if isinstance(chunk.audio, InterleavedAudio):
            if chunk.audio.layout != "interleaved":
                raise StructuralInputError("incorrect layout discriminator")
            self._view(chunk.audio.samples, n * len(chunk.audio.channel_ids))
            size = chunk.audio.samples.nbytes
        else:
            if chunk.audio.layout != "planar" or not isinstance(
                chunk.audio.planes, tuple
            ):
                raise StructuralInputError("incorrect planar layout")
            if len(chunk.audio.planes) != len(chunk.audio.channel_ids):
                raise StructuralInputError("plane count does not match channel IDs")
            for plane in chunk.audio.planes:
                self._view(plane, n)
            size = sum(p.nbytes for p in chunk.audio.planes)
        if chunk.quality is not None:
            from .contracts import AcquisitionQuality

            if not isinstance(chunk.quality, AcquisitionQuality):
                raise StructuralInputError("expected AcquisitionQuality")
            if (
                chunk.quality.overflow is not None
                and type(chunk.quality.overflow) is not bool
            ):
                raise StructuralInputError("overflow must be bool or None")
            for count in (
                chunk.quality.reported_dropped_blocks,
                chunk.quality.reported_dropped_frames,
            ):
                if count is not None:
                    _integer(count, "reported loss")
        return spec, size

    def process(self, observation: AudioChunk | UnavailableObservation) -> OutputBatch:
        self._check_open()
        if isinstance(observation, UnavailableObservation):
            return self._process_unavailable(observation)
        if not isinstance(observation, AudioChunk):
            raise StructuralInputError("expected AudioChunk or UnavailableObservation")
        chunk = observation
        spec, size = self._validate_audio(chunk)
        if spec.ref != chunk.interval.source or spec.ref.source_id not in self._active:
            return self._batch(
                outcome="ignored",
                receipts=(
                    InputReceipt(
                        chunk.observation_id,
                        ignored=(DataSpan(chunk.interval, chunk.audio.channel_ids),),
                        reason_codes=("late_unverifiable", "inactive_or_stale_epoch"),
                    ),
                ),
            )
        if size > self.config.limits.max_chunk_bytes:
            return self._refuse_resource(
                "max_chunk_bytes",
                size,
                self.config.limits.max_chunk_bytes,
                unit="bytes",
                provenance=self._provenance(
                    DataSpan(chunk.interval, chunk.audio.channel_ids),
                    chunk.measured_start,
                    chunk.received_at,
                ),
            )
        # Stage immutable descriptor plans; payload is copied only after preflight.
        lanes = dict(self._lanes)
        borrowed: dict[ChannelKey, memoryview] = {}
        drafts: list[Draft] = []
        accepted, ignored, reasons = [], [], []
        start, stop = chunk.interval.start_frame, chunk.interval.stop_frame
        width = len(chunk.audio.channel_ids)
        timing = _Timing(start, chunk.measured_start, chunk.received_at)
        for position, channel in enumerate(chunk.audio.channel_ids):
            key = ChannelKey(spec.ref.source_id, channel)
            lane = replace(lanes[key])
            lanes[key] = lane
            values = (
                chunk.audio.samples[position::width]
                if isinstance(chunk.audio, InterleavedAudio)
                else chunk.audio.planes[position]
            )
            borrowed[key] = values
            span = DataSpan(chunk.interval, (channel,))
            provenance = self._provenance(span, chunk.measured_start, chunk.received_at)
            valid = all(math.isfinite(v) and -1 <= v <= 1 for v in values)
            # Any provable conflict quarantines this channel's entire incoming block.
            conflict = valid and (
                lane.pending.conflicts(start, values)
                or lane.history.conflicts(start, values)
            )
            if conflict:
                ignored.append(span)
                reasons.append("conflicting_observation")
                drafts.append(
                    (
                        "health",
                        {
                            "event_type": "conflicting_observation",
                            "channels": (key,),
                            "provenance": provenance,
                            "reason_codes": ("incoming_channel_block_quarantined",),
                        },
                    )
                )
                continue
            late_stop = min(stop, lane.watermark)
            if start < late_stop:
                ignored.append(
                    DataSpan(SourceInterval(spec.ref, start, late_stop), (channel,))
                )
                comparable = lane.history.covers(start, late_stop)
                reasons.append("late" if comparable else "late_unverifiable")
            ranges = lane.pending.uncovered(max(start, lane.watermark), stop)
            if ranges:
                lane.pending = lane.pending.added(_planned(ranges, valid, timing))
                new_spans = [
                    DataSpan(SourceInterval(spec.ref, a, b), (channel,))
                    for a, b in ranges
                ]
                accepted.extend(new_spans)
                if not valid:
                    measured = timing.at(ranges[0][0], spec.sample_rate_hz)
                    drafts.append(
                        (
                            "health",
                            {
                                "event_type": "invalid_audio",
                                "channels": (key,),
                                "provenance": Provenance(
                                    tuple(new_spans),
                                    () if measured is None else (measured,),
                                    ()
                                    if timing.received is None
                                    else (timing.received,),
                                ),
                            },
                        )
                    )
                lane.high_water = max(lane.high_water, stop)
                delay = (
                    self.config.max_reorder_delay_ns * spec.sample_rate_hz + 999_999_999
                ) // 1_000_000_000
                frontier = max(lane.watermark, lane.high_water - delay)
                self._finalize_lane(spec, channel, lane, frontier, drafts)
            elif start >= lane.watermark:
                ignored.append(span)
                reasons.append("duplicate")
        count = sum(len(lane.pending) + len(lane.history) for lane in lanes.values())
        if 4 * count > self.config.limits.max_buffer_bytes:
            return self._refuse_resource(
                "max_buffer_bytes",
                4 * count,
                self.config.limits.max_buffer_bytes,
                temporary=True,
                unit="sample_bytes",
                provenance=self._provenance(
                    DataSpan(chunk.interval, chunk.audio.channel_ids),
                    chunk.measured_start,
                    chunk.received_at,
                ),
            )
        for key, values in borrowed.items():
            lane = lanes[key]
            lane.pending = lane.pending.materialized(values, start)
            lane.history = lane.history.materialized(values, start).compacted()
        self._lanes = lanes
        if chunk.quality and (
            chunk.quality.overflow
            or chunk.quality.reported_dropped_frames
            or chunk.quality.reported_dropped_blocks
        ):
            drafts.insert(
                0,
                (
                    "health",
                    {
                        "event_type": "stream_discontinuity",
                        "channels": tuple(
                            ChannelKey(spec.ref.source_id, c)
                            for c in chunk.audio.channel_ids
                        ),
                        "reason_codes": ("adapter_reported_loss",),
                        "provenance": self._provenance(
                            DataSpan(chunk.interval, chunk.audio.channel_ids),
                            chunk.measured_start,
                            chunk.received_at,
                        ),
                    },
                ),
            )
        drafts.extend(self._end_episodes(("max_chunk_bytes", "max_buffer_bytes")))
        outcome = (
            "partially_accepted"
            if accepted and ignored
            else "accepted"
            if accepted
            else "ignored"
        )
        return self._batch(
            drafts,
            outcome=outcome,
            receipts=(
                InputReceipt(
                    chunk.observation_id,
                    tuple(accepted),
                    tuple(ignored),
                    tuple(dict.fromkeys(reasons)),
                ),
            ),
            finalized=tuple(
                SourceWatermark(
                    spec.ref, (c,), lanes[ChannelKey(spec.ref.source_id, c)].watermark
                )
                for c in chunk.audio.channel_ids
            ),
        )

    @staticmethod
    def _provenance(span: DataSpan, measured=None, received=None) -> Provenance:
        return Provenance(
            (span,),
            () if measured is None else (measured,),
            () if received is None else (received,),
        )

    def _finalize_lane(
        self,
        spec: SourceSpec,
        channel: str,
        lane: _Lane,
        frontier: int,
        drafts: list[Draft],
    ) -> None:
        key = ChannelKey(spec.ref.source_id, channel)
        finalized, pending = lane.pending.split(frontier)
        groups = []
        for span in finalized.spans:
            if (
                not groups
                or span.start != groups[-1][-1].stop
                or span.valid != groups[-1][-1].valid
            ):
                groups.append([span])
            else:
                groups[-1].append(span)
        cursor = lane.watermark
        for group in groups:
            first, last = group[0], group[-1]
            a, b = first.start, last.stop
            if a > cursor:
                self._gap(spec, channel, lane, cursor, a, drafts)
            if first.valid:
                provenance = self._provenance(
                    DataSpan(SourceInterval(spec.ref, a, b), (channel,)),
                    first.timing.at(a, spec.sample_rate_hz),
                    first.timing.received,
                )
                if (
                    lane.unavailable
                    and lane.recovery_after is not None
                    and b > lane.recovery_after
                ):
                    recovered_start = max(a, lane.recovery_after)
                    recovered_span = finalized.at(recovered_start)
                    assert recovered_span is not None
                    drafts.append(
                        (
                            "health",
                            {
                                "event_type": "channels_recovered",
                                "channels": (key,),
                                "provenance": self._provenance(
                                    DataSpan(
                                        SourceInterval(spec.ref, recovered_start, b),
                                        (channel,),
                                    ),
                                    recovered_span.timing.at(
                                        recovered_start, spec.sample_rate_hz
                                    ),
                                    recovered_span.timing.received,
                                ),
                                "reason_codes": ("baseline_revalidation_required",),
                            },
                        )
                    )
                    lane.unavailable = False
                    lane.recovery_after = None
                measured = last.timing.at(b - 1, spec.sample_rate_hz)
                stamp = (
                    None
                    if measured is None
                    else ClockStamp(
                        spec.clock_id,
                        measured.ns + 1_000_000_000 // spec.sample_rate_hz,
                    )
                )
                drafts.append(
                    (
                        "salience",
                        {
                            "channels": (key,),
                            "timestamp": stamp,
                            "provenance": provenance,
                        },
                    )
                )
            cursor = b
        if frontier > cursor:
            self._gap(spec, channel, lane, cursor, frontier, drafts)
        lane.pending = pending
        lane.watermark = frontier
        retain = (
            self.config.comparison_retention_ns * spec.sample_rate_hz
        ) // 1_000_000_000
        cutoff = frontier - retain
        lane.history = lane.history.added(finalized.spans).pruned(cutoff)

    def _gap(self, spec, channel, lane, start, stop, drafts) -> None:
        if not lane.unavailable:
            lane.recovery_after = stop
            drafts.append(
                (
                    "health",
                    {
                        "event_type": "channels_unavailable",
                        "channels": (ChannelKey(spec.ref.source_id, channel),),
                        "reason_codes": ("finalized_source_gap",),
                        "provenance": self._provenance(
                            DataSpan(SourceInterval(spec.ref, start, stop), (channel,))
                        ),
                    },
                )
            )
        elif lane.recovery_after is not None:
            lane.recovery_after = max(lane.recovery_after, stop)
        lane.unavailable = True

    def advance_watermark(self, mark: SourceWatermark) -> OutputBatch:
        self._check_open()
        if not isinstance(mark, SourceWatermark):
            raise StructuralInputError("expected SourceWatermark")
        spec = self._source(mark.source)
        self._validate_channels(mark.channel_ids, spec.channel_ids)
        _integer(mark.before_frame, "watermark")
        if spec.ref != mark.source or spec.ref.source_id not in self._active:
            raise StateError("watermark must target an active epoch")
        if any(
            mark.before_frame < self._lanes[ChannelKey(spec.ref.source_id, c)].watermark
            for c in mark.channel_ids
        ):
            raise StructuralInputError("watermarks cannot move backwards")
        lanes = dict(self._lanes)
        drafts: list[Draft] = []
        for channel in mark.channel_ids:
            key = ChannelKey(spec.ref.source_id, channel)
            lanes[key] = replace(lanes[key])
            self._finalize_lane(spec, channel, lanes[key], mark.before_frame, drafts)
            lanes[key].history = lanes[key].history.compacted()
        self._lanes = lanes
        return self._batch(drafts, finalized=(mark,))

    def end_source(self, ref: SourceRef) -> OutputBatch:
        self._check_open()
        spec = self._source(ref)
        if spec.ref != ref or ref.source_id not in self._active:
            raise StateError("source is not active")
        drafts: list[Draft] = []
        marks = []
        for channel in spec.channel_ids:
            lane = self._lanes[ChannelKey(ref.source_id, channel)]
            frontier = max(lane.watermark, lane.high_water)
            self._finalize_lane(spec, channel, lane, frontier, drafts)
            lane.history = lane.history.compacted()
            marks.append(SourceWatermark(ref, (channel,), frontier))
        self._active.remove(ref.source_id)
        return self._batch(drafts, finalized=tuple(marks))

    def _process_unavailable(self, item: UnavailableObservation) -> OutputBatch:
        spec = self._source(item.source)
        current_epoch = item.source == spec.ref
        self._validate_channels(
            item.channel_ids,
            spec.channel_ids if current_epoch else self._identities[spec.ref.source_id],
        )
        if item.reason not in ("missing", "invalid"):
            raise StructuralInputError("unknown unavailability reason")
        if item.observation_id:
            _identity(item.observation_id)
        _stamp(item.measured_start)
        _stamp(item.received_at)
        if (
            current_epoch
            and item.measured_start
            and item.measured_start.clock_id != spec.clock_id
        ):
            raise StructuralInputError("wrong measurement clock")
        if item.interval is not None:
            self._validate_interval(item.interval)
            if item.interval.source != item.source:
                raise StructuralInputError("unavailability interval source mismatch")
        if spec.ref != item.source or item.source.source_id not in self._active:
            return self._batch(
                outcome="ignored",
                receipts=(
                    InputReceipt(
                        item.observation_id, reason_codes=("late_unverifiable",)
                    ),
                ),
            )
        drafts: list[Draft] = []
        for channel in item.channel_ids:
            lane = self._lanes[ChannelKey(spec.ref.source_id, channel)]
            if item.interval is None:
                lane.recovery_after = None
            else:
                lane.recovery_after = max(
                    lane.recovery_after or 0,
                    item.interval.stop_frame,
                )
            lane.unavailable = True
        spans = (
            ()
            if item.interval is None
            else (DataSpan(item.interval, item.channel_ids),)
        )
        drafts.append(
            (
                "health",
                {
                    "event_type": "invalid_audio"
                    if item.reason == "invalid"
                    else "channels_unavailable",
                    "channels": tuple(
                        ChannelKey(spec.ref.source_id, c) for c in item.channel_ids
                    ),
                    "reason_codes": (
                        "adapter_report",
                        "unknown_duration"
                        if item.interval is None
                        else "known_interval",
                    ),
                    "provenance": Provenance(
                        spans,
                        () if item.measured_start is None else (item.measured_start,),
                        () if item.received_at is None else (item.received_at,),
                    ),
                },
            )
        )
        return self._batch(drafts)

    def _configuration_change(
        self, old: SourceSpec | None, new: SourceSpec
    ) -> list[Draft]:
        if old is None:
            return []
        drafts: list[Draft] = []
        channels = tuple(ChannelKey(new.ref.source_id, c) for c in new.channel_ids)
        rate_changed = old.sample_rate_hz != new.sample_rate_hz
        if rate_changed:
            drafts.append(
                ("health", {"event_type": "sample_rate_changed", "channels": channels})
            )
        if old.channel_ids != new.channel_ids and set(old.channel_ids) != set(
            new.channel_ids
        ):
            drafts.append(
                (
                    "health",
                    {"event_type": "channel_membership_changed", "channels": channels},
                )
            )
        for candidate_id, record in tuple(self._baselines.records.items()):
            if not any(c.source_id == new.ref.source_id for c in record.channels):
                continue
            statuses = tuple(
                replace(
                    c,
                    status="unknown" if rate_changed else "revalidate",
                    reason="source epoch compatibility must be declared",
                )
                for c in record.compatibility
            )
            after = replace(record, compatibility=statuses, ready=False)
            self._baselines.records[candidate_id] = after
            drafts.append(
                (
                    "lifecycle",
                    {
                        "event_type": "baseline_compatibility_changed",
                        "before": record,
                        "after": after,
                    },
                )
            )
        return drafts

    def reconfigure_source(self, replacement: SourceSpec) -> OutputBatch:
        """Explicit new epoch; finalize old data, never reinterpret rate in place."""
        self._check_open()
        self._validate_spec(replacement)
        old = self._sources.get(replacement.ref.source_id)
        if old is None or old.ref.source_id not in self._active:
            raise StateError("reconfiguration requires an active source")
        if old.ref == replacement.ref:
            raise StructuralInputError("source reconfiguration requires a new epoch")
        total = sum(len(v) for v in self._identities.values())
        total += len(
            set(replacement.channel_ids) - set(self._identities[old.ref.source_id])
        )
        if total > self.config.limits.max_channels:
            return self._refuse_resource(
                "max_channels",
                total,
                self.config.limits.max_channels,
                operation="reconfigure_source",
            )
        ended = self.end_source(old.ref)
        started = self.start_source(replacement)
        return self._merge(ended, started)

    @staticmethod
    def _merge(first: OutputBatch, second: OutputBatch) -> OutputBatch:
        return OutputBatch(
            second.run_id,
            second.outcome,
            first.salience + second.salience,
            first.health + second.health,
            first.baseline_lifecycle + second.baseline_lifecycle,
            first.input_receipts + second.input_receipts,
            first.refusals + second.refusals,
            second.command_receipt,
            first.finalized + second.finalized,
        )

    def update_geometry(self, geometry: MicrophoneGeometry) -> OutputBatch:
        self._check_open()
        if not isinstance(geometry, MicrophoneGeometry):
            raise StructuralInputError("expected MicrophoneGeometry")
        _identity(geometry.geometry_id)
        _identity(geometry.reference_frame_id)
        if not isinstance(geometry.positions, tuple):
            raise StructuralInputError("geometry positions must be a tuple")
        seen = set()
        for position in geometry.positions:
            self._channel_key(position.channel)
            if position.channel in seen:
                raise StructuralInputError("duplicate geometry channel")
            seen.add(position.channel)
            if (
                not isinstance(position.position_m, tuple)
                or len(position.position_m) != 3
                or any(
                    type(v) not in (int, float) or not math.isfinite(v)
                    for v in position.position_m
                )
            ):
                raise StructuralInputError(
                    "geometry requires finite Cartesian metre positions"
                )
        if geometry == self._geometry:
            return self._batch(outcome="no_change")
        drafts: list[Draft] = [
            (
                "health",
                {
                    "event_type": "geometry_changed",
                    "channels": tuple(seen),
                    "provenance": Provenance(geometry_id=geometry.geometry_id),
                },
            )
        ]
        for candidate_id, record in tuple(self._baselines.records.items()):
            compatibility = tuple(
                replace(c, status="revalidate", reason="explicit geometry change")
                if c.depends_on_geometry
                else c
                for c in record.compatibility
            )
            if compatibility != record.compatibility:
                after = replace(record, compatibility=compatibility, ready=False)
                self._baselines.records[candidate_id] = after
                drafts.append(
                    (
                        "lifecycle",
                        {
                            "event_type": "baseline_compatibility_changed",
                            "before": record,
                            "after": after,
                        },
                    )
                )
        self._geometry = geometry
        return self._batch(drafts)

    def _channel_key(self, channel: ChannelKey) -> None:
        if not isinstance(channel, ChannelKey):
            raise StructuralInputError("expected ChannelKey")
        spec = self._sources.get(channel.source_id)
        if (
            spec is None
            or channel.channel_id not in self._identities[channel.source_id]
        ):
            raise StructuralInputError("unknown source/channel identity")

    def register_feature(self, spec: FeatureSpec) -> OutputBatch:
        """Register prerequisites only; no feature implementation is implied."""
        self._check_open()
        if not isinstance(spec, FeatureSpec):
            raise StructuralInputError("expected FeatureSpec")
        _identity(spec.feature_id)
        if (
            type(spec.continuous_only) is not bool
            or any(
                not isinstance(p, EligibilityPolicy)
                for p in (
                    spec.assessment,
                    spec.established_learning,
                    spec.candidate_learning,
                )
            )
            or (
                spec.alignment is not None
                and not isinstance(spec.alignment, AlignmentPolicy)
            )
        ):
            raise StructuralInputError("invalid feature eligibility/alignment policy")
        if (
            not isinstance(spec.channels, tuple)
            or not spec.channels
            or len(set(spec.channels)) != len(spec.channels)
        ):
            raise StructuralInputError("feature channels must be nonempty and unique")
        for channel in spec.channels:
            self._channel_key(channel)
        if spec.feature_id in self._features:
            raise StructuralInputError("feature ID is already registered")
        if len(spec.channels) > 1:
            _integer(spec.cross_source_wait_horizon_ns, "cross-source wait horizon", 1)
        elif spec.cross_source_wait_horizon_ns is not None:
            _integer(spec.cross_source_wait_horizon_ns, "wait horizon", 1)
        if len(self._features) >= self.config.limits.max_features:
            return self._refuse_resource(
                "max_features",
                len(self._features) + 1,
                self.config.limits.max_features,
                operation="register_feature",
            )
        self._features[spec.feature_id] = spec
        self._sync[spec.feature_id] = _Sync()
        return self._batch()

    def update_alignment(self, evidence: AlignmentEvidence) -> OutputBatch:
        self._check_open()
        if not isinstance(evidence, AlignmentEvidence):
            raise StructuralInputError("expected AlignmentEvidence")
        _identity(evidence.evidence_id)
        _identity(evidence.reference_clock_id)
        if not isinstance(evidence.channels, tuple) or not isinstance(
            evidence.mappings, tuple
        ):
            raise StructuralInputError("alignment channels/mappings must be tuples")
        _integer(evidence.valid_start_ns, "alignment start")
        _integer(evidence.valid_stop_ns, "alignment stop", 1)
        _stamp(evidence.observed_at)
        if evidence.observed_at.ns < self._reference_times.get(
            evidence.reference_clock_id, 0
        ):
            raise StructuralInputError(
                "alignment report predates explicit reference progress"
            )
        if (
            evidence.valid_start_ns >= evidence.valid_stop_ns
            or evidence.observed_at.clock_id != evidence.reference_clock_id
            or not evidence.valid_start_ns
            <= evidence.observed_at.ns
            < evidence.valid_stop_ns
        ):
            raise StructuralInputError(
                "alignment evidence has inconsistent time coordinates"
            )
        if evidence.max_pairwise_error_ns is not None:
            _integer(evidence.max_pairwise_error_ns, "alignment error")
        if not evidence.channels or len(set(evidence.channels)) != len(
            evidence.channels
        ):
            raise StructuralInputError("alignment channels must be unique")
        sources = set()
        for channel in evidence.channels:
            self._channel_key(channel)
            sources.add(channel.source_id)
        mappings = {}
        for mapping in evidence.mappings:
            if not isinstance(mapping, ClockMapping):
                raise StructuralInputError("expected ClockMapping")
            spec = self._source(mapping.source)
            if (
                mapping.source != spec.ref
                or mapping.source.source_id not in self._active
            ):
                raise StructuralInputError("mapping must reference an active epoch")
            if mapping.source_clock_id != spec.clock_id:
                raise StructuralInputError("mapping source clock mismatch")
            if mapping.source.source_id in mappings:
                raise StructuralInputError("duplicate source mapping")
            for value in (mapping.source_anchor_ns, mapping.reference_anchor_ns):
                _integer(value, "mapping anchor")
            for value in (mapping.rate_numerator, mapping.rate_denominator):
                _integer(value, "mapping rate ratio", 1)
            mappings[mapping.source.source_id] = mapping
        if set(mappings) != sources:
            raise StructuralInputError(
                "external mapping required for every participating source"
            )
        affected = [
            f
            for f in self._features.values()
            if f.alignment is not None and set(f.channels) <= set(evidence.channels)
        ]
        for feature in affected:
            old = self._sync[feature.feature_id].evidence
            if old and (
                old.reference_clock_id != evidence.reference_clock_id
                or old.observed_at.ns > evidence.observed_at.ns
            ):
                raise StructuralInputError(
                    "alignment evidence cannot change clock or move backwards"
                )
        drafts: list[Draft] = []
        for feature in affected:
            sync = self._sync[feature.feature_id]
            policy = feature.alignment
            assert policy is not None
            previous = sync.evidence
            broken = previous is not None and (
                evidence.observed_at.ns >= previous.valid_stop_ns
                or {m.source for m in previous.mappings}
                != {m.source for m in evidence.mappings}
            )
            error = evidence.max_pairwise_error_ns
            if sync.enabled and (
                broken or error is None or error > policy.disable_error_ns
            ):
                self._close_alignment_period(
                    feature.feature_id, evidence.observed_at.ns
                )
                sync.enabled = False
                sync.confirmation_start = None
                sync.continuity_revision += 1
                drafts.append(
                    (
                        "health",
                        {
                            "event_type": "alignment_lost",
                            "channels": feature.channels,
                            "feature_ids": (feature.feature_id,),
                        },
                    )
                )
            if not sync.enabled:
                if error is not None and error <= policy.recover_error_ns:
                    if sync.confirmation_start is None or broken:
                        sync.confirmation_start = evidence.observed_at.ns
                    if (
                        evidence.observed_at.ns - sync.confirmation_start
                        >= policy.confirmation_duration_ns
                    ):
                        sync.enabled = True
                        sync.period = _AlignmentPeriod(
                            evidence.observed_at.ns,
                            evidence.valid_stop_ns,
                            evidence,
                            evidence,
                        )
                        drafts.append(
                            (
                                "health",
                                {
                                    "event_type": "alignment_recovered",
                                    "channels": feature.channels,
                                    "feature_ids": (feature.feature_id,),
                                    "reason_codes": ("fresh_continuous_sequence",),
                                },
                            )
                        )
                else:
                    sync.confirmation_start = None
            elif sync.period is not None:
                sync.period = replace(
                    sync.period, stop_ns=evidence.valid_stop_ns, evidence=evidence
                )
            sync.evidence = evidence
            if sync.enabled and sync.period is not None:
                self._remember_alignment_period(
                    feature.feature_id,
                    replace(
                        sync.period,
                        stop_ns=min(sync.period.stop_ns, evidence.observed_at.ns),
                    ),
                )
        return self._batch(drafts)

    def _remember_alignment_period(
        self, feature_id: str, period: _AlignmentPeriod
    ) -> None:
        # At most one immutable supporting period per already bounded pending window.
        for window_id, window in self._windows.items():
            if window.feature_id == feature_id and period.supports(window):
                self._window_alignment.setdefault(window_id, period)

    def _close_alignment_period(self, feature_id: str, stop_ns: int) -> None:
        sync = self._sync[feature_id]
        if sync.period is not None:
            sync.period = replace(
                sync.period, stop_ns=min(sync.period.stop_ns, stop_ns)
            )
            self._remember_alignment_period(feature_id, sync.period)

    def _alignment_for_window(
        self, window: RelationshipWindow
    ) -> _AlignmentPeriod | None:
        period = self._window_alignment.get(window.window_id)
        if period is None:
            period = self._sync[window.feature_id].period
        return period if period is not None and period.supports(window) else None

    def alignment_available(self, feature_id: str, at: ClockStamp) -> bool:
        """Read-only query; expiry transitions occur on explicit reference advance."""
        _stamp(at)
        feature = self._features[feature_id]
        if feature.alignment is None:
            return True
        sync = self._sync[feature_id]
        evidence = sync.evidence
        return bool(
            sync.enabled
            and evidence
            and at.clock_id == evidence.reference_clock_id
            and evidence.valid_start_ns <= at.ns < evidence.valid_stop_ns
            and all(
                self._sources[m.source.source_id].ref == m.source
                and m.source.source_id in self._active
                for m in evidence.mappings
            )
        )

    def feature_status(self, feature_id: str, at: ClockStamp) -> FeatureStatus:
        """Inspect feature-local continuity; no feature values are calculated."""
        sync = self._sync[feature_id]
        return FeatureStatus(
            feature_id,
            self.alignment_available(feature_id, at),
            sync.continuity_revision,
            None if sync.evidence is None else sync.evidence.evidence_id,
        )

    def open_relationship_window(self, window: RelationshipWindow) -> OutputBatch:
        """PROVISIONAL opt-in: caller anchors a finite reference-clock window."""
        self._check_open()
        if not isinstance(window, RelationshipWindow):
            raise StructuralInputError("expected RelationshipWindow")
        for value in (window.window_id, window.feature_id, window.reference_clock_id):
            _identity(value)
        _integer(window.start_ns, "window start")
        _integer(window.stop_ns, "window stop", 1)
        if window.start_ns >= window.stop_ns or window.window_id in self._windows:
            raise StructuralInputError("invalid or already pending relationship window")
        feature = self._features.get(window.feature_id)
        if feature is None or len(feature.channels) < 2:
            raise StructuralInputError("relationship feature must be registered first")
        old_clock = self._feature_clocks.get(window.feature_id)
        if old_clock is not None and old_clock != window.reference_clock_id:
            raise StructuralInputError(
                "a registered feature uses one explicit reference clock"
            )
        horizon = feature.cross_source_wait_horizon_ns
        assert horizon is not None
        if window.stop_ns + horizon <= self._reference_times.get(
            window.reference_clock_id, -1
        ):
            raise StructuralInputError("window deadline is already finalized")
        if len(self._windows) >= self.config.limits.max_relationship_windows:
            return self._refuse_resource(
                "max_relationship_windows",
                len(self._windows) + 1,
                self.config.limits.max_relationship_windows,
                temporary=True,
                operation="open_relationship_window",
            )
        self._windows[window.window_id] = window
        self._feature_clocks[window.feature_id] = window.reference_clock_id
        sync = self._sync[window.feature_id]
        if sync.period is not None:
            settled = sync.period.stop_ns
            if sync.enabled and sync.evidence is not None:
                settled = min(settled, sync.evidence.observed_at.ns)
            self._remember_alignment_period(
                window.feature_id, replace(sync.period, stop_ns=settled)
            )
        return self._batch(self._end_episodes(("max_relationship_windows",)))

    def advance_reference_time(self, at: ClockStamp) -> OutputBatch:
        """Explicit finite-window progress; never reads a microphone or a timer."""
        self._check_open()
        _stamp(at)
        known = {w.reference_clock_id for w in self._windows.values()}
        known.update(self._feature_clocks.values())
        known.update(
            s.evidence.reference_clock_id for s in self._sync.values() if s.evidence
        )
        if at.clock_id not in known:
            raise StructuralInputError(
                "reference clock must belong to a window or alignment"
            )
        if at.ns < self._reference_times.get(at.clock_id, 0):
            raise StructuralInputError("reference progress cannot move backwards")
        self._reference_times[at.clock_id] = at.ns
        drafts: list[Draft] = []
        for feature_id, sync in self._sync.items():
            evidence = sync.evidence
            if (
                evidence
                and evidence.reference_clock_id == at.clock_id
                and at.ns >= evidence.valid_stop_ns
            ):
                sync.confirmation_start = None
                if sync.enabled:
                    self._close_alignment_period(feature_id, evidence.valid_stop_ns)
                    sync.enabled = False
                    sync.continuity_revision += 1
                    drafts.append(
                        (
                            "health",
                            {
                                "event_type": "alignment_lost",
                                "channels": self._features[feature_id].channels,
                                "feature_ids": (feature_id,),
                                "reason_codes": ("alignment_expired",),
                            },
                        )
                    )
        released = False
        for window_id, window in tuple(self._windows.items()):
            feature = self._features[window.feature_id]
            horizon = feature.cross_source_wait_horizon_ns
            assert horizon is not None
            if (
                window.reference_clock_id != at.clock_id
                or at.ns < window.stop_ns + horizon
            ):
                continue
            reasons = ["not_implemented"]
            period = self._alignment_for_window(window)
            if feature.alignment is not None and period is None:
                reasons.append("alignment_unavailable")
            reasons.append("relationship_coverage_not_evaluated")
            supporting = (
                ()
                if period is None
                else tuple(
                    dict.fromkeys(
                        (
                            period.initial_evidence.evidence_id,
                            period.evidence.evidence_id,
                        )
                    )
                )
            )
            source_clocks = (
                ()
                if period is None
                else tuple(
                    (mapping.source, mapping.source_clock_id)
                    for mapping in period.evidence.mappings
                )
            )
            drafts.append(
                (
                    "salience",
                    {
                        "channels": feature.channels,
                        "timestamp": ClockStamp(at.clock_id, window.stop_ns),
                        "unavailable_reasons": tuple(reasons),
                        "provenance": Provenance(
                            alignment_evidence_ids=supporting,
                            source_clocks=source_clocks,
                        ),
                    },
                )
            )
            del self._windows[window_id]
            self._window_alignment.pop(window_id, None)
            released = True
        if released:
            drafts.extend(self._end_episodes(("max_relationship_windows",)))
        # Prune completed reference-clock bookkeeping; retained evidence is feature-bounded.
        keep = {w.reference_clock_id for w in self._windows.values()}
        keep.update(self._feature_clocks.values())
        keep.update(
            s.evidence.reference_clock_id for s in self._sync.values() if s.evidence
        )
        self._reference_times = {
            k: v for k, v in self._reference_times.items() if k in keep
        }
        return self._batch(drafts)

    def candidates(self) -> tuple[BaselineRecord, ...]:
        """Metadata records only; their existence does not imply acoustic learning."""
        return tuple(self._baselines.records.values())

    @staticmethod
    def _compatibility(items: tuple[BaselineCompatibility, ...]) -> None:
        if not isinstance(items, tuple):
            raise StructuralInputError("compatibility must be a tuple")
        seen = set()
        for item in items:
            if not isinstance(item, BaselineCompatibility):
                raise StructuralInputError("expected BaselineCompatibility")
            _identity(item.feature_id)
            if (
                item.feature_id in seen
                or item.status
                not in (
                    "compatible",
                    "revalidate",
                    "unavailable",
                    "incompatible",
                    "unknown",
                )
                or type(item.depends_on_geometry) is not bool
            ):
                raise StructuralInputError("invalid compatibility metadata")
            seen.add(item.feature_id)
            _identity(item.reason)

    def register_candidate(
        self,
        channels: tuple[ChannelKey, ...],
        *,
        compatibility: tuple[BaselineCompatibility, ...] = (),
    ) -> OutputBatch:
        """Explicit extension/test metadata registration; creates no learned model."""
        self._check_open()
        if (
            not isinstance(channels, tuple)
            or not channels
            or len(set(channels)) != len(channels)
        ):
            raise StructuralInputError("candidate channels must be nonempty and unique")
        for channel in channels:
            self._channel_key(channel)
        self._compatibility(compatibility)
        if len(compatibility) > self.config.limits.max_features:
            return self._refuse_resource(
                "max_features",
                len(compatibility),
                self.config.limits.max_features,
                operation="register_candidate",
            )
        if len(self._baselines.administration) >= self.config.limits.max_baselines:
            return self._refuse_resource(
                "max_baselines",
                len(self._baselines.administration) + 1,
                self.config.limits.max_baselines,
                operation="register_candidate",
            )
        candidate_id = new_id()
        record = BaselineRecord(
            new_id(), candidate_id, channels, compatibility=compatibility
        )
        self._baselines.records[candidate_id] = record
        self._baselines.administration[candidate_id] = (0, False)
        return self._batch(
            [("lifecycle", {"event_type": "candidate_created", "after": record})]
        )

    def report_readiness(self, evidence: ReadinessEvidence) -> OutputBatch:
        """Consume an explicit extension decision; no matching/stability heuristic."""
        self._check_open()
        if not isinstance(evidence, ReadinessEvidence):
            raise StructuralInputError("expected ReadinessEvidence")
        self._compatibility(evidence.compatibility)
        _integer(evidence.evidence_revision, "evidence revision")
        if (
            type(evidence.ready) is not bool
            or type(evidence.materially_changed) is not bool
            or type(evidence.explicit_review) is not bool
        ):
            raise StructuralInputError("readiness flags must be bool")
        before = self._baselines.records.get(evidence.candidate_id)
        if before is None:
            return self._batch(
                outcome="refused", refusals=(Refusal("readiness", "unknown_candidate"),)
            )
        if evidence.evidence_revision < before.evidence_revision:
            raise StructuralInputError("evidence revision cannot move backwards")
        if len(evidence.compatibility) > self.config.limits.max_features:
            return self._refuse_resource(
                "max_features",
                len(evidence.compatibility),
                self.config.limits.max_features,
                operation="readiness",
            )
        proposal = before.proposal_id
        suppressed = proposal is not None and proposal == before.rejected_proposal_id
        ready = evidence.ready and (
            not suppressed
            or evidence.explicit_review
            or (
                evidence.materially_changed
                and evidence.evidence_revision > before.evidence_revision
            )
        )
        newly_ready = ready and (not before.ready or proposal is None)
        if newly_ready:
            proposal = new_id()
        activity = before.activity
        if (
            activity == "reactivating"
            and evidence.compatibility
            and all(c.status == "compatible" for c in evidence.compatibility)
        ):
            activity = "active"
        after = replace(
            before,
            evidence_revision=evidence.evidence_revision,
            ready=ready,
            proposal_id=proposal,
            activity=activity,
            compatibility=evidence.compatibility,
        )
        self._baselines.records[evidence.candidate_id] = after
        drafts: list[Draft] = []
        if after.compatibility != before.compatibility:
            drafts.append(
                (
                    "lifecycle",
                    {
                        "event_type": "baseline_compatibility_changed",
                        "before": before,
                        "after": after,
                    },
                )
            )
        if activity != before.activity:
            drafts.append(
                (
                    "lifecycle",
                    {
                        "event_type": "baseline_reactivated",
                        "before": before,
                        "after": after,
                    },
                )
            )
        if newly_ready:
            drafts.append(
                (
                    "lifecycle",
                    {"event_type": "candidate_ready", "before": before, "after": after},
                )
            )
        output = self._batch(
            drafts, outcome="accepted" if after != before else "no_change"
        )
        if (
            self.config.promotion_mode == "automatic"
            and ready
            and not after.promotion_blocked
            and after.role == "candidate"
            and after.activity == "active"
        ):
            output = self._merge(
                output,
                self.command(
                    CandidateCommand(
                        new_id(),
                        after.candidate_id,
                        after.control_revision,
                        "approve",
                        proposal,
                        "explicit_readiness_extension",
                    )
                ),
            )
        return output

    def command(self, command: CandidateCommand) -> OutputBatch:
        self._check_open()
        if isinstance(command, CandidateCommand):
            _integer(command.expected_control_revision, "expected_control_revision")
            for value in (command.command_id, command.candidate_id):
                _identity(value)
            if command.proposal_id is not None:
                _identity(command.proposal_id)
            if command.actor_id is not None:
                _identity(command.actor_id)
        plan = self._baselines.plan(command)
        if plan.prior is not None:
            return self._batch(outcome="no_change", command_receipt=plan.prior)
        drafts: list[Draft] = []
        if plan.event_type:
            drafts.append(
                (
                    "lifecycle",
                    {
                        "event_type": plan.event_type,
                        "before": plan.before,
                        "after": plan.after,
                        "command_id": command.command_id,
                        "candidate_id": command.candidate_id,
                        "control_revision": plan.revision,
                    },
                )
            )
        output = self._batch(
            drafts,
            outcome="refused" if plan.code else "accepted",
            refusals=()
            if plan.code is None
            else (Refusal("candidate_command", plan.code),),
        )
        receipt = self._baselines.commit(
            plan, tuple(e.header.event_id for e in output.ordered)
        )
        return replace(output, command_receipt=receipt)

    def _idle(self) -> None:
        self._check_open()
        if self._active:
            raise StateError("stop all acquisition sources before resetting")

    def reset_learning(self) -> OutputBatch:
        self._idle()
        self._baselines.records.clear()
        self._lanes.clear()
        self._features.clear()
        self._sync.clear()
        self._windows.clear()
        self._window_alignment.clear()
        self._reference_times.clear()
        self._feature_clocks.clear()
        drafts = self._end_episodes(("max_relationship_windows",))
        drafts.append(("lifecycle", {"event_type": "learning_reset"}))
        return self._batch(drafts)

    def factory_reset(self) -> OutputBatch:
        self._idle()
        self._sources.clear()
        self._identities.clear()
        self._lanes.clear()
        self._baselines = BaselineStore(self.config.limits.max_receipts)
        self._features.clear()
        self._sync.clear()
        self._windows.clear()
        self._window_alignment.clear()
        self._reference_times.clear()
        self._feature_clocks.clear()
        self._geometry = None
        drafts = self._end_episodes()
        drafts.append(("lifecycle", {"event_type": "factory_reset"}))
        return self._batch(drafts)

    def save_baselines(self, path: Path) -> None:
        self._check_open()
        from .persistence import save

        save(self, Path(path))

    @classmethod
    def from_baselines(
        cls, path: Path, *, config: DetectorConfig
    ) -> AudioSalienceDetector:
        from .persistence import load

        detector = cls(config)
        load(detector, Path(path))
        return detector

    def close(self) -> OutputBatch:
        if self._closed:
            return self._batch(outcome="no_change")
        output = self._batch(outcome="no_change")
        for source_id in tuple(self._active):
            output = self._merge(output, self.end_source(self._sources[source_id].ref))
        # Abort pending relationships without inventing reference-clock progress.
        for window in tuple(self._windows.values()):
            output = self._merge(
                output,
                self._batch(
                    [
                        (
                            "salience",
                            {
                                "channels": self._features[window.feature_id].channels,
                                "unavailable_reasons": (
                                    "not_implemented",
                                    "source_closed_before_finalization",
                                ),
                            },
                        )
                    ]
                ),
            )
        self._windows.clear()
        self._window_alignment.clear()
        output = self._merge(output, self._batch(self._end_episodes()))
        self._lanes.clear()
        self._closed = True
        return output
