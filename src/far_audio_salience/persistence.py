"""Version-1 bounded JSON metadata snapshots; never pickle or executable data."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from .contracts import (
    BaselineCompatibility,
    BaselineRecord,
    CandidateCommand,
    ChannelKey,
    CommandReceipt,
    MicrophoneGeometry,
    MicrophonePosition,
    SnapshotError,
    SourceRef,
    SourceSpec,
    StructuralInputError,
)

if TYPE_CHECKING:
    from .detector import AudioSalienceDetector


def save(detector: AudioSalienceDetector, path: Path) -> None:
    store = detector._baselines
    payload = {
        "format": "far-contract-metadata",
        "version": 1,
        "sources": [asdict(s) for s in detector.sources],
        "channel_catalog": [
            {"source_id": k, "channel_ids": list(v)}
            for k, v in detector._identities.items()
        ],
        "baselines": [asdict(r) for r in store.records.values()],
        "administration": [
            {"candidate_id": key, "revision": value[0], "blocked": value[1]}
            for key, value in store.administration.items()
        ],
        "receipts": [
            {"command": asdict(c), "receipt": asdict(r)}
            for c, r in store.receipts.values()
        ],
        "geometry": None if detector._geometry is None else asdict(detector._geometry),
    }
    from ._acoustic_snapshot import encode

    acoustic = (
        encode(detector._acoustic)
        if detector._acoustic is not None
        else detector._saved_acoustic
    )
    if acoustic is not None:
        payload["version"] = 2
        payload["acoustic"] = acoustic
    data = json.dumps(
        payload, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if len(data) > detector.config.limits.max_snapshot_bytes:
        raise SnapshotError("snapshot exceeds max_snapshot_bytes; nothing was written")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".far-", delete=False
        ) as f:
            temporary = Path(f.name)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SnapshotError("duplicate JSON key")
        value[key] = item
    return value


def _invalid_constant(value):
    raise SnapshotError(f"nonfinite JSON value: {value}")


def _keys(value, fields):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise SnapshotError("invalid snapshot object fields")


def _id(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 256:
        raise SnapshotError("invalid snapshot identity")
    return value


def _int(value):
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise SnapshotError("invalid snapshot revision")
    return value


def _bool(value):
    if type(value) is not bool:
        raise SnapshotError("invalid snapshot boolean")
    return value


def _list(value, bound):
    if not isinstance(value, list) or len(value) > bound:
        raise SnapshotError("snapshot collection exceeds configured bounds")
    return value


def _channel(value):
    _keys(value, ("source_id", "channel_id"))
    return ChannelKey(_id(value["source_id"]), _id(value["channel_id"]))


def load(detector: AudioSalienceDetector, path: Path) -> None:
    """Validate temporary metadata; never partially restore on failure."""
    try:
        with path.open("rb") as f:
            data = f.read(detector.config.limits.max_snapshot_bytes + 1)
        if len(data) > detector.config.limits.max_snapshot_bytes:
            raise SnapshotError("snapshot exceeds max_snapshot_bytes")
        payload = json.loads(
            data, object_pairs_hook=_object, parse_constant=_invalid_constant
        )
        _restore(detector, payload)
    except SnapshotError:
        raise
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        RecursionError,
        StructuralInputError,
        OverflowError,
    ) as error:
        raise SnapshotError(f"invalid snapshot: {error}") from error


def _restore(detector, payload):
    acoustic = None
    if (
        isinstance(payload, dict)
        and type(payload.get("version")) is int
        and payload["version"] == 2
    ):
        payload = dict(payload)
        if "acoustic" not in payload:
            raise SnapshotError("missing acoustic parameters")
        acoustic = payload.pop("acoustic")
        if acoustic is None:
            raise SnapshotError("invalid acoustic parameters")
        payload["version"] = 1
    _keys(
        payload,
        (
            "format",
            "version",
            "sources",
            "channel_catalog",
            "baselines",
            "administration",
            "receipts",
            "geometry",
        ),
    )
    if (
        payload["format"] != "far-contract-metadata"
        or type(payload["version"]) is not int
        or payload["version"] != 1
    ):
        raise SnapshotError("unsupported snapshot format/version")
    limits = detector.config.limits
    sources = {}
    for value in _list(payload["sources"], limits.max_sources):
        _keys(value, SourceSpec.__dataclass_fields__)
        _keys(value["ref"], SourceRef.__dataclass_fields__)
        ref = SourceRef(**{k: _id(v) for k, v in value["ref"].items()})
        spec = SourceSpec(
            ref,
            _id(value["label"]),
            value["sample_rate_hz"],
            _id(value["clock_id"]),
            tuple(_id(c) for c in _list(value["channel_ids"], limits.max_channels)),
        )
        detector._validate_spec(spec)
        if ref.source_id in sources:
            raise SnapshotError("duplicate source identity")
        sources[ref.source_id] = spec
    if sum(len(s.channel_ids) for s in sources.values()) > limits.max_channels:
        raise SnapshotError("source catalog exceeds channel bound")

    identities = {}
    for item in _list(payload["channel_catalog"], limits.max_sources):
        _keys(item, ("source_id", "channel_ids"))
        key = _id(item["source_id"])
        ids = tuple(_id(c) for c in _list(item["channel_ids"], limits.max_channels))
        if (
            key not in sources
            or key in identities
            or not ids
            or len(ids) != len(set(ids))
        ):
            raise SnapshotError("invalid stable channel catalog")
        if not set(sources[key].channel_ids) <= set(ids):
            raise SnapshotError("current channels absent from identity catalog")
        identities[key] = ids
    if (
        set(identities) != set(sources)
        or sum(map(len, identities.values())) > limits.max_channels
    ):
        raise SnapshotError("invalid channel catalog bounds")

    def validate_channels(channels):
        if not channels or len(set(channels)) != len(channels):
            raise SnapshotError("empty/duplicate baseline channel identities")
        if any(
            c.source_id not in sources or c.channel_id not in identities[c.source_id]
            for c in channels
        ):
            raise SnapshotError("baseline references unknown source/channel")

    administration = {}
    for value in _list(payload["administration"], limits.max_baselines):
        _keys(value, ("candidate_id", "revision", "blocked"))
        key = _id(value["candidate_id"])
        if key in administration:
            raise SnapshotError("duplicate administrative candidate identity")
        administration[key] = (_int(value["revision"]), _bool(value["blocked"]))
    records = {}
    baseline_ids = set()
    for value in _list(payload["baselines"], limits.max_baselines):
        _keys(value, BaselineRecord.__dataclass_fields__)
        fields = dict(value)
        fields["channels"] = tuple(
            _channel(c) for c in _list(value["channels"], limits.max_channels)
        )
        validate_channels(fields["channels"])
        items = []
        for c in _list(value["compatibility"], limits.max_features):
            _keys(c, BaselineCompatibility.__dataclass_fields__)
            items.append(BaselineCompatibility(**c))
        fields["compatibility"] = tuple(items)
        detector._compatibility(fields["compatibility"])
        record = BaselineRecord(**fields)
        for v in (record.baseline_id, record.candidate_id):
            _id(v)
        for v in (record.control_revision, record.evidence_revision):
            _int(v)
        for v in (record.ready, record.promotion_blocked):
            _bool(v)
        for v in (record.proposal_id, record.rejected_proposal_id):
            if v is not None:
                _id(v)
        if record.role not in ("candidate", "established") or record.activity not in (
            "active",
            "archived",
            "reactivating",
        ):
            raise SnapshotError("unknown baseline lifecycle state")
        if record.candidate_id in records or record.baseline_id in baseline_ids:
            raise SnapshotError("duplicate baseline/candidate identity")
        if administration.get(record.candidate_id) != (
            record.control_revision,
            record.promotion_blocked,
        ):
            raise SnapshotError("candidate/control metadata mismatch")
        records[record.candidate_id] = record
        baseline_ids.add(record.baseline_id)
    receipts = []
    command_ids = set()
    for value in _list(payload["receipts"], limits.max_receipts):
        _keys(value, ("command", "receipt"))
        _keys(value["command"], CandidateCommand.__dataclass_fields__)
        _keys(value["receipt"], CommandReceipt.__dataclass_fields__)
        command = CandidateCommand(**value["command"])
        fields = dict(value["receipt"])
        fields["event_ids"] = tuple(_id(v) for v in _list(fields["event_ids"], 1))
        receipt = CommandReceipt(**fields)
        for v in (
            command.command_id,
            command.candidate_id,
            receipt.command_id,
            receipt.candidate_id,
        ):
            _id(v)
        _int(command.expected_control_revision)
        _int(receipt.control_revision)
        for v in (command.proposal_id, command.actor_id):
            if v is not None:
                _id(v)
        if command.action not in (
            "approve",
            "reject",
            "block",
            "unblock",
            "archive",
            "reactivate",
        ):
            raise SnapshotError("unknown command action")
        if receipt.outcome not in ("applied", "no_change", "refused"):
            raise SnapshotError("unknown command receipt outcome")
        if (
            command.command_id != receipt.command_id
            or command.candidate_id != receipt.candidate_id
        ):
            raise SnapshotError("command/receipt identity mismatch")
        if command.command_id in command_ids:
            raise SnapshotError("duplicate command receipt identity")
        if (
            receipt.control_revision
            > administration.get(command.candidate_id, (0, False))[0]
        ):
            raise SnapshotError("receipt revision exceeds persistent guard")
        command_ids.add(command.command_id)
        receipts.append((command, receipt))
    geometry = None
    if payload["geometry"] is not None:
        g = payload["geometry"]
        _keys(g, MicrophoneGeometry.__dataclass_fields__)
        positions = []
        for p in _list(g["positions"], limits.max_channels):
            _keys(p, ("channel", "position_m"))
            channel = _channel(p["channel"])
            validate_channels((channel,))
            xyz = _list(p["position_m"], 3)
            if len(xyz) != 3 or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in xyz
            ):
                raise SnapshotError("invalid geometry position")
            positions.append(MicrophonePosition(channel, tuple(xyz)))
        if len({p.channel for p in positions}) != len(positions):
            raise SnapshotError("duplicate geometry channel")
        geometry = MicrophoneGeometry(
            _id(g["geometry_id"]), _id(g["reference_frame_id"]), tuple(positions)
        )
    from ._acoustic_snapshot import restore, validate

    engine = None
    if acoustic is not None:
        acoustic = validate(
            acoustic, identities, records, limits, detector.config.acoustic
        )
        if detector._acoustic is not None:
            engine = detector._acoustic.fork(set())
            restore(engine, acoustic)
            try:
                engine.reserve([], limits)
            except Exception as error:
                raise SnapshotError(
                    "acoustic restore exceeds resource limits"
                ) from error
    if engine is not None:
        detector._acoustic = engine
    detector._saved_acoustic = acoustic
    detector._sources = sources
    detector._identities = identities
    detector._baselines.records = records
    detector._baselines.administration = administration
    for command, receipt in receipts:
        detector._baselines.receipts[command.command_id] = (command, receipt)
    detector._geometry = geometry
