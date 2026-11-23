"""Strict scalar acoustic parameter codec. No transient evidence is serialized."""

import math
from dataclasses import asdict

from .contracts import ChannelKey, SnapshotError


def encode(engine):
    return {
        "schema": "far-acoustic-parameters-v1",
        "channels": [
            {
                "channel": asdict(key),
                "stats": {fid: asdict(stat) for fid, stat in stats.items()},
            }
            for key, stats in engine.models.items()
        ],
        "candidates": [
            {
                "candidate_id": cid,
                "channel": asdict(value["channel"]),
                "stats": {fid: asdict(stat) for fid, stat in value["stats"].items()},
            }
            for cid, value in engine.candidates.items()
        ],
    }


def validate(value, identities, records, limits, acoustic):
    from .persistence import _channel, _id, _keys, _list

    _keys(value, ("schema", "channels", "candidates"))
    if value["schema"] != "far-acoustic-parameters-v1":
        raise SnapshotError("unsupported acoustic parameter schema")
    count = 0
    seen = set()
    for family, bound in [
        ("channels", limits.max_channels),
        ("candidates", limits.max_baselines),
    ]:
        for row in _list(value[family], bound):
            _keys(
                row,
                ("channel", "stats")
                if family == "channels"
                else ("candidate_id", "channel", "stats"),
            )
            key = _channel(row["channel"])
            if key.channel_id not in identities.get(key.source_id, ()):
                raise SnapshotError("unknown acoustic channel")
            identity = key if family == "channels" else _id(row["candidate_id"])
            if (family, identity) in seen:
                raise SnapshotError("duplicate acoustic model")
            seen.add((family, identity))
            if family == "candidates" and (
                identity not in records or records[identity].channels != (key,)
            ):
                raise SnapshotError("acoustic candidate identity mismatch")
            stats = row["stats"]
            if not isinstance(stats, dict):
                raise SnapshotError("invalid acoustic statistics")
            count += len(stats)
            if count > (acoustic.max_parameter_features if acoustic else 4096):
                raise SnapshotError("too many acoustic parameters")
            for fid, stat in stats.items():
                if (
                    not isinstance(fid, str)
                    or len(fid) != 64
                    or any(c not in "0123456789abcdef" for c in fid)
                ):
                    raise SnapshotError("invalid acoustic feature schema")
                _keys(
                    stat,
                    (
                        "mean",
                        "variance",
                        "exposure",
                        "expected",
                        "rhythm_rate",
                        "proposal_exposure",
                    ),
                )
                if type(stat["expected"]) is not bool:
                    raise SnapshotError("invalid expected presence")
                for name in (
                    "mean",
                    "variance",
                    "exposure",
                    "rhythm_rate",
                    "proposal_exposure",
                ):
                    v = stat[name]
                    if name == "rhythm_rate" and v is None:
                        continue
                    if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                        raise SnapshotError("invalid acoustic statistic")
                if stat["mean"] > 1.01 or stat["variance"] > 1.01:
                    raise SnapshotError("invalid normalized power model")
    return value


def restore(engine, value):
    from ._acoustic import _Stat

    engine.models = {
        ChannelKey(**r["channel"]): {f: _Stat(**s) for f, s in r["stats"].items()}
        for r in value["channels"]
    }
    engine.candidates = {
        r["candidate_id"]: {
            "channel": ChannelKey(**r["channel"]),
            "stats": {f: _Stat(**s) for f, s in r["stats"].items()},
        }
        for r in value["candidates"]
    }
