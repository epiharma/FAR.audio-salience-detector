"""Private immutable acquisition/comparison evidence (ADR-001).

Indexes contain sorted, disjoint half-open spans. No caller memoryview is stored.
Valid spans with no backing exist only in uncommitted admission plans; after
resource preflight they are materialized from the current borrowed input.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from heapq import merge
from operator import attrgetter

from .contracts import ClockStamp

# Implementation parameters, not public resource limits or acoustic thresholds.
BLOCK_BYTES = 16_384
BLOCK_FRAMES = BLOCK_BYTES // 4
COMPACTION_FRACTION = 0.5


@dataclass(frozen=True, slots=True)
class _Timing:
    origin: int
    measured: ClockStamp | None
    received: ClockStamp | None

    def at(self, frame: int, rate: int) -> ClockStamp | None:
        if self.measured is None:
            return None
        return ClockStamp(
            self.measured.clock_id,
            self.measured.ns + (frame - self.origin) * 1_000_000_000 // rate,
        )


@dataclass(frozen=True, slots=True)
class _Span:
    start: int
    stop: int
    valid: bool
    timing: _Timing
    backing: bytes | None = None
    offset: int = 0

    def clipped(self, start: int, stop: int) -> _Span:
        if start == self.start and stop == self.stop:
            return self
        return replace(
            self, start=start, stop=stop, offset=self.offset + start - self.start
        )


@dataclass(frozen=True, slots=True)
class _Index:
    spans: tuple[_Span, ...] = ()
    starts: tuple[int, ...] = ()
    count: int = 0

    def __len__(self) -> int:
        """Logical sample positions, including invalid markers, not descriptors."""
        return self.count

    @classmethod
    def build(cls, spans: Iterable[_Span]) -> _Index:
        values = tuple(spans)
        return cls(
            values,
            tuple(s.start for s in values),
            sum(s.stop - s.start for s in values),
        )

    def at(self, frame: int) -> _Span | None:
        i = bisect_right(self.starts, frame) - 1
        return self.spans[i] if i >= 0 and frame < self.spans[i].stop else None

    def overlaps(self, start: int, stop: int) -> Iterator[_Span]:
        i = max(0, bisect_right(self.starts, start) - 1)
        while i < len(self.spans):
            span = self.spans[i]
            if span.start >= stop:
                break
            if span.stop > start:
                yield span
            i += 1

    def conflicts(self, start: int, values: memoryview) -> bool:
        for span in self.overlaps(start, start + len(values)):
            if not span.valid:
                continue
            assert span.backing is not None
            a, b = max(start, span.start), min(start + len(values), span.stop)
            offset = span.offset + a - span.start
            # Format-f memoryview equality is numeric, including signed zero.
            # Both sides are valid finite float32; no hash replaces exact evidence.
            stored = memoryview(span.backing).cast("f")[offset : offset + b - a]
            if stored != values[a - start : b - start]:
                return True
        return False

    def uncovered(self, start: int, stop: int) -> tuple[tuple[int, int], ...]:
        ranges = []
        cursor = start
        for span in self.overlaps(start, stop):
            if cursor < span.start:
                ranges.append((cursor, span.start))
            cursor = max(cursor, span.stop)
        if cursor < stop:
            ranges.append((cursor, stop))
        return tuple(ranges)

    def covers(self, start: int, stop: int) -> bool:
        return not self.uncovered(start, stop)

    def added(self, spans: tuple[_Span, ...]) -> _Index:
        if not spans:
            return self
        if not self.spans or self.spans[-1].stop <= spans[0].start:
            return self.build((*self.spans, *spans))
        return self.build(merge(self.spans, spans, key=attrgetter("start")))

    def split(self, frontier: int) -> tuple[_Index, _Index]:
        """Transfer a chronological prefix, splitting at most one descriptor."""
        if not self.spans or frontier <= self.spans[0].start:
            return _Index(), self
        if frontier >= self.spans[-1].stop:
            return self, _Index()
        i = bisect_left(self.starts, frontier)
        left, right = self.spans[:i], self.spans[i:]
        if left and left[-1].stop > frontier:
            boundary = left[-1]
            left = (*left[:-1], boundary.clipped(boundary.start, frontier))
            right = (boundary.clipped(frontier, boundary.stop), *right)
        return self.build(left), self.build(right)

    def pruned(self, cutoff: int) -> _Index:
        if not self.spans or cutoff <= self.spans[0].start:
            return self
        if cutoff >= self.spans[-1].stop:
            return _Index()
        i = max(0, bisect_right(self.starts, cutoff) - 1)
        if self.spans[i].stop <= cutoff:
            i += 1
        spans = self.spans[i:]
        if spans[0].start < cutoff:
            spans = (spans[0].clipped(cutoff, spans[0].stop), *spans[1:])
        return self.build(spans)

    def materialized(self, values: memoryview, origin: int) -> _Index:
        """Copy only surviving, newly admitted valid ranges after preflight."""
        if not any(s.valid and s.backing is None for s in self.spans):
            return self
        return self.build(
            replace(
                s,
                backing=values[s.start - origin : s.stop - origin].tobytes(),
                offset=0,
            )
            if s.valid and s.backing is None
            else s
            for s in self.spans
        )

    def compacted(self) -> _Index:
        """Copy a boundary only after at least half its backing prefix expires."""
        if not self.spans:
            return self
        head = self.spans[0]
        if (
            head.backing is None
            or head.offset * 4 < len(head.backing) * COMPACTION_FRACTION
        ):
            return self
        data = head.backing[
            head.offset * 4 : (head.offset + head.stop - head.start) * 4
        ]
        return self.build((replace(head, backing=data, offset=0), *self.spans[1:]))


def _planned(
    ranges: tuple[tuple[int, int], ...], valid: bool, timing: _Timing
) -> tuple[_Span, ...]:
    """Bound valid backing owners; an invalid interval needs no sample payload."""
    spans = []
    for start, stop in ranges:
        step = BLOCK_FRAMES if valid else stop - start
        spans.extend(
            _Span(a, min(a + step, stop), valid, timing)
            for a in range(start, stop, step)
        )
    return tuple(spans)
