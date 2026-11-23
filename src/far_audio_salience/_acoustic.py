"""Transactional per-channel experimental acoustic model (DESIGN-003).

Only finalized evidence is evaluated. Owned ring buffers and scalar models are
forked before evaluation; no operation mutates the live engine until admission.
"""

import math
from collections import deque
from copy import copy, deepcopy
from dataclasses import dataclass, field, replace

from ._acoustic_features import Ring, SpectralPlan, numpy_module
from ._rhythm import Rhythm
from .contracts import (
    AcousticBandChange,
    AcousticBandPower,
    AcousticChange,
    AcousticQuantity,
    AcousticWindow,
    BaselineCompatibility,
    BaselineRecord,
    ChannelKey,
    DataSpan,
    Provenance,
    SourceInterval,
    new_id,
)
from .eligibility import (
    EligibilityPolicy,
    FeatureSpec,
    evaluate_eligibility,
    feature_eligibility,
)


class AcousticLimit(Exception):
    def __init__(self, resource, observed, limit):
        self.resource, self.observed, self.limit = resource, observed, limit


@dataclass
class _Stat:
    mean: float = 0.0
    variance: float = 0.0
    exposure: float = 0.0
    expected: bool = False
    rhythm_rate: float | None = None
    proposal_exposure: float = 0.0

    def learn(self, power, dt, config):
        alpha = (
            1.0 if self.exposure == 0 else -math.expm1(-dt / config.adaptation_seconds)
        )
        delta = power - self.mean
        self.mean += alpha * delta
        self.variance = (1 - alpha) * (self.variance + alpha * delta * delta)
        self.exposure += dt
        self.expected = (
            self.mean >= config.absolute_power_floor
            and self.exposure >= config.maturity_seconds
        )


@dataclass
class _Activity:
    initialized: bool = False
    active: bool = False
    start: int = 0
    rise: int | None = None
    fall: int | None = None
    onset_id: str | None = None
    absent_power: float | None = None
    observed_active: bool = False
    presence_ambiguous: bool = False
    background_share: float = 0.0
    relative: bool = False
    relative_share: float = 0.0
    relative_power: float = 0.0
    absolute_absence: bool = False


@dataclass
class _State:
    source: object
    rate: int
    plan: SpectralPlan
    ring: Ring
    cursor: int | None = None
    next_end: int = 0
    guard: int = 0
    continuity: int = 0
    exposure: float = 0.0
    bands: dict = field(default_factory=dict)
    rhythm: Rhythm = field(default_factory=Rhythm)
    changes: dict = field(default_factory=dict)
    group: dict = field(default_factory=dict)
    transitions: dict = field(default_factory=dict)
    levels: deque = field(default_factory=deque)
    history: deque = field(default_factory=deque)
    last_power: float | None = None
    stable_since: int | None = None
    candidate_since: int | None = None
    reference_id: str | None = None


class AcousticEngine:
    def __init__(self, config):
        self.config = config
        self.np = numpy_module()
        self.states = {}
        self.models = {}
        self.candidates = {}
        self._pending_lifecycle = []
        self._pending_candidate_skips: list[tuple[ChannelKey, Provenance]] = []
        self.window_count = self.revision_count = 0

    def fork(self, keys):
        draft = copy(self)
        draft.states, draft.models, draft.candidates = (
            dict(self.states),
            dict(self.models),
            dict(self.candidates),
        )
        memo = {id(self.np): self.np}
        for key in keys:
            if key in self.states:
                memo[id(self.states[key].plan)] = self.states[key].plan
                draft.states[key] = deepcopy(self.states[key], memo)
            if key in self.models:
                draft.models[key] = deepcopy(self.models[key])
        for cid, value in self.candidates.items():
            if value["channel"] in keys:
                draft.candidates[cid] = deepcopy(value)
        draft._pending_lifecycle = []
        draft._pending_candidate_skips = []
        draft.window_count = draft.revision_count = 0
        return draft

    def reserve(self, specs, limits):
        cfg = self.config
        by_key = {key: state.rate for key, state in self.states.items()}
        by_key.update(
            {ChannelKey(s.ref.source_id, c): s.sample_rate_hz for s, c in specs}
        )
        # Conservative logical reservation, including transaction/workspace and
        # Python descriptor allowance. This is not an OS allocator guarantee.
        size = sum(len(value) for value in self.models.values()) + sum(
            len(c["stats"]) for c in self.candidates.values()
        )
        reserve = sum(
            math.ceil(cfg.frame_seconds * rate) * 80
            + (len(cfg.bands) + 1) * (cfg.max_impulses * 32 + 2048)
            + cfg.max_open_changes * (len(cfg.bands) + 1) * 1024
            + math.ceil(
                max(*cfg.horizons_seconds, cfg.loudness_horizon_seconds)
                / cfg.hop_seconds
            )
            * 128
            for rate in by_key.values()
        )
        reserve += (limits.max_baselines * (len(cfg.bands) + 1) + size) * 1024
        if reserve > cfg.max_state_bytes:
            raise AcousticLimit("acoustic_state_bytes", reserve, cfg.max_state_bytes)
        if size > cfg.max_parameter_features:
            raise AcousticLimit(
                "acoustic_parameter_features", size, cfg.max_parameter_features
            )

    def _state(self, spec, key):
        state = self.states.get(key)
        if state is None or state.source != spec.ref:
            plan = SpectralPlan.create(self.np, self.config, spec.sample_rate_hz)
            state = _State(
                spec.ref,
                spec.sample_rate_hz,
                plan,
                Ring(self.np.empty(plan.size, dtype=self.np.float32)),
            )
            self.states[key] = state
        self.models.setdefault(key, {})
        return state

    def _provenance(self, spec, channel, start, stop, timing=None):
        measured = timing.at(start, spec.sample_rate_hz) if timing is not None else None
        received = timing.received if timing is not None else None
        return Provenance(
            (DataSpan(SourceInterval(spec.ref, start, stop), (channel,)),),
            () if measured is None else (measured,),
            () if received is None else (received,),
            source_clocks=((spec.ref, spec.clock_id),),
        )

    def _publish(self, state, change, output):
        self.revision_count += 1
        if self.revision_count > self.config.max_revisions_per_call:
            raise AcousticLimit(
                "acoustic_revisions_per_call",
                self.revision_count,
                self.config.max_revisions_per_call,
            )
        state.changes[change.change_id] = change
        output.append(change)

    def _new(
        self,
        state,
        spec,
        key,
        kind,
        band,
        start,
        end,
        power,
        delta,
        confidence,
        output,
        *,
        sequence=None,
    ):
        cfg = self.config
        old_id = state.group.get(kind)
        old = state.changes.get(old_id)
        member = AcousticBandChange(
            band, start, end, None, power, None if kind.startswith("tempo") else delta
        )
        if (
            old is not None
            and old.phase != "interrupted"
            and end <= old.confirmed_frame + round(cfg.grouping_seconds * state.rate)
        ):
            bands = (*old.bands, member)
            change = replace(
                old,
                revision=old.revision + 1,
                phase="detected",
                bands=bands,
                last_observed_frame=end,
                end_frame=None,
                duration_frames=None,
            )
        else:
            if len(state.changes) >= cfg.max_open_changes:
                raise AcousticLimit(
                    "acoustic_open_changes",
                    len(state.changes) + 1,
                    cfg.max_open_changes,
                )
            change = AcousticChange(
                new_id(),
                1,
                kind,
                "detected",
                key,
                spec.ref,
                start,
                end,
                end,
                None,
                None,
                (member,),
                (
                    AcousticQuantity(
                        "delta", delta, "events/s" if kind.startswith("tempo") else "dB"
                    ),
                ),
                confidence,
                self._provenance(
                    spec, key.channel_id, start, end, getattr(state, "timing", None)
                ),
                sequence,
            )
            state.group[kind] = change.change_id
        self._publish(state, change, output)
        return change.change_id

    def _finish(
        self, state, cid, end, output, *, band=None, start=None, interrupted=False
    ):
        previous = state.changes.get(cid)
        if previous is None or previous.phase == "interrupted":
            return
        members = tuple(
            replace(b, last_observed_frame=end, end_frame=None if interrupted else end)
            if (band is None or b.band_id == band)
            and (start is None or b.start_frame == start)
            else b
            for b in previous.bands
        )
        complete = all(b.end_frame is not None for b in members)
        finish = max(b.end_frame for b in members) if complete else None
        phase = (
            "interrupted" if interrupted else "completed" if complete else "detected"
        )
        change = replace(
            previous,
            revision=previous.revision + 1,
            phase=phase,
            bands=members,
            last_observed_frame=max(state.cursor or end, end),
            end_frame=finish,
            duration_frames=finish - previous.start_frame
            if finish is not None
            else None,
        )
        self._publish(state, change, output)

    def interrupt(self, spec, channel, boundary, *, guard=None):
        key = ChannelKey(spec.ref.source_id, channel)
        state = self.states.get(key)
        if state is None or state.source != spec.ref:
            return []
        changes = []
        last = state.cursor or 0
        for cid, change in tuple(state.changes.items()):
            if change.phase == "detected":
                self._finish(state, cid, last, changes, interrupted=True)
        state.changes.clear()
        state.group.clear()
        state.transitions.clear()
        state.bands.clear()
        state.rhythm = Rhythm()
        state.ring.filled = state.ring.write = 0
        state.cursor = None
        state.exposure = 0.0
        state.last_power = None
        state.stable_since = None
        state.candidate_since = None
        state.levels.clear()
        state.continuity += 1
        if guard is not None:
            state.guard = max(state.guard, guard)
        if not changes:
            return []
        return [
            (
                "salience",
                dict(
                    channels=(key,),
                    changes=tuple(changes),
                    provenance=self._provenance(
                        spec, channel, min(c.start_frame for c in changes), last
                    ),
                    unavailable_reasons=("observation_interrupted",),
                ),
            )
        ]

    def evaluate(self, work, borrowed, origin, store, limits, promotion_mode):
        output = []
        for spec, channel, spans, old_frontier, frontier in work:
            key = ChannelKey(spec.ref.source_id, channel)
            state = self._state(spec, key)
            produced = False
            cursor = old_frontier
            for span in spans:
                if span.start > cursor:
                    output.extend(self.interrupt(spec, channel, span.start))
                cursor = span.stop
                if not span.valid:
                    output.extend(self.interrupt(spec, channel, span.stop))
                    continue
                start = max(span.start, state.guard)
                if start >= span.stop:
                    continue
                if state.cursor is not None and start != state.cursor:
                    output.extend(self.interrupt(spec, channel, start))
                if state.cursor is None:
                    state.cursor = start
                    state.next_end = (
                        (start + state.plan.size + state.plan.hop - 1) // state.plan.hop
                    ) * state.plan.hop
                values = (
                    borrowed[key][start - origin : span.stop - origin]
                    if span.backing is None
                    else memoryview(span.backing).cast("f")[
                        span.offset + start - span.start : span.offset
                        + span.stop
                        - span.start
                    ]
                )
                array = self.np.asarray(values)
                consumed = 0
                while consumed < len(array):
                    take = min(len(array) - consumed, state.next_end - state.cursor)
                    state.ring.append(array[consumed : consumed + take])
                    consumed += take
                    state.cursor += take
                    if state.cursor == state.next_end:
                        self.window_count += 1
                        if self.window_count > self.config.max_windows_per_call:
                            raise AcousticLimit(
                                "acoustic_windows_per_call",
                                self.window_count,
                                self.config.max_windows_per_call,
                            )
                        output.append(
                            self._window(
                                state,
                                spec,
                                key,
                                span.timing,
                                store,
                                limits,
                                promotion_mode,
                            )
                        )
                        produced = True
                        state.next_end += state.plan.hop
            if frontier > cursor:
                output.extend(self.interrupt(spec, channel, frontier))
            if not produced and any(s.valid for s in spans):
                output.append(
                    (
                        "salience",
                        dict(
                            channels=(key,),
                            unavailable_reasons=("insufficient_contiguous_duration",),
                        ),
                    )
                )
        return output

    def _window(self, state, spec, key, timing, store, limits, promotion_mode):
        cfg, np = self.config, self.np
        state.timing = timing
        frame = state.ring.ordered(np)
        end, start = state.cursor, state.cursor - state.plan.size
        power, peak, powers, hop_power, peak_offset = state.plan.extract(frame)
        state.current_power = power
        dt = state.plan.hop / state.rate
        state.exposure += dt
        confidence = min(1.0, state.exposure / cfg.maturity_seconds)
        reference = self._reference(state, key, powers, store)
        for name, (value, _) in powers.items():
            if value is not None and state.plan.schemas[name] not in reference:
                reference[state.plan.schemas[name]] = self.models[key].setdefault(
                    state.plan.schemas[name], _Stat()
                )
        background = reference[state.plan.schemas["broadband"]]
        baseline_available = background.exposure > 0
        rhythm_reference = background.rhythm_rate
        confidence = min(confidence, background.exposure / cfg.maturity_seconds)
        profile_values = [
            v
            for name, (v, _) in powers.items()
            if name != "broadband" and v is not None
        ]
        total = sum(profile_values)
        profile = tuple(
            v / max(total, cfg.numerical_power_floor) for v in profile_values
        )
        impulse_high = (
            hop_power
            > max(cfg.absolute_power_floor, background.mean * cfg.impulse_ratio)
            and peak >= cfg.impulse_min_peak
        )
        previous_tempo_available = state.rhythm.rate is not None
        analysis_eligible = cfg.min_contiguous_seconds <= state.plan.size / state.rate
        tempo_change = (
            state.rhythm.observe(
                high=impulse_high,
                pulse_frame=end - state.plan.hop + peak_offset,
                end=end,
                rate=state.rate,
                profile=profile,
                config=cfg,
            )
            if analysis_eligible
            else None
        )
        changes = []
        if previous_tempo_available and state.rhythm.rate is None:
            for kind in ("tempo_increase", "tempo_decrease"):
                self._finish(
                    state,
                    state.transitions.pop(kind, None),
                    end,
                    changes,
                    interrupted=True,
                )
        self._expire(state, end)
        if tempo_change:
            direction, magnitude, finished = tempo_change
            kind = "tempo_increase" if direction > 0 else "tempo_decrease"
            if finished:
                self._finish(state, state.transitions.pop(kind, None), end, changes)
            else:
                opposite = "tempo_decrease" if direction > 0 else "tempo_increase"
                self._finish(state, state.transitions.pop(opposite, None), end, changes)
                cid = self._new(
                    state,
                    spec,
                    key,
                    kind,
                    "broadband",
                    end - state.plan.hop,
                    end,
                    power,
                    magnitude,
                    confidence,
                    changes,
                    sequence=state.rhythm.sequence_id,
                )
                state.transitions[kind] = cid
        # Competing common-mode gain hypothesis. Require established audible
        # presence; otherwise below-floor emergence retains the ordinary path.
        invariant = (
            state.plan.level_invariant_bands(
                frame, cfg.absolute_power_floor, cfg.offset_ratio
            )
            if background.expected
            else None
        )
        peak_frame = without_peak = None
        if analysis_eligible and invariant is not None and peak >= cfg.strong_peak:
            offset = state.plan.isolated_peak_offset(
                frame, cfg.strong_peak, cfg.impulse_ratio
            )
            if offset is not None:
                # Verify that this sample supplies the independent band evidence
                # under the same gain compensation as the observed frame.
                counterfactual = frame.copy()
                counterfactual[offset] = (
                    float(frame[offset - 1]) + float(frame[offset + 1])
                ) / 2
                without_peak = state.plan.level_invariant_bands(
                    counterfactual, cfg.absolute_power_floor, cfg.offset_ratio
                )
                peak_frame = start + offset
        band_rows = []
        anomalous = False
        for name, (value, reason) in powers.items():
            schema = state.plan.schemas[name]
            if value is None:
                band_rows.append(
                    AcousticBandPower(name, None, "unavailable", schema, reason)
                )
                continue
            stat = reference[schema]
            activity = state.bands.setdefault(name, _Activity())
            if cfg.min_contiguous_seconds <= state.plan.size / state.rate:
                self._activity(
                    state,
                    spec,
                    key,
                    name,
                    activity,
                    stat,
                    value,
                    peak,
                    start,
                    end,
                    confidence,
                    changes,
                    common_gain=power / background.mean
                    if baseline_available and background.mean > 0
                    else None,
                    onset_power=None
                    if invariant is None or name == "broadband"
                    else min(value, invariant[name][0]),
                    impulse=impulse_high,
                    isolated_peak=(peak_frame, without_peak[name][0])
                    if without_peak is not None and name != "broadband"
                    else None,
                )
            if (
                invariant is not None
                and name != "broadband"
                and value >= cfg.absolute_power_floor
                and invariant[name][0] < cfg.absolute_power_floor
                and not activity.active
            ):
                reason = "common_mode_onset_ambiguous"
            if activity.presence_ambiguous:
                reason = "common_mode_presence_ambiguous"
            relative_error = abs(value - stat.mean) / max(
                stat.mean, cfg.absolute_power_floor
            )
            anomalous |= (
                stat.exposure > 0 and relative_error > cfg.learning_max_relative_error
            )
            label = (
                "observed_active"
                if activity.observed_active and activity.active
                else "active"
                if activity.active
                else "background"
            )
            band_rows.append(AcousticBandPower(name, value, label, schema, reason))
        if cfg.min_contiguous_seconds <= state.plan.size / state.rate:
            self._loudness(state, spec, key, power, start, end, confidence, changes)
        minimum = round(cfg.min_contiguous_seconds * 1e9) or None
        policies = [
            EligibilityPolicy(value, minimum)
            for value in (
                cfg.assessment_min_coverage,
                cfg.established_min_coverage,
                cfg.candidate_min_coverage,
            )
        ]
        feature = FeatureSpec("acoustic", (key,), *policies, True)
        duration = round(state.plan.size * 1e9 / state.rate)
        eligible = feature_eligibility(
            feature, start_ns=0, stop_ns=duration, segments=((0, duration),)
        )
        established = (
            eligible.established_learning.eligible and not changes and not anomalous
        )
        candidate = (
            eligible.candidate_learning.eligible
            and not changes
            and state.exposure >= cfg.maturity_seconds
        )
        deviation = max(
            (
                abs(v - reference[state.plan.schemas[n]].mean)
                / max(reference[state.plan.schemas[n]].mean, cfg.absolute_power_floor)
                for n, (v, _) in powers.items()
                if v is not None and reference[state.plan.schemas[n]].exposure > 0
            ),
            default=0.0,
        )
        if established:
            for name, (value, _) in powers.items():
                if value is not None:
                    stat = reference[state.plan.schemas[name]]
                    stat.learn(value, dt, cfg)
                    if state.rhythm.rate is not None:
                        stat.rhythm_rate = state.rhythm.rate
        lifecycle = self._candidate(
            state,
            key,
            powers,
            dt,
            anomalous and candidate,
            store,
            limits,
            promotion_mode,
        )
        state.history.append((end, power))
        while state.history and (end - state.history[0][0]) / state.rate >= max(
            cfg.horizons_seconds
        ):
            state.history.popleft()
        horizons = []
        for seconds in cfg.horizons_seconds:
            nominal = round(seconds * 1e9)
            rows = [(f, p) for f, p in state.history if f > end - seconds * state.rate]
            segments = tuple(
                (
                    round((f - state.plan.hop) * 1e9 / state.rate),
                    round(f * 1e9 / state.rate),
                )
                for f, _ in rows
            )
            coverage = evaluate_eligibility(
                start_ns=round(end * 1e9 / state.rate) - nominal,
                stop_ns=round(end * 1e9 / state.rate),
                segments=segments,
                policy=policies[0],
                continuous_only=False,
            )
            horizons.append(
                (
                    seconds,
                    sum(p for _, p in rows) / len(rows)
                    if rows and coverage.eligible
                    else None,
                )
            )
        # Evaluate above against pre-update references: score is also driven by
        # recorded transition magnitudes, never by unavailable components.
        deviation = max(
            deviation,
            max(
                (
                    abs(q.value) / 12
                    for c in changes
                    for q in c.quantities
                    if q.unit == "dB"
                ),
                default=0.0,
            ),
        )
        acoustic = (
            deviation / (1 + deviation)
            if eligible.assessment.eligible and baseline_available
            else None
        )
        temporal = None
        if state.rhythm.rate is not None and rhythm_reference is not None:
            diff = abs(state.rhythm.rate - rhythm_reference)
            absolute = (
                cfg.tempo_increase_absolute
                if state.rhythm.rate >= rhythm_reference
                else cfg.tempo_decrease_absolute
            )
            relative = (
                cfg.tempo_increase_relative
                if state.rhythm.rate >= rhythm_reference
                else cfg.tempo_decrease_relative
            )
            temporal = diff / (absolute + relative * rhythm_reference + diff)
        required = [(cfg.acoustic_weight, acoustic), (cfg.temporal_weight, temporal)]
        score = (
            None
            if any(w > 0 and v is None for w, v in required)
            else sum(w * v for w, v in required if w > 0) / sum(w for w, _ in required)
        )
        window = AcousticWindow(
            SourceInterval(spec.ref, start, end),
            math.sqrt(max(0.0, power)),
            power,
            peak,
            tuple(band_rows),
            state.rhythm.rate,
            state.rhythm.reason,
            state.rhythm.sequence_id,
            confidence,
            eligible.assessment.eligible,
            established,
            candidate,
            state.continuity,
            tuple(horizons),
        )
        provenance = self._provenance(spec, key.channel_id, start, end, timing)
        if lifecycle is None:
            self._pending_candidate_skips.append((key, provenance))
        else:
            self._pending_lifecycle.extend(lifecycle)
        return (
            "salience",
            dict(
                channels=(key,),
                timestamp=timing.at(end - 1, state.rate),
                time_scale="acoustic",
                salience_score=score,
                confidence=confidence,
                is_salient=None if score is None else score >= cfg.score_threshold,
                baseline_status="provisional" if confidence < 1 else "established",
                acoustic_deviation=acoustic,
                temporal_deviation=temporal,
                unavailable_reasons=()
                if score is not None
                else ("component_unavailable",),
                changes=tuple(changes),
                windows=(window,),
                provenance=provenance,
            ),
        )

    def _expire(self, state, end):
        for cid, change in tuple(state.changes.items()):
            if change.phase in (
                "completed",
                "interrupted",
            ) and end > change.confirmed_frame + round(
                self.config.grouping_seconds * state.rate
            ):
                del state.changes[cid]

    def _reference(self, state, key, powers, store):
        primary = self.models[key]
        choices = [(None, primary)]
        choices.extend(
            (
                cid,
                {
                    fid: stat
                    for fid, stat in c["stats"].items()
                    if any(
                        compat.feature_id == fid
                        and compat.status in ("compatible", "revalidate")
                        for compat in store.records[cid].compatibility
                    )
                },
            )
            for cid, c in self.candidates.items()
            if c["channel"] == key
            and cid in store.records
            and store.records[cid].role == "established"
            and store.records[cid].activity == "active"
        )

        def distance(row):
            values = [
                abs(
                    math.log(
                        (value + self.config.absolute_power_floor)
                        / (
                            row.get(state.plan.schemas[name], _Stat()).mean
                            + self.config.absolute_power_floor
                        )
                    )
                )
                for name, (value, _) in powers.items()
                if value is not None and state.plan.schemas[name] in row
            ]
            return max(values) if values else float("inf")

        cid, result = min(choices, key=lambda pair: distance(pair[1]))
        state.reference_id = cid
        return {**primary, **result}

    def _activity(
        self,
        state,
        spec,
        key,
        name,
        activity,
        stat,
        power,
        peak,
        start,
        end,
        confidence,
        changes,
        *,
        onset_power=None,
        common_gain=None,
        impulse=False,
        isolated_peak=None,
    ):
        activity.presence_ambiguous = False
        cfg = self.config
        floor = cfg.absolute_power_floor
        share = (
            power / state.current_power
            if name != "broadband" and state.current_power >= floor
            else None
        )
        if not activity.initialized:
            activity.initialized = True
            activity.active = power >= floor
            activity.observed_active = activity.active
            activity.start = start
            activity.background_share = share or 0.0
            return
        db = 10 * math.log10(
            (power + cfg.numerical_power_floor)
            / (stat.mean + cfg.numerical_power_floor)
        )
        relative = (
            activity.active
            and activity.observed_active
            and share is not None
            and (state.cursor - activity.start) / state.rate >= cfg.maturity_seconds
        )
        if activity.relative:
            # End the independently observed contribution without marking its
            # background band absent. A level step with the same spectrum does
            # not meet this test.
            threshold = activity.background_share * min(
                cfg.onset_ratio, cfg.offset_ratio**2
            )
            dt = state.plan.hop / state.rate
            activity.relative_share += -math.expm1(-dt / cfg.frame_seconds) * (
                (share or 0.0) - activity.relative_share
            )
            if power < floor or (
                activity.relative_share < threshold
                and power < activity.relative_power / cfg.offset_ratio
            ):
                if activity.fall is None:
                    activity.fall = end - state.plan.hop
                if (
                    end - activity.fall
                ) / state.rate >= cfg.offset_confirmation_seconds:
                    fall = activity.fall
                    if activity.onset_id:
                        self._finish(
                            state,
                            activity.onset_id,
                            fall,
                            changes,
                            band=name,
                            start=activity.start,
                        )
                        if (
                            fall - activity.start
                        ) / state.rate >= cfg.sustained_seconds:
                            self._new(
                                state,
                                spec,
                                key,
                                "sound_offset",
                                name,
                                fall,
                                end,
                                power,
                                db,
                                confidence,
                                changes,
                            )
                    activity.relative = False
                    activity.onset_id = activity.rise = activity.fall = None
            else:
                activity.fall = None
            return
        if not activity.active or relative:
            base = stat.mean if activity.absent_power is None else activity.absent_power
            if share is not None and activity.observed_active:
                # A short trough is insufficient evidence to lower the onset
                # reference for a band that was already present.
                base = max(base, stat.mean)
            threshold = max(floor, base * cfg.onset_ratio)
            declining = (
                name != "broadband"
                and state.last_power is not None
                and state.current_power
                < state.last_power * cfg.band_onset_decline_ratio
            )
            evidence = power if onset_power is None else onset_power
            different_share = (
                share is None
                or activity.background_share == 0
                or share >= activity.background_share * cfg.onset_ratio
            )
            if (
                power >= threshold
                and evidence >= threshold
                and not declining
                and different_share
            ):
                if activity.rise is None:
                    activity.rise = end - state.plan.hop
                # A strong component elsewhere must not accelerate a weak band's
                # confirmation. sqrt(2*power) is a sinusoidal-equivalent band
                # amplitude, not a reconstructed peak or a source identity.
                intensity = peak if name == "broadband" else math.sqrt(2 * evidence)
                peak_support = (
                    isolated_peak is not None
                    and not activity.active
                    and not activity.observed_active
                    and isolated_peak[1] < threshold
                )
                if peak_support:
                    intensity = cfg.strong_peak
                strong = min(1.0, intensity / max(cfg.strong_peak, 1e-12))
                confirmation = cfg.weak_confirmation_seconds + strong * (
                    cfg.strong_confirmation_seconds - cfg.weak_confirmation_seconds
                )
                # In an occupied band, several overlapping FFT hops can see
                # the same stochastic excursion. Wait for one analysis frame
                # of contrast unless independent hop energy supports a pulse.
                if activity.observed_active and share is not None and not impulse:
                    confirmation = max(confirmation, cfg.frame_seconds)
                detected_start = isolated_peak[0] if peak_support else activity.rise
                if (end - detected_start) / state.rate >= confirmation:
                    activity.active, activity.start, activity.fall = (
                        True,
                        detected_start,
                        None,
                    )
                    activity.relative = bool(
                        activity.observed_active and activity.background_share and share
                    )
                    activity.relative_share = share or 0.0
                    activity.relative_power = power
                    activity.observed_active = activity.relative
                    if not state.rhythm.ever_tracked:
                        activity.onset_id = self._new(
                            state,
                            spec,
                            key,
                            "sound_onset",
                            name,
                            activity.start,
                            end,
                            power,
                            db,
                            confidence,
                            changes,
                        )
            else:
                activity.rise = None
            if not relative or activity.relative:
                return
        if share is not None and activity.observed_active and activity.rise is None:
            # Average nearby spectral shares before comparing a rise. The
            # broadband denominator removes uniform level changes.
            activity.background_share += -math.expm1(
                -state.plan.hop / state.rate / cfg.maturity_seconds
            ) * (share - activity.background_share)
        if activity.active:
            threshold = max(floor, stat.mean / cfg.offset_ratio)
            # Absence is directional: energy above the gain-adjusted expected
            # lower bound supports continuing presence, even if stochastic
            # excess makes it unlike an exact common-mode gain fit. An upper
            # fit bound would turn positive residual energy into disappearance.
            # Keep the existing lower hysteresis bound and raw evidence.
            follows_gain = (
                common_gain is not None
                and 0 < common_gain < 1 / cfg.offset_ratio
                and state.current_power >= floor
                and stat.mean > 0
                and 1 / cfg.offset_ratio <= (power / stat.mean) / common_gain
            )
            activity.presence_ambiguous = power < threshold and follows_gain
            if power < threshold and not follows_gain:
                if activity.fall is None:
                    activity.fall = end - state.plan.hop
                    activity.absolute_absence = True
                activity.absolute_absence &= power < floor
                if (
                    end - activity.fall
                ) / state.rate >= cfg.offset_confirmation_seconds:
                    sustained = (
                        activity.fall - activity.start
                    ) / state.rate >= cfg.sustained_seconds
                    if activity.onset_id:
                        self._finish(
                            state,
                            activity.onset_id,
                            activity.fall,
                            changes,
                            band=name,
                            start=activity.start,
                        )
                    if (
                        sustained and activity.onset_id or stat.expected
                    ) and not state.rhythm.ever_tracked:
                        self._new(
                            state,
                            spec,
                            key,
                            "sound_offset",
                            name,
                            activity.fall,
                            end,
                            power,
                            db,
                            confidence,
                            changes,
                        )
                    activity.active = False
                    activity.absent_power = power
                    activity.onset_id = activity.rise = activity.fall = None
                    if activity.absolute_absence:
                        activity.observed_active = False
                        activity.background_share = 0.0
            else:
                activity.fall = None

    def _loudness(self, state, spec, key, power, start, end, confidence, changes):
        cfg = self.config
        state.levels.append((end, power))
        while (
            len(state.levels) > 1
            and (end - state.levels[0][0]) / state.rate > cfg.loudness_horizon_seconds
        ):
            state.levels.popleft()
        if state.last_power is not None:
            db_last = abs(
                10
                * math.log10(
                    (power + cfg.numerical_power_floor)
                    / (state.last_power + cfg.numerical_power_floor)
                )
            )
            state.stable_since = (
                state.stable_since if db_last < cfg.stable_level_db else None
            )
            if state.stable_since is None and db_last < cfg.stable_level_db:
                state.stable_since = end
            if (
                state.stable_since is not None
                and (end - state.stable_since) / state.rate >= cfg.rearm_seconds
            ):
                for kind in ("loudness_spike", "loudness_drop", "sound_offset"):
                    cid = state.transitions.pop(kind, None) or (
                        state.group.get(kind) if kind == "sound_offset" else None
                    )
                    if (
                        cid
                        and state.changes.get(cid)
                        and state.changes[cid].phase == "detected"
                    ):
                        self._finish(state, cid, end, changes)
        before = state.levels[0][1]
        delta = power - before
        db = 10 * math.log10(
            (power + cfg.numerical_power_floor) / (before + cfg.numerical_power_floor)
        )
        kind = "loudness_spike" if delta > 0 else "loudness_drop"
        relative = cfg.spike_relative_db if delta > 0 else cfg.drop_relative_db
        absolute = cfg.spike_absolute_power if delta > 0 else cfg.drop_absolute_power
        guard = power if delta > 0 else before
        if (
            abs(db) >= relative
            and abs(delta) >= absolute
            and guard >= cfg.loudness_activity_floor
            and kind not in state.transitions
        ):
            # A confirmed reversal ends the prior excursion even when the
            # signal never settles long enough for the stable rearm path.
            opposite = "loudness_drop" if delta > 0 else "loudness_spike"
            previous = state.transitions.pop(opposite, None)
            if previous is not None:
                self._finish(state, previous, end, changes)
                # A subsequent reversal must start its own broadband event,
                # even inside the preceding event's grouping window.
                state.group.pop(opposite, None)
            state.transitions[kind] = self._new(
                state,
                spec,
                key,
                kind,
                "broadband",
                start,
                end,
                power,
                db,
                confidence,
                changes,
            )
        state.last_power = power

    def _candidate(self, state, key, powers, dt, eligible, store, limits, mode):
        cfg = self.config
        if not eligible:
            state.candidate_since = None
            return []
        if state.candidate_since is None:
            state.candidate_since = state.cursor
        if (state.cursor - state.candidate_since) / state.rate < cfg.candidate_seconds:
            return []
        values = {
            state.plan.schemas[n]: p for n, (p, _) in powers.items() if p is not None
        }
        matches = [
            (cid, c)
            for cid, c in self.candidates.items()
            if c["channel"] == key
            and cid in store.records
            and store.records[cid].activity != "archived"
            and all(
                abs(p - c["stats"].get(fid, _Stat()).mean)
                / max(p, cfg.absolute_power_floor)
                < cfg.learning_max_relative_error
                for fid, p in values.items()
            )
        ]
        drafts = []
        if matches:
            cid, model = matches[0]
        else:
            if len(store.administration) >= limits.max_baselines:
                # New learning is unavailable. Existing models and this window
                # can still be committed by the enclosing audio transaction.
                return None
            cid = new_id()
            record = BaselineRecord(
                new_id(),
                cid,
                (key,),
                compatibility=tuple(
                    BaselineCompatibility(fid, "compatible") for fid in values
                ),
            )
            store.records[cid] = record
            store.administration[cid] = (0, False)
            model = {"channel": key, "stats": {fid: _Stat() for fid in values}}
            self.candidates[cid] = model
            drafts.append(
                (
                    "lifecycle",
                    dict(
                        event_type="candidate_created",
                        after=record,
                        candidate_id=cid,
                        control_revision=0,
                    ),
                )
            )
        for fid, value in values.items():
            model["stats"].setdefault(fid, _Stat()).learn(value, dt, cfg)
        before = store.records[cid]
        ready = min(
            s.exposure for s in model["stats"].values()
        ) >= cfg.candidate_ready_seconds and all(
            s.variance
            <= (cfg.candidate_relative_spread * max(s.mean, cfg.absolute_power_floor))
            ** 2
            for s in model["stats"].values()
        )
        if (
            ready
            and not before.ready
            and before.role == "candidate"
            and all(
                stat.exposure - stat.proposal_exposure >= cfg.candidate_ready_seconds
                for stat in model["stats"].values()
            )
        ):
            after = replace(
                before,
                ready=True,
                evidence_revision=before.evidence_revision + 1,
                proposal_id=new_id(),
            )
            for stat in model["stats"].values():
                stat.proposal_exposure = stat.exposure
            store.records[cid] = after
            drafts.append(
                (
                    "lifecycle",
                    dict(
                        event_type="candidate_ready",
                        before=before,
                        after=after,
                        candidate_id=cid,
                    ),
                )
            )
            before = after
        if (
            ready
            and mode == "automatic"
            and not before.promotion_blocked
            and before.role == "candidate"
            and before.ready
        ):
            after = replace(before, role="established", ready=False)
            store.records[cid] = after
            drafts.append(
                (
                    "lifecycle",
                    dict(
                        event_type="candidate_promoted",
                        before=before,
                        after=after,
                        candidate_id=cid,
                    ),
                )
            )
        return drafts
