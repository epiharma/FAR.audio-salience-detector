"""Bounded observation-only traces of frozen validation failures."""

import json
from dataclasses import asdict

from broader_noise import fixtures as noise_fixtures
from broader_validation import fixtures, run_case

from far_audio_salience._acoustic import AcousticEngine
from far_audio_salience._rhythm import Rhythm


def main():
    targets = (
        "development:43:16000:noise_gain",
        "development:43:48000:gradual_faster",
        "development:43:48000:competing_pulses",
        "colored:16031:16000:colored_hum_component_0.06",
        "colored:99277:16000:brown_gain",
        "colored:99277:48000:brown_gain",
    )
    output = []
    for target in targets:
        pool = noise_fixtures() if target.startswith("colored:") else fixtures()
        case = next(c for c in pool if c["name"] == target)
        rows = []
        old_activity = AcousticEngine._activity
        old_observe = Rhythm.observe

        def activity(
            self,
            state,
            spec,
            key,
            name,
            active,
            stat,
            power,
            peak,
            start,
            end,
            confidence,
            changes,
            **kwargs,
        ):
            before = asdict(active)
            n = len(changes)
            old_activity(
                self,
                state,
                spec,
                key,
                name,
                active,
                stat,
                power,
                peak,
                start,
                end,
                confidence,
                changes,
                **kwargs,
            )
            original_noise = (
                case["family"] == "noise_gain"
                and name == "low"
                and end <= 17600
                and len(rows) < 260
            )
            extra_noise = target.startswith("colored:") and (
                ("component" in target and name == "mid")
                or (
                    "brown_gain" in target
                    and (
                        end / case["rate"] < 0.1
                        or 0.55 < end / case["rate"] < 0.72
                        or 1.05 < end / case["rate"] < 1.26
                        or before["active"] != active.active
                        or len(changes) != n
                    )
                )
            )
            if original_noise or (extra_noise and len(rows) < 500):
                rows.append(
                    dict(
                        frame=end,
                        band=name,
                        power=power,
                        broadband_power=state.current_power,
                        reference=asdict(stat),
                        gain=kwargs.get("common_gain"),
                        normalized_onset_power=kwargs.get("onset_power"),
                        confidence=confidence,
                        before=before,
                        after=asdict(active),
                        changes=[asdict(c) for c in changes[n:]],
                    )
                )

        def observe(self, **kwargs):
            rising = kwargs["high"] and not self.pulse_high
            before = dict(
                rate=self.rate,
                reference=self.reference,
                direction=self.direction,
                stable_count=self.stable_count,
                reason=self.reason,
                deadline=self.deadline,
                frames=list(self.times)[-9:],
            )
            result = old_observe(self, **kwargs)
            if (
                case["family"] not in ("noise_gain", "brown_gain")
                and not target.startswith("colored:")
                and (rising or result)
                and len(rows) < 160
            ):
                rows.append(
                    dict(
                        frame=kwargs["end"],
                        pulse_frame=kwargs["pulse_frame"],
                        before=before,
                        after=dict(
                            rate=self.rate,
                            reference=self.reference,
                            direction=self.direction,
                            stable_count=self.stable_count,
                            reason=self.reason,
                            deadline=self.deadline,
                            frames=list(self.times)[-9:],
                        ),
                        transition=result,
                    )
                )
            return result

        AcousticEngine._activity = activity
        Rhythm.observe = observe
        try:
            result = run_case(case)
        finally:
            AcousticEngine._activity = old_activity
            Rhythm.observe = old_observe
        output.append(
            dict(
                name=target,
                metrics=result["metrics"],
                predictions=result["predictions"],
                bounded_trace=rows,
            )
        )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
