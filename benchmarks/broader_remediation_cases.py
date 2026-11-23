"""New procedural counterexamples, scored separately from the frozen 213 cases."""

import json

import numpy as np
from broader_noise import fixtures as colored_fixtures
from broader_noise import make_case
from broader_validation import run_case


def fixtures():
    rate = 32000
    t = np.arange(2 * rate) / rate
    frequency = np.fft.rfftfreq(t.size, 1 / rate)
    for seed in (62013, 86021):
        rng = np.random.default_rng(seed)
        white = rng.normal(size=t.size)
        for color, exponent in (("white", 0), ("pink", 0.5), ("brown", 1)):
            spectrum = np.fft.rfft(white) / np.maximum(frequency, 30) ** exponent
            spectrum[0] = 0
            carrier = np.fft.irfft(spectrum)
            carrier /= np.max(np.abs(carrier))
            for gain_label, gain in (
                ("constant", 0.16),
                (
                    "smooth_gain",
                    np.interp(
                        t,
                        [0, 0.5, 0.7, 1.0, 1.4, 2],
                        [0.21, 0.21, 0.04, 0.04, 0.16, 0.16],
                    ),
                ),
            ):
                yield make_case(
                    f"{color}_{gain_label}",
                    seed,
                    rate,
                    carrier * gain,
                    [],
                    "Continuously available random carrier with no injected component or abrupt step.",
                )
        spectrum = np.fft.rfft(white)
        spectrum[(frequency < 50) | (frequency > 350)] = 0
        background = np.fft.irfft(spectrum)
        background *= 0.012 / np.max(np.abs(background))
        background += 0.06 * np.sin(2 * np.pi * 210 * t)
        first, stop = round(0.65 * rate), round(1.3 * rate)
        for name, component in (
            ("weak_narrowband", 0.003 * np.sin(2 * np.pi * 8000 * t)),
            ("sustained_narrowband", 0.06 * np.sin(2 * np.pi * 1390 * t)),
            ("sustained_broadband", 0.06 * white / np.max(np.abs(white))),
        ):
            x = background.copy()
            x[first:stop] += component[first:stop]
            yield make_case(
                name,
                seed,
                rate,
                x,
                [("sound_onset", first / rate), ("sound_offset", stop / rate)],
                "Injected independent band evidence over continuous low-frequency background.",
            )
        x = background.copy()
        x[first] += 0.7
        yield make_case(
            "impulse",
            seed,
            rate,
            x,
            [("sound_onset", first / rate)],
            "One impulse over continuous stochastic background.",
        )
    original = next(
        c
        for c in colored_fixtures()
        if c["name"] == "colored:16031:48000:colored_hum_component_0.02"
    )
    rate = original["rate"]
    x = original["samples"].copy()
    t = np.arange(len(x)) / rate
    during = (t >= 0.82) & (t < 1.4)
    x[during] += 0.4 * np.sin(2 * np.pi * 4500 * t[during])
    yield make_case(
        "cross_band_dilution",
        16031,
        rate,
        x,
        [
            ("sound_onset", 0.6),
            ("sound_onset", 0.82),
            ("sound_offset", 1.1),
            ("sound_offset", 1.4),
            ("loudness_spike", 0.82),
            ("loudness_drop", 1.4),
        ],
        "A higher band starts while the already observed mid-band contribution continues.",
    )


if __name__ == "__main__":
    print(json.dumps(dict(cases=[run_case(case) for case in fixtures()]), indent=2))
