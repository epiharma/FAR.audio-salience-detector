"""Two USB-microphone-shaped sources with synthetic, asynchronous arrivals."""

from array import array

from example_config import example_config

from far_audio_salience import AudioSalienceDetector, SourceSpec


def main() -> None:
    detector = AudioSalienceDetector(example_config())
    a = SourceSpec.microphone(label="usb-a", sample_rate_hz=16_000)
    b = SourceSpec.microphone(label="usb-b", sample_rate_hz=48_000)
    detector.start_source(a)
    detector.start_source(b)
    assert a.clock_id != b.clock_id
    # The adapter may deliver A repeatedly before B. This implies no B loss.
    arrivals = ((a, 0, 160), (a, 160, 160), (b, 0, 480), (a, 320, 160))
    for source, first_frame, count in arrivals:
        batch = detector.process(
            source.chunk(
                memoryview(array("f", [0.0] * count)),
                first_frame=first_frame,
            )
        )
        assert batch.salience and not batch.health
        print(source.label, len(batch.salience), "unavailable acoustic assessment")
    # No alignment evidence was supplied. No cross-source synchronization is assumed.
    detector.end_source(a.ref)
    detector.end_source(b.ref)
    detector.close()


if __name__ == "__main__":
    main()
