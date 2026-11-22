"""One microphone-shaped synthetic adapter; no device is opened."""

from array import array

from example_config import example_config

from far_audio_salience import AudioSalienceDetector, SourceSpec


def main() -> None:
    detector = AudioSalienceDetector(example_config())
    mic = SourceSpec.microphone(label="desk", sample_rate_hz=16_000)
    detector.start_source(mic)
    # Replace this producer with your external adapter. Coordinates are acquisition
    # positions, not delivery times. Device-format conversion belongs in the adapter.
    for first_frame in (0, 160):
        samples = memoryview(array("f", [0.0] * 160))
        batch = detector.process(mic.chunk(samples, first_frame=first_frame))
        for event in batch.ordered:
            print(type(event).__name__, event.header.emission_sequence)
        assert all(result.salience_score is None for result in batch.salience)
    detector.end_source(mic.ref)
    detector.close()


if __name__ == "__main__":
    main()
