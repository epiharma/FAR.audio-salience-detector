"""Optional example adapter tests: no sounddevice, NumPy, or hardware required."""

import importlib.util
import sys
from array import array
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from far_audio_salience import AudioSalienceDetector

spec = importlib.util.spec_from_file_location(
    "far_live_microphone_example",
    Path(__file__).parents[1] / "examples/live_microphone.py",
)
live = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = live
spec.loader.exec_module(live)


class FakeStream:
    device = 3
    samplerate = 48000
    channels = 1
    dtype = "float32"
    blocksize = 480

    def __init__(self, owner):
        self.owner = owner
        self.reads = iter(owner.blocks)

    def start(self):
        self.owner.actions.append("capture_start")
        if self.owner.fail == "start":
            raise RuntimeError("start failed")

    def read(self, frames):
        self.owner.requests.append(frames)
        action = next(self.reads, KeyboardInterrupt())
        if isinstance(action, BaseException):
            raise action
        return action

    def stop(self):
        self.owner.actions.append("capture_stop")
        if self.owner.fail == "stop":
            raise RuntimeError("stop failed")

    def close(self):
        self.owner.actions.append("capture_close")


class FakeSoundDevice:
    def __init__(self, blocks=(), fail=None):
        self.blocks, self.fail = blocks, fail
        self.actions, self.requests = [], []

    def query_devices(self, *args):
        return (
            dict(name="fake microphone", index=3, max_input_channels=1)
            if args
            else "3 fake microphone"
        )

    def check_input_settings(self, **settings):
        self.settings = settings
        if self.fail == "settings":
            raise RuntimeError("unsupported rate")

    def RawInputStream(self, **settings):
        self.open_settings = settings
        if self.fail == "open":
            raise RuntimeError("open failed")
        stream = FakeStream(self)
        if self.fail == "actual":
            stream.samplerate = 44100
        return stream


class RecordingDetector(AudioSalienceDetector):
    def __init__(self, config, actions, *, refuse=False, reorder=False):
        if refuse:
            config = replace(config, limits=replace(config.limits, max_chunk_bytes=4))
        if reorder:
            config = replace(config, max_reorder_delay_ns=10**9)
        super().__init__(config)
        self.actions, self.inputs, self.started = actions, [], []

    def start_source(self, source):
        self.started.append(source)
        return super().start_source(source)

    def process(self, observation):
        if hasattr(observation, "audio"):
            self.inputs.append((observation, tuple(observation.audio.samples)))
        else:
            self.inputs.append((observation, None))
        return super().process(observation)

    def end_source(self, source):
        self.actions.append("far_end")
        return super().end_source(source)

    def close(self):
        self.actions.append("far_close")
        return super().close()


def args(**kwargs):
    return SimpleNamespace(
        list_devices=False,
        device=3,
        sample_rate=48000,
        block_size=480,
        duration=None,
        **kwargs,
    )


def exercise(sd, **options):
    detectors, messages = [], []

    def factory(config):
        detector = RecordingDetector(config, sd.actions, **options)
        detectors.append(detector)
        return detector

    stats = live.Statistics()
    result = live.run(
        args(),
        sd,
        detector_factory=factory,
        stats=stats,
        emit=messages.append,
        clock=lambda: 123456,
    )
    return result, stats, detectors, messages


def block(values, overflow=False):
    return array("f", values).tobytes(), overflow


def test_native_float32_actual_frames_receipt_time_and_cleanup():
    sd = FakeSoundDevice([block([0.125, 0.25, 0.375]), block([0, -0.5])])
    code, stats, detectors, messages = exercise(sd)
    assert code == 0
    assert stats.captured == stats.accepted == 5
    audio = detectors[0].inputs
    assert [o.interval.start_frame for o, _ in audio] == [0, 3]
    assert audio[0][1] == (0.125, 0.25, 0.375)
    assert all(
        o.measured_start is None and o.received_at.ns == 123456 for o, _ in audio
    )
    assert len({o.received_at.clock_id for o, _ in audio}) == 1
    assert audio[0][0].received_at.clock_id != detectors[0].started[0].clock_id
    with pytest.raises(ValueError):
        len(audio[0][0].audio.samples)  # Borrowed view has been released.
    assert sd.requests == [480, 480, 480]
    assert sd.actions == [
        "capture_start",
        "capture_stop",
        "capture_close",
        "far_end",
        "far_close",
    ]
    assert stats.sequence_valid and stats.results == 2
    assert messages[-1].startswith("Final report:")


def test_overflow_has_unknown_loss_and_new_epoch_with_stable_identities():
    sd = FakeSoundDevice([block([0.125] * 2), block([0.25] * 3, True), block([0])])
    code, stats, detectors, messages = exercise(sd)
    assert code == 0 and stats.overflows == 1 and stats.epochs == 2
    assert stats.captured == stats.accepted == 6
    detector = detectors[0]
    old, new = detector.started
    assert old.ref.source_id == new.ref.source_id and old.channel_ids == new.channel_ids
    assert old.ref.epoch_id != new.ref.epoch_id and old.clock_id != new.clock_id
    missing = detector.inputs[1][0]
    assert missing.interval is None and missing.measured_start is None
    audio = [o for o, samples in detector.inputs if samples is not None]
    assert [o.interval.start_frame for o in audio] == [0, 0, 3]
    quality = audio[1].quality
    assert quality.overflow and quality.reported_dropped_frames is None
    assert quality.reported_dropped_blocks is None
    assert any("unknown loss" in message for message in messages)


def test_refusal_stops_without_retry_and_counts_delivered_frames():
    sd = FakeSoundDevice([block([0.125] * 2), block([0.5])])
    code, stats, detectors, messages = exercise(sd, refuse=True)
    assert code == 1 and stats.refusals == 1
    assert stats.captured == 2 and stats.accepted == 0
    assert sd.requests == [480]
    assert any("max_chunk_bytes" in message for message in messages)
    assert detectors[0]._closed


@pytest.mark.parametrize("failure", ["settings", "open", "start", "actual", "stop"])
def test_failures_close_available_resources(failure):
    sd = FakeSoundDevice(fail=failure)
    code, stats, detectors, messages = exercise(sd)
    assert code == 1
    if failure == "settings":
        assert not detectors and "Available devices" in messages[0]
    else:
        assert detectors[0]._closed and sd.actions[-1] == "far_close"
    if failure not in ("settings", "open"):
        assert "capture_close" in sd.actions


@pytest.mark.parametrize("payload", [b"", b"123", memoryview(bytearray(8))[::2]])
def test_malformed_capture_buffer_stops_cleanly(payload):
    sd = FakeSoundDevice([(payload, False)])
    code, stats, detectors, messages = exercise(sd)
    assert code == 1 and stats.captured == 0
    assert detectors[0]._closed
    assert any("buffer" in message for message in messages)


def test_read_error_consumes_final_pending_output():
    sd = FakeSoundDevice([block([0.125] * 3), RuntimeError("device disconnected")])
    code, stats, detectors, messages = exercise(sd, reorder=True)
    assert code == 1 and stats.results == 1  # Result emitted by end_source().
    assert stats.sequence_valid and detectors[0]._closed
    assert any("device disconnected" in message for message in messages)


def test_statistics_are_bounded_and_worst_is_lifetime():
    stats = live.Statistics()
    for i in range(10000):
        stats.latency.append(i)
    stats.process_calls = 10000
    stats.worst_ns = 50000
    report = stats.report()
    assert len(stats.latency) == 2048
    assert "2048" in report["latency_scope"]
    assert report["process_p99_ms"] <= 0.01 and report["process_worst_ms"] == 0.05


def test_sequence_discontinuity_is_reported():
    detector = AudioSalienceDetector(live.configuration(48000, 480))
    source = live.SourceSpec.microphone(label="fake", sample_rate_hz=48000)
    detector.start_source(source)
    output = detector.process(
        source.chunk(memoryview(array("f", [0.0])), first_frame=0)
    )
    stats = live.Statistics()
    stats.consume(output)
    with pytest.raises(live.AdapterError, match="sequence"):
        stats.consume(output)
    assert not stats.sequence_valid
    detector.close()


def test_duration_uses_time_and_does_not_require_interrupt():
    sd = FakeSoundDevice([block([0.125])])
    settings = args()
    settings.duration = 0.1
    ticks = iter([0, 1, 2, 200000000, 200000001])
    stats = live.Statistics()
    assert (
        live.run(
            settings, sd, clock=lambda: next(ticks), stats=stats, emit=lambda text: None
        )
        == 0
    )
    assert stats.captured == 1


def test_cli_device_ids_listing_and_missing_dependency(monkeypatch, capsys):
    sd = FakeSoundDevice()
    monkeypatch.setattr(live.importlib, "import_module", lambda name: sd)
    assert live.main(["--list-devices"]) == 0
    assert "fake microphone" in capsys.readouterr().out
    assert live.main(["--device", "3"]) == 0
    assert sd.settings["device"] == 3

    def unavailable(name):
        raise ImportError("not installed")

    monkeypatch.setattr(live.importlib, "import_module", unavailable)
    assert live.main([]) == 1
    assert ".[mic]" in capsys.readouterr().err


@pytest.mark.parametrize(
    "option,value",
    [
        ("--duration", "nan"),
        ("--duration", "0"),
        ("--block-size", "0"),
        ("--sample-rate", "-1"),
    ],
)
def test_invalid_cli_arguments(option, value):
    with pytest.raises(SystemExit) as error:
        live.main([option, value])
    assert error.value.code == 2
