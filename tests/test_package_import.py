"""Smoke test for the installed package."""

from importlib import import_module


def test_package_import() -> None:
    """The package must be importable after installation."""
    package = import_module("far_audio_salience")
    assert package.__name__ == "far_audio_salience"
