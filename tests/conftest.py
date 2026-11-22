"""Explicit, deliberately small contract-test limits; not acoustic defaults."""

from dataclasses import replace

import pytest

from far_audio_salience import DetectorConfig, ResourceLimits, ResourceReporting


@pytest.fixture
def config():
    return DetectorConfig(
        ResourceLimits(
            max_sources=4,
            max_channels=12,
            max_chunk_bytes=4096,
            max_buffer_bytes=8192,
            max_baselines=8,
            max_receipts=3,
            max_snapshot_bytes=100_000,
            max_relationship_windows=2,
            max_features=8,
        ),
        max_reorder_delay_ns=0,
        comparison_retention_ns=1_000_000_000,
        reporting=ResourceReporting(aggregate_every=3, materially_worse_by=8),
    )


@pytest.fixture
def reorder_config(config):
    return replace(config, max_reorder_delay_ns=10_000_000_000)
