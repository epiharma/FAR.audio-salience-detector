"""Small demonstration resource settings, not recommended acoustic thresholds."""

from far_audio_salience import DetectorConfig, ResourceLimits, ResourceReporting


def example_config() -> DetectorConfig:
    return DetectorConfig(
        limits=ResourceLimits(
            max_sources=4,
            max_channels=16,
            max_chunk_bytes=65_536,
            max_buffer_bytes=262_144,
            max_baselines=16,
            max_receipts=32,
            max_snapshot_bytes=1_048_576,
            max_relationship_windows=16,
            max_features=16,
        ),
        max_reorder_delay_ns=0,
        comparison_retention_ns=1_000_000_000,
        reporting=ResourceReporting(aggregate_every=10, materially_worse_by=4096),
    )
