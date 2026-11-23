# FAR Local Audio Salience Detector

**Status: Research / Experimental Acoustic Model**

FAR is an independent Python package for stateful, streaming local acoustic
change detection. The opt-in DESIGN-003 model detects sound onset/offset,
tempo increase/decrease, and extreme loudness spike/drop per channel.
It does not classify sounds or interpret global importance.

The default configuration preserves the infrastructure-only contract skeleton:
acoustic fields remain unavailable until `DetectorConfig(acoustic=AcousticConfig(),
...)` explicitly enables DSP. Install with `pip install -e ".[dev,dsp]"`.
NumPy is optional; no capture or AGI SDK is required by the package.
See [the experimental model guide](docs/acoustic-model.md) for contracts,
configuration, persistence, measured pilot results and limitations.

## Scope and design

The approved design targets acoustic and temporal novelty, multi-scale analysis,
selective online baseline adaptation, candidate baselines, provisional confidence,
stream-integrity diagnostics, and persistent learned state. Audio capture belongs
to the caller; the package will remain independent of AGI internals. Semantic
sound classification and speech recognition are outside v1 scope.

See [DESIGN-001](docs/design/DESIGN-001-local-audio-salience.md) and
[DESIGN-002](docs/design/DESIGN-002-streaming-audio-api.md). Section 18 establishes
precedence; Section 19 records implemented signatures and deferred capabilities.

Input uses normalized native float32 through planar/interleaved memoryviews.
Each call may provide a subset of one independently clocked source's channels.
Missing channels from other sources are never inferred from that omission. The
caller owns the buffer until `process()` returns; retained samples are copied.
Structural errors reject atomically. Invalid values exclude the affected channel
block and produce health events while other channels continue.

All calls must be serialized by the application. FAR provides no capture thread,
clock synchronizer, resampler, broker, background timer, or global interpretation.
Snapshots use bounded, versioned JSON with atomic replacement. Version 2 stores
learned scalar acoustic parameters and candidate controls; legacy metadata-only
version 1 remains readable. Restart preserves compatible models and identities,
with no active sources, buffered audio, impulse sequences or alignment claims.

Numerical thresholds and heuristic confidence are experimental, not calibrated
physical sound pressure or deployment accuracy guarantees. Pattern novelty and
joint-channel acoustic inference remain unavailable.

## Development installation

Requires Python 3.11 or newer. `.python-version` selects 3.11 as the development
default; CI checks Python 3.11 and 3.14. Run these commands from the repository root:

```sh
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```sh
source .venv/bin/activate
```

Install the package in editable mode with separate development tools:

```sh
python -m pip install -e ".[dev]"
```

The base package has no runtime dependencies; acoustic tests require `[dsp]`.
Tests require no microphone hardware, API
keys, external databases, or network services.

## Checks and builds

```sh
python -m ruff check .
python -m ruff format --check .
python -m pytest
python -c "import far_audio_salience; print(far_audio_salience.__file__)"
python -m build
```

To apply formatting, run `python -m ruff format .`. Builds produce a wheel and
source distribution in `dist/`.

Runnable [synthetic contract examples](examples/README.md) show one microphone-
shaped source and two independent microphone-shaped sources. Actual live capture
and recorded-audio decoding require external adapters.

## License

The license decision is open. No license has been selected or added.
