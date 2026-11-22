# FAR Local Audio Salience Detector

**Status: Research / Streaming Contract Skeleton**

FAR is an independent Python package intended for research and development of a
stateful, streaming local audio salience detector, with eventual integration into
a larger AGI proof of concept.

**The acoustic detector is not implemented yet.** The package implements typed
streaming contracts, bounded per-source ingestion/finalization, health events,
eligibility and alignment interfaces, administrative baseline/candidate lifecycle,
and metadata snapshots. Finalized valid observations have explicit unavailable
assessments: acoustic scores, confidence, and `is_salient` are `None` with
`not_implemented`. No extractor, model learning, matching/readiness algorithm,
score calibration/composition, or audio capture is supplied.

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
Snapshot files use bounded, versioned JSON metadata with atomic replacement;
they contain no learned acoustic parameters because no learning algorithm exists.
Restore preserves candidate IDs and blocks but starts a new run with no active
sources, buffered audio, or alignment claims. Reset operations never delete files.

Automatic window sizing and score-weight learning remain future enhancements.
Algorithms, scoring, calibration, and quantitative acceptance budgets remain open.
There are no model performance results or P0 live-microphone validation results.

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

There are no runtime dependencies. Tests require no microphone hardware, API
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
