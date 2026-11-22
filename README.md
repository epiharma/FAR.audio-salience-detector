# FAR Local Audio Salience Detector

**Status: Research / Initial Setup**

FAR is an independent Python package intended for research and development of a
stateful, streaming local audio salience detector, with eventual integration into
a larger AGI proof of concept.

**The detector is not implemented yet.** This repository currently contains only
package scaffolding, development tooling, and an import smoke test. There are no
model capabilities, performance measurements, or experimental results to report.

## Scope and design

The approved design targets acoustic and temporal novelty, multi-scale analysis,
selective online baseline adaptation, candidate baselines, provisional confidence,
stream-integrity diagnostics, and persistent learned state. Audio capture belongs
to the caller; the package will remain independent of AGI internals. Semantic
sound classification and speech recognition are outside v1 scope.

See the [architecture specification](docs/design/DESIGN-001-local-audio-salience)
for approved decisions and open questions. The existing specification is stored
without a file extension and is preserved unchanged.

Input/output types, audio formats, algorithms, scoring/calibration, baseline
policies, persistence details, and evaluation thresholds remain open. Automatic
window sizing and score-weight learning are future enhancements. None of these
decisions is resolved by this initial setup.

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

## License

The license decision is open. No license has been selected or added.
