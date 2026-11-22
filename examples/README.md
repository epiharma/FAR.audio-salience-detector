# Streaming contract examples

After the editable development installation, run from the repository root:

```sh
python examples/one_microphone.py
python examples/two_microphones.py
```

Both scripts are runnable **synthetic adapters**. They demonstrate the one-source
helper and asynchronous arrivals from two independent source clocks. They open no
devices, require no audio library, and produce no acoustic scores. Replace their
block producers with an external adapter for actual microphone or recorded input.
The configuration values are demonstration resource settings, not acoustic policy
defaults or measured resource budgets.

Live microphone streaming remains the primary future validation scenario (P0).
Recorded-audio replay and controlled synthetic audio are supporting scenarios.
Actual capture/decoding remains outside FAR; independent clocks require explicit
external alignment evidence for features that depend on synchronization.
