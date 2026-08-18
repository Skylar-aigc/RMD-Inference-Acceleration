# Contributing to RMD

Thank you for improving RMD. Bug fixes, documentation, new hardware support,
and reproducibility reports are welcome.

## Development setup

```bash
git clone <your-fork-url> RMD
cd RMD
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Before opening a pull request

```bash
ruff check src tests benchmarks app.py
ruff format --check src tests benchmarks app.py
pytest -q
```

Please keep pull requests focused and include tests for behavior changes. For
training or inference changes, report the model, device type/count, command,
precision, seed, and the smallest reproducible log excerpt. Do not commit model
weights, generated videos, private datasets, credentials, or machine-specific
paths.

## Documentation

- Keep `README.md` and `README_CN.md` aligned when changing user-facing commands.
- Mark unverified hardware claims clearly.
- Do not add benchmark numbers without the exact prompt set, seeds, dependency
  versions, hardware, and scoring procedure.

By submitting a contribution, you agree that it is licensed under Apache-2.0.
Participation is also subject to the [Code of Conduct](CODE_OF_CONDUCT.md).
