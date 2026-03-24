# Countdown Numbers Game

Small Python implementation of the Countdown numbers game — search and benchmark
algorithms to reach target numbers using arithmetic operations on a set of numbers.

Contents
- `countdown_engine.py` — reference solver
- `countdown_engine_optimized.py` — optimized solver
- `countdown_cli.py` — simple CLI to run the solver
- `benchmarks/` — benchmark results and tools
- `tests/` — test cases

Quick start

1. Create and activate a virtualenv (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # if you add dependencies
```

2. Run the CLI on an example (replace with desired args):

```bash
python countdown_cli.py
```

Benchmarks

- Benchmarks are stored under `benchmarks/` and can be regenerated with the scripts in that folder.
