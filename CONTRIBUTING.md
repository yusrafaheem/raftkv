# Contributing

This is primarily a personal from-scratch implementation project, but issues
and pull requests are welcome if something is genuinely broken or unclear.

## Running the test suite

```
pip install -e ".[dev]"
python -m unittest discover -s tests -v
```

The whole suite runs in a couple of seconds; most tests are deterministic
simulations, so a failure should reproduce every time rather than flake.

## Linting and formatting

CI runs `ruff check` and `black --check --diff` on every push (see
`.github/workflows/ci.yml`). Please match the existing style -- 100-column
lines, `black`-formatted, `ruff`-clean -- before opening a PR.

## Design constraints to keep in mind

`RaftNode` (`src/raftkv/raft/node.py`) is intentionally a pure state machine:
no I/O, no threads, no wall-clock reads. If a change to the consensus core
needs any of those, it's very likely going in the wrong layer -- see the
README's "Architecture" section for where that logic belongs instead.
