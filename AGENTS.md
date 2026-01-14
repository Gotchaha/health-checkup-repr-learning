## Coding Environment Reference

This repository is not intended to run against your current local machine setup. When writing or modifying code, assume the runtime defined by the root `environment.yml` and target that environment’s versions (Python 3.12, PyArrow 20, Pandas 2.2, etc.). Do not introduce dependencies or system-specific behaviors that are not present in `environment.yml`. Prefer libraries already pinned there (for example, use `pyarrow` for Parquet I/O). If behavior depends on package versions, align with those in `environment.yml` rather than your host system.

## General Code Style

Favor an academic, research-oriented style: clarity, flexibility, and simplicity over heavy abstractions. Keep code easy to read and modify for experiments, avoiding over-engineered structures unless clearly warranted by data scale or reproducibility. Write all comments, docstrings, and other natural language in English. Do not use emoji in code or comments. Prefer concise explanations where intent is non-obvious; otherwise, let the code speak for itself. Keep external dependencies minimal and limited to what is specified in `environment.yml`.

## Script Code Style

Scope: This section applies to repository scripts (e.g., under `scripts/`), which are expected to be invoked from the project root. Guidance is principle‑driven; deviate when it improves clarity or fits the task better.

- Structure
  - Begin with a short docstring stating the script’s purpose and scope.
  - Keep a simple flow: imports → small helpers → `parse_args()` → `main()` that orchestrates steps.
  - Use a conventional entrypoint (`if __name__ == "__main__": ...`) when the script is intended to run directly.

- Imports
  - Order imports as standard library → third‑party → project modules.
  - Scripts run from the project root; include a minimal "make project root importable" block, placed between third‑party and project imports. Prefer absolute imports from `src`.

- Logging
  - Use Python’s `logging` module; default to INFO level.
  - Log to console; add file logging for longer‑running or audit‑relevant scripts.
  - Avoid duplicate handlers when re‑executing from notebooks/REPL.

- CLI behavior
  - Use `argparse` with clear help strings and sensible defaults for research workflows.
  - Favor non‑interactive scripts; expose behavior via flags instead of prompts.

- Auditing and provenance
  - For scripts that transform or materialize datasets, prefer emitting a lightweight audit artifact (e.g., JSON) capturing timestamps, inputs/outputs, key parameters, and basic stats.
  - Include file hashes if they materially help traceability; make hashing optional to keep fast paths available.

- Data I/O
  - Prefer `pyarrow` for Parquet I/O to match the pinned environment; be mindful of dtypes and NA handling.
  - Define explicit schemas when writing nested data structures; otherwise keep I/O straightforward.

- Performance
  - For large data, consider chunked/streaming reads and progress logging. For small data, keep it simple.

- Validation and errors
  - Validate required inputs/columns early and fail with helpful error messages.
  - Catch and log exceptions around critical I/O; return an appropriate exit code from `main()`.

- Typing and naming
  - Add type hints where they improve readability; Python 3.12 features are encouraged (e.g., `str | None`).
  - Use descriptive names over single‑letter identifiers except in tight loops/comprehensions.

These guidelines emphasize clarity, reproducibility, and flexibility suited to research code while staying aligned with the pinned environment in `environment.yml`.
