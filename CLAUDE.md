# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`core-carve` is a Python project (AGPL v3). The repository is in early development — update this file as the project takes shape.

## Development Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (once pyproject.toml / requirements.txt exists)
pip install -e ".[dev]"
```

## Common Commands

Once project tooling is configured, expected commands:

```bash
ruff check .          # lint
ruff format .         # format
pytest                # run all tests
pytest tests/test_foo.py::test_bar  # run a single test
```
