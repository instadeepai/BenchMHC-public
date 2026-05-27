# Documentation

This document explains how to build and serve the project's documentation using
the provided `Makefile` targets.

## Building the Documentation (Locally)

To build the documentation locally with docker, use the `build-docs` target:

```bash
make build-docs
```

To build the documentation locally through the uv environment, you can do the following steps:

```bash
uv sync --group doc
source .venv/bin/activate
mkdocs build
```

## Serving the Documentation (Locally)

To serve the documentation locally with docker, use the `docs` target:

```bash
make docs
```

Optionally you can provide a different port if the default one is not available:

```bash
make docs port=8000
```

To serve the documentation locally with the uv environment you can perform the following steps:

```bash
uv sync --group doc
source .venv/bin/activate
mkdocs serve
```
