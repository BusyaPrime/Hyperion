# Release checklist

Use this checklist before tagging or presenting Hyperion as a stable research
artifact.

## Local validation

- [ ] `cd python && pip install -e ".[dev]"`
- [ ] `cd python && ruff check .`
- [ ] `cd python && mypy hyperion_dsl hyperion_trace hyperion_ir hyperion_backends hyperion_inference hyperion_diagnostics --ignore-missing-imports`
- [ ] `cd python && python -m pytest tests/ -v --tb=short`
- [ ] `python scripts/run_demo.py`

## Evidence

- [ ] README examples still match the public API.
- [ ] Diagnostics reports include warnings for unsafe sampler behavior.
- [ ] Benchmarks were re-run when inference hot paths changed.
- [ ] Java/protobuf integration notes were updated when proto contracts changed.

## Packaging

- [ ] Version was bumped intentionally.
- [ ] Dependency ranges still match supported Python versions.
- [ ] Generated or local cache artifacts are not included in the diff.
