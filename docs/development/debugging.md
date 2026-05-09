# Debugging guide

## Trace and DSL issues

Start with a minimal model and inspect the trace:

```bash
cd python
python -m pytest tests/test_trace -v --tb=short
```

If a site is missing, check handler stack ordering and whether the model function
is wrapped by `@model`.

## Backend issues

Reduce the model to one latent variable, then verify:

- latent names in the compiled graph;
- transform forward/inverse round trips;
- potential function values are finite;
- gradients are finite for a small unconstrained vector.

## Inference issues

Run the smallest engine-specific test first:

```bash
cd python
python -m pytest tests/test_inference/test_components.py -v --tb=short
```

For HMC/NUTS, inspect acceptance rate, divergences, tree depth, and BFMI before
tuning sample counts. More samples do not fix a broken geometry.

## Diagnostics issues

Diagnostics should be deterministic. Prefer small NumPy arrays in tests and
assert concrete warning messages, table keys, and summary fields.
