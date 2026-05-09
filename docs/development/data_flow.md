# Model data flow

The core execution path is:

1. A user defines a Python function with `@model`.
2. DSL primitives call the active trace handler.
3. `hyperion_trace` records sample, param, factor, and deterministic sites.
4. `hyperion_ir` compiles the trace into an `IRGraph`.
5. `hyperion_backends.JAXBackend` builds a differentiable potential function.
6. Inference engines consume the backend contract and return `InferenceResult`.
7. Diagnostics summarize samples, convergence, and report warnings.

## Boundary rules

- DSL code should not know about HMC/NUTS internals.
- Backends should not format user-facing diagnostic reports.
- Inference engines should emit structured diagnostics rather than printing.
- Diagnostics should accept plain arrays and `InferenceResult` objects.

These boundaries keep simple models usable in notebooks while preserving a path
to service and experiment execution.
