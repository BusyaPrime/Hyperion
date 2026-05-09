# Architecture overview

Hyperion is organized as a compiler-backed probabilistic programming stack. Each
layer owns one contract and should stay independently testable.

| Layer | Packages | Responsibility |
| --- | --- | --- |
| DSL | `hyperion_dsl` | User-facing `@model`, `sample`, `plate`, distributions, constraints, and transforms. |
| Trace | `hyperion_trace` | Effect handlers, substitution, replay, blocking, and trace collection. |
| IR | `hyperion_ir` | Model graph representation, compilation, and graph optimization. |
| Backend | `hyperion_backends` | JAX potential function, flatten/unflatten, gradients, and prior sampling. |
| Inference | `hyperion_inference` | HMC, NUTS, SMC, VI, flows, Laplace, and high-level MCMC orchestration. |
| Diagnostics | `hyperion_diagnostics` | ESS, R-hat, BFMI, posterior predictive checks, and report generation. |
| API/experiments | `hyperion_api`, `hyperion_exp` | Service boundary, experiment runner, and serialization. |

Keep new features close to the lowest layer that can own the behavior. For
example, a transform validation belongs in DSL/bijectors, while a warning about
divergences belongs in diagnostics.
