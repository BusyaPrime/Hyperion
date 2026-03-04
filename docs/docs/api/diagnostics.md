# Diagnostics API

## summary_table

```python
from hyperion_diagnostics.metrics import summary_table

table = summary_table(
    samples,                          # dict[str, np.ndarray]
    prob=0.9,                         # credible interval width
    samples_by_chain=by_chain,        # optional, для multi-chain ESS
)

# table["mu"] = {"mean": 2.01, "std": 0.45, "2.5%": 1.15, ...}
```

## Метрики

```python
from hyperion_diagnostics.metrics import (
    effective_sample_size,
    effective_sample_size_multichain,
    r_hat,
    split_r_hat,
    autocorrelation,
    energy_diagnostic,  # BFMI
)

ess = effective_sample_size(chain)              # np.ndarray -> float
ess_mc = effective_sample_size_multichain(arr)  # (chains, samples) -> float
rhat = r_hat(chains)                            # (chains, samples) -> float
srhat = split_r_hat(chain)                      # (samples,) -> float
ac = autocorrelation(chain)                     # (samples,) -> np.ndarray
bfmi = energy_diagnostic(log_probs)             # (samples,) -> float
```

## Критерии сходимости

| Метрика | Хорошо | Плохо |
|---------|--------|-------|
| R-hat | < 1.01 | > 1.1 |
| ESS | > 400 | < 100 |
| BFMI | > 0.3 | < 0.2 |
| Accept rate (HMC) | 0.6-0.9 | < 0.4 |
| Accept rate (NUTS) | 0.7-0.95 | < 0.5 |
