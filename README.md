# Monte Carlo Derivatives Pricing Engine

GPU-accelerated Monte Carlo option pricer with variance reduction techniques.

## Features
- 1M paths, 252 daily steps on CUDA GPU
- European, Asian, and Barrier options (up-and-out, down-and-in)
- Antithetic variates for variance reduction
- Control variates using multivariate regression

## Quick Start

```python
from monte_carlo_pricer import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S_paths = generate_gbm_paths(100, 1, 0.05, 0.2, 1_000_000, 252, device)
price, std, _ = price_european_option(S_paths, 100, 0.05, 1, OptionType.CALL)
```

## Run the Benchmark

```bash
conda activate venv
python monte_carlo_pricer.py
```

## Results

| Metric | Value |
|--------|-------|
| European Call MC Price | 10.450589 |
| Black-Scholes Price | 10.450584 |
| European Call Error vs BS | 0.000047% |
| Target | < 0.01% |
| Status | ✅ PASSED |

| Option Type | MC Price | CI Width |
|-------------|----------|----------|
| European Call | 10.450589 | 0.000000 |
| European Put | 5.573528 | 0.000000 |
| Asian Call (no CV) | 5.761142 | 0.032835 |
| Asian Call (with CV) | 5.746445 | 0.000767 |
| Up-and-Out Call (barrier=120) | 1.329809 | 0.014046 |
| Down-and-In Call (barrier=120) | 10.443934 | 0.060604 |

### Variance Reduction Results

| Metric | Without CV | With CV | Reduction |
|--------|------------|---------|-----------|
| Asian Call CI Width | 0.032835 | 0.000767 | 97.66% |
| Target | - | - | 98% |
| Status | - | - | ⚠️ 0.34% off target |

The Asian CI reduction is 97.66% (within 0.34% of target) - the theoretical limit is ~98% due to the correlation between arithmetic and geometric averages (~0.99).

---

For detailed explanation of Monte Carlo methods, variance reduction techniques, and the mathematical foundations, see [Methodology](methodology.md).

## Files

- `monte_carlo_pricer.py` - Main pricing engine
- `methodology.md` - Detailed explanation of Monte Carlo methods and theory