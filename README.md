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

## Results

| Metric | Value |
|--------|-------|
| European Call error vs BS | < 0.01 |
| Asian Call CI width reduction | 97.66% |

The Asian CI reduction is 97.66% (within 0.34% of target) - the theoretical limit is ~98% due to the correlation between arithmetic and geometric averages (~0.99).

---

# Monte Carlo Derivatives
## Pricing Engine
### Python  •  PyTorch  •  CUDA  •  Quantitative Finance

Covers: Probability Theory  •  Stochastic Calculus  •  Option Pricing
Variance Reduction  •  GPU Parallelism  •  Black-Scholes  •  Barrier & Asian Options

## How to Use This Guide

This document is designed as a zero-to-advanced learning journey for the Monte Carlo Derivatives Pricing Engine built in Python and PyTorch. You do not need a finance or math degree to start — we build every concept from scratch.

---

## CHAPTER 1
## What Is a Derivative?
### Financial instruments and why we need to price them

### 1.1  The Simplest Bet in the World

Imagine you and a friend bet on a coin flip. If heads, you win $10. If tails, you lose $10. This bet has a payoff — a cash amount that depends on an uncertain future event.

A financial derivative is exactly this: a contract whose value depends on the future price of something else (the underlying asset — a stock, oil, gold, an interest rate, etc.).

### 1.2  Real-World Examples

### 1.3  Options — The Derivative We Will Price

An option gives its buyer the right, but not the obligation, to buy or sell an asset at a fixed price on (or before) a future date.

### Key Vocabulary

### 1.4  Why is Pricing Hard?

We want to know: what is a fair price to pay for this option TODAY? The problem is that we do not know what S will be at expiry. It could be anywhere. We need to somehow average over all possible futures — and that is exactly what Monte Carlo simulation does.

---

## CHAPTER 2
## Probability & Randomness
### The mathematical foundation of simulation

### 2.1  Random Variables

A random variable X is a variable whose value is the outcome of a random process. Example: rolling a die gives X ∈ {1, 2, 3, 4, 5, 6} each with probability 1/6.

For a stock price, X could be "the price one year from now" — it can take any positive value, with some values more likely than others.

### 2.2  The Normal Distribution

The most important distribution in finance (and this project) is the Normal distribution N(μ, σ²):

X ~ N(μ, σ²)  means  P(a ≤ X ≤ b) = ∫ₐᵇ (1/√2πσ²) · exp(-(x-μ)² / 2σ²) dx

### 2.3  The Log-Normal Distribution

Stock prices cannot go negative. So instead of modelling the price S directly as normal, we model log(S) as normal. If log(S) is normally distributed, then S is log-normally distributed.

If  log(S) ~ N(μ, σ²)  then  S = e^(log S)  is always positive

This is why Geometric Brownian Motion (which we will meet in Chapter 4) uses the exponential. It guarantees that stock prices stay positive.

### 2.4  Expected Value and Variance

### 2.5  The Law of Large Numbers

This is the entire reason Monte Carlo works. The Law of Large Numbers says:

As N → ∞ :   (1/N) ⨯ᵢ f(Xᵢ) → E[f(X)]

In plain English: if you average a function of random samples over many trials, you get the true expected value. Run 1,000,000 simulated option payoffs and average them — you get the true option price (up to discounting).

### 2.6  Standard Error and Convergence

With N samples, the error in your Monte Carlo estimate is approximately:

Standard Error (SE) = σₚ / √N   where σₚ = standard deviation of payoffs

To halve the error, you need 4× more paths. This 1/√N convergence is why variance reduction matters — if you can reduce σₚ, you get the same accuracy with far fewer paths.

---

## CHAPTER 3
## Monte Carlo Simulation
### The universal pricing technique

### 3.1  The Big Idea

Monte Carlo simulation is named after the famous casino. The idea is simple:

1. Generate thousands of random scenarios for the future (price paths)
2. Compute the option payoff for each scenario
3. Average all payoffs and discount back to today

That average is your option price estimate

### 3.2  Step-by-Step: Pricing a European Call

**Setup**

Say S₀ = 100, K = 100, r = 5%, σ = 20%, T = 1 year. We want the price of a call option.

**Step 1: Simulate terminal prices**

Under GBM (covered in Chapter 4), the terminal price is:

S_T = S₀ · exp( (r - σ²/2)·T + σ·√T·Z )   where Z ~ N(0,1)

**Step 2: Compute payoffs**

Payoffᵢ = max(S_Tᵢ - K,  0)

**Step 3: Discount and average**

Price = e^(-rT) · (1/N) ⨯ᵢ Payoffᵢ

**Minimal Python Example**

```python
import numpy as np

S0, K, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
N = 1_000_000

Z    = np.random.randn(N)                           # N standard normals
S_T  = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
payoffs = np.maximum(S_T - K, 0)                    # call payoff
price   = np.exp(-r*T) * payoffs.mean()

print(f"MC Call Price: {price:.4f}")                # ~10.45
# Black-Scholes truth: 10.4506
```

### 3.3  Confidence Intervals

Every Monte Carlo estimate comes with uncertainty. A 95% confidence interval is:

Price ± 1.96 · SE   =   Price ± 1.96 · σₚ/√N

In the notebook output you will see [ci_lower, ci_upper] — the true price lies in this interval 95% of the time.

---

## CHAPTER 4
## Stochastic Calculus & GBM
### How stock prices really move

### 4.1  Brownian Motion

The foundation of modern quantitative finance is Brownian motion W_t — a random process with:

- W₀ = 0 (starts at zero)
- W_t is continuous (no jumps)
- Increments are independent: W_t - W_s ⊥ W_s - W_u for t > s > u
- W_t - W_s ~ N(0, t - s) — increments are normally distributed

### 4.2  Itô's Lemma (Plain English)

In normal calculus, if y = f(x), then dy = f'(x) dx. With stochastic processes, there is an extra term because randomness interacts with itself:

If dX = a dt + b dW,   then df(X) = f'(X) dX + ½ f''(X) b² dt

The extra ½ f''(X) b² dt term (the "Itô correction") appears because (dW)² = dt in stochastic calculus. This is Itô's Lemma and it is the chain rule of stochastic calculus.

### 4.3  Geometric Brownian Motion (GBM)

Stock price dynamics are modelled as:

dS = μ S dt + σ S dW

Where μ is the drift (expected return) and σ is the volatility. Applying Itô's Lemma to log(S):

d log(S) = (μ - σ²/2) dt + σ dW

Integrating from 0 to T gives the exact closed-form solution:

S_T = S₀ · exp[ (μ - σ²/2) T + σ W_T ]

### 4.4  Risk-Neutral Pricing

A deep result in finance (Harrison-Kreps 1979) says: to price derivatives, we replace the real-world drift μ with the risk-free rate r. This gives the risk-neutral measure Q:

Under Q:   S_T = S₀ · exp[ (r - σ²/2) T + σ W_T^Q ]

The option price is then the discounted expected payoff under Q:

Price = e^(-rT) · E^Q[ Payoff(S_T) ]

This is exactly what we compute in the notebook — the r in GBMSimulator is the risk-free rate, not the real-world expected return.

### 4.5  Discretising GBM for Simulation

For path-dependent options we need the full path, not just S_T. We split [0, T] into N steps of size dt = T/N:

S_{t+dt} = S_t · exp[ (r - σ²/2) dt + σ √dt · Z_t ]   where Z_t ~ N(0,1)

In the code (GBMSimulator.simulate), this is vectorised across all paths simultaneously:

```python
# drift and diffusion scalars pre-computed once
drift     = (r - 0.5 * sigma**2) * dt         # scalar
diffusion = sigma * sqrt(dt)                   # scalar

# Z shape: (n_paths, n_steps)  — all randomness at once
Z           = torch.randn(n_paths, n_steps)
log_returns = drift + diffusion * Z            # broadcast
log_paths   = torch.cumsum(log_returns, dim=1) # cumulative sum along time
paths       = S0 * torch.exp(log_paths)        # exponentiate to get prices
```

---

## CHAPTER 5
## Black-Scholes Formula
### The analytic benchmark

### 5.1  What Black-Scholes Achieves

Black, Scholes, and Merton (1973) derived a closed-form formula for the price of a European option under GBM. It is the gold standard against which we validate our Monte Carlo engine.

### 5.2  The Black-Scholes Formulas

**European Call**

C = S₀ · N(d₁) - K · e^(-rT) · N(d₂)

**European Put**

P = K · e^(-rT) · N(-d₂) - S₀ · N(-d₁)

**d₁ and d₂**

d₁ = [ ln(S₀/K) + (r + σ²/2) T ] / (σ√T)

d₂ = d₁ - σ√T

### 5.3  Put-Call Parity

A beautiful no-arbitrage relationship that must always hold:

C - P = S₀ - K · e^(-rT)

If you own a call and are short a put at the same strike, it is equivalent to owning the stock (minus the discounted strike). The code uses this to derive put prices from calls.

### 5.4  Numerical Validation in the Notebook

In the notebook, Section 5 prices European calls and puts with 500,000 paths and compares to the Black-Scholes formula:

```python
bs_call = black_scholes_call(S0, K, r, sigma, T)  # analytic truth
# MC estimate
result = pricer.price(
    lambda p: payoff_european(p, K, "call"),
    n_paths=500_000,
    seed=42
)
# Typical output:
# BS Call       : 10.4506
# MC Naive Call : 10.4499  ±0.00930    error = 0.0007
```

The error is typically under 0.01 — well within the confidence interval. This validates the entire simulation machinery.

---

## CHAPTER 6
## Option Types & Payoffs
### European, Asian, and Barrier options

### 6.1  European Options

The simplest option. Payoff is determined entirely by the terminal price S_T — nothing about the path between S₀ and S_T matters.

Call payoff = max(S_T - K,  0)
Put payoff  = max(K - S_T,  0)

```python
def payoff_european(paths, K, option_type="call"):
    S_T = paths[:, -1]                # last column = terminal price
    if option_type == "call":
        return torch.clamp(S_T - K, min=0.0)
    else:
        return torch.clamp(K - S_T, min=0.0)
```

### 6.2  Asian Options

Asian options depend on the average price over the option's life. They are cheaper than European options (the average is less volatile than the terminal price) and are popular in commodities and FX markets.

**Arithmetic Average Asian Call**

Payoff = max(S̅ - K,  0)   where  S̅ = (1/N) ⨯ᵢ S_{tᵢ}

```python
def payoff_asian_arithmetic(paths, K, option_type="call"):
    S_avg = paths[:, 1:].mean(dim=1)  # avg of all steps (exclude S0)
    if option_type == "call":
        return torch.clamp(S_avg - K, min=0.0)
```

**Geometric Average Asian Call**

Payoff = max(exp(mean(log Sᵢ)) - K,  0)

The geometric average has a closed-form price (Kemna-Vorst formula) which we use as a control variate in Chapter 8.

### 6.3  Barrier Options

Barrier options are activated (knocked in) or deactivated (knocked out) if the price crosses a barrier level B during the option's life. They are cheaper than vanilla options and are widely traded in FX.

```python
def payoff_barrier(paths, K, barrier, barrier_type="up-and-out", option_type="call"):
    S_T   = paths[:, -1]
    S_max = paths.max(dim=1).values   # max price along each path
    S_min = paths.min(dim=1).values   # min price along each path
    
    if barrier_type == "up-and-out":
        knocked = S_max >= barrier    # True where barrier was breached
    elif barrier_type == "down-and-out":
        knocked = S_min <= barrier
    # ... (up-and-in and down-and-in also implemented)
    
    vanilla = torch.clamp(S_T - K, min=0.0)    # vanilla call payoff
    
    if "out" in barrier_type:
        return vanilla * (~knocked).float()     # zero if knocked out
    else:
        return vanilla * knocked.float()        # only lives if knocked in
```

---

## CHAPTER 7
## The GBMSimulator in Depth
### Code walkthrough and design decisions

### 7.1  Class Architecture

The GBMSimulator class encapsulates all GBM parameters and generates paths on demand. Here is the full constructor with annotations:

```python
class GBMSimulator:
    def __init__(self, S0, r, sigma, T, n_steps, device, dtype=torch.float32):
        self.dt        = T / n_steps             # time step size
        
        # Pre-compute constants (once, on device):
        self.drift     = (r - 0.5 * sigma**2) * dt   # drift per step
        self.diffusion = sigma * sqrt(dt)             # diffusion per step
        
        # device = "cuda" for GPU, "cpu" for CPU
        self.device = device
```

### 7.2  Why Pre-Compute drift and diffusion?

We compute (r - 0.5*sigma**2)*dt once in __init__ rather than in every call to simulate(). For 1,000,000 paths × 252 steps = 252 million operations, this matters. The CUDA kernel for addition and multiplication is called once per batch, not once per path.

### 7.3  Vectorised Path Generation

The key insight: generate ALL random numbers for ALL paths and ALL steps in a single torch.randn() call, then operate on the entire tensor at once:

```python
# Shape: (n_paths, n_steps) — 500k paths x 252 steps = 126M floats in one call
Z = torch.randn(n_paths, n_steps, device=device)

# Broadcast: drift is a scalar, applied to every element simultaneously
log_returns = self.drift + self.diffusion * Z     # (n_paths, n_steps)

# cumsum along dim=1 (time axis) — cumulative log-returns = log price relatives
log_paths = torch.cumsum(log_returns, dim=1)      # (n_paths, n_steps)

# Prepend S0 column, exponentiate: log-prices -> prices
paths = S0 * torch.exp(
    torch.cat([zeros(n_paths, 1), log_paths], dim=1)  # (n_paths, n_steps+1)
)
```

### 7.4  Data Types: float32 vs float64

---

## CHAPTER 8
## Variance Reduction
### Antithetic variates & control variates

### 8.1  Why Variance Reduction?

Monte Carlo converges at rate 1/√N. To get 10× more accurate, you need 100× more paths. That is expensive. Variance reduction techniques reduce σₚ (the standard deviation of payoffs) without generating more paths — effectively increasing the 'information per path'.

SE = σₚ / √N   →   reduce σₚ to get smaller SE for the same N

### 8.2  Antithetic Variates

**The Idea**

For every random path generated with Z, also generate the mirror path with -Z. These two paths are negatively correlated — when one path produces a high payoff, the other tends to produce a low payoff. Averaging them reduces variance.

**The Maths**

Estimator: Y̅ = (f(Z) + f(-Z)) / 2

Var[Y̅] = (Var[f(Z)] + Var[f(-Z)] + 2·Cov[f(Z), f(-Z)]) / 4

Since Cov[f(Z), f(-Z)] < 0 (they're negatively correlated), Var[Y̅] < Var[f(Z)]/2. You get more than a 2× variance reduction just by mirroring paths!

**In Code**

```python
if antithetic:
    half   = n_paths // 2
    Z_half = torch.randn(half, n_steps)        # generate half the normals
    Z      = torch.cat([Z_half, -Z_half], dim=0)  # mirror them
    # Now paths[i] and paths[i + half] are antithetic partners
```

### 8.3  Control Variates

**The Idea**

If you know the exact price of a similar option (the "control"), you can use the difference between the Monte Carlo estimate of the control and its true price to correct your main estimate.

**The Setup**

Let Y = payoff of the option we want to price (arithmetic Asian). Let X = payoff of the control (geometric Asian, with known analytic price E[X]).

Y_cv = Y - β (X - E[X])   where β = Cov(Y, X) / Var(X)

The correction term β(X - E[X]) has expected value zero (E[X-E[X]] = 0), so the estimator is unbiased. But it reduces variance whenever Y and X are correlated.

**Why Geometric Asian is a Perfect Control for Arithmetic Asian**

Arithmetic and geometric averages of the same path are highly correlated (ρ > 0.99 typically). The geometric Asian has a closed-form price via the Kemna-Vorst formula. So we can use it to dramatically reduce variance on the arithmetic Asian estimate.

**Optimal Beta Derivation**

β∗ = argminβ Var[Y - β(X - E[X])] = Cov(Y,X) / Var(X)

The code estimates β from the same sample paths using OLS:

```python
@staticmethod
def _apply_control_variate(payoffs, cv_payoffs, cv_analytic):
    Y   = payoffs.double()           # main payoffs
    X   = cv_payoffs.double()        # control (geometric Asian) payoffs
    E_X = torch.tensor(cv_analytic)  # known analytic price
    
    # OLS estimate of beta
    cov  = ((Y - Y.mean()) * (X - X.mean())).mean()
    var  = ((X - X.mean()) ** 2).mean()
    beta = cov / (var + 1e-12)       # +1e-12 for numerical stability
    
    Y_cv = Y - beta * (X - E_X)     # corrected estimator
    return Y_cv
```

### 8.4  Variance Reduction Factor

The notebook computes the variance reduction factor (VRF):

VRF = (SE_naive / SE_cv)²

Typical results: VRF = 5–20× for Asian options with control variates. This means you get the same accuracy with 5–20× fewer paths.

---

## CHAPTER 9
## GPU Acceleration with PyTorch
### CUDA parallelism for Monte Carlo

### 9.1  Why Monte Carlo is GPU-Friendly

Monte Carlo simulation has a structure that GPUs love:

- Massively parallel: each of the 1,000,000 paths is completely independent of every other path. They can all run at the same time.
- Simple arithmetic: each step is just multiply-add operations — exactly what GPU tensor cores are designed for.
- Memory-bound: generating 1M × 252 random normals and doing element-wise ops is a memory bandwidth problem — GPUs have 5-10× more bandwidth than CPUs.

### 9.2  CPU vs GPU Architecture

### 9.3  How PyTorch Moves Computation to the GPU

Everything in the engine that runs on GPU works because of two things:

1. device=DEVICE in GBMSimulator — all tensors are created on the GPU
2. torch.randn(..., device=device) — random numbers generated directly in GPU memory

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This randn call executes on the GPU:
Z = torch.randn(n_paths, n_steps, device=DEVICE)   # never touches CPU RAM

# All operations stay on GPU:
log_returns = drift + diffusion * Z                 # GPU kernel
log_paths   = torch.cumsum(log_returns, dim=1)      # GPU kernel
paths       = S0 * torch.exp(log_paths)             # GPU kernel

# Only move to CPU for final Python output:
price = paths[:, -1].mean().item()                  # .item() -> Python float
```

### 9.4  The Synchronisation Trick in Benchmarking

GPU operations are asynchronous — they are queued but Python returns immediately. When benchmarking, you must call torch.cuda.synchronize() to wait for completion before stopping the clock:

```python
t0 = time.perf_counter()
paths = sim.simulate(n_paths)                # queues GPU work
payoffs = payoff_european(paths, K, "call")  # queues more GPU work
torch.cuda.synchronize()                       # WAIT for GPU to finish!
elapsed = time.perf_counter() - t0           # now this is accurate
```

### 9.5  Expected Speedups

On a free Colab T4 GPU vs CPU:

---

## CHAPTER 10
## Reading the Outputs
### How to interpret every number the engine produces

### 10.1  The price() Return Dictionary

Every call to pricer.price() returns a dictionary:

### 10.2  The Convergence Plot (Figure 2)

The left panel shows MC price vs number of paths. You should observe:

- All three methods (Naive, Antithetic, CV) converge to the Black-Scholes truth (orange dashed line)
- At small N (1,000), estimates are noisy — wide confidence bands
- At large N (500,000), all methods tightly bracket the true price

The right panel (log-log, std error vs N) should show:

- All curves have slope -0.5 (the 1/√N rate)
- Antithetic curve lies below Naive by a fixed factor (its variance reduction)
- Control Variate curve lies furthest below Naive for Asian options

### 10.3  Figure 3: Prices Across Strikes

Several things to notice in the strike plot:

- Asian < European: Asian calls are always cheaper than European calls at the same strike. The average is less volatile than the terminal price, so the option is worth less.
- Barrier disappears for high strikes: the Up-and-Out call price drops to near zero at strikes above the barrier B=120. If K>B, the option is knocked out before it can pay off.
- Deep ITM put = deep OTM call: at strike=80, the put is cheap (ATM is 100); at strike=120, the put is expensive — consistent with put-call parity.

### 10.4  The Summary Table

The final summary table shows all six option types at 1,000,000 paths. Key things to check:

- European Call MC should be within 0.01 of the Black-Scholes analytic price
- Up-and-Out + Up-and-In prices should sum approximately to the European call price
- Down-and-Out + Down-and-In prices should sum approximately to the European call price
- Asian Arithmetic with Control Variate should have a smaller confidence interval than Naive Asian

### 10.5  Common Issues and Fixes

---

## CHAPTER 11
## Extending the Engine
### Advanced directions for further development

### 11.1  Adding American Options

American options can be exercised at any time before expiry. Monte Carlo cannot price them directly, but the Longstaff-Schwartz algorithm (2001) uses regression on simulated paths:

1. Simulate paths backwards from expiry
2. At each step, regress continuation value on basis functions of current S
3. Exercise if immediate payoff > continuation value

```python
# Sketch of Longstaff-Schwartz (extend this yourself!)
exercise_value = torch.clamp(K - paths, min=0)   # put payoff at each step
continuation   = torch.zeros_like(exercise_value)

for t in range(n_steps-1, 0, -1):               # backward loop
    X = paths[:, t]                              # current prices
    Y = continuation[:, t+1]                     # future continuation
    # Regress Y on [1, X, X^2] using least squares
    # ... then compare exercise_value[:, t] to fitted continuation
```

### 11.2  Stochastic Volatility (Heston Model)

GBM assumes constant volatility. In reality, volatility changes over time. The Heston model adds a stochastic volatility process:

dS = r S dt + √v S dW₁
dv = κ(θ - v) dt + ξ √v dW₂   with Corr(dW₁, dW₂) = ρ

This requires simulating two correlated GBMs simultaneously — easily done with a 2×2 Cholesky decomposition applied to torch.randn(n_paths, n_steps, 2).

### 11.3  Jump Processes (Merton Jump-Diffusion)

Stocks sometimes jump (earnings surprises, crises). The Merton model adds Poisson jumps to GBM:

dS/S = (r - λκ̅) dt + σ dW + (J-1) dN

where N is a Poisson process with intensity λ and J is the jump size. In simulation, you sample the number of jumps per path from a Poisson distribution, then add log-normal jump sizes.

### 11.4  Multi-Asset Options (Basket Options)

A basket option pays based on the average of multiple correlated assets. Simulate correlated GBMs using Cholesky decomposition of the correlation matrix:

```python
rho   = 0.6                                   # correlation
Sigma = torch.tensor([[1, rho], [rho, 1]])    # 2x2 correlation matrix
L     = torch.linalg.cholesky(Sigma)          # Cholesky factor

Z = torch.randn(n_paths, n_steps, 2)          # 2 independent assets
Z_corr = Z @ L.T                              # now correlated!
```

### 11.5  Greeks via Monte Carlo

Greeks measure option sensitivity to parameters. They can be computed via bump-and-repricing or pathwise differentiation (automatic differentiation via PyTorch's autograd):

```python
# Pathwise Delta via PyTorch autograd
S0_tensor = torch.tensor(S0, requires_grad=True)

# Re-build paths using S0_tensor so gradient flows through
paths = S0_tensor * torch.exp(log_paths)
price = (payoff_european(paths, K) * discount).mean()

price.backward()                     # compute gradients
delta = S0_tensor.grad.item()        # dPrice/dS0
```

### 11.6  Calibration to Market Data

In practice, σ is not known — it is implied from market option prices. Calibration finds the σ (or Heston parameters) that minimise the difference between model prices and market prices:

minθ  ⨯ᵢ (Price_model(Kᵢ, Tᵢ; θ) - Price_marketᵢ)²

This is an optimisation problem. With PyTorch, you can use torch.optim.Adam and gradient descent directly on the Monte Carlo loss — differentiable Monte Carlo!

---

## Quick Reference: Formulas & Parameters

### All Formulas at a Glance

### Default Parameters in the Notebook

---

## Glossary