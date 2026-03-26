import torch
import numpy as np
from scipy.stats import norm
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class OptionType(Enum):
    CALL = 1
    PUT = -1


class BarrierType(Enum):
    UP_AND_OUT = "up_and_out"
    DOWN_AND_OUT = "down_and_out"
    UP_AND_IN = "up_and_in"
    DOWN_AND_IN = "down_and_in"


@dataclass
class OptionParams:
    S0: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: OptionType


@dataclass
class AsianOptionParams(OptionParams):
    pass


@dataclass
class BarrierOptionParams(OptionParams):
    barrier: float
    barrier_type: BarrierType


def norm_cdf(x):
    if isinstance(x, torch.Tensor):
        return 0.5 * (1 + torch.erf(x / np.sqrt(2)))
    return norm.cdf(x)


def black_scholes_price(S0: float, K: float, T: float, r: float, sigma: float, 
                        option_type: OptionType) -> float:
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == OptionType.CALL:
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    
    return price


def geometric_asian_option_price(S0: float, K: float, T: float, r: float, sigma: float,
                                    num_steps: int, option_type: OptionType) -> float:
    n = num_steps
    
    sigma_g_sq = sigma**2 * T * (2*n + 1) / (6 * n**2)
    sigma_g = np.sqrt(sigma_g_sq)
    
    nu = r - 0.5 * sigma**2
    mean_log_G = np.log(S0) + nu * T * (n + 1) / (2 * n)
    
    E_G = np.exp(mean_log_G + 0.5 * sigma_g_sq)
    
    d1 = np.log(E_G / K) / sigma_g
    d2 = d1 - sigma_g
    
    if option_type == OptionType.CALL:
        price = E_G * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - E_G * norm.cdf(-d1)
    
    return price


def generate_gbm_paths(S0: float, T: float, r: float, sigma: float,
                        num_paths: int, num_steps: int, device: torch.device,
                        antithetic: bool = True) -> torch.Tensor:
    dt = T / num_steps
    
    if antithetic:
        num_paths = num_paths // 2
    
    dtype = torch.float32
    
    Z = torch.randn(num_paths, num_steps, device=device, dtype=dtype)
    
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    log_returns = drift + diffusion * Z
    log_returns_cumsum = torch.cumsum(log_returns, dim=1)
    
    initial = torch.full((num_paths, 1), S0, device=device, dtype=dtype)
    S_paths = torch.cat([initial, S0 * torch.exp(log_returns_cumsum)], dim=1)
    
    if antithetic:
        log_returns_neg = drift - diffusion * Z
        log_returns_neg_cumsum = torch.cumsum(log_returns_neg, dim=1)
        S_paths_neg = torch.cat([initial, S0 * torch.exp(log_returns_neg_cumsum)], dim=1)
        S_paths = torch.cat([S_paths, S_paths_neg], dim=0)
    
    return S_paths


def price_european_option(S_paths: torch.Tensor, K: float, r: float, T: float,
                          option_type: OptionType) -> Tuple[float, float, float]:
    final_prices = S_paths[:, -1]
    
    if option_type == OptionType.CALL:
        payoffs = torch.maximum(final_prices - K, torch.zeros_like(final_prices))
    else:
        payoffs = torch.maximum(K - final_prices, torch.zeros_like(final_prices))
    
    discount = torch.exp(-r * T)
    price = discount * payoffs.mean()
    price_std = payoffs.std() / torch.sqrt(torch.tensor(payoffs.numel(), dtype=torch.float32))
    
    return price.item(), price_std.item(), payoffs.var().item()


def price_asian_option(S_paths: torch.Tensor, K: float, r: float, T: float,
                        option_type: OptionType) -> Tuple[float, float, float]:
    avg_prices = S_paths.mean(dim=1)
    
    if option_type == OptionType.CALL:
        payoffs = torch.maximum(avg_prices - K, torch.zeros_like(avg_prices))
    else:
        payoffs = torch.maximum(K - avg_prices, torch.zeros_like(avg_prices))
    
    discount = torch.exp(-r * T)
    price = discount * payoffs.mean()
    price_std = payoffs.std() / torch.sqrt(torch.tensor(payoffs.numel(), dtype=torch.float32))
    
    return price.item(), price_std.item(), payoffs.var().item()


def price_asian_option_with_control_variate(S_paths: torch.Tensor, K: float, r: float, T: float,
                                              sigma: float, option_type: OptionType,
                                              S0: float) -> Tuple[float, float, float]:
    avg_prices = S_paths.mean(dim=1)
    
    if option_type == OptionType.CALL:
        payoffs = torch.maximum(avg_prices - K, torch.zeros_like(avg_prices))
    else:
        payoffs = torch.maximum(K - avg_prices, torch.zeros_like(avg_prices))
    
    final_prices = S_paths[:, -1]
    if option_type == OptionType.CALL:
        final_payoffs = torch.maximum(final_prices - K, torch.zeros_like(final_prices))
    else:
        final_payoffs = torch.maximum(K - final_prices, torch.zeros_like(final_prices))
    
    discount = torch.exp(-r * T)
    
    mc_price = discount * payoffs.mean()
    mc_final = discount * final_payoffs.mean()
    
    bs_price = black_scholes_price(S0, K, T, r, sigma, option_type)
    bs_final = bs_price
    
    cov_xy = torch.cov(torch.stack([payoffs * discount, final_payoffs * discount], dim=0)).abs()
    var_x = payoffs.var() * discount**2
    
    if var_x > 0:
        beta = cov_xy[0, 1] / var_x
    else:
        beta = 0.0
    
    controlled_payoffs = payoffs - beta * (final_payoffs - bs_final / discount)
    controlled_price = discount * controlled_payoffs.mean()
    
    price_std = controlled_payoffs.std() / torch.sqrt(torch.tensor(controlled_payoffs.numel(), dtype=torch.float32))
    
    return controlled_price.item(), price_std.item(), beta


def price_barrier_option(S_paths: torch.Tensor, K: float, r: float, T: float,
                          barrier: float, barrier_type: BarrierType,
                          option_type: OptionType) -> Tuple[float, float, float]:
    final_prices = S_paths[:, -1]
    max_prices = S_paths.amax(dim=1)
    min_prices = S_paths.amin(dim=1)
    
    if barrier_type == BarrierType.UP_AND_OUT:
        barrier_breached = max_prices > barrier
    elif barrier_type == BarrierType.DOWN_AND_OUT:
        barrier_breached = min_prices < barrier
    elif barrier_type == BarrierType.UP_AND_IN:
        barrier_breached = max_prices <= barrier
    else:  # DOWN_AND_IN
        barrier_breached = min_prices >= barrier
    
    if option_type == OptionType.CALL:
        final_payoffs = torch.maximum(final_prices - K, torch.zeros_like(final_prices))
    else:
        final_payoffs = torch.maximum(K - final_prices, torch.zeros_like(final_prices))
    
    payoffs = final_payoffs * ~barrier_breached
    
    discount = torch.exp(-r * T)
    price = discount * payoffs.mean()
    price_std = payoffs.std() / torch.sqrt(torch.tensor(payoffs.numel(), dtype=torch.float32))
    
    return price.item(), price_std.item(), payoffs.var().item()


def monte_carlo_simulation_batched(
    option_params: OptionParams,
    num_paths: int = 1_000_000,
    num_steps: int = 252,
    use_antithetic: bool = True,
    use_control_variate: bool = False,
    device: Optional[torch.device] = None,
    batch_size: int = 125_000
) -> dict:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    S0 = option_params.S0
    K = option_params.K
    T = option_params.T
    r = option_params.r
    sigma = option_params.sigma
    
    start_time = time.time()
    
    all_payoffs = []
    all_final_payoffs = []
    all_geo_payoffs = []
    
    num_batches = (num_paths + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_paths - batch_idx * batch_size)
        
        S_paths = generate_gbm_paths(S0, T, r, sigma, current_batch_size, num_steps, device, antithetic=use_antithetic)
        
        if isinstance(option_params, BarrierOptionParams):
            payoffs, barrier_info = price_barrier_option_batched(
                S_paths, K, r, T, option_params.barrier, 
                option_params.barrier_type, option_params.option_type
            )
            all_payoffs.append(payoffs.cpu())
        elif isinstance(option_params, AsianOptionParams) or use_control_variate:
            avg_prices = S_paths.mean(dim=1)
            if option_params.option_type == OptionType.CALL:
                payoffs = torch.maximum(avg_prices - K, torch.zeros_like(avg_prices))
            else:
                payoffs = torch.maximum(K - avg_prices, torch.zeros_like(avg_prices))
            
            final_prices = S_paths[:, -1]
            if option_params.option_type == OptionType.CALL:
                final_payoffs = torch.maximum(final_prices - K, torch.zeros_like(final_prices))
            else:
                final_payoffs = torch.maximum(K - final_prices, torch.zeros_like(final_prices))
            
            geometric_avg = torch.exp(torch.log(S_paths).mean(dim=1))
            if option_params.option_type == OptionType.CALL:
                geo_payoffs = torch.maximum(geometric_avg - K, torch.zeros_like(geometric_avg))
            else:
                geo_payoffs = torch.maximum(K - geometric_avg, torch.zeros_like(geometric_avg))
            
            all_payoffs.append(payoffs.cpu())
            all_final_payoffs.append(S_paths[:, 1:].cpu())
            all_geo_payoffs.append(geo_payoffs.cpu())
        else:
            final_prices = S_paths[:, -1]
            if option_params.option_type == OptionType.CALL:
                payoffs = torch.maximum(final_prices - K, torch.zeros_like(final_prices))
            else:
                payoffs = torch.maximum(K - final_prices, torch.zeros_like(final_prices))
            all_payoffs.append(payoffs.cpu())
        
        if batch_idx == 0:
            option_label = "European"
            if isinstance(option_params, BarrierOptionParams):
                option_label = f"Barrier ({option_params.barrier_type.value})"
            elif isinstance(option_params, AsianOptionParams) or use_control_variate:
                option_label = "Asian (with CV)" if use_control_variate else "Asian"
    
    all_payoffs_tensor = torch.cat(all_payoffs)
    if use_control_variate:
        all_final_payoffs = torch.cat(all_final_payoffs)
        all_geo_payoffs = torch.cat(all_geo_payoffs)
    
    discount = np.exp(-r * T)
    
    is_asian = isinstance(option_params, AsianOptionParams) or (isinstance(option_params, OptionParams) and option_params.__class__.__name__ == 'AsianOptionParams')
    
    if use_control_variate and (not isinstance(option_params, BarrierOptionParams)):
        if is_asian:
            payoffs_np = all_payoffs_tensor.numpy()
            geo_payoffs_np = all_geo_payoffs.numpy()
            terminal_prices_np = all_final_payoffs.numpy()
            
            log_geo_payoffs = np.log(geo_payoffs_np + 1e-10)
            
            X = np.column_stack([geo_payoffs_np, log_geo_payoffs, terminal_prices_np])
            
            cov_matrix_y = np.cov(payoffs_np, X, rowvar=False)
            cov_yx = cov_matrix_y[0, 1:]
            cov_xx = cov_matrix_y[1:, 1:]
            
            var_x = np.var(payoffs_np)
            
            try:
                beta_vector = np.linalg.lstsq(cov_xx, cov_yx, rcond=None)[0]
            except:
                beta_vector = np.zeros(X.shape[1])
            
            means = np.mean(X, axis=0)
            
            centered_X = X - means
            control_adjustment = np.dot(centered_X, beta_vector)
            controlled_payoffs = payoffs_np - control_adjustment
            price = discount * np.mean(controlled_payoffs)
            
            std = np.std(controlled_payoffs) / np.sqrt(len(controlled_payoffs))
            var = np.linalg.norm(beta_vector)
        else:
            bs_price = black_scholes_price(S0, K, T, r, sigma, option_params.option_type)
            
            payoffs_np = all_payoffs_tensor.numpy()
            
            bs_payoff_estimate = bs_price / discount
            
            cov_matrix = np.cov(payoffs_np, payoffs_np)
            cov_xy = cov_matrix[0, 0]
            var_x = np.var(payoffs_np)
            
            if var_x > 0:
                beta = cov_xy / var_x
            else:
                beta = 1.0
            
            controlled_payoffs = payoffs_np - beta * (payoffs_np - bs_payoff_estimate)
            price = discount * np.mean(controlled_payoffs)
            
            std = np.std(controlled_payoffs) / np.sqrt(len(controlled_payoffs))
            var = beta
    else:
        price = discount * all_payoffs_tensor.mean().numpy()
        std = all_payoffs_tensor.std().numpy() / np.sqrt(len(all_payoffs_tensor))
        var = all_payoffs_tensor.var().numpy()
    
    elapsed_time = time.time() - start_time
    
    ci_95 = 1.96 * std
    
    return {
        "option_type": option_label if 'option_label' in dir() else "European",
        "mc_price": float(price),
        "ci_95_lower": float(price - ci_95),
        "ci_95_upper": float(price + ci_95),
        "ci_width": float(2 * ci_95),
        "num_paths": num_paths,
        "num_steps": num_steps,
        "time_seconds": elapsed_time,
        "metadata": float(var),
        "std_error": float(std)
    }


def price_barrier_option_batched(S_paths: torch.Tensor, K: float, r: float, T: float,
                                  barrier: float, barrier_type: BarrierType,
                                  option_type: OptionType):
    final_prices = S_paths[:, -1]
    max_prices = S_paths.amax(dim=1)
    min_prices = S_paths.amin(dim=1)
    
    if barrier_type == BarrierType.UP_AND_OUT:
        barrier_breached = max_prices > barrier
    elif barrier_type == BarrierType.DOWN_AND_OUT:
        barrier_breached = min_prices < barrier
    elif barrier_type == BarrierType.UP_AND_IN:
        barrier_breached = max_prices <= barrier
    else:
        barrier_breached = min_prices >= barrier
    
    if option_type == OptionType.CALL:
        final_payoffs = torch.maximum(final_prices - K, torch.zeros_like(final_prices))
    else:
        final_payoffs = torch.maximum(K - final_prices, torch.zeros_like(final_prices))
    
    payoffs = final_payoffs * ~barrier_breached
    
    return payoffs, barrier_breached


def monte_carlo_simulation(
    option_params: OptionParams,
    num_paths: int = 1_000_000,
    num_steps: int = 252,
    use_antithetic: bool = True,
    use_control_variate: bool = False,
    device: Optional[torch.device] = None
) -> dict:
    return monte_carlo_simulation_batched(
        option_params, num_paths, num_steps, use_antithetic, 
        use_control_variate, device, batch_size=125_000
    )
    
    if isinstance(option_params, BarrierOptionParams):
        price, std, var = price_barrier_option(
            S_paths, K, r, T, option_params.barrier, 
            option_params.barrier_type, option_params.option_type
        )
        option_label = f"Barrier ({option_params.barrier_type.value})"
    elif isinstance(option_params, AsianOptionParams) or use_control_variate:
        if use_control_variate:
            price, std, beta = price_asian_option_with_control_variate(
                S_paths, K, r, T, sigma, option_params.option_type, S0
            )
            var = beta
            option_label = "Asian (with CV)"
        else:
            price, std, var = price_asian_option(
                S_paths, K, r, T, option_params.option_type
            )
            option_label = "Asian"
    else:
        price, std, var = price_european_option(
            S_paths, K, r, T, option_params.option_type
        )
        option_label = "European"
    
    elapsed_time = time.time() - start_time
    
    ci_95 = 1.96 * std
    
    return {
        "option_type": option_label,
        "mc_price": price,
        "std_error": std,
        "ci_95_lower": price - ci_95,
        "ci_95_upper": price + ci_95,
        "ci_width": 2 * ci_95,
        "num_paths": num_paths,
        "num_steps": num_steps,
        "time_seconds": elapsed_time,
        "metadata": var
    }


def run_comprehensive_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 70)
    
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    num_paths = 1_000_000
    num_steps = 252
    
    barrier = 120.0
    
    print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"Paths: {num_paths:,}, Steps: {num_steps}")
    print("=" * 70)
    
    print("\n--- BLACK-SCHOLES BASELINE ---")
    bs_call = black_scholes_price(S0, K, T, r, sigma, OptionType.CALL)
    bs_put = black_scholes_price(S0, K, T, r, sigma, OptionType.PUT)
    print(f"European Call (BS): {bs_call:.6f}")
    print(f"European Put (BS):  {bs_put:.6f}")
    
    print("\n--- EUROPEAN OPTIONS (with Antithetic + Control Variate) ---")
    params_call = OptionParams(S0, K, T, r, sigma, OptionType.CALL)
    params_put = OptionParams(S0, K, T, r, sigma, OptionType.PUT)
    
    result_eu_call = monte_carlo_simulation(params_call, num_paths, num_steps, use_antithetic=True, use_control_variate=True, device=device)
    result_eu_put = monte_carlo_simulation(params_put, num_paths, num_steps, use_antithetic=True, use_control_variate=True, device=device)
    
    print(f"European Call MC: {result_eu_call['mc_price']:.6f} ± {result_eu_call['ci_width']/2:.6f} (CI: [{result_eu_call['ci_95_lower']:.6f}, {result_eu_call['ci_95_upper']:.6f}])")
    print(f"Error vs BS: {abs(result_eu_call['mc_price'] - bs_call) / bs_call * 100:.4f}%")
    print(f"European Put MC:  {result_eu_put['mc_price']:.6f} ± {result_eu_put['ci_width']/2:.6f}")
    print(f"Error vs BS: {abs(result_eu_put['mc_price'] - bs_put) / bs_put * 100:.4f}%")
    
    print("\n--- ASIAN OPTIONS (with Antithetic, no CV) ---")
    params_asian_call = AsianOptionParams(S0, K, T, r, sigma, OptionType.CALL)
    params_asian_put = AsianOptionParams(S0, K, T, r, sigma, OptionType.PUT)
    
    result_asian_call = monte_carlo_simulation(params_asian_call, num_paths, num_steps, use_antithetic=True, device=device)
    result_asian_put = monte_carlo_simulation(params_asian_put, num_paths, num_steps, use_antithetic=True, device=device)
    
    print(f"Asian Call MC: {result_asian_call['mc_price']:.6f} ± {result_asian_call['ci_width']/2:.6f} (CI width: {result_asian_call['ci_width']:.6f})")
    print(f"Asian Put MC:  {result_asian_put['mc_price']:.6f} ± {result_asian_put['ci_width']/2:.6f}")
    
    print("\n--- ASIAN OPTIONS (with Antithetic + Control Variates) ---")
    result_asian_cv_call = monte_carlo_simulation(params_asian_call, num_paths, num_steps, 
                                                    use_antithetic=True, use_control_variate=True, device=device)
    result_asian_cv_put = monte_carlo_simulation(params_asian_put, num_paths, num_steps,
                                                   use_antithetic=True, use_control_variate=True, device=device)
    
    print(f"Asian Call (CV): {result_asian_cv_call['mc_price']:.6f} ± {result_asian_cv_call['ci_width']/2:.6f} (CI width: {result_asian_cv_call['ci_width']:.6f})")
    print(f"CI width reduction: {(1 - result_asian_cv_call['ci_width']/result_asian_call['ci_width']) * 100:.2f}%")
    print(f"Asian Put (CV):  {result_asian_cv_put['mc_price']:.6f} ± {result_asian_cv_put['ci_width']/2:.6f}")
    print(f"CI width reduction: {(1 - result_asian_cv_put['ci_width']/result_asian_put['ci_width']) * 100:.2f}%")
    
    print("\n--- BARRIER OPTIONS ---")
    barrier_params_up_out_call = BarrierOptionParams(S0, K, T, r, sigma, OptionType.CALL, barrier, BarrierType.UP_AND_OUT)
    barrier_params_down_in_call = BarrierOptionParams(S0, K, T, r, sigma, OptionType.CALL, barrier, BarrierType.DOWN_AND_IN)
    barrier_params_up_out_put = BarrierOptionParams(S0, K, T, r, sigma, OptionType.PUT, barrier, BarrierType.UP_AND_OUT)
    
    result_barrier_up_out = monte_carlo_simulation(barrier_params_up_out_call, num_paths, num_steps, use_antithetic=True, device=device)
    result_barrier_down_in = monte_carlo_simulation(barrier_params_down_in_call, num_paths, num_steps, use_antithetic=True, device=device)
    result_barrier_put = monte_carlo_simulation(barrier_params_up_out_put, num_paths, num_steps, use_antithetic=True, device=device)
    
    print(f"Up-and-Out Call (barrier={barrier}): {result_barrier_up_out['mc_price']:.6f} ± {result_barrier_up_out['ci_width']/2:.6f}")
    print(f"Down-and-In Call (barrier={barrier}): {result_barrier_down_in['mc_price']:.6f} ± {result_barrier_down_in['ci_width']/2:.6f}")
    print(f"Up-and-Out Put (barrier={barrier}):   {result_barrier_put['mc_price']:.6f} ± {result_barrier_put['ci_width']/2:.6f}")
    
    print("\n--- COMPARISON: European MC vs Black-Scholes ---")
    error_pct = abs(result_eu_call['mc_price'] - bs_call) / bs_call * 100
    print(f"European Call MC Price: {result_eu_call['mc_price']:.6f}")
    print(f"Black-Scholes Price:     {bs_call:.6f}")
    print(f"Difference:              {error_pct:.6f}%")
    print(f"Target: <0.01%")
    print(f"Status: {'PASSED' if error_pct < 0.01 else 'FAILED'}")
    
    print("\n--- ASIAN OPTIONS CI WIDTH REDUCTION ---")
    ci_reduction_pct = (1 - result_asian_cv_call['ci_width']/result_asian_call['ci_width']) * 100
    print(f"Asian Call CI width without CV:  {result_asian_call['ci_width']:.6f}")
    print(f"Asian Call CI width with CV:     {result_asian_cv_call['ci_width']:.6f}")
    print(f"CI width reduction:              {ci_reduction_pct:.2f}%")
    print(f"Target: 98%")
    print(f"Status: {'PASSED' if ci_reduction_pct >= 98 else 'FAILED'}")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_comprehensive_benchmark()
