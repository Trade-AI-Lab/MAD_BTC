import torch
from torch import erf, sqrt

def black_scholes_call(S, K, T, r, sigma):
    S, K, T, r, sigma = map(torch.tensor, [S, K, T, r, sigma])
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    N_d1 = 0.5 * (1 + erf(d1 / sqrt(torch.tensor(2))))
    N_d2 = 0.5 * (1 + erf(d2 / sqrt(torch.tensor(2))))
    call_price = S * N_d1 - K * torch.exp(-r * T) * N_d2
    return call_price.item()  # Return the scalar value

def black_scholes_put(S, K, T, r, sigma):
    S, K, T, r, sigma = map(torch.tensor, [S, K, T, r, sigma])
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    N_minus_d1 = 0.5 * (1 - erf(d1 / sqrt(torch.tensor(2))))
    N_minus_d2 = 0.5 * (1 - erf(d2 / sqrt(torch.tensor(2))))
    put_price = K * torch.exp(-r * T) * N_minus_d2 - S * N_minus_d1
    return put_price.item()  # Return the scalar value

# Example usage
S = 100.0   # Current stock price
K = 100.0   # Strike price
T = 1.0     # Time to maturity
r = 0.05    # Risk-free rate
sigma = 0.2 # Volatility

call_price = black_scholes_call(S, K, T, r, sigma)
put_price = black_scholes_put(S, K, T, r, sigma)

print(f"Black-Scholes Call Option Price: {call_price}")
print(f"Black-Scholes Put Option Price: {put_price}")

import torch
from torch import erf, sqrt

def black_scholes_call(S, K, T, r, sigma):
    S, K, T, r, sigma = map(torch.tensor, [S, K, T, r, sigma])
    dT_sqrt = sqrt(T)  # Square root of time to maturity
    Z = torch.randn(1)  # Generate a random number from a standard normal distribution
    S_T = S * torch.exp((r - 0.5 * sigma**2) * T + sigma * dT_sqrt * Z)
    call_price = torch.exp(-r * T) * torch.max(S_T - K, torch.tensor(0.0))
    return call_price.item()  # Return the scalar value

def black_scholes_put(S, K, T, r, sigma):
    S, K, T, r, sigma = map(torch.tensor, [S, K, T, r, sigma])
    dT_sqrt = sqrt(T)  # Square root of time to maturity
    Z = torch.randn(1)  # Generate a random number from a standard normal distribution
    S_T = S * torch.exp((r - 0.5 * sigma**2) * T + sigma * dT_sqrt * Z)
    put_price = torch.exp(-r * T) * torch.max(K - S_T, torch.tensor(0.0))
    return put_price.item()  # Return the scalar value

# Example usage
S = 100.0   # Current stock price
K = 100.0   # Strike price
T = 1.0     # Time to maturity
r = 0.05    # Risk-free rate
sigma = 0.2 # Volatility

call_price = black_scholes_call(S, K, T, r, sigma)
put_price = black_scholes_put(S, K, T, r, sigma)

print(f"Black-Scholes Call Option Price with Stochastic Component: {call_price}")
print(f"Black-Scholes Put Option Price with Stochastic Component: {put_price}")

