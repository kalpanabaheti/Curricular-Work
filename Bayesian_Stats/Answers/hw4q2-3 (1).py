import numpy as np
from rich.progress import track

# HW4Q2

rng = np.random.default_rng(11)

samples = 50000
burn = 500
accepted = np.zeros(samples)

y_mean = 105.5
μ = 110
τ = 120
n = 10

θ = μ  # init
θ_samples = np.zeros(samples)

λ = 1  # init
λ_samples = np.zeros(samples)

for i in track(range(samples)):
    θ_mean = τ * y_mean / (τ + λ * 90 / n) + λ * 90 * μ / (n * τ + λ * 90)
    θ_var = τ * 90 / (n * τ + λ * 90)
    λ_mean = (τ + (θ - μ) ** 2) / (2 * τ)

    θ_new = rng.normal(θ_mean, θ_var**0.5)
    λ_new = rng.exponential(1/λ_mean)

    θ_samples[i] = θ = θ_new
    λ_samples[i] = λ = λ_new

θ_samples = θ_samples[burn:]

print(f"{np.mean(θ_samples)=}")
print(f"{np.var(θ_samples)=}")
print(f"{np.quantile(θ_samples, [.03, .97])=}")
