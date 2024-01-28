import numpy as np
import matplotlib.pyplot as plt
from rich.progress import track

# HW4 Q1


def calc_hdi(samples: np.ndarray, alpha: float = 0.05) -> tuple:
    """
    Calculate minimum-width credible interval (HPD credible set)

    from https://areding.github.io/6420-pymc/unit4/Unit4-GammaGamma.html
    based on arviz.hdi function

    samples: samples from posterior
    alpha: credibility of the interval == 1 - alpha

    returns tuple of the lower and upper bounds of the interval
    """
    n = len(samples)
    x = np.sort(samples)

    lower_idx = int(np.floor(alpha * n))
    x_left = x[:lower_idx]
    x_right = x[n - lower_idx :]

    idx = np.argmin(x_right - x_left)

    upper_bound = x_right[idx]
    lower_bound = x_left[idx]

    return lower_bound, upper_bound


rng = np.random.default_rng(1)

samples = 100000
burn = 500
accepted = np.zeros(samples)

theta = 1  # init
theta_samples = np.zeros(samples)

# independent proposals
alpha = 1
beta = 3
proposals = rng.gamma(alpha, 1 / beta, size=samples)
unifs = rng.random(size=samples)


def target(theta):
    x = -2  # given observation
    return np.exp(-theta * (1 + 0.5 * x**2))


def propprop(theta, alpha, beta):
    """
    proportional to proposal
    """
    return theta ** (alpha - 1) * np.exp(-beta * theta)


for i in track(range(samples)):
    theta_prop = proposals[i]

    ar = (target(theta_prop) * propprop(theta, alpha, beta)) / (
        target(theta) * propprop(theta_prop, alpha, beta)
    )

    rho = min(1, ar)

    if unifs[i] < rho:
        theta = theta_prop
        accepted[i] = 1

    theta_samples[i] = theta


print(f"acceptance rate: {accepted.sum()/samples}")
theta_samples = theta_samples[burn:]

# posterior densities
plt.hist(theta_samples, 80)
plt.xlabel("theta")
plt.show()

# traceplots
plt.plot(range(samples - burn), theta_samples, linewidth=0.1)
plt.ylabel("theta")
plt.xlabel("iteration")
plt.show()

# traceplot over last 1000 samples
plt.plot(
    range((samples - burn) - 1000, samples - burn), theta_samples[-1000:], linewidth=0.5
)
plt.ylabel("theta")
plt.xlabel("iteration")
plt.show()

print(f"{np.mean(theta_samples)=}")
print(f"{calc_hdi(theta_samples, alpha=.03)=}")
