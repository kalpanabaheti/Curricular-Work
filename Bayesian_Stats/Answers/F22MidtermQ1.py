# -*- coding: utf-8 -*-
# @Author: Aaron Reding
# @Date:   2022-10-26 10:01:07
# @Last Modified by:   aaronreding
# @Last Modified time: 2022-10-26 10:22:58
import numpy as np
from scipy.stats import beta


# from https://areding.github.io/6420-pymc/unit4/Unit4-GammaGamma.html
def calc_hdi(samples: np.ndarray, alpha: float = 0.05) -> tuple:
    """
    Calculate minimum-width credible interval (HPD credible set)

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


def sampling_method():
    dist_A = beta(17, 283)
    dist_B = beta(25, 275)

    weight_A = 0.1023
    weight_B = 1 - weight_A

    total_sample_ct = 1e8
    A_sample_ct = int(weight_A * total_sample_ct)
    B_sample_ct = int(weight_B * total_sample_ct)

    mixture_samples = np.append(dist_A.rvs(A_sample_ct), dist_B.rvs(B_sample_ct))

    eqt_set = np.quantile(mixture_samples, [0.025, 0.975])
    hdi_set = calc_hdi(mixture_samples)

    print(f"Equi-tailed credible interval: {eqt_set}")
    print(f"HDI credible interval: {hdi_set}")


if __name__ == "__main__":
    sampling_method()
