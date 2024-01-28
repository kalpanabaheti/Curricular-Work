# -*- coding: utf-8 -*-
# @Author: Aaron Reding
# @Date:   2022-11-12 09:29:36
# @Last Modified by:   aaronreding
# @Last Modified time: 2022-11-12 10:25:15
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az


def main():
    # load data
    #############################################################################
    data = pd.read_csv("Q2data.csv")

    censored = data["censored"].to_numpy()
    treatment = data["treatment"].to_numpy()
    months = data["months"].to_numpy()
    age = data["age"].to_numpy()

    # run model
    #############################################################################
    with pm.Model() as m:
        alpha = pm.Exponential("alpha", 0.5)
        beta0 = pm.Normal("beta0", 0, 100)
        beta1 = pm.Normal("beta1", 0, 100)
        beta2 = pm.Normal("beta2", 0, 100)

        beta = pm.math.exp(beta0 + beta1 * treatment + beta2 * age) ** (-1 / alpha)

        obs_latent = pm.Weibull.dist(alpha=alpha, beta=beta)
        pm.Censored("lik", obs_latent, lower=None, upper=censored, observed=months)

        trace = pm.sample(3000, init="jitter+adapt_diag_grad", target_accept=0.9)

    # print posterior summary
    #############################################################################
    print(az.summary(trace, hdi_prob=0.95))


if __name__ == "__main__":
    main()
