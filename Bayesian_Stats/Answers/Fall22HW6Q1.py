# -*- coding: utf-8 -*-
# @Author: Aaron Reding
# @Date:   2022-11-12 09:02:29
# @Last Modified by:   aaronreding
# @Last Modified time: 2022-11-12 09:50:02
import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az


def load_data():
    # load data
    #############################################################################
    data = pd.read_csv("Q1data.csv")

    temp = np.nan_to_num(data["temp"].to_numpy(), nan=-1)
    develop = np.nan_to_num(data["develop"].to_numpy(), nan=-1)
    temp = np.ma.masked_values(temp, value=-1)
    develop = np.ma.masked_values(develop, value=-1)

    return temp, develop


def Q1_ab(X, y):
    # run model (part a)
    #############################################################################
    with pm.Model() as m:
        beta0 = pm.Normal("beta0", 0, 100)
        beta1 = pm.Normal("beta1", 0, 100)
        sigma = pm.Exponential("tau", 0.05)

        x_imputed = pm.Normal("x_imputed", 80, 10, observed=X)
        mu = beta0 + beta1 * x_imputed
        pm.Normal("lik", mu, sigma=sigma, observed=y)

        trace = pm.sample(3000, target_accept=0.9)
        pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    # print posterior summary (this answers part b)
    #############################################################################
    print(az.summary(trace, hdi_prob=0.90))

    # plot posterior predictive distribution (this will help with c)
    #############################################################################
    az.plot_ppc(trace, num_pp_samples=100)
    plt.show()


def Q1_c(X, y):
    # clearly that value of 1.4 for one of the missing develop values is too low.
    # the posterior predictive also doesn't fit the data well.
    # this part was open-ended, so your solution may be different.
    # what if we transform the y data? log(y) seems like it could be decent fit

    with pm.Model() as m2:
        beta0 = pm.Normal("beta0", 0, 100)
        beta1 = pm.Normal("beta1", 0, 100)
        sigma = pm.Exponential("tau", 0.05)

        x_imputed = pm.Normal("x_imputed", 80, 10, observed=X)
        mu = beta0 + beta1 * x_imputed
        pm.LogNormal("lik", mu, sigma=sigma, observed=y)

        trace = pm.sample(3000, target_accept=0.9)
        #pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    # print posterior summary (see if missing values fit better)
    #############################################################################
    print(az.summary(trace, hdi_prob=0.90))

    # plot posterior predictive distribution again
    #############################################################################
    #az.plot_ppc(trace, num_pp_samples=100)
    #plt.show()


if __name__ == "__main__":
    X, y = load_data()
    Q1_ab(X, y)
    Q1_c(X, y)


# Results part a
#                          mean      sd   hdi_5%  hdi_95%
# beta0                 116.950  10.896  100.371  135.625
# beta1                  -1.287   0.152   -1.539   -1.044
# x_imputed_missing[0]   71.552   4.342   64.425   78.472
# x_imputed_missing[1]   70.946   4.299   63.524   77.531
# lik_missing[0]         29.960   6.073   19.201   39.188
# lik_missing[1]         19.104   6.120    9.146   29.119
# lik_missing[2]          1.482   6.581   -9.556   11.944
# lik_missing[3]         20.508   6.048   10.947   30.842
# tau                     5.852   0.946    4.329    7.261


# Results part c (LogNormal likelihood)
#                         mean     sd  hdi_5%  hdi_95%
# beta0                  6.657  0.312   6.141    7.162
# beta1                 -0.050  0.004  -0.057   -0.042
# x_imputed_missing[0]  68.964  3.309  63.352   74.093
# x_imputed_missing[1]  68.470  3.294  63.207   73.909
# tau                    0.165  0.027   0.123    0.207
# lik_missing[0]        27.754  4.802  20.039   35.345
# lik_missing[1]        18.283  3.173  12.987   23.125
# lik_missing[2]         9.311  1.811   6.315   12.066
# lik_missing[3]        19.361  3.343  14.121   24.609
