# -*- coding: utf-8 -*-
# @Author: Aaron Reding
# @Date:   2022-11-05 08:44:56
# @Last Modified by:   aaronreding
# @Last Modified time: 2022-11-05 13:01:40
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az


def deaggregate():
    logarea = np.array([1.35, 1.60, 1.75, 1.85, 1.95, 2.05, 2.15, 2.25, 2.35])
    survived = np.array([13, 19, 67, 45, 71, 50, 35, 7, 1])
    died = np.array([0, 0, 2, 5, 8, 20, 31, 49, 12])

    assert logarea.shape == survived.shape == died.shape
    trials = survived + died

    deaggregated = []
    for la, surv, dead in zip(logarea, survived, died):
        if surv:
            for _ in range(surv):
                row = [la, 1]
                deaggregated.append(row)
        if dead:
            for _ in range(dead):
                row = [la, 0]
                deaggregated.append(row)

    assert len(deaggregated) == trials.sum()

    return pd.DataFrame(deaggregated, columns=["log(area+1)", "survived"])


def BernoulliVersion():
    print("Bernoulli model results:")
    # load data
    #############################################################################
    data = deaggregate()

    # run model
    #############################################################################
    with pm.Model() as m:
        y_data = pm.Data("y_data", data["survived"].to_numpy(), mutable=False)
        x_data = pm.Data("x_data", data["log(area+1)"].to_numpy(), mutable=True)

        beta0 = pm.Normal("beta0", 0, 50)
        beta1 = pm.Normal("beta1", 0, 50)

        p = pm.Deterministic("p", pm.math.invlogit(beta0 + x_data * beta1))

        pm.Bernoulli("lik", p, observed=y_data)

        trace = pm.sample(3000)

    # print posterior summary and deviance
    #############################################################################
    print(az.summary(trace, var_names="~p"))
    print(az.waic(trace, scale="deviance"))

    # prediction
    #############################################################################
    new_obs = np.array([2.0])
    pm.set_data({"x_data": new_obs}, model=m)
    # turning off the progress bar because of a weird formatting error on print
    ppc = pm.sample_posterior_predictive(
        trace, model=m, predictions=True, progressbar=False
    )

    pred_p = az.summary(ppc.predictions)["mean"].mean()

    print(f"\nPredicted probability when log(area + 1) = 2:\n {pred_p:.3f}")


def BinomialVersion():
    print("\nBinomial model results:")
    # load data
    #############################################################################
    logarea = np.array([1.35, 1.60, 1.75, 1.85, 1.95, 2.05, 2.15, 2.25, 2.35])
    survived = np.array([13, 19, 67, 45, 71, 50, 35, 7, 1])
    died = np.array([0, 0, 2, 5, 8, 20, 31, 49, 12])

    assert logarea.shape == survived.shape == died.shape
    trials = survived + died

    # run model
    #############################################################################
    with pm.Model() as m2:
        p_data = pm.Data("p_data", survived, mutable=False)
        x_data = pm.Data("x_data", logarea, mutable=True)

        beta0 = pm.Normal("beta0", 0, 50)
        beta1 = pm.Normal("beta1", 0, 50)

        p = pm.Deterministic("p", pm.math.invlogit(beta0 + x_data * beta1))

        pm.Binomial("lik", n=trials, p=p, observed=p_data)

        trace = pm.sample(3000)

    # print posterior summary and deviance
    #############################################################################
    print(az.summary(trace, var_names="~p"))
    print(az.waic(trace, scale="deviance"))

    # prediction
    #############################################################################
    new_obs = np.array([2.0])
    pm.set_data({"x_data": new_obs}, model=m2)
    # turning off the progress bar because of a weird formatting error on print
    ppc = pm.sample_posterior_predictive(
        trace, model=m2, predictions=True, progressbar=False
    )

    pred_p = az.summary(ppc.predictions)["mean"].sum() / trials.sum()

    print(f"\nPredicted probability when log(area + 1) = 2:\n {pred_p:.3f}")


if __name__ == "__main__":
    BernoulliVersion()
    BinomialVersion()
