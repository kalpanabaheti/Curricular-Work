# -*- coding: utf-8 -*-
# @Author: Aaron Reding
# @Date:   2022-11-05 08:41:30
# @Last Modified by:   aaronreding
# @Last Modified time: 2022-11-05 12:45:07
import pandas as pd
import numpy as np
import pymc as pm
from pymc.math import switch, ge
import arviz as az


def main():
    # load data
    #############################################################################
    paddy_df = pd.read_csv("paddy.dat", header=None, delim_whitespace=True)

    # adhesion to steel
    ats = paddy_df[0].to_numpy()
    # adhesion to rubber
    atr = paddy_df[1].to_numpy()

    # run model
    #############################################################################
    with pm.Model() as m:
        x = pm.Data("x", ats, mutable=True)
        y = pm.Data("y", atr, mutable=False)

        intercept = pm.Normal("intercept", 0, 20)
        beta = pm.Normal("beta", 0, 10)
        sigma = pm.Exponential("sigma", 0.05)
        mu = intercept + beta * x

        pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y)

        pm.Deterministic("prob_H1", switch(ge(intercept, 0), 1, 0))
        pm.Deterministic("prob_H2", switch(ge(beta, 1), 1, 0))

        trace = pm.sample(3000)

        pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    # print posterior summary and R2 score
    #############################################################################
    print(az.summary(trace, hdi_prob=0.95))
    y_true = paddy_df[1].to_numpy()
    y_pred = trace.posterior_predictive.stack(sample=("chain", "draw"))[
        "likelihood"
    ].values.T
    print(az.r2_score(y_true=ats, y_pred=y_pred))

    # prediction
    #############################################################################
    print("Prediction:\n")
    new_obs = np.array([2.0])
    pm.set_data({"x": new_obs}, model=m)
    ppc = pm.sample_posterior_predictive(trace, model=m, predictions=True)

    print(az.summary(ppc.predictions, hdi_prob=0.95, kind="stats").mean())


if __name__ == "__main__":
    main()
