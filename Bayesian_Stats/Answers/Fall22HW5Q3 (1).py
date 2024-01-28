# -*- coding: utf-8 -*-
# @Author: Aaron Reding
# @Date:   2022-11-05 08:45:10
# @Last Modified by:   aaronreding
# @Last Modified time: 2022-11-05 12:56:51
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

def main():
    # load data
    #############################################################################
    data = pd.read_csv("hospitaladmissions.dat", header=None, delim_whitespace=True)

    # run model
    #############################################################################
    with pm.Model() as m:
        SO_2 = pm.Data("SO_2", data[3].to_numpy(), mutable=True)
        NO_2 = pm.Data("NO_2", data[4].to_numpy(), mutable=True)
        y = pm.Data("admission", data[5].to_numpy(), mutable=False)

        beta0 = pm.Normal("beta0", 0, 100)
        beta1 = pm.Normal("beta1", 0, 100)
        beta2 = pm.Normal("beta2", 0, 100)

        λ = pm.math.exp(beta0 + beta1 * SO_2 + beta2 * NO_2)

        pm.Poisson("likelihood", λ, observed=y)

        trace = pm.sample(3000)

    # print posterior summary
    #############################################################################
    print(az.summary(trace, hdi_prob=0.95))

    # prediction
    #############################################################################
    new_SO2 = np.array([44.0])
    new_NO2 = np.array([100.0])
    pm.set_data({"SO_2": new_SO2, "NO_2": new_NO2}, model=m)
    ppc = pm.sample_posterior_predictive(trace, model=m, predictions=True)
    print(az.summary(ppc.predictions, hdi_prob=0.95, kind="stats").mean())


if __name__ == "__main__":
    main()
