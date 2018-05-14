#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 16:46:46 2018

@author: bitzer
"""

import pystan
import numpy as np
import pandas as pd


#%% definition of Stan model
modelcode = """
data {
    int<lower=1> A;
    int<lower=1> AY;
    int<lower=1> N;
    
    int<lower=1> athletes[N, AY];
    vector[N] records;
    real years[N];
    int<lower=0> recordsa[N];
}

transformed data {
    real delta = 1e-9;
}

parameters {
    real average;
        
    real rho_raw;
    real<lower=0> alpha;
    vector[N] eta;

    real<lower=0> ability_std;
    vector[A] abilities;
    vector[A] astds_raw;
}

transformed parameters {
    real rho = exp(rho_raw + log(20));
    vector[A] ameans = abilities * ability_std;
    vector[A] astds = exp(astds_raw + log(0.2));
    vector[N] f;
    {
        matrix[N, N] L_K;
        matrix[N, N] K = cov_exp_quad(years, alpha, rho);
        // diagonal elements
        for (n in 1:N)
            K[n, n] = K[n, n] + delta;
        L_K = cholesky_decompose(K);
        f = L_K * eta;
    }
}

model {
    eta ~ normal(0, 1);
    
    // should probably be < 1, but use larger std to a bit less informative
    alpha ~ normal(0, 2);
    
    // effective prior for rho: lognormal(log(20), 1)
    rho_raw ~ normal(0, 1);

    // effective prior: normal(10, 1)
    average ~ normal(10, 1);
    
    ability_std ~ normal(0, 3);
    abilities ~ normal(0, 3);
    
    // effective prior: lognormal(log(0.2), 1)
    astds_raw ~ normal(0, 1);
    
    for (n in 1:N) {
        for (a in 1:AY) {
            int athlete = athletes[n, a];
            real amean = f[n] + average + ameans[athlete];
            
            if (recordsa[n] == athlete)
                records[n] ~ normal(amean, astds[athlete]);
            else
                target += normal_lccdf(records[n] | amean, astds[athlete]);
        }
    }
}"""

sm = pystan.StanModel(model_code=modelcode)

def init(N, A, maxT):
    """Custom initialisation for GP records model.
    
    this custom initialisation tries to ensure that the inital values
    for the average tend to be above the world record times,
    otherwise the initialisation should be very similar to the standard
    random initialisation within [-2, 2]
    """
    average = np.random.rand() * 2 + maxT
    ability_std = .1 + np.random.rand() * 0.1
    abilities = np.random.rand(A) * 4 - 2
    
#    ind = (abilities * ability_std + average + 10) < maxT
#    abilities[ind] = np.abs(abilities[ind])
    
    return dict(
            eta=np.random.rand(N) * 4 - 2,
            alpha=np.random.rand() * 2,
            rho_raw=np.random.rand() * 4 - 2,
            average=average,
            ability_std=ability_std,
            abilities=abilities,
            astds_raw=np.random.rand(A) * 4 - 2)


def wpi_quantiles(fit, years):
    """Returns inferred quantiles of hidden world performance index."""
    
    samples = fit.extract(['f', 'average'])
    
    samples = pd.DataFrame(samples['average'][:, None] + samples['f'], 
                           columns=years)

    return samples.stack().groupby('Date').quantile(
            [0.025, 0.5, 0.975]).unstack('Date').T