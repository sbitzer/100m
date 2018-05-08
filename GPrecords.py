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
functions {
    real GaussRecord_lpdf(real rec, int reca, int[] athletes, vector ameans, 
                          vector astds){
        int A = num_elements(athletes);
        real lp = 0;
        int athlete;

        for (a in 1:A) {
            athlete = athletes[a];
            if (athlete == reca)
                lp += -log(astds[athlete]) 
                      - square(rec - ameans[athlete]) 
                      / square(astds[athlete]) / 2;
            else
                lp += log1m(Phi((rec - ameans[athlete]) / astds[athlete]));
        }
        
        return lp;
    }
}

data {
    int<lower=1> A;
    int<lower=1> AY;
    int<lower=1> N;
    
    int<lower=1> athletes[N, AY];
    vector[N] records;
    real years[N];
    int<lower=0> recordsa[N];
    
    real<lower=0> astd_std;
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
    vector<lower=0>[A] inconsistencies;
}

transformed parameters {
    real rho = exp(rho_raw + log(10));
    vector[A] ameans = abilities * ability_std;
    vector[A] astds = inconsistencies * astd_std;
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
    alpha ~ normal(0, 1);
    
    // effective prior for rho: lognormal(log(10), 1)
    rho_raw ~ normal(0, 1);

    // effective prior: normal(10, 1)
    average ~ normal(0, 1);
    
    ability_std ~ normal(0, 1);
    abilities ~ normal(0, 1);
    
    // effective prior: normal(0, astd_std)
    inconsistencies ~ normal(0, 1);
    
    for (n in 1:N) {
        records[n] ~ GaussRecord(recordsa[n], athletes[n], 
                                 f[n] + average + 10 + ameans, astds);
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
    average = np.random.rand() * 2 + maxT - 10
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
            inconsistencies=np.random.rand(A) * 4 + 1)


def wpi_quantiles(fit, years):
    """Returns inferred quantiles of hidden world performance index."""
    
    samples = fit.extract(['f', 'average'])
    
    samples = pd.DataFrame(samples['average'][:, None] + samples['f'] + 10, 
                           columns=years)

    return samples.stack().groupby('Date').quantile(
            [0.025, 0.5, 0.975]).unstack('Date').T