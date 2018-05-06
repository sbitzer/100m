#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 14:18:09 2018

@author: bitzer
"""

import pystan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#%% create some artificial records data
N = 12
A = 15

ryears = np.random.rand(N) < 0.4
ryears[0] = True

ras = np.zeros(N, dtype=int)
ras[ryears] = np.random.randint(1, A+1, ryears.sum())

records = np.ones(N) * np.random.normal(9.8, 0.01)
for r in range(1, N):
    records[r] = records[r-1]
    if ryears[r]:
        records[r] -= np.abs(np.random.normal(0, 0.01))

Rs = pd.Series([np.sum(ras == a) for a in range(1, A+1)],
               index=np.arange(1, A+1))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(records)
Rs.plot.bar(ax=axes[1])


#%% compile the Stan model
code = """functions {
    real GaussRecord_lpdf(real rec, int reca, vector ameans, vector astds){
        int A = num_elements(ameans);
        real lp = 0;

        for (a in 1:A) {
            if (a == reca)
                lp += -log(astds[a]) - square(rec - ameans[a]) / square(astds[a]) / 2;
            else
                lp += log1m(Phi((rec - ameans[a]) / astds[a]));
        }
        
        return lp;
    }
}

data {
    int<lower=1> A;
    int<lower=1> N;
    
    vector[N] records;
    int<lower=0> recordsa[N];
    
    real<lower=0> astd_std;
}

parameters {
    real worldspeed;
    vector[A] abilities;
    vector<lower=0>[A] inconsistencies;
}

transformed parameters {
    vector[A] ameans = worldspeed + 10 + abilities;
    vector[A] astds = inconsistencies * astd_std;
}

model {
    worldspeed ~ normal(0, 1);
    abilities ~ normal(0, 1);
    inconsistencies ~ normal(0, 1);
    
    for (n in 1:N)
        records[n] ~ GaussRecord(recordsa[n], ameans, astds);
}"""

sm = pystan.StanModel(model_code=code)


#%% sample
standata = dict(A=A, N=N, records=records, recordsa=ras, astd_std=0.1)
fit = sm.sampling(standata, iter=2000, chains=4)


#%% check the results
samples = fit.extract(('ameans', 'astds', 'worldspeed'))

def tidy(param):
    param = pd.DataFrame(
            samples[param], 
            columns=pd.Index(np.arange(1, A+1), name='athlete'))
    param = param.stack()
    param.name = 'sample'
    param = param.reset_index('athlete')
    param['R'] = param.athlete.map(Rs)
    
    return param

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(25, 10))

for param, ax in zip(['ameans', 'astds'], axes):
    sns.violinplot(x="athlete", y="sample", hue="R", data=tidy(param), 
                   ax=ax);
    ax.set_ylabel(param)
    
axes[1].set_xlabel('athlete')