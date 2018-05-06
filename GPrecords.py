#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 16:46:46 2018

@author: bitzer
"""

import pystan


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
    vector[N] years;
    int<lower=0> recordsa[N];
    
    real<lower=0> astd_std;
}

parameters {
    real worldspeed;
    real<lower=0> ability_std;
    vector[A] abilities;
    vector<lower=0>[A] inconsistencies;
}

transformed parameters {
    vector[A] ameans = worldspeed + 10 + abilities * ability_std * 2;
    vector[A] astds = inconsistencies * astd_std;
}

model {
    worldspeed ~ normal(0, 1);
    ability_std ~ normal(0, 1);
    abilities ~ normal(0, 1);
    inconsistencies ~ normal(0, 1);
    
    for (n in 1:N)
        records[n] ~ GaussRecord(recordsa[n], athletes[n], ameans, astds);
}"""

sm = pystan.StanModel(model_code=modelcode)
