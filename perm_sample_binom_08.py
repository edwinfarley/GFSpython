#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:52:11 2017

@author: Edwin
"""

'''
With a permutation, sample B.
Now sample permutations using metropolis method. End on some permutation: this permutation goes with that B.
Start over and repeat a set number of times.
'''


import numpy as np
import pandas as pd
import pymc3 as pm
import random

from master import build_permutation


eps_sigma_sq = 1
v=1


def simulate_data_logistic(N, B):
    """
    Simulate a random dataset using a noisy
    linear process.

    N: Number of data points to simulate
    B: Vector of regression parameters. Defines number of covariates. (Include B_0)
    """
    seed = 7
    df = pd.DataFrame(
        {str("x1"): np.random.RandomState(seed).uniform(size = N)})
    for i in range(2, len(B)):
        print(i)
        df_i = pd.DataFrame(
                {str("x" + str(i)): np.random.RandomState(seed+i).uniform(size = N)})
        df = pd.concat([df, df_i], axis = 1)


    # Use a linear model (y ~ beta_0 + beta_1*x + epsilon) to
    # generate a column 'y' of responses based on 'x'
    #Betas are normally distributed with mean 0 and variance eps_sigma_sq
    p = np.exp(B[0] + np.matmul(pd.DataFrame.as_matrix(df), np.transpose(B[1:])) \
               + np.random.RandomState(42).normal(0, eps_sigma_sq, N))
    p = p/(1+p)
    df["y"] = np.round(p)

    return df

def logistic_likelihood(x, y):
    exp_x = min(max(np.exp(x), 1e-16), 1e16)
    P = min(exp_x/(1+exp_x), 1 - 1e-16)
    l = ((np.log(P)*y)+(np.log(1-P)*(1-y)))
    return(l)
    
def logistic_likelihood_swap(i, j, x_i, x_j, x_i_swap, x_j_swap, y):
    #Calculate log-likelihoods (terms that cancel are not included)
    #New likelihood with swapped values
    new_l = logistic_likelihood(x_i_swap, y[j])*np.isfinite(y[j]) \
        + logistic_likelihood(x_j_swap, y[i])*np.isfinite(y[i])
    #Likelihood without swapped values
    old_l = logistic_likelihood(x_i, y[i])*np.isfinite(y[i]) \
        + logistic_likelihood(x_j, y[j])*np.isfinite(y[j])
    return([new_l, old_l])
