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

from scipy.stats import norm
#from localsearch import *

eps_sigma_sq = 1
v=1


def simulate_data_normal(N, B):
    """
    Simulate a random dataset using a noisy
    linear process.

    N: Number of data points to simulate
    B: Vector of regression parameters. Defines number of covariates. (Include B_0)
    """
    # Create a pandas DataFrame with column 'x' containing
    # N uniformly sampled values between 0.0 and 1.0
    seed = 7
    df = pd.DataFrame(
        {str("x1"): np.random.RandomState().normal(
                0, v, N)})
    for i in range(2, len(B)):
        print(i)
        df_i = pd.DataFrame(
                {str("x" + str(i)): np.random.RandomState().normal(
                0, v, N)})
        df = pd.concat([df, df_i], axis = 1)


    # Use a linear model (y ~ beta_0 + beta_1*x + epsilon) to
    # generate a column 'y' of responses based on 'x'
    #Betas are normally distributed with mean 0 and variance eps_sigma_sq
    y = B[0] + np.matmul(pd.DataFrame.as_matrix(df), np.transpose(B[1:])) + np.random.RandomState(42).normal(0, eps_sigma_sq, N)
    df["y"] =  y

    return df

def normal_likelihood(x, y):
    #Calculate log-likelihoods (terms that cancel are not included)
    #New likelihood with swapped values
    l = norm.pdf(x, y, 1)
    return(np.log(l))
    
def normal_likelihood_swap(i, j, x_i, x_j, x_i_swap, x_j_swap, y):
    #Calculate log-likelihoods (terms that cancel are not included)
    #New likelihood with swapped values
    new_l = sum([np.log(l) for l in norm.pdf([x_i_swap, x_j_swap], [y[j], y[i]], 1)\
                 *(np.isfinite([y[j], y[i]]))])
    #Likelihood without swapped values
    old_l = sum([np.log(l) for l in norm.pdf([x_i, x_j], [y[i], y[j]], 1)\
                 *(np.isfinite([y[i], y[j]]))])
    return([new_l, old_l])
