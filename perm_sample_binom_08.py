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
    print(1)
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
    P = np.exp(x)/(1+np.exp(x))
    l = ((np.log(P)*y)+(np.log(1-P)*(1-y)))
    return(l)
    
def logistic_likelihood_swap(i, j, x_i, x_j, x_i_swap, x_j_swap, y):
    P_i = np.exp(x_i)/(1+np.exp(x_i))
    P_j = np.exp(x_j)/(1+np.exp(x_j))
    P_i_swap = np.exp(x_i_swap)/(1+np.exp(x_i_swap))
    P_j_swap = np.exp(x_j_swap)/(1+np.exp(x_j_swap))

    #Calculate log-likelihoods (terms that cancel are not included)
    #New likelihood with swapped values
    new_l = ((np.log(P_i_swap)*y[j])+(np.log(1-P_i_swap)*(1-y[j])))*np.isfinite(y[j]) + ((np.log(P_j_swap)*y[i])+(np.log(1-P_j_swap)*(1-y[i])))*np.isfinite(y[i])
    #Likelihood without swapped values
    old_l = ((np.log(P_i)*y[i])+(np.log(1-P_i)*(1-y[i])))*np.isfinite(y[i]) + ((np.log(P_j)*y[j])+(np.log(1-P_j)*(1-y[j])))*np.isfinite(y[j])
    return([new_l, old_l])
