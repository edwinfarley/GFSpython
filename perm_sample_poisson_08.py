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
import math

from master import build_permutation

def poisson_likelihood(x, y):
    l = (y*x) - min(np.exp(x), 1e16)
    return(l)

def poisson_likelihood_swap(i, j, x_i, x_j, x_i_swap, x_j_swap, y):
    #Calculate log-likelihoods (terms that cancel are not included)
    #New likelihood with swapped values
    new_l = poisson_likelihood(x_i_swap, y[j])*np.isfinite(y[j]) \
        + poisson_likelihood(x_j_swap, y[i])*np.isfinite(y[i])
    #Likelihood without swapped values
    old_l = poisson_likelihood(x_i, y[i])*np.isfinite(y[i]) \
        + poisson_likelihood(x_j, y[j])*np.isfinite(y[j])
    return([new_l, old_l])
