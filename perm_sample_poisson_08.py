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

def poisson_likelihood(i, j, x_i, x_j, x_i_swap, x_j_swap, y):
    l = ((y*x) - np.exp(x))
    return(l)

def poisson_likelihood_swap(i, j, x_i, x_j, x_i_swap, x_j_swap, y):
    #Calculate log-likelihoods (terms that cancel are not included)
    #New likelihood with swapped values
    new_l = (((y[j]*x_i_swap) - np.exp(x_i_swap))*np.isfinite(y[j])) + (((y[i]*x_j_swap) - np.exp(x_j_swap))*np.isfinite(y[i]))
    #Likelihood without swapped values
    old_l = (((y[i]*x_i) - np.exp(x_i))*np.isfinite(y[i])) + (((y[j]*x_j) - np.exp(x_j))*np.isfinite(y[j]))
    return([new_l, old_l])
