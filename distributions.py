# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 18:56:04 2018

@author: Edwin
"""
import numpy as np
import math
from scipy.stats import norm

# Normal Family

def normal_likelihood(x, y):
    #Calculate log-likelihoods (terms that cancel are not included)
    #New likelihood with swapped values
    l = norm.pdf(x, y, 1)
    return(np.log(l))
    
def normal_likelihood_swap(i, j, x_i, x_j, x_i_swap, x_j_swap, y):
    #Calculate log-likelihoods (terms that cancel are not included)
    #New likelihood with swapped values
    new_l = normal_likelihood(x_i_swap, y[j])*np.isfinite(y[j]) + normal_likelihood(x_j_swap, y[i])*np.isfinite(y[i])
    #Likelihood without swapped values
    old_l = normal_likelihood(x_i, y[i])*np.isfinite(y[i]) + normal_likelihood(x_j, y[j])*np.isfinite(y[j])
    return([new_l, old_l])
    
# Logistic Family

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
    
# Poisson family

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
