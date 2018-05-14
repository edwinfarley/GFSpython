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

from master import build_permutation, glm_mcmc_inference


def poisson_permute(A, b, y, p, T, m, Y):
    #Performs the Metropolis-Hastings step.
   
    #T number of iterations  
    for t in range(T): 
        i, j = np.random.choice(m, 2, replace = False)
        
        x_i = np.matmul(A[i, :], b)
        x_j = np.matmul(A[j, :], b)
        
        #Switch relevant (Y) covariates
        for _, ix in Y[:-1]:
                temp = A[i, ix]
                A[i, ix] = A[j, ix]
                A[j, ix] = temp
        x_i_swap = np.matmul(A[i, :], b)
        x_j_swap = np.matmul(A[j, :], b)
        
        #Calculate log-likelihoods (terms that cancel are not included)
        #New likelihood with swapped values
        new_l = (((y[j]*x_i_swap) - np.exp(x_i_swap))*np.isfinite(y[j])) + (((y[i]*x_j_swap) - np.exp(x_j_swap))*np.isfinite(y[i]))
        #Likelihood without swapped values
        old_l = (((y[i]*x_i) - np.exp(x_i))*np.isfinite(y[i])) + (((y[j]*x_j) - np.exp(x_j))*np.isfinite(y[j]))
        
        #Probability of accepting proposed swap from definition of Metropolis-Hastings
        choice = min(1, np.exp(new_l - old_l))
        rand = np.random.rand()
        #Accept or reject swap
        if rand <= choice:
            temp = y[i]
            y[i] = y[j]
            y[j] = temp
            
            temp = p[i]
            p[i] = p[j]
            p[j] = temp
        else:
            #Switch Y covariates back
            for _, ix in Y[:-1]:
                temp = A[i, ix]
                A[i, ix] = A[j, ix]
                A[j, ix] = temp
    
    #Returns final permutation of input and the permuted y         
    return(p, y)
            
def permute_search_pois(df, block, formula, Y, N, I, T, burnin, interval):
    #N: Number of permutations
    #I: Number of samples in sampling Betas
    #T: Number of iterations in row swapping phase

    #X is the first dataset and Y is the second dataset that contains the response.

    y1 = formula.split(' ~ ')[0]
    covariates = formula.split(' ~ ')[1].split(' + ')
    num_X = len(covariates) - len(Y)
    
    #Isolate current block
    block_df = pd.DataFrame(df[block[0]:block[1]]).reset_index(drop=True)

    #Missing values: Find indices of missing values and how many there are.
    X_missing = np.where(np.isnan(block_df[covariates[0]]))[0]
    num_X_missing = len(X_missing)
    num_finite = len(block_df) - num_X_missing
    num_Y_missing = sum(np.isnan(block_df[y1]))
    m, n = len(block_df), len(block_df.columns)+1

    #Remove NaNs outside of current block
    df = pd.concat([df[0:block[0]].dropna(), df[block[0]:block[1]], df[block[1]:].dropna()])
    block_size = min(sum(block_df[y1].notnull()), sum(block_df[covariates[0]].notnull()))
    print(block_size)
    
    #P: Permutations after I iterations for each set of Betas
    P = np.zeros((N, block_size)).astype(int)
    #B: Betas for T samplings
    B = [0 for i in range(n)]*N
    
    #Make a copy of the block as it was given in the input.
    original_block = pd.DataFrame(block_df)
    #Sample missing X's
    for i in X_missing:
                r = int(num_finite * random.random())
                #print((block_df.sort_values(by = y1).drop(y1, 1))[r:r+1])
                #print(block_df.loc[i, :][:num_X])
                block_df.loc[i, :][:num_X] = list((block_df.sort_values(by = y1).drop(y1, 1)).loc[r,:][:num_X])

    for t in range(burnin + (N*interval)):
        #Input is the data in the order of the last permutation
        if t > 0:
            #Update block_df with the new permutation of y
            df[y1].loc[block[0]:block[1]-1] = new_y
            block_df[y1] = new_y
            #Update columns from Y that correspond to covariates that need to be permuted.
            for col, _ in Y:
                new_col = build_permutation(P_t, list(original_block[col]))
                df[col].loc[block[0]:block[1]-1] = new_col
                block_df[col] = new_col    
            #Sample missing X's from response values estimated with most recent Betas.
            if num_missing:
                block_df['y_b'] = np.matmul(A, b)
                print(block_df)
                for i in X_missing:
                    r = int(num_finite * random.random())
                    block_df.loc[i, :][:num_X] = list((block_df.sort_values(by = 'y_b').drop([y1, 'y_b'], 1)).loc[r,:][:num_X])
                block_df = block_df.drop('y_b', 1)

        #Sample Betas and search for permutations
        trace = glm_mcmc_inference(df, formula, pm.glm.families.Poisson(), I)
        beta_names = ['Intercept']
        beta_names.extend(formula.split(' ~ ')[1].split(' + '))
        b = np.transpose([trace.get_values(s)[-1] for s in beta_names])

        A = pd.DataFrame.as_matrix(block_df.drop(y1, 1))
        A = np.concatenate([np.ones((m, 1)), A], 1)
        B[((t)*n):((t+1)*n)] = b
        
        #Go to Metropolis-Hastings step.
        if t == 0:
            P_t, new_y = poisson_permute(A, b, np.array(block_df[y1]), np.arange(0, m), T, m, Y)
            #Save permutation
            if burnin == 0:
                if num_X_missing:
                    P[0, :] = P_t[:-num_X_missing]
                elif num_Y_missing:
                    temp = np.array(P_t)
                    temp[np.where(np.isnan(new_y))] = -(block[0]+1)
                    P[0, :] = temp
                else:
                    P[0, :] = P_t
        else:
            P_t, new_y = poisson_permute(A, b, np.array(new_y), P_t, T, m, Y)
            #Save permutation
            if t >= burnin:
                if (t-burnin)%interval == 0:
                    if num_X_missing:
                        P[int((t-burnin)/interval), :] = P_t[:-num_X_missing]
                    elif num_Y_missing:
                        temp = np.array(P_t)
                        temp[np.where(np.isnan(new_y))] = -(block[0]+1)
                        P[int((t-burnin)/interval), :] = temp
                    else:
                        P[int((t-burnin)/interval), :] = P_t
                        
    return([B, P])