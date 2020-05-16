#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 18:56:04 2018

@author: Edwin
"""
from __future__ import print_function

import numpy as np
import pandas as pd
import pymc3 as pm
import sys

def glm_mcmc_inference(df, formula, family, I):
    """
    Calculates the Markov Chain Monte Carlo trace of
    a Generalised Linear Model Bayesian linear regression
    model on supplied data.

    df: DataFrame containing the data
    formula: Regressing equation in terms of columns of DataFrame df
    family: Type of liner model. Takes a pymc object (pm.glm.families).
    I: Number of iterations for MCMC
    """
    
    if family.lower() == 'normal':
        family_object = pm.glm.families.Normal()
    elif family.lower() == 'logistic':
        family_object = pm.glm.families.Binomial()
    elif family.lower() == 'poisson':
        family_object = pm.glm.families.Poisson()
    else:
        print("Family {} is not a supported family".format(family))
        raise(NameError("Invalid family"))

    # Use PyMC3 to construct a model context
    basic_model = pm.Model()
    with basic_model:
        # Create the glm using the Patsy model syntax
        pm.glm.GLM.from_formula(str(formula), df.dropna(), family=family_object)
        step = pm.NUTS()

        trace = pm.sample(I, step, progressbar=False)
        return(trace)

def build_permutation(p, arr):
    #For indices
    new = [np.nan for i in range(sum(np.isfinite(p)))]
    l = len(p)
    nans = 0
    for i in range(l):
        if np.isfinite(p[i]):
            if p[i] < 0:
                new[i-nans] = -1
            else:
                ix = int(p[i])
                new[i-nans] = arr[ix]
        else:
            nans = nans+1
    return(new)

def create_blocks(df1, df2):
    #blocks = np.unique(df1['block'])
    df1 = df1.sort_values(by = ['block']).reset_index()
    df2 = df2.sort_values(by = ['block']).reset_index()
    if df2['block'][0] != df1['block'][0]:
        if df2['block'][0] < df1['block'][0]:
            df1 = pd.concat([pd.DataFrame(np.full((1, len(df1.columns)), np.nan), columns = list(df1.columns)), df1], sort=True).reset_index(drop = True)
            df1['block'][0] = df2['block'][0]
        else:
            df2 = pd.concat([pd.DataFrame(np.full((1, len(df2.columns)), np.nan), columns = list(df2.columns)), df2], sort=True).reset_index(drop = True)
            df2['block'][0] = df1['block'][0]
    n1 = len(df1)
    n2 = len(df2)
    current_block = df2['block'][0]
    i = 1
    while i < min(n1, n2):
        if df2['block'][i] != current_block:
            if df1['block'][i] == current_block:
                df2 = pd.concat([df2[0:i], pd.DataFrame(np.full((1, len(df2.columns)), np.nan), columns = list(df2.columns)), df2[i:]], sort=True).reset_index(drop = True)
                df2['block'][i] = df1['block'][i]
                n2 = n2 + 1
            else:
                current_block = df2['block'][i]
                i = i + 1
        elif df1['block'][i] != current_block:
            df1 = pd.concat([df1[0:i], pd.DataFrame(np.full((1, len(df1.columns)), np.nan), columns = list(df1.columns)), df1[i:]], sort=True).reset_index(drop = True)
            df1['block'][i] = df2['block'][i]
            n1 = n1 + 1
        else:
            i = i + 1
    if n1 > n2:
        df2 = pd.concat([df2, pd.DataFrame(np.full((n1-i, len(df2.columns)), np.nan), columns = list(df2.columns))], sort=True).reset_index(drop = True)
        df2.loc[i:,'block'] = df1.loc[i:,'block']
    elif n1 < n2:
        df1 = pd.concat([df1, pd.DataFrame(np.full((n2-i, len(df1.columns)), np.nan), columns = list(df1.columns))], sort=True).reset_index(drop = True)
        df1.loc[i:,'block'] = df2.loc[i:,'block']

    blocks = [[0,0] for i in range(0, len(np.unique(df1['block'])))]
    current_block = (df1['block'][0], 0)
    for i in range(0, len(df1['block'])):
        if df1['block'][i] != current_block[0]:
            blocks[current_block[1]][1] = i
            current_block = (df1['block'][i], current_block[1] + 1)
            blocks[current_block[1]][0] = i
            
    blocks[-1][1] = i + 1
    df1 = df1.drop('block', 1)
    df2 = df2.drop('block', 1)
    
    return([df1, df2, blocks])

def create_df(df1, df2, covs):
    new_df = pd.DataFrame()
    columns1 = [c for c in covs if c in df1.columns]
    columns2 = [c for c in covs if c in df2.columns]
    np_df = np.transpose(np.array([df1[c] for c in columns1] + \
                                  [df2[c] for c in columns2]))
    new_df = pd.DataFrame(np_df.copy(), columns = columns1 + columns2)
    return(new_df)

def general_permute(block_df, family_dict, p, T, m, block_size, num_X_missing):
    #Performs the Metropolis-Hastings step.
    #T: number of iterations
    y_dict = {family : np.array(block_df[info["y1"]]) for family, info in family_dict.items()}
    for t in range(T * m):
        i, j = np.random.choice(m, 2, replace = False)
        
        new_l = 0
        old_l = 0
        for family, info in family_dict.items():
            A = info["A"]
            b = info["beta"]
            y = y_dict[family]
            Y = info["Y"]
            x_i = np.matmul(A[i, :], b)
            x_j = np.matmul(A[j, :], b)

            #Switch relevant (Y) covariates
            for _, ix in Y:
                    temp = A[i, ix]
                    A[i, ix] = A[j, ix]
                    A[j, ix] = temp
            x_i_swap = np.matmul(A[i, :], b)
            x_j_swap = np.matmul(A[j, :], b)
            
            if family == "normal":
                [new_l_family, old_l_family] = normal_likelihood_swap(i, j, x_i, x_j, x_i_swap, x_j_swap, y)
            elif family == "logistic":
                [new_l_family, old_l_family] = logistic_likelihood_swap(i, j, x_i, x_j, x_i_swap, x_j_swap, y)
            else:
                [new_l_family, old_l_family] = poisson_likelihood_swap(i, j, x_i, x_j, x_i_swap, x_j_swap, y)
                
            new_l = new_l + new_l_family
            old_l = old_l + old_l_family
        
        #Probability of accepting proposed swap from definition of Metropolis-Hastings
        choice = min(1, np.exp(new_l - old_l))
        rand = np.random.rand()
        #Accept or reject swap
        if rand <= choice:
            for family, info in family_dict.items():
                #Update family info
                y = y_dict[family]
                temp = y[i]
                y[i] = y[j]
                y[j] = temp
                y_dict[family] = y
                
            #Update permutation
            temp = p[i]
            p[i] = p[j]
            p[j] = temp
        else:
            #Switch Y covariates back
            for family, info in family_dict.items():
                A = info["A"]
                Y = info["Y"]
                for _, ix in Y:
                    temp = A[i, ix]
                    A[i, ix] = A[j, ix]
                    A[j, ix] = temp
                info["A"] = A

    if num_X_missing:
        l = np.array([0 for _ in range(0, block_size)])
        for f, info in family_dict.items():
            complete_A = info["A"][0:block_size, :]
            b = info["beta"]
            y = y_dict[f]
            for i in range(0, block_size):
                x_i = np.matmul(complete_A[i,:], b)
                if f == "normal":
                    l[i] += normal_likelihood(x_i, y[i])
                elif family == "logistic":
                    l[i] += logistic_likelihood(x_i, y[i])
                else:
                    l[i] += poisson_likelihood(x_i, y[i])
        
        l = np.exp(l)
        l = l/sum(l)
        cum = 0
        for i in range(0, len(l)):
            old = l[i]
            l[i] += cum
            cum += old
        # Check if we have run into an error and use uniform weighting if we have
        if np.isnan(cum):
            l = np.array([1 for _ in range(0, block_size)]) / block_size
        else:
            assert(abs(cum - 1) < 0.0000001)
    else:
        l = []
                    
    #Returns final permutation of input
    return(p, l)

def permute_search_general(block_df, family_dict, primary_family, block, block_dict, covs_Y, N, I, T, burnin, interval, t, P_t):
    #N: Number of permutations
    #I: Number of samples in sampling Betas
    #T: Number of iterations in row swapping phase

    #X is the first dataset and Y is the second dataset that contains the response.
    
    block_size = block_dict["block_size"]
    original_block = block_dict["original_block"]
    X_missing = block_dict["X_missing"]
    Y_missing = block_dict["Y_missing"]
    num_X_missing = block_dict["num_X_missing"]
    num_Y_missing = block_dict["num_Y_missing"]

    m = len(block_df)
    n = len(block_df.columns)+1

    #for t in range(burnin + (N*interval)):
    #Input is the data in the order of the last permutation
    if m > 1:
        for f, info in family_dict.items():
            family_df = pd.DataFrame(block_df, columns = info["covs"])
            info["A"] = np.concatenate([np.ones((m, 1)), family_df.values], 1)
            
        #Go to Metropolis-Hastings step.
        P_t, l = general_permute(block_df, family_dict, P_t, T, m, block_size, num_X_missing)

        if num_X_missing:
            P = P_t[:-num_X_missing]
        #elif num_Y_missing:
        #    temp = np.array(P_t)
        #    temp[np.where(np.isnan(new_y))] = -(block[0]+1)
        #    P = temp
        else:
            P = P_t
            
        #Update columns from Y that correspond to covariates that need to be permuted.
        #all_Y = np.concatenate([info["Y"] for _, info in family_dict.items()])
        for col in covs_Y:
            new_col = build_permutation(P_t, list(original_block[col]))
            block_df[col] = new_col
        #Sample missing X's from response values estimated with most recent Betas.
        if num_X_missing:
            num_finite = len(block_df) - num_X_missing
            #Use primary_family
            sample_df = block_df
            #sample_df["y_b"] = np.matmul(family_dict[primary_family]["A"], family_dict[primary_family]["beta"])
            #sample_df = sample_df.sort_values(by = "y_b")
            #Get picks for each extra row
            r_dict = {row : np.searchsorted(l, np.random.rand()) for row in np.unique(X_missing[0])}
            #Fill missing values in X part of block
            for ix in X_missing[0]:
                r = r_dict[ix]
                for jx in X_missing[1]:
                    block_df.loc[ix][jx] = sample_df.loc[r][jx]
    else:
        P = [0]
        P_t = [0]

    return([P, P_t, block_df])

def sample(df1, df2, formula_array, family_array, N, I, T, burnin, interval):
    N = int(N)
    I = int(I)
    T = int(T)
    burnin = int(burnin)
    interval = int(interval)
    df1, df2, blocks = create_blocks(df1, df2)

    index = np.array(df2['index'])
    true_index = np.array(df1['index'])
    df1 = df1.drop('index', 1)
    df2 = df2.drop('index', 1)
    
    all_covs = set()
    family_dict = {}
    primary_family = str.lower(family_array[0])
    Y = []
    for i in range(0, len(family_array)):
        family = str.lower(family_array[i])
        new_dict = {}
        
        formula = formula_array[i]
        formula = formula.replace(' ', '')
        new_dict["formula"] = formula
        
        covs = formula.split("~")[1].split('+')
        new_dict["covs"] = covs        
        
        #account for intercept column in A matrix with +1
        family_Y = [(covs[c], c+1) for c in range(len(covs)) if covs[c] in df2.columns]
        new_dict["Y"] = family_Y
        Y = Y + family_Y
        
        y1 = formula.split('~')[0]
        new_dict["y1"] = y1
        
        all_covs.update(covs)
        all_covs.add(y1)
        
        family_dict[family] = new_dict
        
    covs_X = [c for c in list(all_covs) if c in df1.columns]
    covs_Y = [c for c in list(all_covs) if c in df2.columns]
    merged_df = create_df(df1, df2, covs_X + covs_Y)

    len_P = len(merged_df) - sum(np.isnan(df1[df1.columns[0]])) #- sum(np.isnan(df2[df2.columns[0]]))
    B_dict = {}
    P_dict = {}

    P_last = {}
    block_dict = {}

    df = merged_df
    for i in range(0, len(blocks)):
        #Call sampling function from given family of distribution
        block = [blocks[i][0],blocks[i][1]]
        block_df = pd.DataFrame(df[block[0]:block[1]]).reset_index(drop=True)
        
        #Make a copy of the block as it was given in the input.
        original_block = pd.DataFrame(block_df.copy())
        #Get X columns
        block_df_X = pd.DataFrame(block_df.copy(), columns = covs_X)
        block_df_Y = pd.DataFrame(block_df.copy(), columns = covs_Y)
        #Get missing entries
        X_missing = np.where(np.isnan(block_df_X)) #[rows, columns]
        num_X_missing = max(np.isnan(block_df_X).sum(axis = 0))
        Y_missing = np.where(np.isnan(block_df_Y)) #[rows, columns]
        num_Y_missing = max(np.isnan(block_df_Y).sum(axis = 0))
        
        block_size = len(block_df) - num_X_missing

        block_dict[str(i)] = {"block" : block, "block_size" : block_size, "original_block" : original_block, "X_missing" : X_missing, "Y_missing" : Y_missing, "num_X_missing" : num_X_missing, "num_Y_missing" : num_Y_missing}
        
        num_finite = len(block_df) - num_X_missing
        if num_X_missing:
            y1_temp = family_dict[primary_family]["y1"]
            sample_df = block_df.copy()
            #sample_df = sample_df.sort_values(by = y1_temp)
            r_dict = {row : int(np.floor(num_finite * np.random.rand())) for row in np.unique(X_missing[0])}
            #fill missing values in X part of block
            for ix in X_missing[0]:
                    r = r_dict[ix]
                    for jx in X_missing[1]:
                        block_df.loc[ix][jx] = sample_df.loc[r][jx]
        
        if i == 0:
            new_df = block_df
        else:
            new_df = new_df.append(block_df, ignore_index=True)
        
        P_dict[str(i)] = []
        
    df = new_df
    P = None
    for t in range(burnin + (N*interval)):
        #Sample Betas for current permutation of data
        for family, info in family_dict.items():
            trace = glm_mcmc_inference(df, info["formula"], family, I)
            b = np.transpose([trace.get_values(s)[-1] for s in ["Intercept"] + info["covs"]])
            info["beta"] = b
        #b = np.transpose([(trace.get_values(s)[-1] if s in trace.varnames else 0 ) for s in beta_names])
        #Loop through blocks
        for i in range(0, len(blocks)):
            #Call sampling function from given family of distribution
            block = [blocks[i][0],blocks[i][1]]
            block_df = pd.DataFrame(df.copy()[block[0]:block[1]]).reset_index(drop=True)
            
            block_dict_i = block_dict[str(i)]
            block_size = block_dict_i["block_size"]
            original_block = block_dict_i["original_block"]
            X_missing = block_dict_i["X_missing"]
            Y_missing = block_dict_i["Y_missing"]
            num_X_missing = block_dict_i["num_X_missing"]
              
            if t == 0:
                P_t = np.arange(0, len(block_df))
            else:
                P_t = P_last[str(i)]
                
            P, P_t, df_i = permute_search_general(block_df, family_dict, primary_family, block, block_dict[str(i)], covs_Y, N, I, T, burnin, interval, t, P_t)
            P_last[str(i)] = P_t
            #Construct new data set
            if i == 0:
                new_df = df_i
            else:
                new_df = new_df.append(df_i, ignore_index=True)

            if t >= burnin and (t-burnin)%interval == 0:
                if P_dict[str(i)] == []:
                    P_dict[str(i)] = np.array([P])
                else:
                    P_dict[str(i)] = np.concatenate((P_dict[str(i)], [P]), 0)
        df = new_df

    #Compile permutations from blocks to obtain full permutations
    full_P = np.zeros((N, len_P))
    for i in range(0, N):
        full_P_i = []
        block_count = 0
        for key in P_dict:
            block_dict_key = block_dict[key]
            block_key = block_dict_key["block"]
            new_P = P_dict[key][i, :]
            new_P = new_P.astype(int)
            num_Y_missing = block_dict_key["num_Y_missing"]
            #Set entries that correspond to added rows in Y to NaN.
            #These will be the last num_Y_missing indices.
            #new_P = [np.nan if val >= len(new_P) - num_Y_missing else val for val in new_P]
            #Adjust indices based on start of block to correspond to full data set
            new_P_adj = block_key[0] + np.array(new_P)
            full_P_i = np.concatenate((full_P_i, new_P_adj), 0)
            block_count = block_count + 1
        
        temp = build_permutation(full_P_i, index)
        F_i  = build_permutation(true_index, temp)
        full_P[i, :] = F_i

    return(full_P.astype(int))


from perm_sample_norm_09 import *
from perm_sample_binom_08 import *
from perm_sample_poisson_08 import *
