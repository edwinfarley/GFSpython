#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 10:28:52 2018

@author: Edwin
"""
import sys, os
import pandas as pd

if __name__ == '__main__':

    _, package_dir, R_wd, temp_dir = sys.argv
    sys.path.append(package_dir + '/python')
    sys.path.append(R_wd)

    file = open(temp_dir + '/input.txt', 'r')
    df1_path = R_wd + '/' + file.readline().split('\n')[0]
    df2_path = R_wd + '/' + file.readline().split('\n')[0]
    formula = file.readline().split('\n')[0].replace(' ','').split(',')
    family = file.readline().split('\n')[0].replace(' ','').split(',')
    N = int(file.readline().split('\n')[0])
    I = int(file.readline().split('\n')[0])
    T = int(file.readline().split('\n')[0])
    burnin = int(file.readline().split('\n')[0])
    interval = int(file.readline().split('\n')[0])
    block_name = file.readline().split('\n')[0]

    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)

    df1.rename(index=str, columns={block_name: 'block'})
    df2.rename(index=str, columns={block_name: 'block'})
    from master import *
    sys.stdout = open(os.devnull, 'w')
    out = sample(df1, df2, formula, family, N, I, T, burnin, interval)
    sys.stdout = sys.__stdout__
    out = pd.DataFrame(np.transpose(out))
    out.to_csv(temp_dir + '/permutations.csv')
