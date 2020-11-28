# -*- coding: utf-8 -*-
"""
This file is part of GFS (python component).

GFS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GFS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GFS. If not, see <https://www.gnu.org/licenses/>.

Created on Sun May 13 10:28:52 2018
"""
import sys, os
import pandas as pd

if __name__ == '__main__':

    _, package_dir, R_wd, temp_dir = sys.argv
    sys.path.append(package_dir + '/python')
    sys.path.append(R_wd)

    file = open(temp_dir + '/input.txt', 'r')
    df1_path = file.readline().split('\n')[0]
    df2_path = file.readline().split('\n')[0]
    formula = file.readline().split('\n')[0].replace(' ','').split(',')
    family = file.readline().split('\n')[0].replace(' ','').split(',')
    N = int(file.readline().split('\n')[0])
    I = int(file.readline().split('\n')[0])
    T = int(file.readline().split('\n')[0])
    burnin = int(file.readline().split('\n')[0])
    interval = int(file.readline().split('\n')[0])
    block_name = file.readline().split('\n')[0]

    try:
        df1 = pd.read_csv(R_wd + '/' + df1_path)
    except:
        df1 = pd.read_csv(df1_path)
    try:
        df2 = pd.read_csv(R_wd + '/' + df2_path)
    except:
        df2 = pd.read_csv(df2_path)

    df1.rename(index=str, columns={block_name: 'block'})
    df2.rename(index=str, columns={block_name: 'block'})
    from master import *
    sys.stdout = open(os.devnull, 'w')
    out = sample(df1, df2, formula, family, N, I, T, burnin, interval)
    sys.stdout = sys.__stdout__
    out = pd.DataFrame(np.transpose(out))
    out.to_csv(temp_dir + '/permutations.csv')
