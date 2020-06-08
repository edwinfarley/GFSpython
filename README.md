# GFS
The permutation sampling procedure is broken into 3 main parts:
1. The execution file--exec.py.
2. The master file--master.py.
3. The file containing the likelihood functions for the supported families of distributions--distributions.py

1. The exec.py file is called from R. It reads in the inputs from the input.txt
file generated by permute_inputs in R. Datasets are loaded into pandas dataframes
from the paths given: df1_path, df2_path. These objects, along with the formula
string, family, and sampling parameters are passed into the sample function loaded
from master.py.
2. The master.py file contains functions that handle permutations and blocking
in addition to the main sample function. The two datasets along with formula,
family, and sampling parameters are passed to the sample procedure, which first
formats the data by arranging rows by block, adding in extra rows so that blocks
have equal numbers of observations, and combining into one dataframe the columns
referenced in the formula input. Based on the chosen family, the sampling
procedure calls the corresponding sampling function with the combined dataframe,
formula, and sampling parameters. Upon completion of sampling, permutations for
individual blocks are combined into a permutation for the full dataset with
respect to the rows of the first dataset at df1_path.
3. The distributions.py file contains the likilihood function methods that are used during
the Metropolis-Hastings step. For each distribution there is a method to calculate the 
likelihood of a covariates/response pair, and a method to compute the likelihood of a block 
under a swap of two rows. This result will be used in a ratio of likelihoods, so only 
the likelihood only needs to be computed for the two rows that are being swapped, both
with and without the swap, because the likelihood of all other rows will be unchanged by 
the swap and thus cancel in the ratio computaion.
