#!/usr/bin/env python
import numpy as np

'''
Created on Aug 1, 2016
@author: skarumbaiah

Computes Fleiss' Kappa 
Joseph L. Fleiss, Measuring Nominal Scale Agreement Among Many Raters, 1971.
'''

def checkInput(rate, n):
    """ 
    Check correctness of the input matrix
    @param rate - ratings matrix
    @return n - number of raters
    @throws AssertionError 
    """
    N = len(rate)
    k = len(rate[0])
    assert all(len(rate[i]) == k for i in range(k)), "Row length != #categories)"
    assert all(isinstance(rate[i][j], int) for i in range(N) for j in range(k)), "Element not integer" 
    assert all(sum(row) == n for row in rate), "Sum of ratings != #raters)"

def fleissKappa(rate,n):
    """ 
    Computes the Kappa value
    @param rate - ratings matrix containing number of ratings for each subject per category 
    [size - N X k where N = #subjects and k = #categories]
    @param n - number of raters   
    @return fleiss' kappa
    """

    N = len(rate)
    k = len(rate[0])
    print("#raters = ", n, ", #subjects = ", N, ", #categories = ", k)
    checkInput(rate, n)

    #mean of the extent to which raters agree for the ith subject 
    PA = sum([(sum([i**2 for i in row])- n) / (n * (n - 1)) for row in rate])/N
    print("PA = ", PA)
    
    # mean of squares of proportion of all assignments which were to jth category
    PE = sum([j**2 for j in [sum([rows[i] for rows in rate])/(N*n) for i in range(k)]])
    print("PE =", PE)
    
    kappa = -float("inf")
    try:
        kappa = (PA - PE) / (1 - PE)
        kappa = float("{:.5f}".format(kappa))
    except ZeroDivisionError:
        print("Expected agreement = 1")

    print("Fleiss' Kappa =", kappa)
    
    return kappa


if __name__ == "__main__":
    import pandas as pd
    file = pd.read_csv("results/agreement/experts_agreement_200.csv", index_col=0, header=0)
    set_cols = [["FP_A", "FP_B", "FP_C"], ["FN_A", "FN_B", "FN_C"]]
    cols = []
    for col in set_cols:
        col_set = pd.DataFrame(file[col])
        col_set = col_set.sum(axis=1)
        col_set.columns = ["A", "B", "C"]
        cols.append(col_set)
    one = pd.concat(cols, axis=0).to_numpy()
    print(one)
    zero = 3 - one
    
    rate = [(int(zero[i]), int(one[i])) for i in range(len(one))]
    n = 3
    print("experts_agreement_200.csv")
    fleissKappa(rate, n)
