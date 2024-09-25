import pandas as pd
import numpy as np

def find_xtest(xtrain, xtest, variable):
    # create xtest_rm
    xtest_rm = xtest.copy()

    # define types of cat_features
    cat_variables = ['object', 'category']

    # replace variable with mode or mean based on its type
    if variable not in cat_variables:
        mean_value = xtrain[variable].mean()
        xtest_rm[variable] = mean_value
    else:
        mode_value = xtrain[variable].mode()[0]
        xtest_rm[variable] = mode_value
    return xtest_rm