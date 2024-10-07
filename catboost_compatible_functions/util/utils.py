import pandas as pd
import numpy as np

def manipulate_testdata(xtrain, xtest, model, variable):
    # create xtest_rm
    xtest_rm = xtest.copy()
    cat_indices = model.get_cat_feature_indices()
    # replace variable with mode or mean based on its type
    if xtrain.columns.get_loc(variable) not in cat_indices:
        mean_value = xtrain[variable].mean()
        xtest_rm[variable] = mean_value
    else:
        mode_value = xtrain[variable].mode()[0]
        xtest_rm[variable] = mode_value
    return xtest_rm

def convert_to_dataframe(*args):
    """Convert inputs to DataFrames."""
    return [pd.DataFrame(arg).reset_index(drop=True) for arg in args]

def validate_variables(variables, xtrain):
    """Check if variables are valid and exist in the train dataset."""
    if not isinstance(variables, list):
        raise ValueError("Variables input must be a list")
    for var in variables:
        if var not in xtrain.columns:
            raise ValueError(f"{var} is not in the variables")

def check_nan(*dataframes):
    for i, df in enumerate(dataframes, start=1):
        if df.isna().sum().sum() > 0:
            raise ValueError(f"DataFrame {i} contains missing values.")
