import pandas as pd
import numpy as np
from .core import rga
from .util.utils import manipulate_testdata, validate_variables, convert_to_dataframe, check_nan


def compute_single_variable_rge(xtrain, xtest, yhat, model, variables):
    """Compute RGE for single variable contribution."""
    # Convert inputs to DataFrames and concatenate them
    xtrain, xtest, yhat = convert_to_dataframe(xtrain, xtest, yhat)
    # check for missing values
    check_nan(xtrain, xtest, yhat)
    # variables should be a list
    validate_variables(variables, xtrain)
    # find RGEs
    rge_list = []
    for i in variables:
        xtest_rm = manipulate_testdata(xtrain, xtest, model, i)
        yhat_rm = [x[1] for x in model.predict_proba(xtest_rm)]
        rge_list.append(1 - (rga(yhat, yhat_rm)))
    rge_df = pd.DataFrame(rge_list, index=variables, columns=["RGE"]).sort_values(by="RGE", ascending=False)
    return rge_df


def compute_group_variable_rge(xtrain, xtest, yhat, model, variables):
    """Compute RGE for group variable contribution."""
    # Convert inputs to DataFrames and concatenate them
    xtrain, xtest, yhat = convert_to_dataframe(xtrain, xtest, yhat)
    # check for missing values
    check_nan(xtrain, xtest, yhat)
    # variables should be a list
    validate_variables(variables, xtrain)
    # find RGEs
    for i in variables:
        xtest_rm = manipulate_testdata(xtrain, xtest, model, i)
    yhat_rm = [x[1] for x in model.predict_proba(xtest_rm)]
    rge = 1 - (rga(yhat, yhat_rm))
    rge_df = pd.DataFrame(rge, index=[str(variables)], columns=["RGE"])
    return rge_df


def compute_full_single_rge(xtrain, xtest, yhat, model):
    """Compute RGE when no variables are specified."""
    # Convert inputs to DataFrames and concatenate them
    xtrain, xtest, yhat = convert_to_dataframe(xtrain, xtest, yhat)
    # check for missing values
    check_nan(xtrain, xtest, yhat)
    # find RGEs
    rge_list = []
    for i in xtest.columns:
        xtest_rm = manipulate_testdata(xtrain, xtest, model, i)
        yhat_rm = [x[1] for x in model.predict_proba(xtest_rm)]
        rge_list.append(1 - (rga(yhat, yhat_rm)))
    rge_df = pd.DataFrame(rge_list, index= xtest.columns, columns=["RGE"]).sort_values(by="RGE", ascending=False)
    return rge_df


