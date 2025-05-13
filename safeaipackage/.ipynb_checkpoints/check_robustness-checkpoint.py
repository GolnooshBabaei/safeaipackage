import pandas as pd
import numpy as np
from typing import Union
from .core import rga
from .util.utils import validate_variables, convert_to_dataframe, check_nan, find_yhat
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.base import is_classifier, is_regressor
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier, XGBRegressor


def perturb(data: pd.DataFrame, 
            variable: str, 
            perturbation_percentage= 0.05):
    """
    Function to perturb a single variable based on the replacement of the two percentiles 
    selected using the perturbation_percentage of the object.

    Parameters
    ----------
    data : pd.DataFrame
            A dataframe including data.
    variable: str 
            Name of variable.
    perturbation_percentage: float
            A percentage value for perturbation. 

    Returns
    -------
    pd.DataFrame
            The perturbed data.
    """ 
    if perturbation_percentage > 0.5 or perturbation_percentage < 0:
        raise ValueError("The perturbation percentage should be between 0 and 0.5.")
        
    data = data.reset_index(drop=True)
    perturbed_variable = data.loc[:,variable]
    vals = [[i, values] for i, values in enumerate(perturbed_variable)]
    indices = [x[0] for x in sorted(vals, key= lambda item: item[1])]
    sorted_variable = [x[1] for x in sorted(vals, key= lambda item: item[1])]
    percentile_5_index = int(np.ceil(perturbation_percentage * len(sorted_variable)))
    percentile_95_index = int(np.ceil((1-perturbation_percentage) * len(sorted_variable)))
    values_before_5th_percentile = sorted_variable[:percentile_5_index]
    values_after_95th_percentile = sorted_variable[percentile_95_index:]
    n = min([len(values_before_5th_percentile), len(values_after_95th_percentile)])
    lowertail_indices = indices[0:n]
    uppertail_indices = (indices[-n:])
    uppertail_indices = uppertail_indices[::-1]
    new_variable = perturbed_variable.copy()
    for j in range(n):
        new_variable[lowertail_indices[j]] = perturbed_variable[uppertail_indices[j]]
        new_variable[uppertail_indices[j]] = perturbed_variable[lowertail_indices[j]]
    data.loc[:,variable] = new_variable
    return data


def compute_rgr_values(xtest: pd.DataFrame, 
                                yhat: list, 
                                model, 
                                variables: list, 
                                perturbation_percentage= 0.05,
                                group: bool = False):
    """
    Compute RANK GRADUATION Robustness (RGR) MEASURE towards the perturbations in the given variables or groups of variables.

    Parameters
    ----------
    xtest : pd.DataFrame
            A dataframe including test data.
    yhat : list
            A list of predicted values.
    model : Union[CatBoostClassifier, CatBoostRegressor, XGBClassifier, XGBRegressor, BaseEstimator]
            A trained model, which could be a classifier or regressor. 
    variables: list 
            A list of variables.
    perturbation_percentage: float
            A percentage value for perturbation .
    group : bool
            If True, calculate RGR for the group of variables as a whole; otherwise, calculate for each variable.

    Returns
    -------
    pd.DataFrame
            The RGR values for each variable or for the group.
    """
    # Convert inputs to DataFrames and concatenate them
    xtest, yhat = utils.convert_to_dataframe(xtest, yhat)
    # check for missing values
    utils.check_nan(xtest, yhat)
    # variables should be a list
    utils.validate_variables(variables, xtest)
    # find RGRs

    if group:
        for variable in variables:
            xtest = utils.manipulate_testdata(xtrain, xtest, model, variable)
        
        # Calculate yhat after manipulating all variables in the group
        yhat_rm = utils.find_yhat(model, xtest)
        # Calculate a single RGR for the entire group
        rgr = core.rga(yhat, yhat_rm)
        return pd.DataFrame([rgr], index=[str(variables)], columns=["RGR"])

    else:
        rgr_list = []
        for variable in variables:
            xtest_pert = perturb(xtest, variable, perturbation_percentage)
            yhat_pert = utils.find_yhat(model, xtest_pert)
            rgr_list.append(core.rga(yhat, yhat_pert))
        rgr_df = pd.DataFrame(rgr_list, index= list(variables), columns=["RGR"]).sort_values(by="RGR", ascending=False)
            
    return rgr_df

    


