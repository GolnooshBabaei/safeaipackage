import pandas as pd
import numpy as np
from .core import rga
from .util.utils import validate_variables, convert_to_dataframe, check_nan

def perturb(data, variable, perturbation_percentage= 0.05):
    """
    Function to perturb a single variable based on the replacement of the two percentiles 
    selected using the perturbation_percentage of the object.
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

def compute_single_variable_rgr(xtest, yhat, model, variables, perturbation_percentage= 0.05):
    # Convert inputs to DataFrames and concatenate them
    xtest, yhat = convert_to_dataframe(xtest, yhat)
    # check for missing values
    check_nan(xtest, yhat)
    # variables should be a list
    validate_variables(variables, xtest)
    # find RGRs
    rgr_list = []
    for i in variables:
        xtest_pert = perturb(xtest, i, perturbation_percentage)
        yhat_pert = [x[1] for x in model.predict_proba(xtest_pert)]
        rgr_list.append(rga(yhat, yhat_pert))

    rgr_df = pd.DataFrame(rgr_list, index= list(variables), columns=["RGR"]).sort_values(by="RGR",
                                                                                ascending=False)
    return rgr_df

def compute_full_single_rgr(xtest, yhat, model, perturbation_percentage= 0.05):  
    # Convert inputs to DataFrames and concatenate them
    xtest, yhat = convert_to_dataframe(xtest, yhat)
    # check for missing values
    check_nan(xtest, yhat)
    # find RGRs 
    rgr_list = []
    for i in xtest.columns:
        xtest_pert = perturb(xtest, i, perturbation_percentage)
        yhat_pert = [x[1] for x in model.predict_proba(xtest_pert)] 
        rgr_list.append(rga(yhat, yhat_pert))     
    rgr_df = pd.DataFrame(rgr_list, index= xtest.columns, columns=["RGR"]).sort_values(by="RGR", 
                                                                                ascending=False)
    return rgr_df
    


