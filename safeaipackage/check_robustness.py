import pandas as pd
import numpy as np
import scipy
from .utils.util import _rga_delta_function, _rga
import matplotlib.pyplot as plt


def rgr_single(xtrain, xtest, ytrain, ytest, model, variable, perturbation_percentage= 0.05):
    """
     ### RANK GRADUATION Robustness (RGR) MEASURE ###
     Function for the RGR measure computation regarding perturbation of a single variable
     Inputs: xtrain -> A dataframe including xtrain data used for traing the model; 
             xtest -> A dataframe including xtest data used for testing the model;
             ytrain -> Actual values of the target variable for traing the model; 
             ytest -> Actual values of the target variable for testing the model;
             model -> Model could be a regressor such as MLPRegressor() from sklearn.neural_network
             or a classifier for example RandomForestClassifier() from sklearn.ensemble;
             variable -> A string representing name of the column of interest for which we want to find the RGR value;
             perturbation_percentage -> The percentage indicating the percentile of interest for perturbation, by 
             default equal to 0.05. Therefore, the 5th and 95th percentiles are considered for the perturbation if a 
             specific percentage is not defined. The maximum value allowed to set for this input variable is 0.5.
    """ 

    if variable not in xtrain.columns:
        raise ValueError("The selected variable is not available.")
    if perturbation_percentage > 0.5 or perturbation_percentage < 0:
        raise ValueError("The perturbation percentage should be between 0 and 0.5.")
    xtrain = pd.DataFrame(xtrain).reset_index(drop=True)
    xtest = pd.DataFrame(xtest).reset_index(drop=True)
    ytrain = pd.DataFrame(ytrain).reset_index(drop=True)
    ytest = pd.DataFrame(ytest).reset_index(drop=True)
    model_full = model.fit(xtrain, ytrain)
    yhat = model_full.predict(xtest)
    rgr_vals = []
    perturbed_xtrain = xtrain.reset_index(drop=True)
    variable_perturbed = perturbed_xtrain.loc[:,variable]
    sorted_data = np.sort(variable_perturbed)
    percentile_5_index = int(np.ceil(perturbation_percentage * len(sorted_data)))
    percentile_95_index = int(np.ceil(1-perturbation_percentage * len(sorted_data)))
    values_before_5th_percentile = sorted_data[:percentile_5_index]
    values_after_95th_percentile = sorted_data[percentile_95_index:]
    vals = [[i, values] for i, values in enumerate(variable_perturbed)]
    indices = [x[0] for x in sorted(vals, key= lambda item: item[1])]
    lower_tail = indices[0:percentile_5_index]
    upper_tail = (indices[-percentile_5_index:])
    upper_tail = upper_tail[::-1]
    for j in range(len(lower_tail)):
        variable_perturbed[lower_tail[j]] = variable_perturbed[upper_tail[j]]
        variable_perturbed[upper_tail[j]] = variable_perturbed[lower_tail[j]]
    perturbed_xtrain.loc[:,variable] = variable_perturbed
    rf_perturbed = model.fit(perturbed_xtrain, ytrain)
    yhat_pert = rf_perturbed.predict(xtest)
    rgr_val = _rga(yhat, yhat_pert)
    return rgr_val


def rgr_all(xtrain, xtest, ytrain, ytest, model, perturbation_percentage= 0.05):
    """
     ### RANK GRADUATION Robustness (RGR) MEASURE ###
     Function for the RGR measure computation regarding perturbation of all variables
     Inputs: xtrain -> A dataframe including xtrain data used for traing the model; 
             xtest -> A dataframe including xtest data used for testing the model;
             ytrain -> Actual values of the target variable for traing the model; 
             ytest -> Actual values of the target variable for testing the model;
             model -> Model could be a regressor such as MLPRegressor() from sklearn.neural_network
             or a classifier for example RandomForestClassifier() from sklearn.ensemble;
             perturbation_percentage -> The percentage indicating the percentile of interest for perturbation, by 
             default equal to 0.05. Therefore, the 5th and 95th percentiles are considered for the perturbation if a 
             specific percentage is not defined. The maximum value allowed to set for this input variable is 0.5.
    """ 
    if perturbation_percentage > 0.5 or perturbation_percentage < 0:
        raise ValueError("The perturbation percentage should be between 0 and 0.5.")
    xtrain = pd.DataFrame(xtrain).reset_index(drop=True)
    xtest = pd.DataFrame(xtest).reset_index(drop=True)
    ytrain = pd.DataFrame(ytrain).reset_index(drop=True)
    ytest = pd.DataFrame(ytest).reset_index(drop=True)
    model_full = model.fit(xtrain, ytrain)
    yhat = model_full.predict(xtest)
    rgr_vals = []
    perturbed_xtrain = xtrain.reset_index(drop=True)
    for i in range(xtrain.shape[1]):
        variable_perturbed = perturbed_xtrain.iloc[:,i]
        sorted_data = np.sort(variable_perturbed)
        percentile_5_index = int(np.ceil(perturbation_percentage * len(sorted_data)))
        percentile_95_index = int(np.ceil(1-perturbation_percentage * len(sorted_data)))
        values_before_5th_percentile = sorted_data[:percentile_5_index]
        values_after_95th_percentile = sorted_data[percentile_95_index:]
        vals = [[i, values] for i, values in enumerate(variable_perturbed)]
        indices = [x[0] for x in sorted(vals, key= lambda item: item[1])]
        lower_tail = indices[0:percentile_5_index]
        upper_tail = (indices[-percentile_5_index:])
        upper_tail = upper_tail[::-1]
        for j in range(len(lower_tail)):
            variable_perturbed[lower_tail[j]] = variable_perturbed[upper_tail[j]]
            variable_perturbed[upper_tail[j]] = variable_perturbed[lower_tail[j]]
        perturbed_xtrain.iloc[:,i] = variable_perturbed
    rf_perturbed = model.fit(perturbed_xtrain, ytrain)
    yhat_pert = rf_perturbed.predict(xtest)
    rgr_ = _rga(yhat, yhat_pert)
    return rgr_


def perturb(data, variable):
    data = data.reset_index(drop=True)
    perturbed_variable = data.loc[:,variable]
    sorted_variable = np.sort(perturbed_variable)
    percentile_5_index = int(np.ceil(0.15 * len(sorted_variable)))
    percentile_95_index = int(np.ceil(0.85 * len(sorted_variable)))
    values_before_5th_percentile = sorted_variable[:percentile_5_index]
    values_after_95th_percentile = sorted_variable[percentile_95_index:]
    vals = [[i, values] for i, values in enumerate(perturbed_variable)]
    indices = [x[0] for x in sorted(vals, key= lambda item: item[1])]
    lower_tail = indices[0:10]
    upper_tail = (indices[-10:])
    upper_tail = upper_tail[::-1]
    new_variable = perturbed_variable.copy()
    n = min([len(lower_tail), len(upper_tail)])
    for j in range(n):
        new_variable[lower_tail[j]] = perturbed_variable[upper_tail[j]]
        new_variable[upper_tail[j]] = perturbed_variable[lower_tail[j]]
    data.loc[:,variable] = new_variable
    return data


def rgr_statistic_test(yhat_mod1,yhat_mod2,yhat_pert_mod1,yhat_pert_mod2):
        """
        RGR based test for comparing the robustness of a model with that of a further compared model
        """
        yhat_mod1 = pd.DataFrame(yhat_mod1).reset_index(drop=True)
        yhat_mod2 = pd.DataFrame(yhat_mod2).reset_index(drop=True)
        yhat_pert_mod1 = pd.DataFrame(yhat_pert_mod1).reset_index(drop=True)
        yhat_pert_mod2 = pd.DataFrame(yhat_pert_mod2).reset_index(drop=True)
        jk_mat = pd.concat([yhat_mod1,yhat_mod2,yhat_pert_mod1,yhat_pert_mod2], axis=1, keys=["yhat_mod1", 
                                                                                        "yhat_mode2", 
                                                                                        "yhat_pert_mode1",
                                                                                        "yhat_pert_mode2"])
        n = len(jk_mat)
        index = np.arange(n)
        jk_results = []
        for i in range(n):
            jk_sample = jk_mat.iloc[[x for x in index if x != i],:]
            jk_sample.reset_index(drop=True, inplace=True)
            jk_statistic = _rga_delta_function(jk_sample, _rga)
            jk_results.append(jk_statistic)
        se = np.sqrt(((n-1)/n)*(sum([(x-np.mean(jk_results))**2 for x in jk_results])))
        z = (_rga(yhat_mod1, yhat_pert_mod1)- _rga(yhat_mod2, yhat_pert_mod2))/se
        p_value = 2*scipy.stats.norm.cdf(-abs(z))
        return p_value

