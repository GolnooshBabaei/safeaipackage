import pandas as pd
import numpy as np
import scipy
from .utils.util import _rga_delta_function, _rga


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
    perturbed_xtrain = perturb(xtrain, variable, perturbation_percentage)
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
    perturbed_xtrain = xtrain.copy()
    for i in xtrain.columns:
        perturbed_xtrain[i] = perturb(xtrain, i, perturbation_percentage)[i]
    rf_perturbed = model.fit(perturbed_xtrain, ytrain)
    yhat_pert = rf_perturbed.predict(xtest)
    rgr_val= _rga(yhat, yhat_pert)
    return rgr_val


def perturb(data, variable, perturbation_percentage= 0.05):
    """
     Function to perturb a single variable based on the replacement of the two percentiles selected using the 
     given perturbation_percentage.
     Inputs: data -> A dataframe including xtrain data used for traing the model; 
             variable -> A dataframe including xtest data used for testing the model;
             perturbation_percentage -> The percentage indicating the percentile of interest for perturbation, by 
             default equal to 0.05. Therefore, the 5th and 95th percentiles are considered for the perturbation if a 
             specific percentage is not defined. The maximum value allowed to set for this input variable is 0.5.
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

