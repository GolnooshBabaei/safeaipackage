import pandas as pd
import numpy as np
import scipy
from .utils.util import _rga, _delta_function

def rgf(xtrain, xtest, ytrain, ytest, protectedvariables, model):
    """
     ### RANK GRADUATION FAIRNESS (RGF) MEASURE ###
     Function for the RGF measure computation
     Inputs: xtrain, xtest, ytrain, ytest -> the train and test data selected for training and testing the model;
             protectedvariables -> a list of variables should be considered as the protected variables; 
             model -> a classification or regression model. For example RandomForestClassifier() in ensemble module of sklearn.
    """    
    if not isinstance(protectedvariables, list):
        raise ValueError("Protectedvariables input must be a list")
    xtrain = pd.DataFrame(xtrain).reset_index(drop=True)
    xtest = pd.DataFrame(xtest).reset_index(drop=True)
    ytrain = pd.DataFrame(ytrain).reset_index(drop=True)
    ytest = pd.DataFrame(ytest).reset_index(drop=True)
    for i in range(len(list(protectedvariables))):
        if protectedvariables[i] not in xtrain:
            raise ValueError(f"{protectedvariables[i]} is not in the variables")
    rgf_list = []
    model_full = model.fit(xtrain, ytrain)
    yhat = model_full.predict(xtest)
    for i in protectedvariables:
         xtrain_pr = xtrain.drop(i, axis=1)
         xtest_pr = xtest.drop(i, axis=1)
         model_pr = model.fit(xtrain_pr, ytrain)
         yhat_pr = model_pr.predict(xtest_pr)
         rgf_list.append(_rga(yhat, yhat_pr))
    return pd.DataFrame(rgf_list, index=protectedvariables, columns=["RGF"]).sort_values(by="RGF", ascending=False)


def rgf_statistic_test(y, yhat_rm, yhat_cm):
        """
        RGF based test for comparing the ordering of the ranks related to the full model with that of the
        reduced model without the group variable of interest
        """
        y = pd.DataFrame(y).reset_index(drop=True)
        yhat_rm = pd.DataFrame(yhat_rm).reset_index(drop=True)
        yhat_cm = pd.DataFrame(yhat_cm).reset_index(drop=True)
        jk_mat = pd.concat([y,yhat_rm, yhat_cm], axis=1, keys=["y", "yhat_rm", "yhat_cm"])
        n = len(jk_mat)
        index = np.arange(n)
        jk_results = []
        for i in range(n):
            jk_sample = jk_mat.iloc[[x for x in index if x != i],:]
            jk_sample.reset_index(drop=True, inplace=True)
            jk_statistic = _delta_function(jk_sample, _rga)
            jk_results.append(jk_statistic)
        se = np.sqrt(((n-1)/n)*(sum([(x-np.mean(jk_results))**2 for x in jk_results])))
        z = (_rga(y,yhat_cm)-_rga(y,yhat_rm))/se
        p_value = 2*scipy.stats.norm.cdf(-abs(z))
        return p_value