import pandas as pd
import numpy as np
import scipy
from .utils.util import _rga, _num, _den, _rge_delta_function

class Fairness:
    def __init__(self, xtrain, xtest, ytrain, ytest, model):
        """
        Inputs: xtrain, xtest, ytrain, ytest -> the train and test data selected for training and testing the model; 
        model -> a classification or regression model. For example RandomForestClassifier() in ensemble module of sklearn.
        """
        self.xtrain = pd.DataFrame(xtrain).reset_index(drop=True)
        self.xtest = pd.DataFrame(xtest).reset_index(drop=True)
        self.ytrain = pd.DataFrame(ytrain).reset_index(drop=True)
        self.ytest = pd.DataFrame(ytest).reset_index(drop=True)
        self.model = model
        model_full = self.model.fit(xtrain, ytrain)
        self.yhat = pd.DataFrame(model_full.predict(xtest)).reset_index(drop=True)
         
    def rgf(self, protectedvariable):
        """
        RANK GRADUATION FAIRNESS (RGF) MEASURE 
        Function for the RGF measure computation
        
        Returns:
        RGF  : Calculated RGF measure
        """   
        if not isinstance(protectedvariable, list):
            raise ValueError("Protectedvariables input must be a list")
        for i in range(len(list(protectedvariable))):
            if protectedvariable[i] not in self.xtrain:
                raise ValueError(f"{protectedvariable[i]} is not in the variables")
        rgf_list = []
        for i in protectedvariable:
            xtrain_pr = self.xtrain.drop(i, axis=1)
            xtest_pr = self.xtest.drop(i, axis=1)
            model_pr = self.model.fit(xtrain_pr, self.ytrain)
            yhat_pr = model_pr.predict(xtest_pr)
            rgf_list.append(_rga(self.yhat, yhat_pr))
        return pd.DataFrame(rgf_list, index=protectedvariable, columns=["RGF"]).sort_values(by="RGF", ascending=False)

    
    def rgf_statistic_test(self, protectedvariable):
        """
        RGF based test for comparing the ordering of the ranks related to the full model with that of the
        reduced model without the protected variable 
        
        Returns:
        p_value : p-value for the statistical test
        """
        if protectedvariable not in self.xtrain.columns:
            raise ValueError("protected variable is not available.")
        xtrain_pr = self.xtrain.drop(protectedvariable, axis=1)
        xtest_pr = self.xtest.drop(protectedvariable, axis=1)
        model_pr = self.model.fit(xtrain_pr, self.ytrain)
        yhat_pr = pd.DataFrame(model_pr.predict(xtest_pr)).reset_index(drop=True)
        jk_mat = pd.concat([self.yhat, yhat_pr], axis=1)
        jk_mat.columns = ["yhat", "yhat_rm"]
        n = len(jk_mat)
        index = np.arange(n)
        jk_results = []
        for i in range(n):
            jk_sample = jk_mat.iloc[[x for x in index if x != i],:]
            jk_sample.reset_index(drop=True, inplace=True)
            jk_statistic = _rge_delta_function(jk_sample)
            jk_results.append(jk_statistic)
        se = np.sqrt(((n-1)/n)*(sum([(x-np.mean(jk_results))**2 for x in jk_results])))
        z = (_den(self.yhat)-_num(self.yhat, yhat_pr))/se
        p_value= 2*scipy.stats.norm.cdf(-abs(z))
        return p_value