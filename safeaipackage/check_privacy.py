import pandas as pd
import numpy as np
import scipy
from .utils.util import _delta_function, _rga_random, _rga, _num, _den, _rge_delta_function

class Privacy:
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
         
    def rgp(self, index):
        """
        ### RANK GRADUATION PRIVACY (RGP) MEASURE ###
        Function for the RGP measure computation
        """      
        if not isinstance(index, list):
            raise ValueError("Index input must be a list")
        for i in index:
            if i not in self.xtrain.index:
                raise ValueError(f"index {i} is not available.")
        rgp_list = []
        for i in index:
            if i in self.xtrain.index:
                xtrain_rm = self.xtrain.drop(index=i, axis=0)
                ytrain_rm = self.ytrain.drop(index=i, axis=0)
                model_rm = self.model.fit(xtrain_rm, ytrain_rm)
                yhat_rm = model_rm.predict(self.xtest)
                rgp_list.append((_rga(self.yhat, yhat_rm)))
            else:
                raise ValueError(f"index {i} is not available")
        return pd.DataFrame(rgp_list, index=index, columns=["RGP"])

    
    def rgp_statistic_test(self, index):
        """
        RGP based test for comparing the ordering of the ranks related to the full model with that of the
        reduced model without the observation of interest
        """
        xtrain_rm = self.xtrain.drop(index, axis=0)
        ytrain_rm = self.ytrain.drop(index, axis=0)
        model_rm = self.model.fit(xtrain_rm, ytrain_rm)
        yhat_rm = pd.DataFrame(model_rm.predict(self.xtest)).reset_index(drop=True)
        jk_mat = pd.concat([self.yhat, yhat_rm], axis=1)
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
        z = (_den(self.yhat)-_num(self.yhat, yhat_rm))/se
        p_value = 2*scipy.stats.norm.cdf(-abs(z))
        return p_value