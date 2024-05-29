import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from .utils.util import _rga, _num, _den, _rge_delta_function

class Explainability:
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
         
    def rge(self):
        """
        RANK GRADUATION EXPLAINABILITY (RGE) MEASURE
        Function for the RGE measure computation
        
        Returns:
        RGE  : Calculated RGE measure
        """ 
        rge_list = []
        model_full = self.model.fit(self.xtrain, self.ytrain)
        yhat = model_full.predict(self.xtest)
        for i in self.xtrain.columns:
            xtrain_rm = self.xtrain.drop(i, axis=1)
            xtest_rm = self.xtest.drop(i, axis=1)
            model_rm = self.model.fit(xtrain_rm, self.ytrain)
            yhat_rm = model_rm.predict(xtest_rm)
            rge_list.append(1-(_rga(yhat, yhat_rm)))
        rge_df = pd.DataFrame(rge_list, index=self.xtest.columns, columns=["RGE"]).sort_values(by="RGE", ascending=False)
        plt.figure(figsize=(10, 6))
        plt.barh(rge_df.index, rge_df["RGE"])
        plt.xlabel("RGE (Feature Importance)")
        plt.ylabel("Feature")
        plt.title("RGE")
        plt.show()
        return rge_df
    
    def rge_statistic_test(self, variable):
        """
        RGE based test for comparing the ordering of the ranks related to the full model with that of the
        reduced model without the predictor of interest
        
        Returns:
        p_value : p-value for the statistical test
        """
        if variable not in self.xtrain.columns:
            raise ValueError("protected variable is not available.")
        xtrain_rm = self.xtrain.drop(variable, axis=1)
        xtest_rm = self.xtest.drop(variable, axis=1)
        model_rm = self.model.fit(xtrain_rm, self.ytrain)
        yhat_xk = pd.DataFrame(model_rm.predict(xtest_rm)).reset_index(drop=True)
        jk_mat = pd.concat([self.yhat, yhat_xk], axis=1, keys=["yhat", "yhat_xk"])
        n = len(jk_mat)
        index = np.arange(n)
        jk_results = []
        for i in range(n):
            jk_sample = jk_mat.iloc[[x for x in index if x != i],:]
            jk_sample.reset_index(drop=True, inplace=True)
            jk_statistic = _rge_delta_function(jk_sample)
            jk_results.append(jk_statistic)
        se = np.sqrt(((n-1)/n)*(sum([(x-np.mean(jk_results))**2 for x in jk_results])))
        z = (_den(self.yhat)-_num(self.yhat,yhat_xk))/se
        p_value = 2*scipy.stats.norm.cdf(-abs(z))
        return p_value

