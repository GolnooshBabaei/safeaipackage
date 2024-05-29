import pandas as pd
import numpy as np
import scipy
from sklearn.dummy import DummyClassifier, DummyRegressor
from .utils.util import _delta_function, _rga

class Accuracy:
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
        self.yhat_cm = pd.DataFrame(model_full.predict(xtest)).reset_index(drop=True)
         
    def rga(self):
        """
        RANK GRADUATION ACCURACY (RGA) MEASURE
        Function for the RGA measure computation

        Returns:
        RGA  : Calculated RGA measure
        """ 
        df = pd.concat([self.ytest,self.yhat_cm], axis=1)
        df.columns = ["y", "yhat"]
        ryhat = self.yhat_cm.rank(method="min")
        df["ryhat"] = ryhat
        support = df.groupby('ryhat')['y'].mean().reset_index(name='support')
        rord = list(range(len(self.ytest)))
        for jj in range(len(rord)):
            for ii in range(len(support)):
                    if df["ryhat"][jj]== support['ryhat'][ii]:
                        rord[jj] = support['support'][ii]
        vals = [[i, values] for i, values in enumerate(df["yhat"])]
        ranks = [x[0] for x in sorted(vals, key= lambda item: item[1])]
        ystar = [rord[i] for i in ranks]
        I = list(range(len(self.ytest)))
        conc = sum([I[i]*ystar[i] for i in range(len(I))])
        dec= sum([sorted(df["y"], reverse=True)[i]*I[i] for i in range(len(I))]) 
        inc = sum([sorted(df["y"])[i]*I[i] for i in range(len(I))]) 
        RGA=(conc-dec)/(inc-dec)
        return RGA
    
    
    def rga_statistic_test(self, problemtype, variable= np.nan):
        """
        RGA based test for comparing the predictive accuracy of a reduced model with that of a more complex model Or 
        compare the model with a random model
        
        Returns:
        p_value : p-value for the statistical test
        """
        if problemtype not in ["classification", "prediction"]:
            raise ValueError("problemtype should be classification or prediction")
                             
        if variable is not np.nan: 
            xtrain_rm = self.xtrain.drop(variable, axis=1)
            xtest_rm = self.xtest.drop(variable, axis=1)
            model_rm = self.model.fit(xtrain_rm, self.ytrain)
            yhat_rm = pd.DataFrame(model_rm.predict(xtest_rm)).reset_index(drop=True)

        else:
            if problemtype == "classification":
                model_2 = DummyClassifier(strategy="most_frequent", random_state=1)
            else:
                model_2 = DummyRegressor(strategy="mean")
                                                
            model_rm = model_2.fit(self.xtrain, self.ytrain)
            yhat_rm = pd.DataFrame(model_rm.predict(self.xtest)).reset_index(drop=True)

        jk_mat = pd.concat([self.ytest, yhat_rm, self.yhat_cm], axis=1)
        jk_mat.columns = ["y", "yhat_rm", "yhat_cm"]
        n = len(jk_mat)
        index = np.arange(n)
        jk_results = []
        for i in range(n):
            jk_sample = jk_mat.drop(labels= i)
            jk_sample.reset_index(drop=True, inplace=True)
            jk_statistic = _delta_function(jk_sample, _rga)
            jk_results.append(jk_statistic)
        se = np.sqrt(((n-1)/n)*(sum([(x-np.mean(jk_results))**2 for x in jk_results])))
        ## when random, se 
        z = (_rga(self.ytest,self.yhat_cm)-_rga(self.ytest,yhat_rm))/se
        p_value = 2*scipy.stats.norm.cdf(-abs(z)) 
        return p_value