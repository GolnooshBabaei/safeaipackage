import pandas as pd
import numpy as np
import scipy
from sklearn.dummy import DummyClassifier, DummyRegressor
from .utils.util import _delta_function, _rga_random, _rga

class Robustness():
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
        
    def perturb(self, data, variable, perturbation_percentage= 0.05):
        """
         Function to perturb a single variable based on the replacement of the two percentiles selected using the
         perturbation_percentage of the object.
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

    def rgr_single(self, variable, perturbation_percentage= 0.05):
        """
         ### RANK GRADUATION Robustness (RGR) MEASURE ###
         Function for the RGR measure computation regarding perturbation of a single variable
        """ 
        if perturbation_percentage > 0.5 or perturbation_percentage < 0:
            raise ValueError("The perturbation percentage should be between 0 and 0.5.")
            
        if variable not in self.xtrain.columns:
            raise ValueError("The selected variable is not available.")

        perturbed_xtrain = self.perturb(self.xtrain, variable, perturbation_percentage)
        rf_perturbed = self.model.fit(perturbed_xtrain, self.ytrain)
        yhat_pert = rf_perturbed.predict(self.xtest)
        rgr_val = _rga(self.yhat, yhat_pert)
        return rgr_val
    
    
    def rgr_all(self, perturbation_percentage= 0.05):
        """
         ### RANK GRADUATION Robustness (RGR) MEASURE ###
         Function for the RGR measure computation regarding perturbation of all variables
        """ 
        if perturbation_percentage > 0.5 or perturbation_percentage < 0:
            raise ValueError("The perturbation percentage should be between 0 and 0.5.")    
        perturbed_xtrain = self.xtrain.copy()
        for i in self.xtrain.columns:
            perturbed_xtrain[i] = self.perturb(self.xtrain, i, perturbation_percentage)[i]
        rf_perturbed = self.model.fit(perturbed_xtrain, self.ytrain)
        yhat_pert = rf_perturbed.predict(self.xtest)
        rgr_val= _rga(self.yhat, yhat_pert)
        return rgr_val
    
    def rgr_statistic_test(self, problemtype, secondmodel= np.nan, variable= np.nan, perturbation_percentage= 0.05):
        """
        RGR based test for comparing the robustness of a model with that of a further model if a value is set
        for the secondmodel. In this case, variable should be set as well. If the second model and variable are
        not given by the user, this test function compares the main model of the class object with a baseline model
        depending on the type of the problem defined by the problemtype, either classification or prediction.
        """
        if problemtype not in ["classification", "prediction"]:
            raise ValueError("problemtype should be classification or prediction")
        if secondmodel is not np.nan:
            secondmodel = secondmodel
        else:
            if problemtype == "classification":
                secondmodel = DummyClassifier(strategy="most_frequent", random_state=1)
            else:
                secondmodel = DummyRegressor(strategy="mean")
        mod2 = secondmodel.fit(self.xtrain, self.ytrain)
        yhat_mod2 = pd.DataFrame(mod2.predict(self.xtest)).reset_index(drop=True)
        
        if variable is not np.nan:
            #perturb the selected variable
            perturbed_xtrain = self.perturb(self.xtrain, variable, perturbation_percentage)
            rf_perturbed = self.model.fit(perturbed_xtrain, self.ytrain)
            yhat_pert = pd.DataFrame(rf_perturbed.predict(self.xtest)).reset_index(drop=True)
            mode2_pert = secondmodel.fit(perturbed_xtrain, self.ytrain)
            yhat_mode2_pert = pd.DataFrame(mode2_pert.predict(self.xtest)).reset_index(drop=True)

            jk_mat = pd.concat([self.yhat,yhat_mod2,yhat_pert,yhat_mode2_pert], axis=1)
            jk_mat.columns = ["yhat_mod1", "yhat_mode2", "yhat_pert_mode1", "yhat_pert_mode2"]
            n = len(jk_mat)
            index = np.arange(n)
            jk_results = []
            for i in range(n):
                jk_sample = jk_mat.iloc[[x for x in index if x != i],:]
                jk_sample.reset_index(drop=True, inplace=True)
                jk_statistic = _delta_function(jk_sample, _rga)
                jk_results.append(jk_statistic)
            se = np.sqrt(((n-1)/n)*(sum([(x-np.mean(jk_results))**2 for x in jk_results])))
            z = (_rga(self.yhat, yhat_pert)- _rga(yhat_mod2, yhat_mode2_pert))/se
            p_value = 2*scipy.stats.norm.cdf(-abs(z))
            
            
        else:
            #perturb all the variables
            perturbed_xtrain = self.xtrain.copy()
            for i in self.xtrain.columns:
                perturbed_xtrain[i] = self.perturb(self.xtrain, i, perturbation_percentage)[i]
            rf_perturbed = self.model.fit(perturbed_xtrain, self.ytrain)
            yhat_pert = pd.DataFrame(rf_perturbed.predict(self.xtest)).reset_index(drop=True)
            mode2_pert = secondmodel.fit(perturbed_xtrain, self.ytrain)
            yhat_mode2_pert = pd.DataFrame(mode2_pert.predict(self.xtest)).reset_index(drop=True)
         
            jk_mat = pd.concat([self.yhat,yhat_mod2,yhat_pert,yhat_mode2_pert], axis=1)
            jk_mat.columns = ["yhat_mod1", "yhat_mode2", "yhat_pert_mode1", "yhat_pert_mode2"]
            n = len(jk_mat)
            index = np.arange(n)
            jk_results = []
            for i in range(n):
                jk_sample = jk_mat.iloc[[x for x in index if x != i],:]
                jk_sample.reset_index(drop=True, inplace=True)
                jk_statistic = _delta_function(jk_sample, _rga)
                jk_results.append(jk_statistic)
            se = np.sqrt(((n-1)/n)*(sum([(x-np.mean(jk_results))**2 for x in jk_results])))
            z = (_rga(self.yhat, yhat_pert)- _rga_random(yhat_mod2, yhat_mode2_pert))/se
            p_value = 2*scipy.stats.norm.cdf(-abs(z))
        return p_value