import pandas as pd
import numpy as np
import scipy
from .utils.util import _delta_function, _rga
import matplotlib.pyplot as plt


def rgr(xtrain, xtest, ytrain, ytest, model, perturbationtimes = 10):    
    xtrain = pd.DataFrame(xtrain).reset_index(drop=True)
    xtest = pd.DataFrame(xtest).reset_index(drop=True)
    ytrain = pd.DataFrame(ytrain).reset_index(drop=True)
    ytest = pd.DataFrame(ytest).reset_index(drop=True)
    model_full = model.fit(xtrain, ytrain)
    yhat = model_full.predict(xtest)
    rgr_list = []
    xtrain_pert = xtrain.copy()
    for i in xtrain.columns:
        rgr_vals = []
        for t in range(perturbationtimes):
            xtrain_pert[i] = np.random.permutation(xtrain[i])
            rf_perturbed = model.fit(xtrain_pert, ytrain)
            yhat_pert = rf_perturbed.predict(xtest)
            rgr_vals.append(_rga(yhat, yhat_pert))
        rgr_list.append(np.mean(rgr_vals))
    rgr_df = pd.DataFrame(rgr_list, index=xtest.columns, columns=["RGR"]).sort_values(by="RGR", ascending=False)
    return rgr_df

def rgr_testfunc(yhat, yhat_pert):
    yhat = pd.DataFrame(yhat).reset_index(drop=True)
    yhat_pert = pd.DataFrame(yhat_pert).reset_index(drop=True)
    df = pd.concat([yhat,yhat_pert], axis=1)
    df.columns = ["yhat", "yhat_pert"]
    ryhat_pert = yhat_pert.rank(method="min")
    ryhat_pert.columns = ["ryhat_pert"]
    df["ryhat_pert"] = ryhat_pert
    support = df.groupby('ryhat_pert')['yhat'].mean().reset_index(name='support')
    rord = list(range(len(yhat)))
    for jj in range(len(rord)):
        for ii in range(len(support)):
                if df["ryhat_pert"][jj]== support['ryhat_pert'][ii]:
                    rord[jj] = support['support'][ii]         
    vals = [[i, values] for i, values in enumerate(df["yhat_pert"])]
    ranks = [x[0] for x in sorted(vals, key= lambda item: item[1])]
    yhatstar = [rord[i] for i in ranks]
    I = list(range(1,len(yhatstar)+1))   
    conc = 2*sum([I[i]*yhatstar[i] for i in range(len(I))])
    dec = 2*sum([sorted(df["yhat"], reverse=True)[i]*I[i] for i in range(len(I))]) 
    inc = 2*sum([sorted(df["yhat"])[i]*I[i] for i in range(len(I))]) 
    RGR = (conc-dec)/(inc-dec)
    return RGR 

def rgr_statistic_test(yhat_mod1,yhat_mod2,yhat_pert_mod1,yhat_pert_mod2):
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
            jk_statistic = _delta_function(jk_sample, rgr_testfunc)
            jk_results.append(jk_statistic)
        se = np.sqrt(((n-1)/n)*(sum([(x-np.mean(jk_results))**2 for x in jk_results])))
        z = (rgr_testfunc(yhat_mod1, yhat_pert_mod1)- rgr_testfunc(yhat_mod2, yhat_pert_mod2))/se
        p_value = 2*scipy.stats.norm.cdf(-abs(z))
        return p_value

