import pandas as pd
import numpy as np
import scipy
from .utils.util import _rga, _delta_function, _num, _den 
import matplotlib.pyplot as plt


def rge(xtrain, xtest, ytrain, ytest, model):
    """
     ### RANK GRADUATION EXPLAINABILITY (RGE) MEASURE ###
     Function for the RGE measure computation
     Inputs: xtrain, xtest, ytrain, ytest -> the train and test data selected for training and testing the model; 
             model -> a classification or regression model. For example RandomForestClassifier() in ensemble module of sklearn.
    """    
    xtrain = pd.DataFrame(xtrain).reset_index(drop=True)
    xtest = pd.DataFrame(xtest).reset_index(drop=True)
    ytrain = pd.DataFrame(ytrain).reset_index(drop=True)
    ytest = pd.DataFrame(ytest).reset_index(drop=True)
    rge_list = []
    model_full = model.fit(xtrain, ytrain)
    yhat = model_full.predict(xtest)
    for i in xtrain.columns:
         xtrain_rm = xtrain.drop(i, axis=1)
         xtest_rm = xtest.drop(i, axis=1)
         model_rm = model.fit(xtrain_rm, ytrain)
         yhat_rm = model_rm.predict(xtest_rm)
         rge_list.append(1-(_rga(yhat, yhat_rm)))
    rge_df = pd.DataFrame(rge_list, index=xtest.columns, columns=["RGE"]).sort_values(by="RGE", ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(rge_df.index, rge_df["RGE"])
    plt.xlabel("RGE (Feature Importance)")
    plt.ylabel("Feature")
    plt.title("RGE")
    plt.show()
    return rge_df


def rge_num(yhat, yhat_xk):
    yhat = pd.DataFrame(yhat).reset_index(drop=True)
    yhat_xk = pd.DataFrame(yhat_xk).reset_index(drop=True)
    df = pd.concat([yhat,yhat_xk], axis=1)
    df.columns = ["yhat", "yhat_xk"]
    ryhat_xk = yhat_xk.rank(method="min")
    df["ryhat_xk"] = ryhat_xk
    support = df.groupby('ryhat_xk')['yhat'].mean().reset_index(name='support')
    rord = list(range(len(yhat)))
    for jj in range(len(rord)):
        for ii in range(len(support)):
                if df["ryhat_xk"][jj]== support['ryhat_xk'][ii]:
                    rord[jj] = support['support'][ii]
    vals = [[i, values] for i, values in enumerate(df["yhat_xk"])]
    ranks = [x[0] for x in sorted(vals, key= lambda item: item[1])]
    ystar = [rord[i] for i in ranks]
    I = list(range(len(ystar)))
    conc = 2*sum([I[i]*ystar[i] for i in range(len(I))])
    dec= 2*sum([sorted(df["yhat"], reverse=True)[i]*I[i] for i in range(len(I))]) 
    inc = 2*sum([sorted(df["yhat"])[i]*I[i] for i in range(len(I))]) 
    RGE_num=(conc-dec)
    return RGE_num 

def rge_den(yhat):
    yhat = pd.DataFrame(yhat)
    yhat.columns = ["yhat"]
    I = list(range(len(yhat)))
    dec= 2*sum([yhat.sort_values(by="yhat", ascending=False, ignore_index=True).iloc[i,0]*I[i] for i in range(len(I))]) 
    inc = 2*sum([yhat.sort_values(by="yhat", ignore_index=True).iloc[i,0]*I[i] for i in range(len(I))]) 
    RGE_den=(inc-dec)
    return RGE_den

def delta_function(data):
    return rge_den(data.iloc[:,0])-rge_num(data.iloc[:,0], data.iloc[:,1])

def rge_statistic_test(yhat, yhat_xk):
     """
     RGE based test for comparing the ordering of the ranks related to the full model with that of the
     reduced model without the predictor of interest
     """
     yhat = pd.DataFrame(yhat).reset_index(drop=True)
     yhat_xk = pd.DataFrame(yhat_xk).reset_index(drop=True)
     jk_mat = pd.concat([yhat, yhat_xk], axis=1, keys=["yhat", "yhat_xk"])
     n = len(jk_mat)
     index = np.arange(n)
     jk_results = []
     for i in range(n):
          jk_sample = jk_mat.iloc[[x for x in index if x != i],:]
          jk_sample.reset_index(drop=True, inplace=True)
          jk_statistic = _delta_function(jk_sample)
          jk_results.append(jk_statistic)
     se = np.sqrt(((n-1)/n)*(sum([(x-np.mean(jk_results))**2 for x in jk_results])))
     z = (_den(yhat)-_num(yhat,yhat_xk))/se
     p_value = 2*scipy.stats.norm.cdf(-abs(z))
     return p_value