import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from .util import _rga 
import matplotlib.pyplot as plt


def rge(xtrain, xtest, ytrain, ytest, model):    
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
