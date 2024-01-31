import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from .util import rf_modeling
from .check_accuracy import rga 
import matplotlib.pyplot as plt

#def rge(yhat_rm, yhat):    
#    yhat_rm = pd.DataFrame(yhat_rm).reset_index(drop=True)
#    yhat = pd.DataFrame(yhat).reset_index(drop=True)
#    df = pd.concat([yhat_rm,yhat], axis=1)
#    df.columns = ["yhat_rm", "yhat"]
#    ryhat_rm = yhat_rm.rank(method="min")
#    df["ryhat_rm"] = ryhat_rm
#    support = df.groupby('ryhat_rm')['yhat'].mean().reset_index(name='support')
#    rord = list(range(len(yhat)))
#    for jj in range(len(rord)):
#        for ii in range(len(support)):
#                if df["ryhat_rm"][jj]== support['ryhat_rm'][ii]:
#                    rord[jj] = support['support'][ii]
#    vals = [[i, values] for i, values in enumerate(df["yhat_rm"])]
#    ranks = [x[0] for x in sorted(vals, key= lambda item: item[1])]
#    ystar = [rord[i] for i in ranks]
#    I = list(range(len(yhat)))
#    conc = 2*sum([I[i]*ystar[i] for i in range(len(I))])
#    dec= 2*sum([sorted(df["yhat"], reverse=True)[i]*I[i] for i in range(len(I))]) 
#    inc = 2*sum([sorted(df["yhat"])[i]*I[i] for i in range(len(I))]) 
#    RGE=(conc-dec)/(inc-dec)
#    return 1-RGE 

#def rge(X, y, testsize = 0.3):
#    rf_predictions_full = rf_modeling(X,y, testsize)
#    rge_values = []
#    for i in X.columns:
#         X_ = X.drop(i, axis=1)
#         rf_predictions_rm = rf_modeling(X_, y, testsize)
#         rge = rga(rf_predictions_rm, rf_predictions_full)
#         rge_values.append(1-rge)
#    rge_df = pd.DataFrame(rge_values, index= X.columns, columns= ["RGE"]).sort_values(by="RGE", ascending=False)
#    plt.figure(figsize=(10, 6))
#    plt.barh(rge_df.index, rge_df["RGE"])
#    plt.xlabel("RGE (Feature Importance)")
#    plt.ylabel("Feature")
#    plt.title("RGE(yhat_reducedmodel, yhat_fullmodel)")
#    plt.show()
#    return rge_df


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
         rge_list.append(1-(rga(yhat, yhat_rm)))
    rge_df = pd.DataFrame(rge_list, index=xtest.columns, columns=["RGE"]).sort_values(by="RGE", ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(rge_df.index, rge_df["RGE"])
    plt.xlabel("RGE (Feature Importance)")
    plt.ylabel("Feature")
    plt.title("RGE")
    plt.show()
    return rge_df
