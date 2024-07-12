"""This is a helper module with utility classes and functions."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def _delta_function(data, func):
        result = (func(data.iloc[:,0], data.iloc[:,2]))-(func(data.iloc[:,0], data.iloc[:,1]))
        return result


def _rge_delta_function(data):
    return _den(data.iloc[:,0])-_num(data.iloc[:,0], data.iloc[:,1])


def _rga(y, yhat):
    y = pd.DataFrame(y).reset_index(drop=True)
    yhat = pd.DataFrame(yhat).reset_index(drop=True)
    df = pd.concat([y,yhat], axis=1)
    df.columns = ["y", "yhat"]
    df["ryhat"] = df["yhat"].rank(method="min")
    support = df.groupby('ryhat')['y'].mean().reset_index(name='support')
    df = pd.merge(df, support, on= "ryhat")
    
    inc = np.sum(np.arange(1,len(df)+1) * sorted(df["y"]), dtype=np.int64)
    dec = np.sum(np.arange(1,len(df)+1) * sorted(df["y"], reverse= True), dtype=np.int64)
    
    ystar = df.sort_values(by= "ryhat")["support"]
    ystar.reset_index(drop=True, inplace=True)
    conc = np.sum(np.arange(1,len(df)+1) * ystar, dtype=np.int64)
    RGA = (conc-dec)/(inc-dec)
    return RGA


def _num(yhat, yhat_xk):
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


def _den(yhat):
    yhat = pd.DataFrame(yhat)
    yhat.columns = ["yhat"]
    I = list(range(len(yhat)))
    dec= 2*sum([yhat.sort_values(by="yhat", ascending=False, ignore_index=True).iloc[i,0]*I[i] for i in range(len(I))]) 
    inc = 2*sum([yhat.sort_values(by="yhat", ignore_index=True).iloc[i,0]*I[i] for i in range(len(I))]) 
    RGE_den=(inc-dec)
    return RGE_den


