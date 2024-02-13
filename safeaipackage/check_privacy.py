import pandas as pd
from .utils.util import _rga

def rgp(xtrain, xtest, ytrain, ytest, model):    
    xtrain = pd.DataFrame(xtrain).reset_index(drop=True)
    xtest = pd.DataFrame(xtest).reset_index(drop=True)
    ytrain = pd.DataFrame(ytrain).reset_index(drop=True)
    ytest = pd.DataFrame(ytest).reset_index(drop=True)
    rgp_list = []
    model_full = model.fit(xtrain, ytrain)
    yhat = model_full.predict(xtest)
    for i in range(len(xtrain)):
        indices = xtrain.index
        xtrain_rm = xtrain.drop(index=indices[i], axis=0)
        ytrain_rm = ytrain.drop(index=indices[i], axis=0)
        model_rm = model.fit(xtrain_rm, ytrain_rm)
        yhat_rm = model_rm.predict(xtest)
        rgp_list.append((1-_rga(yhat, yhat_rm)))
    return pd.DataFrame(rgp_list, index=indices, columns=["RGP"])


def rgp_statistic_test():
    pass