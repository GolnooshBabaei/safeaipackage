import pandas as pd
from .check_accuracy import rga 

#def rgf(yhat_rm, yhat):    
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
#    RGF=(conc-dec)/(inc-dec)
#    return 1-RGF

def rgf(xtrain, xtest, ytrain, ytest, protectedvariables, model):
    if not isinstance(protectedvariables, list):
        raise ValueError("Protectedvariables input must be a list")
    xtrain = pd.DataFrame(xtrain).reset_index(drop=True)
    xtest = pd.DataFrame(xtest).reset_index(drop=True)
    ytrain = pd.DataFrame(ytrain).reset_index(drop=True)
    ytest = pd.DataFrame(ytest).reset_index(drop=True)
    for i in range(len(list(protectedvariables))):
        if protectedvariables[i] not in xtrain:
            raise ValueError(f"{protectedvariables[i]} is not in the variables")
    rgf_list = []
    model_full = model.fit(xtrain, ytrain)
    yhat = model_full.predict(xtest)
    for i in protectedvariables:
         xtrain_pr = xtrain.drop(i, axis=1)
         xtest_pr = xtest.drop(i, axis=1)
         model_pr = model.fit(xtrain_pr, ytrain)
         yhat_pr = model_pr.predict(xtest_pr)
         rgf_list.append(rga(yhat, yhat_pr))
    return pd.DataFrame(rgf_list, index=protectedvariables, columns=["RGF"]).sort_values(by="RGF", ascending=False)