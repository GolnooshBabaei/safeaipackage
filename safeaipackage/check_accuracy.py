import pandas as pd
import numpy as np
import scipy

def rga(y, yhat):
    y = pd.DataFrame(y).reset_index(drop=True)
    yhat = pd.DataFrame(yhat).reset_index(drop=True)
    df = pd.concat([y,yhat], axis=1)
    df.columns = ["y", "yhat"]
    ryhat = yhat.rank(method="min")
    df["ryhat"] = ryhat
    support = df.groupby('ryhat')['y'].mean().reset_index(name='support')
    rord = list(range(len(y)))
    for jj in range(len(rord)):
        for ii in range(len(support)):
                if ryhat[jj]== support['ryhat'][ii]:
                    rord[jj] = support['support'][ii]
    vals = [[i, values] for i, values in enumerate(yhat)]
    ranks = [x[0] for x in sorted(vals, key= lambda item: item[1])]
    ystar = [rord[i] for i in ranks]
    I = list(range(1,len(y)+1))
    conc = 2*sum([I[i]*ystar[i] for i in range(len(I))])
    dec= 2*sum([sorted(y, reverse=True)[i]*I[i] for i in range(len(I))]) # second term of the RGA numerator and denominator (dual Lorenz)
    inc = 2*sum([sorted(y)[i]*I[i] for i in range(len(I))]) # first term of the RGA denominator (Lorenz)
    RGA=(conc-dec)/(inc-dec)
    return RGA 

def delta_function(data):
        result = (rga(data.iloc[:,0], data.iloc[:,2]))-(rga(data.iloc[:,0], data.iloc[:,1]))
        return result
    
def rga_statistic_test(y, yhat_rm, yhat_cm):
        jk_mat = pd.concat([y,yhat_rm, yhat_cm], axis=1, keys=["y", "yhat_rm", "yhat_cm"])
        n = len(jk_mat)
        index = np.arange(n)
        jk_results = []
        for i in range(n):
            jk_sample = jk_mat.iloc[[x for x in index if x != i],:]
            jk_sample.reset_index(drop=True, inplace=True)
            jk_statistic = delta_function(jk_sample)
            jk_results.append(jk_statistic)
        se = np.sqrt(((n-1)/n)*(sum([(x-np.mean(jk_results))**2 for x in jk_results])))
        z = (rga(y,yhat_cm)-rga(y,yhat_rm))/se
        p_value = 2*scipy.stats.norm.cdf(-abs(z))
        return p_value