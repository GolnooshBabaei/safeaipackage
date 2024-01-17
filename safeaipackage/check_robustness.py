import pandas as pd
import numpy as np
import scipy




def rgr(yhat, yhat_pert):
    df = pd.concat([yhat,yhat_pert], axis=1, keys=['yhat', 'yhat_pert'])
    ryhat_pert = yhat_pert.rank(method="min")
    df["ryhat_pert"] = ryhat_pert
    support = df.groupby('ryhat_pert')['yhat'].mean().reset_index(name='support')
    rord = list(range(len(yhat)))
    for jj in range(len(rord)):
        for ii in range(len(support)):
                if ryhat_pert[jj]== support['ryhat_pert'][ii]:
                    rord[jj] = support['support'][ii]         
    vals = [[i, values] for i, values in enumerate(yhat_pert)]
    ranks = [x[0] for x in sorted(vals, key= lambda item: item[1])]
    yhatstar = [rord[i] for i in ranks]
    I = list(range(1,len(yhatstar)+1))   ########by this line everything is correct######
    conc = 2*sum([I[i]*yhatstar[i] for i in range(len(I))])
    dec = 2*sum([sorted(yhat, reverse=True)[i]*I[i] for i in range(len(I))]) 
    inc = 2*sum([sorted(yhat)[i]*I[i] for i in range(len(I))]) 
    RGR = (conc-dec)/(inc-dec)
    return RGR
    
def delta_function(data):
        result = (rgr(data.iloc[:,0], data.iloc[:,2]))-(rgr(data.iloc[:,0], data.iloc[:,1]))
        return result
    
    
def rgr_statistic_test(yhat_mod1,yhat_mod2,yhat_pert_mod1,yhat_pert_mod2):
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
            jk_statistic = delta_function(jk_sample)
            jk_results.append(jk_statistic)
        se = np.sqrt(((n-1)/n)*(sum([(x-np.mean(jk_results))**2 for x in jk_results])))
        z = (rgr(yhat_mod1, yhat_pert_mod1)- rgr(yhat_mod2, yhat_pert_mod2))/se
        p_value = 2*scipy.stats.norm.cdf(-abs(z))
        return p_value


