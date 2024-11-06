import pandas as pd
import numpy as np
from typing import Union
from .core import rga
from .util.utils import convert_to_dataframe, check_nan, find_yhat
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.base import is_classifier, is_regressor
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier, XGBRegressor

def compute_rga_parity(xtrain: pd.DataFrame, 
                       xtest: pd.DataFrame, 
                       ytest: list, 
                       yhat: list, 
                       model: Union[CatBoostClassifier, CatBoostRegressor, XGBClassifier, XGBRegressor, BaseEstimator], 
                       protectedvariable: str):
    """
    Compute RGA-based imparity MEASURE. 

    Parameters
    ----------
    xtrain : pd.DataFrame
            A dataframe including train data.
    xtest : pd.DataFrame
            A dataframe including test data.
    ytest : list
            A list of actual values.
    yhat : list
            A list of predicted values.
    model : Union[CatBoostClassifier, CatBoostRegressor, XGBClassifier, XGBRegressor, BaseEstimator]
            A trained model, which could be a classifier or regressor. 
    protectedvariable: str 
            Name of the protected (sensitive) variable.

    Returns
    -------
    str
            RGA-based imparity score.
    """
    # check if the protected variable is available in data
    if protectedvariable not in xtrain.columns:
        raise ValueError(f"{protectedvariable} is not in the variables")
    xtrain, xtest, ytest, yhat = convert_to_dataframe(xtrain, xtest, ytest, yhat)
    # check for missing values
    check_nan(xtrain, xtest, ytest, yhat)
    # find protected groups
    protected_groups = xtrain[protectedvariable].value_counts().index
    # measure RGA for each group
    rgas = []
    for i in protected_groups:
        xtest_pr = xtest[xtest[protectedvariable]== i]
        ytest_pr = ytest.loc[xtest_pr.index]
        yhat_pr = find_yhat(model, xtest_pr)         
        rga_value = rga(ytest_pr, yhat_pr)
        rgas.append(rga_value)            
    return f"The RGA-based imparity between the protected gorups is {max(rgas)-min(rgas)}."    
 

