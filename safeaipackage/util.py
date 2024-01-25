"""This is a helper module with utility classes and functions."""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def delta_function(data, func):
        result = (func(data.iloc[:,0], data.iloc[:,2]))-(func(data.iloc[:,0], data.iloc[:,1]))
        return result
    
def rf_modeling(X, y, testsize):
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size= testsize, random_state=1, stratify=y)
    rf_model =  RandomForestClassifier(random_state=1).fit(xtrain, ytrain)
    rf_predictions = rf_model.predict(xtest)
    return rf_predictions