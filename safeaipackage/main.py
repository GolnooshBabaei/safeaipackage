from check_accuracy import rga, rga_statistic_test
from check_explainability import rge
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    data = pd.read_excel("D:/files/research_activities/ORGANIZED_FILES/safeaipackage/employee.xlsx")
    data["gender_"] = np.where(data["gender"]=="m", 0, 1)
    data["minority_"] = np.where(data["minority"]=="no_min", 0, 1)
    data = pd.get_dummies(data, columns=["jobcat"])
    data.drop(["gender", "minority"], axis=1, inplace=True)
    data["promoted"] = np.where(data["salary"]/data["startsal"] > 2,1,0)
    X = data.drop(["promoted", "salary", "startsal"], axis=1)
    y = data["promoted"]
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=1)
    rf_model = RandomForestClassifier(random_state=1).fit(xtrain, ytrain)
    yhat = rf_model.predict(xtest)
    #print(rga(ytest,yhat))
    X_prevexp = X.drop("prevexp", axis=1)
    xtrain_prevexp, xtest_prevexp, ytrain, ytest = train_test_split(X_prevexp, y, test_size=0.3, random_state=1)
    rfmodel_prevexp = RandomForestClassifier(random_state=1).fit(xtrain_prevexp, ytrain)
    rf_predictions_prevexp = rfmodel_prevexp.predict(xtest_prevexp)
    rga_prevexp = rga(ytest, rf_predictions_prevexp)
    #print(rga_statistic_test(ytest, rf_predictions_prevexp, yhat))
    X_gender = X.drop("gender_", axis=1)
    xtrain_gender, xtest_gender, ytrain, ytest = train_test_split(X_gender, y, test_size=0.3, random_state=1)
    rfmodel_gender = RandomForestClassifier(random_state=1).fit(xtrain_gender, ytrain)
    yhat_gender = rfmodel_gender.predict(xtest_gender)
    rge_gender = rge(yhat_gender, yhat)
    print(rge_gender)