from enum import Enum


class ExperimentDataType(str, Enum):
    """_summary_"""

    TABULAR = "tabular"
    TEXT = "text"
    IMAGE = "image"
    ### Blah Blah Blah


class ModelClassifier(str, Enum):
    """_summary_"""

    ## Match model names from sklearn library
    RANDOM_FOREST = "RandomForest"
    XGBOOST = "XGBoost"
    LOGISTIC_REGRESSION = "LogisticRegression"
    # Blah Blah Blah
