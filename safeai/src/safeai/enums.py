from enum import Enum


class SafeAIEnum(str, Enum):
    """_summary_"""

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return str(self.value)

    @classmethod
    def to_list(cls) -> list[str]:
        """_summary_"""
        return [str(enum.value) for enum in cls]


class ExperimentDataType(SafeAIEnum):
    """_summary_"""

    TABULAR = "Tabular"
    TEXT = "Text"
    IMAGE = "Image"
    ### Blah Blah Blah


class PredictionType(SafeAIEnum):
    """_summary_"""

    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"
    ### Blah Blah Blah


class ModelRegressor(SafeAIEnum):
    """_summary_"""

    CATBOOSTREGRESSOR = "CatBoostRegressor"
    XGBREGRESSOR = "XGBRegressor"
    LOGISTICREGRESSION = "LogisticRegression"


class ModelClassifier(SafeAIEnum):
    """_summary_"""

    # TODO: Seperate Classifiers and Regressors
    ## Match model names from sklearn library
    CATBOOSTCLASSIFIER = "CatBoostClassifier"
    XGBCLASSIFIER = "XGBClassifier"
    RANDOMFORESTCLASSIFIER = "RandomForestClassifier"
    # Blah Blah Blah


class SafeAILLMS(SafeAIEnum):
    OPENAI = "OpenAI"
    GPT2 = "GPT2"
    GPT3 = "GPT3"
    # Blah Blah Blah
