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


class ModelClassifier(SafeAIEnum):
    """_summary_"""

    ## Match model names from sklearn library
    RANDOM_FOREST = "RandomForest"
    XGBOOST = "XGBoost"
    LOGISTIC_REGRESSION = "LogisticRegression"
    # Blah Blah Blah


class SafeAILLMS(SafeAIEnum):
    OPENAI = "OpenAI"
    GPT2 = "GPT2"
    GPT3 = "GPT3"
    # Blah Blah Blah
