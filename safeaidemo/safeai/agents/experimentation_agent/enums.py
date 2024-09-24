from safeai.enums import SafeAIStrEnum


class EvaluationMetric(SafeAIStrEnum):
    ACCURACY = "accuracy"
    EXPLAINABILITY = "explainability"
    FAIRNESS = "fairness"
    SECURITY = "security"
