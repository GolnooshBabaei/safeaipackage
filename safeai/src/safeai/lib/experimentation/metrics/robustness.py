from pandas import DataFrame
from safeai.lib.experimentation.base import SafeAIMetric


class Robustness(SafeAIMetric):
    """_summary_: A metric to evaluate the robustness of a model"""

    def pertub(self) -> DataFrame:
        """_summary_: Pertub the data"""
        return list(
            self.experiment_job.data.select_dtypes(include="number").mean(axis=1).values
        )

    def compute_single_variable_rgr(self) -> DataFrame:
        """_summary_: Computes the RGR for a single variable"""
        return list(
            self.experiment_job.data.select_dtypes(include="number").mean(axis=1).values
        )

    def compute_full_single_rgr(self) -> DataFrame:
        """_summary_: Computes the Full Single RGR"""
        return list(
            self.experiment_job.data.select_dtypes(include="number").mean(axis=1).values
        )
