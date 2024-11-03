from pandas import DataFrame
from pydantic import computed_field
from safeai.lib.experimentation.base import SafeAIMetric


class Explainability(SafeAIMetric):
    """_summary_: A metric to evaluate the explainability of a model"""

    @computed_field
    @property
    def compute_single_variable_rge(self) -> DataFrame:
        """_summary_: Computes the RGE for a single variable"""
        return list(
            self.experiment_job.data.select_dtypes(include="number").mean(axis=1).values
        )

    @computed_field
    @property
    def compute_group_variable_rge(self) -> DataFrame:
        """_summary_: Computes the RGE for a group of variables"""
        return list(
            self.experiment_job.data.select_dtypes(include="number").mean(axis=1).values
        )

    @computed_field
    @property
    def compute_single_variable_pdp(self) -> DataFrame:
        """_summary_: Computes the PDP for a single variable"""
        return list(
            self.experiment_job.data.select_dtypes(include="number").mean(axis=1).values
        )
