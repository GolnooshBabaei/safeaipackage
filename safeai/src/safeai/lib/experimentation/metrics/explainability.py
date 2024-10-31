from pandas import DataFrame
from pydantic import Field, computed_field
from safeai.lib.experimentation.base import SafeAIMetric


class Explainability(SafeAIMetric):
    """_summary_: A metric to evaluate the explainability of a model"""

    """variables: list[str] = Field(
        default=None,
        description="The variables to compute the explainability for"
    )"""

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
        return (
            self.experiment_job.data.select_dtypes(include="number").mean(axis=1).values
        )

    @computed_field
    @property
    def compute_single_variable_pdp(self) -> DataFrame:
        """_summary_: Computes the PDP for a single variable"""
        return list(
            self.experiment_job.data.select_dtypes(include="number").mean(axis=1).values
        )
