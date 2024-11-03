from pandas import DataFrame
from pydantic import computed_field
from safeai.lib.experimentation.base import SafeAIMetric


class Fairness(SafeAIMetric):
    """_summary_: A metric to evaluate the fairness of a model"""

    @computed_field
    @property
    def compute_rga_parity(self) -> DataFrame:
        """_summary_: Computes the RGA Parity"""
        self.rga
        return list(
            self.experiment_job.data.select_dtypes(include="number").mean(axis=1).values
        )
