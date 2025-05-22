from typing import TypeVar
from pydantic import BaseModel, computed_field, Field
import numpy as np
from pandas import DataFrame, Series, merge

from safeai.base import SafeAIJob
from safeai.lib.jobs.tabular import TabularJob


T_Job = TypeVar("T_Job", bound=SafeAIJob)


class SafeAIMetric(BaseModel):
    """_summary_: A metric for the SafeAI Experiment"""

    experiment_job: TabularJob

    class Config:
        arbitrary_types_allowed = True

    @computed_field
    @property
    def rga(self) -> float:
        """_summary_: Returns the RGA"""
        return self.get_rga(
            self.experiment_job.predictions_test["y"],
            self.experiment_job.predictions_test["yhat"],
        )

    def get_rga(
        self, y: Series | np.ndarray | list, yhat: Series | np.ndarray | list
    ) -> float:
        """_summary_: Returns the RGA"""
        predictions = DataFrame({"y": y, "yhat": yhat})
        predictions["ryhat"] = predictions["yhat"].rank(method="min")

        # Merge support back to the original dataframe
        predictions = (
            merge(
                predictions,
                predictions.groupby("ryhat")["y"].mean().reset_index(name="support"),
                on="ryhat",
                how="left",
            )
            .sort_values(by="ryhat")
            .reset_index(drop=True)
            .sort_values(by="yhat")
            .reset_index(drop=True)
        )

        _Y = np.sort(predictions["y"])
        inc, dec = (
            np.sum(predictions.index * _Y),
            np.sum(predictions.index * _Y[::-1]),
        )
        return (np.sum(predictions.index * predictions.support.values) - dec) / (
            inc - dec
        )
