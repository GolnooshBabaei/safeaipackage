from typing import TypeVar
from pydantic import BaseModel, computed_field, Field
import numpy as np
from pandas import DataFrame, merge

from safeai.base import SafeAIJob
from safeai.lib.jobs.tabular import TabularJob
from safeai.lib.jobs.text import TextJob
from safeai.lib.jobs.image import ImageJob


T_Job = TypeVar("T_Job", bound=SafeAIJob)


class SafeAIMetric(BaseModel):
    """_summary_: A metric for the SafeAI Experiment"""

    experiment_job: TabularJob | TextJob | ImageJob = Field(
        ..., description="The job for the experiment", exclude=True
    )

    class Config:
        arbitrary_types_allowed = True

    @computed_field
    @property
    def rga(self) -> DataFrame:
        """_summary_: Returns the RGA"""
        predictions = self.experiment_job.predictions_train.copy()
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
        inc, dec = np.sum(predictions.index * _Y), np.sum(predictions.index * _Y[::-1])
        return (np.sum(predictions.index * predictions.support.values) - dec) / (
            inc - dec
        )
