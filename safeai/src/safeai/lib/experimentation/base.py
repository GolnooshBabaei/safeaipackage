from typing import TypeVar
from pydantic import BaseModel, computed_field, Field
from pandas import DataFrame

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
        ## self.model.fit(self.x_train, self.y_train)
        return self.experiment_job.data.select_dtypes(include="number").mean(axis=1)
