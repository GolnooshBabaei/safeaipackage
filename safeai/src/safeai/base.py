from uuid import uuid4
from datetime import datetime
from abc import abstractmethod

from functools import cached_property
from typing import Any, Self, TypeVar
from pydantic import BaseModel, computed_field, Field, model_validator

from requests import head
from pandas import DataFrame

from sklearn.model_selection import train_test_split
from crewai import Agent, Crew, Task

from safeai.enums import ExperimentDataType, SafeAILLMS


def output_object_factory(properties: list[tuple[str, type]]) -> type[BaseModel] | None:
    """_summary_: Dynamically Create Pydantic Models for Output"""

    class NewModel(BaseModel):
        """_summary_: Dynamically Create Pydantic Models for Output"""

        __annotations__ = {prop[0]: prop[1] for prop in properties}

    return NewModel


class SafeAICrew(Crew):
    """_summary_"""

    artefact_path: str = Field(default=".", description="The path to the artefacts")

    @computed_field
    @property
    def job_id(self) -> str:
        """_summary_"""
        return f"job_{str(self.id)}"


class SafeAIAgent(Agent):
    """_summary_"""

    @computed_field
    @property
    def agent_id(self) -> str:
        """_summary_"""
        return f"agent_{str(self.id)}"


class SafeAITask(Task):
    """_summary_"""

    job_id: str = Field(
        ...,
        description="The job id",
    )
    agent_id: str = Field(
        ...,
        description="The agent id",
    )
    output_name: str | None = Field(
        default=None,
        description="The name of the output file",
    )
    output_properties: list[tuple[str, type]] | None = Field(
        default=None,
        description="The properties of the output file",
    )

    @computed_field
    @property
    def task_id(self) -> str:
        """_summary_"""
        return f"task_{str(self.id)}"

    @computed_field
    @property
    def created_at(self) -> datetime:
        """_summary_"""
        return datetime.now()


class SafeAIJob(BaseModel):
    """_summary_

    Model executes steps we need to control and sends output to the crew

    """

    @cached_property
    def job_id(self) -> str:
        """_summary_: Create an UUID4 ID for new Jobs"""
        return f"job_{str(uuid4())}"

    source: str = Field(..., description="URL to the dataset")
    test_size: float = Field(default=0.2, description="The size of the test dataset")
    target: str = Field(
        default=None, description="The target column for the classification task"
    )

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        """_summary_"""
        if not self.source.startswith("http"):
            raise ValueError("The source must be a URL")
        try:
            response = head(self.source, allow_redirects=True, timeout=10)
            if response.status_code != 200:
                raise ValueError(
                    f"An error occurred while downloading the data: {response.status_code}"
                )
        except Exception as e:
            raise ValueError("An error occurred while downloading the data") from e
        return self

    @cached_property
    def train_test_data(self) -> list[DataFrame]:
        """_summary_: Splits @self.data into training and testing sets"""
        return train_test_split(self.data, test_size=self.test_size)

    @cached_property
    @abstractmethod
    def read_source(self) -> Any:
        """_summary_: Reads Data from @self.source"""
        raise NotImplementedError("This method is not implemented")

    @cached_property
    @abstractmethod
    def data(self) -> DataFrame:
        """_summary_: Returns the Pandas DataFrame with data for experiment"""
        raise NotImplementedError("This method is not implemented")

    @cached_property
    def x_train(self) -> DataFrame:
        """_summary_: Returns the training data"""
        raise NotImplementedError("This method is not implemented")

    @cached_property
    def x_test(self) -> DataFrame:
        """_summary_: Returns the testing data"""
        raise NotImplementedError("This method is not implemented")

    @cached_property
    def y_train(self) -> DataFrame:
        """_summary_: Returns the training target"""
        raise NotImplementedError("This method is not implemented")

    @cached_property
    def y_test(self) -> DataFrame:
        """_summary_: Returns the testing target"""
        raise NotImplementedError("This method is not implemented")


class SafeAIExperiment(BaseModel):
    """_summary_: Class for Creating Experiments"""

    @computed_field
    @property
    def experiment_id(self) -> str:
        """_summary_: Create an UUID4 ID for new Experiments"""
        return f"exp_{str(uuid4())}"

    experiment_job: SafeAIJob

    experiment_name: str = Field(
        default="A Name for the experiment", description="The name of the experiment"
    )

    experiment_iterations: int = Field(
        default=10, description="The number of iterations for the experiment"
    )

    experiment_llm: SafeAILLMS = Field(
        default=SafeAILLMS.GPT2, description="The LLM for the experiment"
    )

    experiment_type: ExperimentDataType = Field(
        default=ExperimentDataType.TABULAR,
        description="The type of data for the experiment",
    )

    @computed_field
    @property
    def experiment_start(self) -> datetime:
        """_summary_: The start time of the experiment"""
        return datetime.now()

    @computed_field
    @property
    def experiment_status(self) -> str:
        """_summary_: The status of the experiment"""
        return "Running"

    @computed_field
    @property
    def experiment_end(self) -> datetime | None:
        """_summary_: The end time of the experiment"""
        if self.experiment_status == "Running":
            return None
        return datetime.now()

    @abstractmethod
    def metrics(self) -> Any:
        """_summary_: Returns the metric for the experiment"""
        raise NotImplementedError("This method is not implemented")


T_Experiment = TypeVar("T_Experiment", bound=SafeAIExperiment)
