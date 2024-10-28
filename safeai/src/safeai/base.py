from abc import abstractmethod
from functools import cached_property
import os
from datetime import datetime
from pathlib import Path

from typing import Any, Self
from pydantic import BaseModel, computed_field, Field, model_validator

from pandas import DataFrame
from requests import head

from sklearn.model_selection import train_test_split
from crewai import Agent, Crew, Task

os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

base = Path(__file__).parent.stem


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

    name: str = Field(default="SafeAIJobConfig", description="The name of the job")
    source: str = Field(..., description="URL to the dataset")
    test_size: float = Field(default=0.2, description="The size of the test dataset")
    target: str = Field(
        default=None, description="The target column for the classification task"
    )

    job: SafeAICrew = Field(
        default_factory=lambda: SafeAICrew(
            agents=[
                SafeAIAgent(
                    role="Greet World",
                    goal="Say Hello World",
                    backstory="Alien lands on Earth",
                )
            ],
            tasks=[
                SafeAITask(
                    job_id="job_1",
                    agent_id="agent_1",
                    expected_output="Hello World",
                    description="Hello World",
                    agent=SafeAIAgent(
                        role="Greet World",
                        goal="Say Hello World",
                        backstory="Alien lands on Earth",
                    ),
                )
            ],
            verbose=True,
        ),
        description="The job id",
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
    def train_test_data(self) -> list:
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
