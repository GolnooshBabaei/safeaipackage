import os
from uuid import uuid4
from functools import cached_property
from enum import Enum
from requests import head
from typing import Self
from pydantic import BaseModel, computed_field, Field, model_validator
from pandas import DataFrame, read_csv, get_dummies, concat
from crewai import Agent, Crew, Task

os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"


class ClassifierType(str, Enum):
    RANDOM_FOREST = "RandomForest"
    XGBOOST = "XGBoost"
    LogisticRegression = "LogisticRegression"





class SafeAIAgent(Agent):
    """_summary_"""
    job_id: str
    @computed_field
    @property
    def agent_id(self) -> str:
        return f"agent_{str(self.id)}"

class SafeAITask(Task):
    """_summary_"""
    job_id: str
    @computed_field
    @property
    def task_id(self) -> str:
        return f"task_{str(self.id)}"
    
    
class SafeAICrew(Crew):
    """_summary_"""
    job_id: str
    @computed_field
    @property
    def crew_id(self) -> str:
        return f"crew_{str(self.id)}"


class SafeAIJobConfig(BaseModel):
    """_summary_
    
        Model executes steps we need to control and sends output to the crew
    
    """
    source: str = Field(
        ...,
        description="URL to the dataset"
    )
    target: str = Field(
        ...,
        description="The target column for the classification task"
    )
    drops: list[str] | None = Field(
        default=None,
        description="Columns to drop from the dataset"
    )
    encodes: list[str] | None = Field(
        default=None,
        description="Columns to encode from the dataset"
    )
    keeps: list[str] | None = Field(
        default=None,
        description="Columns to keep from the dataset"
    )
    test_size: float = Field(
        default=0.2,
        description="The size of the test dataset"
    )
    sep: str = Field(
        default=',',
        description="The delimeter of the dataset"
    )
    delimeter: str | None = Field(
        default=None,
        description="The delimeter of the dataset"
    )
    header: int = Field(
        default=0,
        description="The header of the dataset"
    )
    classifier: ClassifierType = Field(
        default=ClassifierType.LogisticRegression,
        description="The classifier to use for the classification task"
    )
    
    @model_validator(mode="after")
    def validate_source(self) -> Self:
        """_summary_"""
        if not self.source.startswith("http"):
            raise ValueError("The source must be a URL")
        response = head(self.source, allow_redirects=True, timeout=10)
        if response.status_code != 200:
            raise ValueError(f"An error occurred while downloading the data: {response.status_code}")
        return self
        

    @cached_property
    def raw_data(self) -> DataFrame:
        return read_csv(
            str(self.source),
            sep=self.sep,
            delimiter=self.delimeter,
            header=self.header
        )
        
    @cached_property
    def encoded(self) -> DataFrame:
        return get_dummies(self.raw_data, columns=self.encodes)
    
    @cached_property
    def data(self) -> DataFrame:
        _data = self.raw_data
        if self.keeps:
            _data = _data[self.keeps]
        if self.drops:
            _data = _data.drop(columns=self.drops, errors='ignore', axis=1)
        if self.encodes:
            _dummies = get_dummies(_data, columns=self.encodes)
            _data = _data.drop(columns=self.encodes, errors='ignore', axis=1)
            _data = concat([_data, _dummies], axis=1)
        return _data
    
    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """_summary_"""
        if self.target not in self.raw_data.columns:
            raise ValueError(f"Target column {self.target} not found in the dataset")
        if self.keeps:
            if any(col not in self.raw_data.columns for col in self.keeps):
                raise ValueError(f"Column not found in the dataset")
        if self.drops:
            if any(col not in self.raw_data.columns for col in self.drops):
                raise ValueError(f"Column not found in the dataset")
        if self.encodes:
            if any(col not in self.raw_data.columns for col in self.encodes):
                raise ValueError(f"Column not found in the dataset")
        return self

    @computed_field
    @property
    def id(self) -> str:
        return str(uuid4())
    
class ValidatorOutput(BaseModel):
    """_summary_"""
    is_unique: bool
    y_true: list[str]
    accuracy: float = Field(
        default=0.0,
        description="The accuracy of the model"
    )
    explainability: float = Field(
        default=0.0,
        description="The explainability of the model"
    )
    