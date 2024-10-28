from functools import cached_property
from typing import Callable, Self

from pydantic import Field, model_validator
from pandas import DataFrame, read_csv, get_dummies, concat

from sklearn.model_selection import train_test_split

from safeai.enums import ModelClassifier
from safeai.base import SafeAIJob



class SafeAITabularJob(SafeAIJob):
    """_summary_

    Model executes steps we need to control and sends output to the crew

    """

    column_names: list[str] | None = Field(
        default=None, description="Columns in the dataset"
    )
    drops: list[str] | None = Field(
        default=None, description="Columns to drop from the dataset"
    )
    encodes: list[str] | None = Field(
        default=None, description="Columns to encode from the dataset"
    )
    keeps: list[str] | None = Field(
        default=None, description="Columns to keep from the dataset"
    )
    sep: str = Field(default=",", description="The delimeter of the dataset")
    delimeter: str | None = Field(
        default=None, description="The delimeter of the dataset"
    )
    header: int = Field(default=0, description="The header of the dataset")
    classifier: ModelClassifier = Field(
        default=ModelClassifier.LOGISTIC_REGRESSION,
        description="The classifier to use for the classification task",
    )
    balance_target: bool = Field(
        default=False, description="Whether to balance the target column"
    )
    impute_missing_data: bool = Field(
        default=False, description="Whether to impute missing data"
    )
    cleaner: Callable[..., DataFrame] | None = Field(
        default=None, description="The function to clean the dataset"
    )

    @cached_property
    def read_source(self) -> DataFrame:
        """_summary_: Reads the source"""
        _data = read_csv(
            str(self.source),
            sep=self.sep,
            delimiter=self.delimeter,
            header=self.header
        )
        if self.column_names:
            _data.columns = self.column_names
        return _data

    @cached_property
    def dummies(self) -> DataFrame:
        """_summary_: Encodes columns in @self.encodes"""
        # TODO: Detrmine the type of encoding to use based on the number of unique values in the column
        return get_dummies(self.read_source, columns=self.encodes)

    @cached_property
    def cleaned(self) -> DataFrame:
        """_summary_: Cleans and encodes @self.read_source"""
        _data = self.read_source

        if self.keeps:
            _data = _data[self.keeps]
        if self.drops:
            _data = _data.drop(columns=self.drops, errors="ignore", axis=1)
        if self.cleaner:
            _data = self.cleaner(_data)
        if self.balance_target:
            # TODO: Implement balancing of target column
            pass
        if self.encodes:
            _data = _data.drop(columns=self.encodes, errors="ignore", axis=1)
            _data = concat([_data, self.dummies], axis=1)
        return _data

    @cached_property
    def data(self) -> DataFrame:
        """_summary_: Imputes missing data in @self.cleaned"""
        if self.impute_missing_data:
            return self.cleaned.fillna(self.cleaned.mean())
        return self.cleaned.dropna(axis=0)

    @cached_property
    def percentage_missing(self) -> float:
        """_summary_: Calculates percentage of missing values in @self.cleaned"""
        return self.cleaned.isnull().mean().mean() * 100

    @cached_property
    def outliers(self) -> DataFrame:
        """_summary_: Identifies outliers in @self.data"""
        raise NotImplementedError("This method is not implemented")

    @cached_property
    def train_test_data(self) -> list[DataFrame]:
        """_summary_: Splits @self.data into training and testing sets"""
        return train_test_split(self.data, test_size=self.test_size, random_state=42)

    @model_validator(mode="after")
    def validate_job_configuration(self) -> Self:
        """_summary_"""

        if self.column_names and len(self.column_names) != len(
            self.read_source.columns
        ):
            raise ValueError(f"""
                The number of column names {len(self.column_names)} does not match 
                the number of columns in the dataset: {len(self.read_source.columns)}\n.
                {len(self.column_names)} != {len(self.read_source.columns)}\n.
            """)

        if self.target not in self.read_source.columns:
            raise ValueError(f"""
                The specfied target column {self.target} was not found in available
                columns: {self.read_source.columns}.
            """)

        if self.drops and self.target in self.drops:
            raise ValueError(f"The target {self.target} column cannot be dropped.")

        if self.drops:
            if any(col not in self.read_source.columns for col in self.drops):
                raise ValueError(f"""
                    Some drop columns not found in the dataset. 
                    {set(self.drops) - set(self.read_source.columns)} not found in dataset.
                """)

        if self.keeps:
            if any(col not in self.read_source.columns for col in self.keeps):
                raise ValueError(f"""
                    Some keep columns not found in the dataset. 
                    {set(self.keeps) - set(self.read_source.columns)} not found in dataset.
                """)

        if self.encodes:
            if any(col not in self.read_source.columns for col in self.encodes):
                raise ValueError(f"""
                    Some keep columns not found in the dataset. 
                    {set(self.encodes) - set(self.read_source.columns)} not found in dataset.
                """)

        # TODO: Validate all columns to encode are categorical and with varied values
        return self
    