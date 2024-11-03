from functools import cached_property
from typing import Any, Callable, Self, Union

from pydantic import Field, model_validator, computed_field
from pandas import DataFrame, Series, read_csv, get_dummies, concat

from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor


from safeai.enums import ModelClassifier
from safeai.base import SafeAIJob


class TabularJob(SafeAIJob):
    """_summary_

    Model executes steps we need to control and sends output to the crew

    """

    new_cols: str = Field(default=None, description="Columns in the dataset")
    drops: list[str] | None = Field(
        default=None, description="Columns to drop from the dataset"
    )
    protected_variables: list[str] | None = Field(
        default=None, description="Columns to protect from the dataset"
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
    header: int | None = Field(default=None, description="The header of the dataset")
    classifier: ModelClassifier = Field(
        default=ModelClassifier.LOGISTICREGRESSION,
        description="The classifier to use for the classification task",
    )
    balance_target: bool = Field(
        default=False, description="Whether to balance the target column"
    )
    impute_missing_data: bool = Field(
        default=False, description="Whether to impute missing data"
    )

    @computed_field
    @property
    def column_names(self) -> list[str] | None:
        """_summary_: Returns the column names"""
        return self.new_cols.split(",") if self.new_cols else None

    @cached_property
    def read_source(self) -> DataFrame:
        """_summary_: Reads the source"""
        _data = read_csv(str(self.source), sep=self.sep)
        if self.column_names:
            _data.columns = self.column_names
        return _data

    @cached_property
    def cleaned(self) -> DataFrame:
        """_summary_: Cleans and encodes @self.read_source"""
        _data = self.read_source
        if self.drops:
            _data = _data.drop(columns=self.drops, errors="ignore", axis=1)

        if self.keeps:
            _data = _data[self.keeps]

        if self.encodes:
            _data = get_dummies(_data, columns=self.encodes)

        if self.balance_target:
            # TODO: Implement balancing of target column
            pass

        return _data

    @cached_property
    def data(self) -> DataFrame:
        """_summary_: Imputes missing data in @self.cleaned"""
        if self.impute_missing_data:
            # TODO: Implement imputing of missing data or Group means
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
    def y(self) -> Series:
        """_summary_: Returns the target column"""
        return self.data[self.target]

    @cached_property
    def x(self) -> DataFrame:
        """_summary_: Returns the feature columns"""
        return self.data.drop(columns=[self.target])

    @cached_property
    def train_test_data(self) -> list[DataFrame]:
        """_summary_: Splits @self.data into training and testing sets"""
        return train_test_split(
            self.x, self.y, test_size=self.test_size, random_state=42
        )

    @cached_property
    def x_train(self) -> DataFrame:
        """_summary_: Returns the training data"""
        return self.train_test_data[0]

    @cached_property
    def x_test(self) -> DataFrame:
        """_summary_: Returns the testing data"""
        return self.train_test_data[1]

    @cached_property
    def y_train(self) -> DataFrame:
        """_summary_: Returns the training target"""
        return self.train_test_data[2]

    @cached_property
    def y_test(self) -> DataFrame:
        """_summary_: Returns the testing target"""
        return self.train_test_data[3]

    @model_validator(mode="after")
    def validate_job_configuration(self) -> Self:
        """_summary_"""

        if self.column_names and len(self.column_names) != len(
            self.read_source.columns.to_list()
        ):
            raise ValueError(f"""
                The number of column names {len(self.column_names)} does not match 
                the number of columns in the dataset: {len(self.read_source.columns)}\n.
                {len(self.column_names)} != {len(self.read_source.columns)}\n.
            """)

        if self.target not in self.read_source.columns.to_list():
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

        if self.protected_variables:
            if any(
                col not in self.read_source.columns for col in self.protected_variables
            ):
                raise ValueError(f"""
                    Some protected columns not found in the dataset. 
                    {set(self.protected_variables) - set(self.read_source.columns)} not found in dataset.
                """)

        if self.protected_variables and self.target in self.protected_variables:
            raise ValueError(f"The target {self.target} column cannot be protected.")

        if self.protected_variables:
            if any(
                list(
                    map(
                        lambda variable: self.x[variable].nunique() > 2,
                        self.protected_variables,
                    )
                )
            ):
                raise ValueError("""
                    The protected variables must be categorical and have more than 2 unique values.
                """)

        # TODO: Validate all columns to encode are categorical and with varied values
        return self

    @cached_property
    def fit(
        self,
    ) -> Union[
        LogisticRegression,
        RandomForestClassifier,
        CatBoostClassifier,
        CatBoostRegressor,
        XGBClassifier,
        XGBRegressor,
    ]:
        """_summary_: Fits the model to the data"""
        return self.clf().fit(self.x_train, self.y_train)

    @cached_property
    def predictions_train(self) -> DataFrame:
        """_summary_: Predicts the probability of the training data"""
        return DataFrame({"y": self.y_train, "yhat": self.fit.predict(self.x_train)})

    @cached_property
    def predictions_test(self) -> DataFrame:
        """_summary_: Predicts the probability of the testing data"""
        return DataFrame({"y": self.y_test, "yhat": self.fit.predict(self.x_test)})

    def clf(
        self,
    ) -> Union[
        LogisticRegression,
        RandomForestClassifier,
        CatBoostClassifier,
        CatBoostRegressor,
        XGBClassifier,
        XGBRegressor,
    ]:
        """_summary_: Returns the model to use for the classification task"""
        if self.classifier == ModelClassifier.LOGISTICREGRESSION:
            return LogisticRegression(random_state=42, penalty="l2", solver="liblinear")
        if self.classifier == ModelClassifier.RANDOMFORESTCLASSIFIER:
            return RandomForestClassifier(random_state=42, criterion="gini")
        if self.classifier == ModelClassifier.CATBOOSTCLASSIFIER:
            return CatBoostClassifier(
                iterations=100, learning_rate=0.1, depth=2, loss_function="Logloss"
            )
        if self.classifier == ModelClassifier.CATBOOSTREGRESSOR:
            return CatBoostRegressor(
                iterations=100, learning_rate=0.1, depth=2, loss_function="RMSE"
            )
        if self.classifier == ModelClassifier.XGBCLASSIFIER:
            return XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=2,
                objective="binary:logistic",
                enable_categorical=True,
            )
        if self.classifier == ModelClassifier.XGBREGRESSOR:
            return XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=2,
                objective="reg:squarederror",
            )
