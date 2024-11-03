from functools import cached_property
from typing import Self, Union

from numpy import array, ndarray
from pydantic import Field, model_validator, computed_field
from pandas import DataFrame, Series, read_csv, get_dummies

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor


from safeai.enums import ModelClassifier, PredictionType, ModelRegressor
from safeai.base import SafeAIJob


class TabularJob(SafeAIJob):
    """_summary_: Preprocessing Class for Tabular Data

    Args:
        SafeAIJob (_type_): _description_

    Raises:
        NotImplementedError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: TabularJob
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
    sep: str = Field(default=",", description="The delimeter of the dataset")
    delimeter: str | None = Field(
        default=None, description="The delimeter of the dataset"
    )
    header: int | None = Field(default=None, description="The header of the dataset")
    model: ModelClassifier | ModelRegressor = Field(
        default=ModelClassifier.CATBOOSTCLASSIFIER,
        description="The classifier to use for the classification task",
    )
    prediction_type: PredictionType = Field(
        default=PredictionType.CLASSIFICATION,
        description="The prediction type for the classification task"
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

        _label_encoder = LabelEncoder()

        
        if self.protected_variables:
            for variable in self.protected_variables:
                if _data[variable].dtype.name.startswith("object"):
                    _data[variable] = _label_encoder.fit_transform(_data[variable])

        if self.encodes:
            for variable in self.encodes:
                if _data[variable].dtype.name.startswith("object"):
                    if _data[variable].dropna().nunique() == 2:
                        _data[variable] = _label_encoder.fit_transform(_data[variable])
                    else:
                        _data = get_dummies(_data, columns=[variable])
                        
        for variable in _data.columns:
            if variable != self.target:
                if _data[variable].dtype.name.startswith("object"):
                    _data.drop(columns=[variable], inplace=True)
            
        

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
                
        if self.protected_variables and self.encodes:
            if any(
                col in self.protected_variables for col in self.encodes
            ):
                raise ValueError(f"""
                    Some protected columns are also encoded. 
                    {set(self.protected_variables) & set(self.encodes)} are also encoded.
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
        return self.clf().fit(self.xtrain, self.ytrain)

    @cached_property
    def predictions_train(self) -> DataFrame:
        """_summary_: Predicts the probability of the training data"""
        return DataFrame({
            "y": self.ytrain,
            "yhat": self.fit.predict(self.xtrain),
            "proba": self.predict_proba(self.xtrain)
        })

    @cached_property
    def predictions_test(self) -> DataFrame:
        """_summary_: Predicts the probability of the testing data"""
        return DataFrame({
            "y": self.ytest,
            "yhat": self.fit.predict(self.xtest),
            "proba": self.predict_proba(self.xtest)
        })
    
        
    def predict_proba(self, x:DataFrame) -> ndarray:
        """_summary_: Predicts the probability of the training data"""
        try:
            return self.fit.predict_proba(x).max(axis=1)
        except Exception as e:
            print(f"Error: {str(e)}")
            return self.fit.predict(x)

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
        if self.model == ModelRegressor.LOGISTICREGRESSION:
            return LogisticRegression(random_state=42, penalty="l2", solver="liblinear")
        if self.model == ModelClassifier.RANDOMFORESTCLASSIFIER:
            return RandomForestClassifier(random_state=42, criterion="gini")
        if self.model == ModelClassifier.CATBOOSTCLASSIFIER:
            return CatBoostClassifier(
                iterations=100, learning_rate=0.1, depth=2, loss_function="Logloss"
            )
        if self.model == ModelRegressor.CATBOOSTREGRESSOR:
            return CatBoostRegressor(
                iterations=100, learning_rate=0.1, depth=2, loss_function="RMSE"
            )
        if self.model == ModelClassifier.XGBCLASSIFIER:
            return XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=2,
                objective="binary:logistic",
                enable_categorical=True,
            )
        if self.model == ModelRegressor.XGBREGRESSOR:
            return XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=2,
                objective="reg:squarederror",
            )
