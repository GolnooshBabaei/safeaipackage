from abc import abstractmethod
from datetime import date, datetime
from typing import Union

from safeai.enums import Country
from safeai.utils.base import (
    st,
    Base,
    DeltaGenerator,
    Field,
    model_validator,
    computed_field
)
from safeai.agents.generator_agent.enums import (
    ColumnDataType,
    TimeFrequency,
    TextType,
    Emotion
)


class ColumnDescriptorFactory(Base):
    """_summary_"""
    data_num_rows: int = Field(
        100,
        title="Number of Samples",
        description="The Number of Data Samples to Provide"
    )
    data_sample_country: list[Country] = Field(
        [Country.GLOBAL],
        title="Sample Country",
        description="The Country where the Sample Data is Located  "
    )
    data_sample_description: str | None = Field(
        None,
        title="Sample Description",
        description="A Description of the Sample Data"
    )
    
    @classmethod
    @abstractmethod
    def form(cls, c_name: str) -> DeltaGenerator:
        """_summary_"""
        raise NotImplementedError("Method form not implemented")
    
    
    @model_validator(mode="before")
    @classmethod
    @abstractmethod
    def validate_data(cls, data:dict) -> dict:
        """_summary_"""
        raise NotImplementedError("Method validate_data not implemented")


class DateDataDescription(ColumnDescriptorFactory):
    """_summary_"""

    start_date: date | datetime = Field(
        default_factory=date.today,
        title="Minimum Value",
        description="The Minimum Value in the Column"
    )
    end_date: date | datetime = Field(
        default_factory=date.today,
        title="Maximum Value",
        description="The Maximum Value in the Column"
    )
    frequency: TimeFrequency = Field(
        default="minutes",
        title="Frequency",
        description="The Frequency of the Date Column"
    )

    @model_validator(mode="before")
    @classmethod
    def validate_data(cls, data:dict) -> dict:
        """_summary_"""
        if data["start_date"] > data["end_date"]:
            raise ValueError("Minimum Date must be less than Maximum Date")
        return data

    @classmethod
    def form(cls, c_name: str) -> DeltaGenerator:
        form_key = f"description_{c_name}"
        form = st.form(key=form_key)
        form.date_input(
            "Minimum Value",
            key="start_date",
            value=date.today()
        )
        form.date_input(
            "Maximum Value",
            key="end_date",
            value=date.today()
        )
        form.selectbox(
            "Frequency",
            TimeFrequency.to_list(),
            key="frequency",
            index=0
        )
        return form


class NumericDataDescription(ColumnDescriptorFactory):
    """__summary__"""
    min: float = Field(
        0.0,
        title="Minimum Value",
        description="The Minimum Value in the Column"
    )
    max: float = Field(
        100.0,
        title="Maximum Value",
        description="The Maximum Value in the Column"
    )
    mean: float = Field(
        50.0,
        title="Mean Value",
        description="The Mean Value in the Column"
    )
    median: float = Field(
        50.0,
        title="Median Value",
        description="The Median Value in the Column"
    )
    iqr1: float = Field(
        25.0,
        title="First Quartile",
        description="The First Quartile of the Column"
    )
    iqr3: float = Field(
        75.0,
        title="Third Quartile",
        description="The Third Quartile of the Column"
    )
    std_dev: float = Field(
        1.0,
        title="Standard Deviation",
        description="The Standard Deviation of the Column"
    )
    skewness: float = Field(
        0.0,
        title="Skewness",
        description="The Skewness of the Column"
    )
    kurtosis: float = Field(
        0.0,
        title="Kurtosis",
        description="The Kurtosis of the Column"
    )

    @model_validator(mode="before")
    @classmethod
    def validate_data(cls, data:dict) -> dict:
        if data["min"] > data["max"]:
            raise ValueError("Minimum Value must be less than Maximum Value")
        if data["iqr1"] > data["iqr3"]:
            raise ValueError("First Quartile must be less than Third Quartile")
        if data["mean"] > data["max"] or data["mean"] < data["min"]:
            raise ValueError("Mean Value must be between Minimum and Maximum Value")
        return data

    @classmethod
    def form(cls, c_name: str) -> DeltaGenerator:
        form_key = f"description_{c_name}"
        form = st.form(key=form_key)
        for k, v in cls.model_fields.items():
            form.number_input(
                v.title or k,
                key=k,
                value=v.default
            )
        return form        


class CategoricalDataDescription(ColumnDescriptorFactory):
    """__summary__"""

    unique_values: list[str] = Field(
        ...,
        title="Unique Values",
        description="The Unique Values in the Column"
    )
    value_coverage: dict[str, int] = Field(
        ...,
        title="Value Counts",
        description="'%' of Values in the Column"
    )

    @model_validator(mode="before")
    @classmethod
    def validate_data(cls, data:dict) -> dict:
        if data["unique_values"]:
            data["unique_values"] = data["unique_values"].split(",")
        if data["value_coverage"]:
            data["value_coverage"] = dict(
                [i.split(":") for i in data["value_coverage"].split(",")]
            )
        return data

    @classmethod
    def form(cls, c_name: str) -> DeltaGenerator:
        form_key = f"description_{c_name}"
        form = st.form(key=form_key)
        for k, v in cls.model_fields.items():
            form.text_input(
                v.title or k,
                key=k,
                value=v.default,
                placeholder="Value1:30,Value2:50,Value3:20"
            )
        return form


class TextDataDescription(ColumnDescriptorFactory):
    """For Text Generation"""

    text_type: TextType = Field(
        TextType.TWEET,
        title="Text Type",
        description="The Type of the Text Domain"
    )
    min_length: int | None = Field(
        None,
        title="Minimum Length",
        description="The Minimum Length of the Text Domain"
    )
    max_length: int | None = Field(
        None,
        title="Maximum Length",
        description="The Maximum Length of the Text Domain"
    )
    emotions: list[Emotion] | None = Field(
        None,
        title="Emotions",
        description="The Emotions of the Text Domain"
    )

    @model_validator(mode="before")
    @classmethod
    def validate_data(cls, data:dict) -> dict:
        return data

    @classmethod
    def form(cls, c_name: str) -> DeltaGenerator:
        form_key = f"description_{c_name}"
        form = st.form(key=form_key)
        tt = form.selectbox(
            "Text Type",
            TextType.to_list(),
            key="text_type",
            index=0
        )
        if tt and any([i in tt.lower() for i in  ['review', 'news', 'blog', 'article']]):
            form.slider(
                "Minimum Length",
                min_value=1,
                max_value=1000,
                key="min_length",
                value=None
            )
            form.slider(
                "Maximum Length",
                min_value=1,
                max_value=1000,
                key="max_length",
                value=None
            )
        form.multiselect(
            "Emotions",
            Emotion.to_list(),
            key="emotions",
            default=None
        )
        return form


S = Union[
        DateDataDescription,
        NumericDataDescription,
        CategoricalDataDescription,
        TextDataDescription
    ]

class DescriptorModel(Base):
    """_summary_"""
    col_name: str
    col_desc: str
    model: type[S]
    state:dict
    
    @computed_field
    @property
    def form(self) -> DeltaGenerator:
        """_summary_"""
        return self.model.form(self.col_name)

    @computed_field
    @property
    def data(self) -> S:
        """_summary_"""
        return self.state.get(f"{self.col_name}_data_desc")



class ColumnDescriptorStrategy:
    """_summary_"""
    def __init__(self, c_name:str, c_desc:str, c_type:ColumnDataType,state:dict) -> None:
        self.c_name = c_name
        self.c_desc = c_desc
        self.c_type = c_type
        self.state = state

    def __call__(self, *args, **kwds) -> DescriptorModel:
        if (self.c_type == ColumnDataType.DATE
                or self.c_type == ColumnDataType.TIMESTAMP):
            return DescriptorModel(
                col_name=self.c_name,
                col_desc=self.c_desc,
                model=DateDataDescription,
                state=self.state
            )
        
        elif (self.c_type == ColumnDataType.CATEGORICAL or
                self.c_type == ColumnDataType.ORDINAL or
                    self.c_type == ColumnDataType.BOOLEAN):
            return DescriptorModel(
                col_name=self.c_name,
                col_desc=self.c_desc,
                model=CategoricalDataDescription,
                state=self.state
            )
        
        elif (self.c_type == ColumnDataType.INTEGER
                or self.c_type == ColumnDataType.DECIMAL):
            return DescriptorModel(
                col_name=self.c_name,
                col_desc=self.c_desc,
                model=NumericDataDescription,
                state=self.state
            )
        
        else:
            return DescriptorModel(
                col_name=self.c_name,
                col_desc=self.c_desc,
                model=TextDataDescription,
                state=self.state
            )

    
#https://gist.github.com/CHerSun/156b2aec324903a70738fd7858126584
