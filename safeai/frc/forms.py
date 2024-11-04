import streamlit as st
import pandas as pd

from safeai.enums import ModelClassifier, ExperimentDataType, ModelRegressor, SafeAILLMS, PredictionType


data: pd.DataFrame | None = None


def experimentation_form():
    st.text_input(
        "Experiment Name",
        value="Titanic Experiment",
        placeholder="Give your experiment a name",
        key="experiment_name",
    )
    st.selectbox(
        "Experiment Type",
        options=ExperimentDataType.to_list(),
        key="experiment_type",
        index=0,
    )
    st.text_input(
        "Source Data URL",
        value="https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv",
        placeholder="https://source_of_data.csv",
        key="source",
    )
    st.slider(
        "Experiment Iterations",
        min_value=0,
        max_value=100,
        value=10,
        step=1,
        key="experiment_iterations",
    )
    st.slider(
        "Experiment Test Size",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
        key="experiment_test_size",
    )

    prediction_type = st.selectbox(
        "Select the Prediction Type",
        options=[i.value for i in PredictionType],
        key="prediction_type",
        index=0,
    )
    
    if prediction_type == PredictionType.CLASSIFICATION:
        st.selectbox(
            "Select the Classifier",
            options=[i.value for i in ModelClassifier if "classifier" in i.value.lower()],
            key="model",
            index=0,
        )
    else:
        st.selectbox(
            "Select the Regressor",
            options=[i.value for i in ModelRegressor if "regress" in i.value.lower()],
            key="model",
            index=0,
        )
    st.multiselect(
        "Select the LLM",
        options=[i.value for i in SafeAILLMS],
        key="experiment_llm",
        default=[SafeAILLMS.GPT2.value],
    )

def tabular_config_form(source_data: pd.DataFrame):
    _columns = set(source_data.columns.tolist())
    sep = st.text_input("File Separator", value=",", placeholder=",", key="sep")
    st.text_area("Column Names", value="", placeholder="", key="new_cols")
    st.checkbox("First Row as Header?", value=True, key="header")

    drops = st.multiselect(
        "Select columns to drop", options=list(_columns), key="drops", default=None
    )
    
    st.multiselect(
        "Select columns to encode", options=list(_columns), key="encodes", default=None
    )
    if drops:
        _columns -= set(drops)
    
    st.multiselect(
        "Select columns to protect",
        options=list(_columns),
        key="protected_variables",
        default=None,
    )
    st.slider(
        "Perturbation",
        min_value=0.0,
        max_value=0.5,
        value=0.01,
        step=0.01,
        key="perturbation"
    )

    target = st.selectbox("Target Column", options=_columns, key="target")

    if target:
        _columns.remove(target)
    
    st.checkbox("Balance the target column", key="balance_target", value=False)
    st.checkbox("Impute missing data", key="impute_missing_data", value=False)


def image_config_form():
    st.text_input(
        "Enter the source of the image",
        value="https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png",
        placeholder="https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png",
        key="source",
    )


def text_config_form():
    st.text_area(
        "Enter the text", value="Hello World", placeholder="Hello World", key="source"
    )
