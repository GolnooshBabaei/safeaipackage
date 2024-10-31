import streamlit as st
import pandas as pd

from src.main import Experimentation
from src.safeai.enums import ExperimentDataType
from src.safeai.lib.jobs.tabular import TabularJob
from src.safeai.lib.config.experiments import TabularExperiment
from frc.forms import experimentation_form, tabular_config_form


source_data = None


st.header("SafeAI Demo")
with st.sidebar:
    st.title("SafeAI Experimentation Configuration")

    with st.form(key="experimentation_config_form"):
        experimentation_form()
        _experiment_config_button = st.form_submit_button(
            "Submit",
            type="primary",
            use_container_width=True,
            icon=":material/upload_2:",
        )

        if _experiment_config_button:
            st.session_state["submitted_experiment_config"] = True

configure, source_tab, results_tab = st.tabs(
    ["Configure Experiment", "Source", "Results"]
)

if st.session_state.get("submitted_experiment_config", False):
    if st.session_state.get("experiment_type") == ExperimentDataType.TABULAR:
        source_data = pd.read_csv(st.session_state["source"])

        with configure:
            with st.form("tabular_data_form"):
                tabular_config_form(source_data=source_data)
                _tabular_config_button = st.form_submit_button(
                    "Submit",
                    type="primary",
                    use_container_width=True,
                    icon=":material/upload_2:",
                )
            if _tabular_config_button:
                st.session_state["submitted_tabular_config"] = True

            if st.session_state.get("submitted_tabular_config", False):
                experiment = TabularExperiment(
                    experiment_job=TabularJob(**st.session_state).model_dump(),
                    experiment_name=st.session_state["experiment_name"],
                    experiment_iterations=st.session_state["experiment_iterations"],
                    experiment_type=st.session_state["experiment_type"],
                    experiment_test_size=st.session_state["experiment_test_size"],
                    experiment_llm=st.session_state["experiment_llm"],
                )
                with source_tab:
                    st.write(experiment.experiment_job.cleaned)

                with results_tab:
                    experimentation = Experimentation(experiment=experiment)

                    with st.spinner("Running Experiment"):
                        results = experimentation.start_experiment()
                    if results:
                        st.write(experimentation.get_experiment_outputs(results))

    if st.session_state.get("experiment_type") == ExperimentDataType.IMAGE:
        configure.empty()
        source_tab.empty()
        results_tab.empty()
        with configure:
            st.write("This feature is not yet implemented")
        with source_tab:
            st.write("Please upload an image data source")
        with results_tab:
            st.write("This feature is not yet implemented")

    if st.session_state.get("experiment_type") == ExperimentDataType.TEXT:
        configure.empty()
        source_tab.empty()
        results_tab.empty()
        with configure:
            st.write("This feature is not yet implemented")
        with source_tab:
            st.write("Please upload an image data source")
        with results_tab:
            st.write("This feature is not yet implemented")
