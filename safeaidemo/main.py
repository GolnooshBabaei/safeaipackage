import pandas as pd
from safeai.agents.generator_agent.enums import ColumnDataType
from safeai.enums import FileExtension, AgentTask
from safeai.agents.generator_agent.models.columns import ColumnDescriptorStrategy
from safeai.lib.forms import st, company_profile, safeai_agent, experiment
from safeai.agents.generator_agent.forms import (
    get_data_descriptor_form,
    get_data_config_editor,
)
from safeai.lib.models import Chat
from safeai.models import CompanyProfile
from safeai.agents.generator_agent.generator import SyntheticTabularGenerator

# https://generativeai.pub/an-in-depth-introduction-to-the-atomic-agents-multi-agent-ai-framework-b621f14df454


def dialog(c_name: str, c_desc: str, c_type: ColumnDataType):
    """_summary_"""

    @st.dialog(title=c_name.capitalize())
    def _dialog():
        descriptor = ColumnDescriptorStrategy(
            c_name, c_desc, c_type, st.session_state.to_dict()
        )()
        form = descriptor.form
        s1 = form.form_submit_button("Submit", type="primary", use_container_width=True)
        if s1:
            setattr(
                st.session_state,
                f"{c_name}_data_desc",
                descriptor.model(**st.session_state.to_dict()),
            )

    _dialog()


def save_company_profile():
    """_summary_"""
    st.session_state.company_profile = CompanyProfile(**st.session_state.to_dict())


with st.sidebar:
    with st.form(key="company_config"):
        with st.expander("Profile"):
            company_profile()

        with st.expander("Task"):
            agent_task = st.radio(
                "", AgentTask.to_list(), key="agent_evaluation_task", index=None
            )

        with st.expander("Agent"):
            safeai_agent()

        with st.expander("Experiment"):
            experiment()

        submit = st.form_submit_button(
            "Submit",
            type="primary",
            use_container_width=True,
            on_click=save_company_profile,
        )


st.title("Safe AI Demo")


if not st.session_state.agent_evaluation_task:
    st.image("media/bg.png", use_column_width=True)


if st.session_state.agent_evaluation_task == AgentTask.DATA_GENERATION:
    tabular, image, audio, video = st.tabs(["Tabular", "Image", "Audio", "Video"])
    with tabular:
        st.subheader("Describe your Data")
        sample_form = get_data_descriptor_form()
        submit_sample_form = sample_form.form_submit_button(
            "Submit", use_container_width=True
        )

        if submit_sample_form:
            st.session_state["_configure_data_columns"] = True

        st.write("---")
        if st.session_state.get("_configure_data_columns"):
            st.subheader("Data Columns")

            data_config_editor = get_data_config_editor()
            configure_submit = st.button(
                "Configure Data", key="configure_data_fields", use_container_width=True
            )
            if configure_submit:
                st.session_state["_configure_fields"] = True

            if st.session_state.get("_configure_fields"):
                st.subheader("Data Fields")
                if rows := list(data_config_editor.iterrows()):
                    for i in range(0, len(rows), 3):
                        if _this := list(rows[i : i + 3]):
                            for c, j in enumerate(st.columns(len(_this))):
                                j.button(
                                    f"Configure {_this[c][1][0]}",
                                    key=f"{_this[c][1][0]}_submit",
                                    on_click=dialog,
                                    args=(
                                        _this[c][1][0],
                                        _this[c][1][2],
                                        _this[c][1][1],
                                    ),
                                )
                            continue
                st.write("---")
                finetune_button = st.button(
                    "Fine Tune Data Fields",
                    key="fine_tune_data_fields",
                    use_container_width=True,
                )
                if finetune_button:
                    st.session_state["_fine_tune_fields"] = True

                st.write("---")
                if st.session_state.get("_fine_tune_fields"):
                    st.subheader("Upload Data")
                    uploaded_file = st.file_uploader(
                        "Choose a file",
                        type=FileExtension.to_list(),
                        key="uploaded_file",
                    )

            with st.spinner("Generating Data..."):
                generate_synthetic_data = st.button(
                    "Generate Synthetic Data",
                    key="generate_synthetic_data",
                    use_container_width=True,
                )
                if generate_synthetic_data:
                    sy = SyntheticTabularGenerator(
                        dataframe=data_config_editor, state=st.session_state.to_dict()
                    )
                    response = sy.generate()
                    st.write(Chat(role="assistant", content=response.content))
            st.write("Success")

if st.session_state.agent_evaluation_task == AgentTask.PREDICTION:
    company_data_file = st.file_uploader(
        "Choose a file", type=FileExtension.to_list(), key="company_data_file"
    )
    if company_data_file is not None:
        dataframe = pd.read_csv(company_data_file)
