from pandas import DataFrame

from safeai.enums import Country
from safeai.agents.generator_agent.enums import ColumnDataType

import streamlit as st


def get_data_descriptor_form():
    """_summary_"""
    form = st.form(key="get_data_descriptor_form")
    form.text_area(
        "Describe the kind of Data you would like to generate",
        placeholder="Toxic tweets from elected officials\n",
        key="data_sample_description",
    )
    form.slider(
        "Number of Samples to generate",
        min_value=1,
        max_value=1000,
        key="data_num_rows",
        value=100,
    )
    form.multiselect(
        "Samples Should be drawn from [Optional]",
        Country.to_list(),
        key="data_sample_country",
    )
    form.slider(
        "Percentage of Missing Values",
        min_value=0,
        max_value=100,
        key="percentage_missing_values",
    )
    return form


def get_data_config_editor():
    """_summary_"""
    data_config_frame = DataFrame(
        [
            {
                "col_name": "tweet",
                "col_type": ColumnDataType.TEXT,
                "col_description": "A toxic tweet from an elected official",
            },
            {
                "col_name": "timestamp",
                "col_type": ColumnDataType.TIMESTAMP,
                "col_description": "The time the tweet was posted",
            },
            {
                "col_name": "username",
                "col_type": ColumnDataType.TEXT,
                "col_description": "The username of the elected official",
            },
        ]
    )

    return st.data_editor(
        data_config_frame,
        column_config=dict(
            col_name=st.column_config.TextColumn(width="small"),
            col_type=st.column_config.SelectboxColumn(
                options=ColumnDataType.to_list(),
                default=ColumnDataType.TEXT,
                width="small",
            ),
            col_description=st.column_config.TextColumn(width="large"),
        ),
        key="data_config_editor",
        num_rows="dynamic",
        use_container_width=True,
    )
