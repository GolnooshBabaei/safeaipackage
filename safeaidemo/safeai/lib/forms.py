from datetime import date, timedelta
import streamlit as st

from safeai.enums import (
    Country,
    AgentDomain,
    AgentModel
)
from safeai.agents.experimentation_agent.enums import EvaluationMetric

def company_profile():
    """_summary_"""
    st.text_input(
        "Company Name",
        placeholder="Safe AI Inc",
        key="company_name",
        value="Safe AI Inc"
    )
    st.text_area(
        "Company Description",
        placeholder="We offer clean energy solutions",
        key="company_description"
    )
    st.text_input(
        "Company Website",
        placeholder="https://safeai.com",
        key="company_website",
        value="https://safeai.com"
    )
    st.multiselect(
        "Company Country",
        Country.to_list(),
        placeholder="Select a Country",
        key="company_country",
        default=Country.default()
    )
    st.multiselect(
        "Company Business Sector",
        AgentDomain.to_list(),
        placeholder="Select a Business Sector",
        key="company_business_sector",
        default=AgentDomain.default()
    )
    


def safeai_agent():
    """_summary_"""
    st.radio(
        "",
        AgentModel.to_list(),
        key="agent_model",
        index=0
    )
    cols = st.columns(2)
    with cols[0]:
        st.date_input(
            "Context Start",
            key="agent_domain_start",
            value=date.today()
        )
    with cols[1]:
        st.date_input(
            "Context End",
            key="agent_domain_end",
            value=date.today()-timedelta(days=365)
        )
        
def experiment():
    """_summary_"""
    st.slider(
        "Number of Trials",
        min_value=1,
        max_value=1000,
        key="exp_trials",
        value=100
    )
    st.multiselect(
        "Evaluate On:",
        EvaluationMetric.to_list(),
        key="exp_metrics",
        default=EvaluationMetric.default()
    )