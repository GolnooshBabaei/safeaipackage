from datetime import date, timedelta
from pydantic_settings import BaseSettings
from safeai.agents.experimentation_agent.enums import EvaluationMetric
from safeai.enums import (
    AgentTask,
    Country,
    AgentDomain,
    AgentModel
)
from safeai.utils.base import Base, Field

class Env(BaseSettings):
    """_summary_"""
    openai_api_key: str
    palm_api_key: str
    anthropic_api_key: str
    llama_api_key: str
    
class CompanyProfile(Base):
    """_summary_"""
    company_name: str = Field(
        default="Safe AI Inc",
        title="Company Name",
        description="The Name of the Company"
    )
    company_description: str = Field(
        default="We offer clean energy solutions",
        title="Company Description",
        description="A Brief Description of the Company"
    )
    company_website: str | None = Field(
        default="https://safeai.com",
        title="Company Website",
        description="The Company's Website"
    )
    company_country: list[Country] = Field(
        default=Country.UNITED_KINGDOM,
        title="Company Country",
        description="The Country where the Company is Located"
    )
    company_business_sector: list[AgentDomain] = Field(
        default=AgentDomain.ENERGY,
        title="Company Business Sector",
        description="The Sector of the Company"
    )
    agent_model: AgentModel = Field(
        default=AgentModel.GPT4,
        title="LLM Model",
        description="The Language Model to Evaluate"
    )
    agent_domain_start: date = Field(
        default=date.today()-timedelta(days=365),
        title="Agent Context Start",
        description="The Start Date of the Agent Context"
    )
    agent_domain_end: date = Field(
        date.today(),
        title="Agent Context End",
        description="The End Date of the Agent Context"
    )
    agent_evaluation_task: AgentTask = Field(
        default=AgentTask.DATA_GENERATION,
        title="Agent Task",
        description="The Task of the Agent"
    )
    exp_metrics: list[EvaluationMetric] = Field(
        default=EvaluationMetric.default(),
        title="Evaluation Metrics",
        description="The Metrics to Evaluate the Agent"
    )
    exp_trials: int = Field(
        1000,
        title="Number of Trials",
        description="The Number of Trials to Run"
    )


class DataProfile(Base):
    """_summary_"""
    data_sample_description: str = Field(
        default="This is a sample data description",
        title="Data Sample Description",
        description="A Description of the Data Sample"
    )
    data_num_rows: int = Field(
        default=1000,
        title="Number of Rows",
        description="The Number of Rows in the Data Sample"
    )
    