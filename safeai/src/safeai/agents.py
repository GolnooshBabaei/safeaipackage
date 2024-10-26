from safeai.models import (
    SafeAIJobConfig,
    SafeAIAgent,
)


class SafeAIAgents:
    """_summary_"""

    def __init__(self, safeai_config: SafeAIJobConfig) -> None:
        self.safeai_config = safeai_config

    def data_downloader_agent(self) -> SafeAIAgent:
        """Validates rows in the dataset are unique"""
        return SafeAIAgent(
            job_id=f"job_{self.safeai_config.id}",
            role="Download and Validate Data from a URL",
            goal=f"""
                Validate, extract target column and count missing
                values in data at URL {self.safeai_config.source}
            """,
            backstory=f"""
                You are a seasoned data scientist with a keen eye for detail.
                You have downloaded the data from the source {self.safeai_config.source}
                and stored it in the pandas dataframe {self.safeai_config.data}. You will
                refer to this dataframe as 'safeai_dataset'. 
                A dataframe is a 2-dimensional labeled data structure with rows and columns 
                of potentially different types. Each row in a dataframe is a unique record. 
                A row has a value for each column in the dataframe. A column in a dataframe
                is a series of values. A column has a name and a data type. The columns in
                a dataframe are called features.
                
                Your dataframe 'safeai_dataset' has {self.safeai_config.data.shape[0]}
                rows and {self.safeai_config.data.shape[1]} columns. The columns in 'safeai_dataset'
                are {self.safeai_config.data.columns}.
                
                You are now tasked with validating that each row in a dataframe is unique,
                extracting the target column, {self.safeai_config.target} and counting the
                number of missing values in the dataframe.
            """,
            # llm=somellm,
            #
        )

    def data_preprocessor_agent(self) -> SafeAIAgent:
        """Preprocesses the data"""
        raise NotImplementedError("This method is not implemented")

    def data_sampler_agent(self) -> SafeAIAgent:
        """Splits the data into training and testing sets"""
        raise NotImplementedError("This method is not implemented")

    def data_classifier_agent(self) -> SafeAIAgent:
        """Classifies the data"""
        raise NotImplementedError("This method is not implemented")

    def responsible_ai_agent(self) -> SafeAIAgent:
        """Responsible for the entire process"""
        raise NotImplementedError("This method is not implemented")

