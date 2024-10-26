from crewai import Process
from safeai.models import (
    SafeAIJobConfig,
    SafeAIAgent,
    SafeAITask,
    SafeAICrew,
    TaskOutputHandler,
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


class SafeAITasks(SafeAIAgents):
    """_summary_"""

    def data_validator_task(self) -> SafeAITask:
        """Creates the Safeai crew"""
        handler = TaskOutputHandler(
            job_id=self.safeai_config.id,
            name="DataValidatorOutput",
            properties=[("is_unique", bool), ("total_duplicate_rows", int)],
            output_file_name="unique.json",
        )
        return SafeAITask(
            job_id=f"job_{self.safeai_config.id}",
            description="""
                A dataframe that has all unique rows, has no 2 or more rows
                with the same values in corresponding columns.
                
                You are responsible for validating that each row in 'safeai_dataset'
                is unique. Count the number of rows in 'safeai_dataset' that are have
                duplicates.
            """,
            expected_output="""
                A json object two keys: 'is_unique', 'total_duplicate_rows'.
                The value for 'is_unique' should be a boolean. If every row
                in 'safeai_dataset' is unique, the value should be True.
                Otherwise, the value should be False.
                
                The value for 'total_duplicate_rows' should be an integer.
                The integer should be the number of rows in the dataframe that 
                are duplicates. If there are no duplicate rows in 'safeai_dataset',
                the value should be 0.
            """,
            agent=self.data_downloader_agent(),
            output_json=handler.output_format_factory,
            output_file=handler.output_path,
        )

    def data_target_extract_task(self) -> SafeAITask:
        """Creates the Safeai crew"""
        handler = TaskOutputHandler(
            job_id=self.safeai_config.id,
            name="DataFormatterOutput",
            properties=[("y_true", list[str])],
            output_file_name="ytrue.json",
        )
        return SafeAITask(
            job_id=f"job_{self.safeai_config.id}",
            description=f"""
                    For each row in the pandas dataframe 'safeai_dataset', the value in the target 
                    column {self.safeai_config.target} should be extracted and stored in a
                    list named 'y_true'.
                    
                    The list 'y_true' should contain the value of the target column for each row in
                    'safeai_dataset'. You must extract the value of the target column for every row in
                    'safeai_dataset'. The total number of items in the list 'y_true' should be equal 
                    to the number of rows in 'safeai_dataset'.
            """,
            expected_output="""
                    A json object with one key: 'y_true'. The value for 'y_true' should be a list.
                    The list should contain the value of the target column for each row in 'safeai_dataset'.
            """,
            agent=self.data_downloader_agent(),
            output_json=handler.output_format_factory,
            output_file=handler.output_path,
        )

    def data_nan_counter_task(self) -> SafeAITask:
        """Creates the Safeai crew"""
        handler = TaskOutputHandler(
            job_id=self.safeai_config.id,
            name="DataMissingValuesOutput",
            properties=[("nans", list[dict[str, list[str]]])],
            output_file_name="missing.json",
        )
        return SafeAITask(
            job_id=f"job_{self.safeai_config.id}",
            description="""
                    On each row in the pandas dataframe 'safeai_dataset', there might be missing values.
                    Missing values are either '', Nan or nan. You are tasked with counting the
                    number of missing values in each row.
                    
                    For each row in the pandas dataframe 'safeai_dataset', extract the row index
                    and the column names of the columns with missing values for that row. 
                    
                    The value for 'row_index' should be an integer that represents the index of that row.
                    The value for 'nan_columns' should a list of column names with missing values for that row.
                    
                    
                    For each row, extract these values as a json object with 2 keys: 'row_index'
                    and 'nan_columns'. You must extract these values for every row in 'safeai_dataset'. 
                    
                    If there are no missing values for a row, the value for 'nan_columns' for that row should
                    be None.
                    
                    Store the json object for all rows in a list called 'nans'. The total number 
                    of json objects in 'nans' should be equal to the number of rows in 'safeai_dataset'.
            """,
            expected_output="""
                    A json object with one key called 'nans' whose value is the 'nans' list.
                    Each json object in 'nans' list corresponds to one row in 'safeai_dataset'. 
                    
                    Each json object in the 'nans' list should have 2 keys. Each key should have a value. 
                    The first key is 'row_index'. The value for 'row_index' is an integer that
                    represents the index of that row in 'safeai_dataset'.
                    
                    The second key is 'nan_columns'. The value for 'nan_columns' is a list of the column names
                    of columns with missing values for that row. If there are no columns with missing values 
                    in that row, the value for the second key should be None.
            """,
            agent=self.data_downloader_agent(),
            output_json=handler.output_format_factory,
            output_file=handler.output_path,
        )


class SafeAIExperimentJob(SafeAITasks):
    """_summary_"""

    def job(self) -> SafeAICrew:
        """Creates the Safeai crew"""
        return SafeAICrew(
            job_id=f"job_{self.safeai_config.id}",
            agents=[self.data_downloader_agent()],
            tasks=[
                self.data_validator_task(),
                self.data_target_extract_task(),
                self.data_nan_counter_task(),
            ],
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )

    def replay(self, task_id: str) -> None:
        """_summary_"""
        self.job().replay(task_id=task_id)
