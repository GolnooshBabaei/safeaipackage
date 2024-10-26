from safeai.models import (
    SafeAITask,
    TaskOutputHandler
)
from safeai.agents import SafeAIAgents


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