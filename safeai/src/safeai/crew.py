from crewai import Process
from safeai.models import (
    SafeAIJobConfig,
    SafeAIAgent,
    SafeAITask,
    SafeAICrew,
    ValidatorOutput
)

class SafeCrew():
    def __init__(self, safeai_config:SafeAIJobConfig) -> None:
        super().__init__()
        self.safeai_config = safeai_config
    
    def data_downloader(self) -> SafeAIAgent:
        """Validates rows in the dataset are unique"""
        return SafeAIAgent(
            role = "Download CSV Data from a URL",
            goal= f"Download, validate and format the data {self.safeai_config.data}",
            backstory=f"""
                You are a seasoned data scientist with a keen eye for detail.
                You are responsible for downloading, validating and formatting the data from the source {self.safeai_config.source}.
                The dataset is expected to have unique rows and columns.
                The dataset is also expected to have a target column: {self.safeai_config.target}.
            """,
            #llm=somellm,
            #
        )
        
    def download_validator_task(self) -> SafeAITask:
        """Creates the Safeai crew"""
        return SafeAITask(
			description=f"Validate each row in this pandas dataframe: {self.safeai_config.data} is unique",
			expected_output="A Json object with a boolean value indicating if the data is unique or not",
			agent=self.data_downloader(),
   			output_json=ValidatorOutput,
      		output_file="experiment_results.json"
		)
        
    def data_formatter_task(self) -> SafeAITask:
        """Creates the Safeai crew"""
        return SafeAITask(
      		description = f"For each row in the pandas dataframe {self.safeai_config.data},\
            				the value in the target column {self.safeai_config.target} should be extracted\
                    		and stored in a list named 'y_true'",
            expected_output=f"A comma seperated list named 'y_true' containing the values of the target column\
                			for each row in the dataframe {self.safeai_config.data}",
            agent=self.data_downloader(),
            output_json=ValidatorOutput,
            output_file="experiment_results.json"
		)
                
                
    def crew(self) -> SafeAICrew:
        """Creates the Safeai crew"""
        return SafeAICrew(
			agents=[self.data_downloader()],
			tasks=[
       			self.download_validator_task(),
          		self.data_formatter_task()
        	],
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
        
    def replay(self, task_id:str) -> None:
        self.crew().replay(task_id=task_id)
        
    def run(self) -> None:
        self.crew().kickoff()