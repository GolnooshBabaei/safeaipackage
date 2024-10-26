from crewai import Process
from safeai.models import SafeAICrew
from safeai.tasks import SafeAITasks

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
