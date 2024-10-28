from functools import cached_property

from crewai import Process
from safeai.base import SafeAICrew

from safeai.lib.experiments.base import SafeAIExperiment
from safeai.lib.jobs.models.tabular import SafeAITabularJob
from safeai.lib.jobs.models.text import SafeAITextJob
from safeai.lib.jobs.models.image import SafeAIImageJob


class TabularExperiment(SafeAIExperiment):
    """_summary_: An experiment for tabular data"""

    @cached_property
    def job(self) -> SafeAICrew:
        """_summary_: Create a new SafeAI Experiment Crew"""
        setattr(
            self.safeai_config.job,
            "agents",
            [self.data_downloader_agent()]
        )
        setattr(
            self.safeai_config.job,
            "tasks",
            [
                self.data_validator_task(),
                self.data_target_extract_task(),
                self.data_nan_counter_task(),
            ],
        )
        setattr(
            self.safeai_config.job,
            "process",
            Process.sequential
        )
        setattr(
            self.safeai_config.job,
            "artefact_path",
            self.artefact_path
        )
        return self.safeai_config.job
    
    def run_experiment(self, job: SafeAITabularJob) -> None:
        """_summary_: Run the job"""
        job.job.kickoff()


class TextExperiment(SafeAIExperiment):
    """_summary_: An experiment for text data"""

    def run_experiment(self, job: SafeAITextJob) -> None:
        """_summary_: Run the job"""
        job.job.kickoff()


class ImageExperiment(SafeAIExperiment):
    """_summary_: An experiment for image data"""

    def run_experiment(self, job: SafeAIImageJob) -> None:
        """_summary_: Run the job"""
        job.job.kickoff()
