from abc import abstractmethod
from functools import cached_property

from crewai import Process
from safeai.base import SafeAICrew, SafeAIExperiment
from safeai.lib.config.tasks import SafeAITasks
from safeai.lib.jobs.tabular import TabularJob


class TabularExperiment(SafeAIExperiment):
    """_summary_: An experiment for tabular data"""

    experiment_job: TabularJob

    @cached_property
    def tasks(self) -> SafeAITasks:
        """_summary_: Create a new SafeAI Task Configuration"""
        return SafeAITasks(self.experiment_job)

    @cached_property
    def job(self) -> SafeAICrew:
        """_summary_: Create a new SafeAI Experiment Job"""
        return SafeAICrew(
            agents=[
                self.tasks.experiment_configuration_agent(),
                self.tasks.data_downloader_agent(),
            ],
            tasks=[
                self.tasks.experiment_logger_task(),
                self.tasks.data_validator_task(),
            ],
            process=Process.sequential,
            verbose=True,
            memory=True,
            full_output=True,
        )


class TextExperiment(SafeAIExperiment):
    """_summary_: An experiment for text data"""

    @abstractmethod
    def tasks(self) -> SafeAITasks:
        """_summary_: Create a new SafeAI Task Configuration"""
        raise NotImplementedError("Text Experiment Not Implemented")

    @abstractmethod
    def job(self) -> SafeAICrew:
        """_summary_: Create a new SafeAI Experiment Job"""
        raise NotImplementedError("Text Experiment Not Implemented")


class ImageExperiment(SafeAIExperiment):
    """_summary_: An experiment for image data"""

    @abstractmethod
    def tasks(self) -> SafeAITasks:
        """_summary_: Create a new SafeAI Task Configuration"""
        raise NotImplementedError("Text Experiment Not Implemented")

    @abstractmethod
    def job(self) -> SafeAICrew:
        """_summary_: Create a new SafeAI Experiment Job"""
        raise NotImplementedError("Text Experiment Not Implemented")
