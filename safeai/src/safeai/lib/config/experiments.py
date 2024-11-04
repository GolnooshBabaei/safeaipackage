from abc import abstractmethod
from functools import cached_property
from pydantic import Field

from crewai import Process
from safeai.base import SafeAICrew, SafeAIExperiment
from safeai.lib.config.tasks import SafeAITasks
from safeai.lib.jobs.tabular import TabularJob
from safeai.lib.experimentation.metrics.explainability import Explainability
from safeai.lib.experimentation.metrics.fairness import Fairness
from safeai.lib.experimentation.metrics.robustness import Robustness


class TabularExperiment(SafeAIExperiment):
    """_summary_: An experiment for tabular data"""

    # TODO: Define correct types and generics
    experiment_job: TabularJob = Field(
        ..., description="The job for the experiment", exclude=True
    )

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
                self.tasks.data_unique_validator_task(),
                self.tasks.data_column_text_describer_task(),
            ],
            process=Process.sequential,
            verbose=True,
            memory=True,
            full_output=True,
        )

    @cached_property
    def explainability(self):
        """_summary_: Returns the explainability of the experiment"""
        return Explainability(experiment_job=self.experiment_job)

    @cached_property
    def fairness(self):
        """_summary_: Returns the fairness of the experiment"""
        return Fairness(experiment_job=self.experiment_job)

    @cached_property
    def robustness(self):
        """_summary_: Returns the robustness of the experiment"""
        return Robustness(experiment_job=self.experiment_job)

    @cached_property
    def metrics(self):
        """_summary_: Returns the metrics of the experiment"""
        return {
            "rga": self.fairness.rga,
            "fairness": self.fairness.compute_rga_parity,
            "robustness": self.robustness.compute_rgr,
            "explainability": self.explainability.compute_rge,
        }


class TextExperiment(SafeAIExperiment):
    """_summary_: An experiment for text data"""

    @abstractmethod
    def tasks(self) -> SafeAITasks:
        """_summary_: Create a new SafeAI Task Configuration"""
        raise NotImplementedError("Text Experiment Not Implemented")

    @cached_property
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
