import os
from uuid import uuid4

from abc import abstractmethod
from functools import cached_property

from pydantic import Field

from safeai.base import SafeAICrew
from safeai.lib.config.tasks import SafeAITasks


class SafeAIExperiment(SafeAITasks):
    """_summary_: Base Class for Creating Experiments"""

    name: str = Field(
        default="A Name for the experiment",
        description="The name of the experiment"
    )

    @cached_property
    def experiment_id(self) -> str:
        """_summary_: Create an UUID4 ID for new Experiments"""
        return f"exp_{str(uuid4())}"
    
    @cached_property
    def agents(self) -> list[str]:
        """_summary_: Returns list of all agents in this experiment"""
        return [str(agent.id) for agent in self.job.agents]

    @cached_property
    def tasks(self) -> list[str]:
        """_summary_: Returns list of all tasks in this experiment"""
        return [str(task.id) for task in self.job.tasks]

    @cached_property
    def artefact_path(self) -> str:
        """_summary_: Creates a path for storing artefacts from this experiment"""
        # TODO: Use Pathlib
        _experiment_path = f"artefacts/experiments/{self.experiment_id}"
        if not os.path.exists(_experiment_path):
            os.makedirs(_experiment_path)

        _jobs_path = f"{_experiment_path}/jobs/{self.job.job_id}"
        if not os.path.exists(_jobs_path):
            os.makedirs(_jobs_path)

        return _jobs_path

    def replay(self, task_id: str) -> None:
        """_summary_: Replay a task"""
        self.job.replay(task_id=task_id)

    @cached_property
    @abstractmethod
    def job(self) -> SafeAICrew:
        """_summary_: Create a new SafeAI Experiment Job"""
        raise NotImplementedError("This method is not implemented")
