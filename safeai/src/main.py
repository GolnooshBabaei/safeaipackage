#!/usr/bin/env python
from concurrent.futures import ThreadPoolExecutor
from typing import Generic, Iterator
from crewai.crew import CrewOutput, TaskOutput
from safeai.base import T_Experiment


class Experimentation(Generic[T_Experiment]):
    def __init__(self, experiment: T_Experiment) -> None:
        self.experiment = experiment

    def start_experiment(self) -> Iterator[CrewOutput]:
        with ThreadPoolExecutor() as iterator:
            results = iterator.map(
                self.experiment.job.kickoff,
                [
                    {
                        "iteration": iteration,
                    }
                    for iteration in range(self.experiment.experiment_iterations)
                ],
            )
            iterator.shutdown(wait=True)
        return results

    def get_experiment_outputs(self, results: Iterator[CrewOutput]) -> list[TaskOutput]:
        """_summary_: Returns a list of all task outputs"""
        results = [[r.json_dict for r in result.tasks_output] for result in results]
        return [
            {
                "job_id": self.experiment.job.job_id,
                "crew_id": str(self.experiment.job.id),
                "experiment_id": self.experiment.experiment_id,
                "experiment_name": self.experiment.experiment_name,
                "iteration": index + 1,
                "tasks": [
                    {
                        "agent_id": str(t.agent.id),
                        "task_id": str(t.id),
                        "task_result": t.output.json_dict,
                    }
                    for t in self.experiment.job.tasks
                ],
            }
            for index, result in enumerate(results)
        ]
