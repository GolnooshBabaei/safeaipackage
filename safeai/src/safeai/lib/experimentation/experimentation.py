#!/usr/bin/env python
from concurrent.futures import ThreadPoolExecutor
from typing import Generic, Iterator
from crewai.crew import CrewOutput, TaskOutput
from safeai.base import T_Experiment


class Experimentation(Generic[T_Experiment]):
    """_summary_: A class to handle the experimentation of a model"""

    def __init__(self, experiment: T_Experiment) -> None:
        self.experiment = experiment

    def start_experiment(self) -> Iterator[CrewOutput]:
        """_summary_: Starts the experiment"""
        with ThreadPoolExecutor() as iterator:
            results = iterator.map(
                self.experiment.job.kickoff,
                [
                    {"iteration": iteration}
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
                "metrics": self.experiment.metrics,
            }
            for index, result in enumerate(results)
        ]
