from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, TypeVar, Generic

from crewai.crew import CrewOutput

from safeai.lib.experiments.base import SafeAIExperiment
from safeai.enums import ExperimentDataType
from safeai.base import SafeAIJob
from safeai.lib.experiments.models import (
    TabularExperiment,
    ImageExperiment,
    TextExperiment,
)


T_Job = TypeVar("T_Job", bound=SafeAIJob)
T_Experiment = TypeVar("T_Experiment", bound=SafeAIExperiment)


class Experimentation(Generic[T_Job, T_Experiment]):
    """_summary_: An experiment to predict tabular data"""

    def __init__(
        self,
        configs: list[T_Job],
        iterations: int = 3,
        experiment_type: ExperimentDataType = ExperimentDataType.TABULAR,
    ) -> None:
        self.configs = configs
        self.iterations = iterations
        self.experiment_type = experiment_type

    def _get_experiment(self, config:T_Job) -> SafeAIExperiment:
        if self.experiment_type == ExperimentDataType.TABULAR:
            return TabularExperiment(config)
        if self.experiment_type == ExperimentDataType.IMAGE:
            return ImageExperiment(config)
        if self.experiment_type == ExperimentDataType.TEXT:
            return TextExperiment(config)
        raise NotImplementedError("This experiment type is not implemented")

    def _run_experiment(self, config: T_Job) -> Iterator[CrewOutput]:
        _experiment = self._get_experiment(config)
        with ThreadPoolExecutor() as iterator:
            results = iterator.map(_experiment.job.kickoff, range(self.iterations))
            iterator.shutdown(wait=True)
        return results

    def start(self) -> Iterator[Iterator[CrewOutput]]:
        with ThreadPoolExecutor() as executor:
            results = executor.map(self._run_experiment, self.configs)
            executor.shutdown(wait=True)
        return results
