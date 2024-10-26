#!/usr/bin/env python
from src.safeai.models import SafeAIJobConfig
from src.safeai.job import SafeAIExperimentJob


def run():
    """
    Run the crew.
    """
    config = SafeAIJobConfig(
        source="https://raw.githubusercontent.com/MainakRepositor/Datasets/refs/heads/master/HR%20Data/train.csv",
        target="department",
    )

    SafeAIExperimentJob(safeai_config=config).job().kickoff()


if __name__ == "__main__":
    run()
