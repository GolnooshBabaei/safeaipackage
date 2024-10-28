#!/usr/bin/env python
from safeai.lib.jobs.models.tabular import SafeAITabularJob
from safeai.lib.experiments.experiments import Experimentation


tabular_experiments: list[SafeAITabularJob] = [
    SafeAITabularJob(
        name="HR Data",
        source="https://raw.githubusercontent.com/MainakRepositor/Datasets/refs/heads/master/HR%20Data/train.csv",
        target="department",
        drops=["employee_id"],
        encodes=[
            "region",
            "education",
            "recruitment_channel",
        ],
    ),
    SafeAITabularJob(
        name="Titanic",
        source="https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv",
        target="Survived",
        drops=["PassengerId", "Name", "Ticket", "Cabin"],
        encodes=[
            "Embarked",
        ],
    ),
    SafeAITabularJob(
        name="Iris",
        source="https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv",
        target="species",
    ),
]

if __name__ == "__main__":
    experimentation = Experimentation(tabular_experiments)
    experimentation.start()
