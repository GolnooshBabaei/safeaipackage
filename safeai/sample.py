#!/usr/bin/env python
from src.safeai.models import SafeAIJobConfig
from src.safeai.crew import SafeCrew

def run():
    """
    Run the crew.
    """
    config = SafeAIJobConfig(
        source = "https://raw.githubusercontent.com/MainakRepositor/Datasets/refs/heads/master/HR%20Data/train.csv",
        target = "department"
    )
    
    SafeCrew(safeai_config=config).run()


if __name__ == "__main__":
    run()
