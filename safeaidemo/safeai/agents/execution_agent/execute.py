from random import choice


def make_randon_float():
    return choice([i for i in range(100)]) / 100


def test_run_experiment():
    return {
        "accuracy": make_randon_float(),
        "f1": make_randon_float(),
        "precision": make_randon_float(),
        "recall": make_randon_float(),
    }
