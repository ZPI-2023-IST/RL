import numpy as np

from rl.algorithms import Algorithm, algorithm_manager


class Trainer:
    pass


class LearningAlgorithm(Algorithm):
    def __init__(self, logger) -> None:
        super().__init__(logger)
        self.trainer = Trainer()


@algorithm_manager.registered_algorithm("dqn")
class DQN(LearningAlgorithm):
    def __init__(self, logger) -> None:
        super().__init__(logger)

    def make_action(self, state: list, actions: list[list]) -> list:
        return [1, 2, 3]

    @staticmethod
    def get_configurable_parameters() -> dict:
        return {"train": {"lr": "float"}, "test": {}}
