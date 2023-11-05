import numpy as np

from rl.algorithms import Algorithm, algorithm_manager


@algorithm_manager.registered_algorithm("random")
class RandomAlgorithm(Algorithm):
    def __init__(self, logger) -> None:
        super().__init__(logger)

    def make_action(self, state: list, actions: list[list]) -> list:
        return [1]

    @staticmethod
    def get_configurable_parameters() -> dict:
        return {"train": {}, "test": {}}
