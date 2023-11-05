import numpy as np

from rl.algorithms import Algorithm, algorithm_manager
from rl.algorithms import ParameterType

@algorithm_manager.registered_algorithm("random")
class RandomAlgorithm(Algorithm):
    def __init__(self, logger) -> None:
        super().__init__(logger)

    def make_action(self, state: list, actions: list[list]) -> list:
        return [1]
    
    def get_reward(self, reward: float) -> float:
        return reward

    @classmethod
    def _get_train_params(cls) -> dict:
        return {"abc": (ParameterType.INT.name, 0)}
    
    @classmethod
    def _get_test_params(cls) -> dict:
        return {}
