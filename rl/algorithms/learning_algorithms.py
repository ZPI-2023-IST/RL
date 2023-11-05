import numpy as np

from rl.algorithms import Algorithm, algorithm_manager
from rl.algorithms import ParameterType


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

    def get_reward(self, reward: float) -> float:
        return reward

    @classmethod
    def _get_train_params(cls) -> dict:
        return {"lr": (ParameterType.FLOAT.name, 0.0)}
    
    @classmethod
    def _get_test_params(cls) -> dict:
        return {}
    