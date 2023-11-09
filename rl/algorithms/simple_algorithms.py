import random

from rl.algorithms import Algorithm, algorithm_manager
from rl.algorithms import ParameterType


@algorithm_manager.registered_algorithm("random")
class RandomAlgorithm(Algorithm):
    def __init__(self, logger) -> None:
        super().__init__(logger)

    def make_action(self, state: list, actions: list[list]) -> list:
        random.seed(self.config.seed)
        return random.choice(actions)

    def store_memory(self, state: list, reward: float) -> None:
        pass

    @classmethod
    def _get_train_params(cls) -> dict:
        return {"seed": (ParameterType.INT.name, None, None, None)}

    @classmethod
    def _get_test_params(cls) -> dict:
        return {"seed": (ParameterType.INT.name, None, None, None)}
