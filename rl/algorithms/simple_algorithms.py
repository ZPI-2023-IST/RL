import random

from rl.algorithms import Algorithm, algorithm_manager
from rl.algorithms import ParameterType


@algorithm_manager.register_algorithm("random")
class RandomAlgorithm(Algorithm):
    def __init__(self, logger) -> None:
        super().__init__(logger)

    def forward(self, state, actions, reward: float) -> int:
        random.seed(self.config.seed)
        return random.choice(actions)

    @classmethod
    def _get_train_params(cls) -> dict:
        return {"seed": (ParameterType.INT.name, None, None, None)}

    @classmethod
    def _get_test_params(cls) -> dict:
        return {"seed": (ParameterType.INT.name, None, None, None)}
