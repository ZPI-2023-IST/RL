import random

from rl.algorithms import Algorithm, algorithm_manager, Parameter
from rl.algorithms import ParameterType


@algorithm_manager.register_algorithm("random")
class RandomAlgorithm(Algorithm):
    def __init__(self, logger) -> None:
        super().__init__(logger)

    def forward(self, state, actions, reward: float) -> int:
        random.seed(self.config.seed)
        return random.choice(actions)

    @classmethod
    def get_configurable_parameters(cls) -> dict:
        return {"seed": Parameter(ParameterType.INT.name, None, 0, 1000, "Random seed")}
