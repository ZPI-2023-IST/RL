from rl.algorithms import Algorithm, algorithm_manager
from rl.algorithms import ParameterType


@algorithm_manager.register_algorithm("random")
class RandomAlgorithm(Algorithm):
    def __init__(self, logger) -> None:
        super().__init__(logger)

    def make_action(self, state: list, actions: list[list]) -> list:
        return [1]

    def store_reward(self, reward: float) -> float:
        return reward

    @classmethod
    def _get_train_params(cls) -> dict:
        return {"abc": (ParameterType.STRING.name, "aaa", None, None)}

    @classmethod
    def _get_test_params(cls) -> dict:
        return {}
