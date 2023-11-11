import torch

from rl.algorithms import Algorithm, algorithm_manager
from rl.algorithms import ParameterType
from rl.algorithms.modules.SimpleNet import SimpleNet


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
        self.model = SimpleNet([10, 20, 30])

    def make_action(self, state: list, actions: list[list]) -> list:
        t = torch.randn(10)
        t = self.model(t)
        return t.tolist()

    def store_reward(self, reward: float) -> None:
        pass

    @classmethod
    def _get_train_params(cls) -> dict:
        return {"lr": (ParameterType.FLOAT.name, 1e-5, 0, 1)}

    @classmethod
    def _get_test_params(cls) -> dict:
        return {}
