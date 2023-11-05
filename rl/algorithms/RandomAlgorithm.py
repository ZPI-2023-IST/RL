import numpy as np

from rl.algorithms.Algorithm import Algorithm


class RandomAlgorithm(Algorithm):
    def __init__(self, logger) -> None:
        super().__init__(logger)

    def make_action(self, state: np.ndarray, actions: list[np.ndarray]) -> np.ndarray:
        return np.random.uniform(-1, 1, 2)

    @staticmethod
    def get_configurable_parameters() -> dict:
        return {"train": {}, "test": {}}

    def create_config(self, config: dict) -> None:
        super().create_config(config)
