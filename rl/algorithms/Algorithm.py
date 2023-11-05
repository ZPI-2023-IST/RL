import numpy as np
from abc import ABC, abstractmethod

from rl.algorithms.Config import Config


class Algorithm(ABC):
    def __init__(self, logger) -> None:
        self.logger = logger
        self.config = None

    @abstractmethod
    def make_action(self, state: np.ndarray, actions: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    @staticmethod
    def get_configurable_parameters() -> dict:
        raise NotImplementedError

    def create_config(self, config: dict) -> None:
        self.config = Config.from_dict(config)
