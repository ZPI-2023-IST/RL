import numpy as np
from abc import ABC, abstractmethod

from rl.algorithms.Config import Config


class Algorithm(ABC):
    def __init__(self, logger) -> None:
        self.logger = logger
        self.config = None
    
    @abstractmethod
    def get_reward(self, reward: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def make_action(self, state: list, actions: list[list]) -> list:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _get_train_params(cls) -> dict:
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def _get_test_params(cls) -> dict:
        raise NotImplementedError

    @classmethod
    def get_configurable_parameters(cls) -> dict:
        return {
            "train": cls._get_train_params(),
            "test": cls._get_test_params()
        }

    def create_config(self, config: dict) -> None:
        self.config = Config.from_dict(config)
