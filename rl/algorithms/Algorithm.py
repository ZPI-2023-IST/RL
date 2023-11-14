from abc import ABC, abstractmethod
from collections import namedtuple

from rl.algorithms.Config import Config

Parameter = namedtuple("Parameter", ("type", "default", "min", "max", "help"))


class Algorithm(ABC):
    def __init__(self, logger) -> None:
        self.logger = logger
        self.config = None

        config = {k: v[1] for k, v in self.get_configurable_parameters().items()}
        self.config_model(config)

    @abstractmethod
    def forward(self, state: list, actions: list, reward: float, game_status: str) -> int:
        """
        This method is called to perform one iteration of model
        """
        pass

    @classmethod
    @abstractmethod
    def get_configurable_parameters(cls) -> dict:
        pass

    @abstractmethod
    def get_model(self) -> object:
        pass

    @abstractmethod
    def set_params(self, params) -> object:
        pass

    def config_model(self, config: dict) -> None:
        self.config = Config.from_dict(config)

    def restart(self) -> None:
        """
        Restart the model (this is not the same as again initialization)
        """
        pass
