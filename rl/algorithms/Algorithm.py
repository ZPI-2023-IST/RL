from abc import ABC, abstractmethod
from collections import namedtuple

from rl.algorithms.Config import Config

Parameter = namedtuple(
    "Parameter", ("type", "default", "min", "max", "help", "modifiable")
)


class Algorithm(ABC):
    def __init__(self, logger) -> None:
        self.logger = logger
        self.config = None

    @abstractmethod
    def forward(self, state: list, actions: list, reward: float) -> int:
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
    def set_params(self, params) -> None:
        pass

    def config_model(self, config: dict) -> None:
        self.config = Config.from_dict(config)

    def restart(self) -> None:
        """
        Restart the model (this is not the same as again initialization)
        """
        pass
    
    def update_config(self, config: dict) -> None:
        self.config.update(config)
