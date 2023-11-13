from abc import ABC, abstractmethod

from rl.algorithms.Config import Config


class Algorithm(ABC):
    def __init__(self, logger) -> None:
        self.logger = logger
        self.config = None
        
        config = {
            k: v[1] for k, v in self.get_configurable_parameters().items()
        }
        self.config_model(config)

    @abstractmethod
    def forward(self, state: list, actions: list, reward: float) -> int:
        """
        This method is called to perform one iteration of model
        """
        pass

    @classmethod
    @abstractmethod
    def get_configurable_parameters(cls) -> dict:
        return {}

    def config_model(self, config: dict) -> None:
        self.config = Config.from_dict(config)
