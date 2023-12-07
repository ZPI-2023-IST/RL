from abc import ABC, abstractmethod
from collections import namedtuple

from rl.algorithms.Config import Config, ParameterType

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

    @abstractmethod
    def get_model(self) -> object:
        pass

    @abstractmethod
    def set_params(self, params) -> None:
        pass

    @classmethod
    def get_configurable_parameters(cls) -> dict:
        return {
            "timeout_steps": Parameter(
                ParameterType.INT.name,
                100,
                1,
                None,
                "Number of steps after which the game will be terminated",
                True,
            ),
            "timeout_penalty": Parameter(
                ParameterType.FLOAT.name,
                2,
                0,
                None,
                "Penalty for timeout",
                True,
            ),
        }

    def config_model(self, config: dict) -> None:
        self.config = Config.from_dict(config)

    def restart(self) -> None:
        """
        Restart the model (this is not the same as again initialization)
        """
        pass

    def update_config(self, config: dict) -> None:
        self.config.update(config)
