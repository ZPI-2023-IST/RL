from abc import ABC, abstractmethod

from rl.algorithms.Config import Config


class Algorithm(ABC):
    def __init__(self, logger) -> None:
        self.logger = logger
        self.config = None

    @abstractmethod
    def store_memory(self, state: list, reward: float) -> None:
        """
        This method is called when we want to save things in memory.
        It should store all the necessary things for for future training.
        """
        raise NotImplementedError

    @abstractmethod
    def make_action(self, state: list, actions: list[list]) -> list:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _get_train_params(cls) -> dict:
        """
        Train params should be returned as a dict of tuples,
        where the first element is the type of the parameter and
        the second element is the default value, third element is the
        minimum value and the fourth element is the maximum value.
        Key is the name of the parameter.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _get_test_params(cls) -> dict:
        """
        Test params should be returned as a dict of tuples,
        where the first element is the type of the parameter and
        the second element is the default value, third element is the
        minimum value and the fourth element is the maximum value.
        Key is the name of the parameter.
        """
        raise NotImplementedError

    @classmethod
    def get_configurable_parameters(cls) -> dict:
        return {"train": cls._get_train_params(), "test": cls._get_test_params()}

    def config_model(self, config: dict) -> None:
        self.config = Config.from_dict(config)
