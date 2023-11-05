from abc import ABC, abstractmethod

from rl.algorithms.Config import Config


class Algorithm(ABC):
    def __init__(self, logger) -> None:
        self.logger = logger
        self.config = None

    @abstractmethod
    def get_reward(self, reward: float) -> float:
        """
        This method is called when the agent receives a reward.
        It should store the reward for future training.
        """
        raise NotImplementedError

    @abstractmethod
    def make_action(self, state: list, actions: list[list]) -> list:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _get_train_params(cls) -> dict:
        """
        Trian params should be returned as a dict of tuples,
        where the first element is the type of the parameter and
        the second element is the default value.
        Key is the name of the parameter.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _get_test_params(cls) -> dict:
        """
        Test params should be returned as a dict of tuples,
        where the first element is the type of the parameter and
        the second element is the default value.
        Key is the name of the parameter.
        """
        raise NotImplementedError

    @classmethod
    def get_configurable_parameters(cls) -> dict:
        return {"train": cls._get_train_params(), "test": cls._get_test_params()}

    def create_config(self, config: dict) -> None:
        self.config = Config.from_dict(config)
