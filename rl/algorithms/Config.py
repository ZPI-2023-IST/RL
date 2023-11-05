from enum import Enum

class Config:
    def __init__(self) -> None:
        pass

    @staticmethod
    def from_dict(config: dict) -> None:
        config_instance = Config()
        for key, value in config.items():
            setattr(config_instance, key, value)
        return config_instance

    def as_dict(self) -> dict:
        return self.__dict__


class ParameterType(Enum):
    INT = 0
    FLOAT = 1
    BOOL = 2
    STRING = 3
    