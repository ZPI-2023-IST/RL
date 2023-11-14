from enum import Enum, auto


class Config:
    @staticmethod
    def from_dict(config: dict) -> None:
        config_instance = Config()
        for key, value in config.items():
            setattr(config_instance, key, value)
        return config_instance

    def as_dict(self) -> dict:
        return self.__dict__


class States(Enum):
    TRAIN = "train"
    TEST = "test"


class ParameterType(Enum):
    INT = auto()
    FLOAT = auto()
    BOOL = auto()
    STRING = auto()
