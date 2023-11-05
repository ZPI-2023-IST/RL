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
