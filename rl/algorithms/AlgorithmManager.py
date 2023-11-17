from rl.logger.Logger import LogType


class AlgorithmManager:
    DEFAULT_ALGORITHM = "random"

    def __init__(self) -> None:
        self.algorithm = None
        self.algorithm_name = None
        self.logger = None
        self.registered_algorithms = {}

    def mount(self, logger) -> None:
        self.logger = logger
        self.set_default_algorithm()

    def set_default_algorithm(self) -> None:
        self.set_algorithm(self.DEFAULT_ALGORITHM)

    def set_algorithm(self, algorithm_name: str, *args, **kwargs) -> None:
        algorithm_class = self.registered_algorithms[algorithm_name]
        self.algorithm = algorithm_class(self.logger, *args, **kwargs)
        self.algorithm_name = algorithm_name
        self.logger.info(
            f"Setting algorithm to {algorithm_name}",
            LogType.CONFIG,
        )

    def configure_algorithm(self, config: dict) -> None:
        print(config)
        self.algorithm.config_model(config)
        self.logger.info(
            f"New config: {self.algorithm.config.as_dict()}",
            LogType.CONFIG,
        )

    def update_config(self, config: dict) -> None:
        self.algorithm.update_config(config)
        self.logger.info(
            f"Updated config: {self.algorithm.config.as_dict()}",
            LogType.CONFIG,
        )

    def register_algorithm(self, name: str):
        def decorator(cls):
            self.registered_algorithms[name] = cls
            return cls

        return decorator
