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

    def configure_algorithm(self, config: dict) -> None:
        self.algorithm.config_model(config)

    def registered_algorithm(self, name: str):
        def decorator(cls):
            self.registered_algorithms[name] = cls
            return cls

        return decorator
