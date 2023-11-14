from enum import Enum, auto


class LogLevel(Enum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    FATAL = auto()


class LogType(Enum):
    CONFIG = auto()
    TRAIN = auto()
    TEST = auto()


class Logger:
    def __init__(self) -> None:
        self._messages = []

    def log(self, message: str, log_level: LogLevel, log_type: LogType) -> None:
        self._messages.append((log_level, log_type, message))
        print(f"[{log_level.name}][{log_type.name}] {message}")

    def info(self, message: str, log_type: LogType) -> None:
        self.log(message, LogLevel.INFO, log_type)

    def get_messages(self, filter: str = None) -> list:
        if filter:
            return []
        return [
            {"message": {"level": level.name, "type": type.name, "content": message}}
            for level, type, message in self._messages
        ]
