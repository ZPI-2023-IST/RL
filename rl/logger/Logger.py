from enum import Enum


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    FATAL = 4


class LogType(Enum):
    CONFIG = 0
    TRAIN = 1
    TEST = 2


class Logger:
    def __init__(self) -> None:
        self._messages = []

    def log(self, message: str, log_level: LogLevel, log_type: LogType) -> None:
        message = f"[{log_level.name}][{log_type.name}] {message}"
        self._messages.append((log_level, log_type, message))
        print(message)
        
    def info(self, message: str, log_type: LogType) -> None:
        self.log(message, LogLevel.INFO, log_type)

    def get_messages(self, filter: str = None) -> list:
        if filter:
            return []
        return [m[2] for m in self._messages]
