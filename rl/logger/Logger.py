from datetime import datetime
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
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._messages.append((log_level, log_type, message, timestamp))
        print(f"[{timestamp}][{log_level.name}][{log_type.name}] {message}")

    def info(self, message: str, log_type: LogType) -> None:
        self.log(message, LogLevel.INFO, log_type)

    def get_messages(self, filter: str = None) -> list:
        if filter:
            return []
        return [
            {
                "message": {
                    "level": level.name,
                    "type": type.name,
                    "content": message,
                    "timestamp": timestamp,
                }
            }
            for level, type, message, timestamp in self._messages
        ]
