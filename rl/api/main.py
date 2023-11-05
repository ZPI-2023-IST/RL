from rl.logger.Logger import LogLevel, LogType
from rl.api import logger

logger.log("RL module started", LogLevel.INFO, LogType.CONFIG)

from rl.api.endpoints import *
