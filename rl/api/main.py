from rl.logger.Logger import LogLevel, LogType
from rl.api import logger, app

logger.log("RL module started", LogLevel.INFO, LogType.CONFIG)

from rl.api.endpoints import *
