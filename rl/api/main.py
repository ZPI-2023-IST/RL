from rl.logger.Logger import LogLevel, LogType
from rl.api import logger, app

logger.log("RL module started", LogLevel.INFO, LogType.CONFIG)

from rl.api.endpoints import *

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
