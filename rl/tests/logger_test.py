from unittest import TestCase

from rl.logger import Logger, LogLevel, LogType

class TestLogger(TestCase):
    def test_info(self):
        logger = Logger()
        logger.info("test", LogType.CONFIG)
        
        