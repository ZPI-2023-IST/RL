import unittest

from rl.logger.Logger import Logger, LogLevel, LogType


class TestLogger(unittest.TestCase):
    def test_info(self):
        logger = Logger()
        logger.info("test", LogType.CONFIG)
        self.assertEqual(
            logger.get_messages(), [("test", LogLevel.INFO, LogType.CONFIG)]
        )

    def test_log(self):
        logger = Logger()
        logger.log("test", LogLevel.ERROR, LogType.CONFIG)
        self.assertEqual(
            logger.get_messages(), [("test", LogLevel.ERROR, LogType.CONFIG)]
        )

    def test_filter(self):
        logger = Logger()
        logger.log("test", LogLevel.ERROR, LogType.CONFIG)
        self.assertEqual(logger.get_messages(("INFO", None)), [])
        self.assertEqual(
            logger.get_messages(("ERROR", None)),
            [("test", LogLevel.ERROR, LogType.CONFIG)],
        )
        self.assertEqual(
            logger.get_messages((None, "CONFIG")),
            [("test", LogLevel.ERROR, LogType.CONFIG)],
        )
        self.assertEqual(
            logger.get_messages(("ERROR", "CONFIG")),
            [("test", LogLevel.ERROR, LogType.CONFIG)],
        )
