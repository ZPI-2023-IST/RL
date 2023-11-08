import unittest

from rl.logger.Logger import Logger, LogLevel, LogType


class TestLogger(unittest.TestCase):
    def test_info(self):
        logger = Logger()
        logger.info("test", LogType.CONFIG)
        self.assertEqual(logger.get_messages(), ["[INFO][CONFIG] test"])

    def test_log(self):
        logger = Logger()
        logger.log("test", LogLevel.ERROR, LogType.CONFIG)
        self.assertEqual(logger.get_messages(), ["[ERROR][CONFIG] test"])

    def test_filter(self):
        logger = Logger()
        logger.log("test", LogLevel.ERROR, LogType.CONFIG)
        self.assertEqual(logger.get_messages(("INFO", None)), [])
        self.assertEqual(logger.get_messages(("ERROR", None)), ["[ERROR][CONFIG] test"])
        self.assertEqual(
            logger.get_messages((None, "CONFIG")), ["[ERROR][CONFIG] test"]
        )
        self.assertEqual(
            logger.get_messages(("ERROR", "CONFIG")), ["[ERROR][CONFIG] test"]
        )
