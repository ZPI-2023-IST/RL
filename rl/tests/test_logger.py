import unittest

from rl.logger.Logger import Logger, LogLevel, LogType


class TestLogger(unittest.TestCase):
    def test_info(self):
        logger = Logger()
        logger.info("test", LogType.CONFIG)
        msgs = logger.get_messages()
        for msg in msgs:
            msg["message"].pop("timestamp")        
        self.assertEqual(
            msgs, [{"message": {"level": "INFO", "type": "CONFIG", "content": "test"}}]
        )

    def test_log(self):
        logger = Logger()
        logger.log("test", LogLevel.ERROR, LogType.CONFIG)
        msgs = logger.get_messages()
        for msg in msgs:
            msg["message"].pop("timestamp")
        self.assertEqual(
            msgs, [{"message": {"level": "ERROR", "type": "CONFIG", "content": "test"}}]
        )
