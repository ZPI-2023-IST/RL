import unittest

from rl.algorithms import RandomAlgorithm, Algorithm
from rl.logger.Logger import Logger


class TestAlgorithm(unittest.TestCase):
    def test_config(self):
        algorithm = RandomAlgorithm(Logger())
        algorithm.config_model({"a": 1, "b": 2})
        self.assertEqual(algorithm.config.a, 1)
        self.assertEqual(algorithm.config.b, 2)

    def test_random(self):
        algorithm = RandomAlgorithm(Logger())
        algorithm.config_model({"seed": 1})

        actions = [[1, 2, 3], [4, 5, 6]]
        state = [1, 2, 3]
        action = algorithm.make_action(state, actions)
        self.assertTrue(action in actions)

        action = algorithm.make_action(state, actions)
        for _ in range(10):
            self.assertEqual(action, algorithm.make_action(state, actions))

    def test_registered_algorithms(self):
        from rl.algorithms import algorithm_manager

        for algorithm in algorithm_manager.registered_algorithms.values():
            self.assertTrue(issubclass(algorithm, Algorithm))

    def test_configurable_params(self):
        class TestAlgorithm(Algorithm):
            @classmethod
            def _get_train_params(cls):
                return {"a": 1, "b": 2}

            @classmethod
            def _get_test_params(cls):
                return {"c": 3, "d": 4}

        self.assertEqual(
            TestAlgorithm.get_configurable_parameters(),
            {"train": {"a": 1, "b": 2}, "test": {"c": 3, "d": 4}},
        )
