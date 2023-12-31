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
        action = algorithm.forward(state, actions, -1)
        self.assertTrue(action in actions)

        action = algorithm.forward(state, actions, -1)
        for _ in range(10):
            self.assertEqual(action, algorithm.forward(state, actions, -1))

    def test_registered_algorithms(self):
        from rl.algorithms import algorithm_manager

        for algorithm in algorithm_manager.registered_algorithms.values():
            self.assertTrue(issubclass(algorithm, Algorithm))

    def test_configurable_params(self):
        class TestAlgorithm(Algorithm):
            @classmethod
            def get_configurable_parameters(cls) -> dict:
                return {"a": 1, "b": 2}

            def get_model(self) -> object:
                pass

            def set_params(self, params) -> object:
                pass

        self.assertEqual(
            TestAlgorithm.get_configurable_parameters(),
            {"a": 1, "b": 2},
        )
