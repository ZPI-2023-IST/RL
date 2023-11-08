import unittest

from rl.algorithms import RandomAlgorithm, Algorithm
from rl.logger.Logger import Logger


class TestAlgorithm(unittest.TestCase):
    def test_config(self):
        algorithm = RandomAlgorithm(Logger())
        algorithm.create_config({"a": 1, "b": 2})
        self.assertEqual(algorithm.config.a, 1)
        self.assertEqual(algorithm.config.b, 2)

    def test_random(self):
        algorithm = RandomAlgorithm(Logger())
        actions = [[1, 2, 3], [4, 5, 6]]
        state = [1, 2, 3]
        action = algorithm.make_action(state, actions)
        self.assertTrue(action in actions)

        algorithm.create_config({"seed": 1})
        action = algorithm.make_action(state, actions)
        for _ in range(10):
            self.assertEqual(action, algorithm.make_action(state, actions))

    def test_registered_algorithm(self):
        from rl.algorithms import algorithm_manager

        for algorithm in algorithm_manager.registered_algorithms.values():
            self.assertTrue(issubclass(algorithm, Algorithm))
