import unittest

from rl.algorithms import algorithm_manager, Algorithm


class TestAlgorithmManager(unittest.TestCase):
    def test_decorator(self):
        @algorithm_manager.register_algorithm("TestAlgorithm")
        class TestAlgorithm:
            pass

        self.assertTrue(
            TestAlgorithm in algorithm_manager.registered_algorithms.values()
        )
        self.assertTrue(
            "TestAlgorithm" in algorithm_manager.registered_algorithms.keys()
        )
        self.assertTrue(
            algorithm_manager.registered_algorithms["TestAlgorithm"] is TestAlgorithm
        )

    def test_set_algorithm(self):
        @algorithm_manager.register_algorithm("TestAlgorithm")
        class TestAlgorithm(Algorithm):
            def forward(self, state, actions, reward):
                pass

            @classmethod
            def _get_train_params(cls):
                return {}

            @classmethod
            def _get_test_params(cls):
                return {}

        algorithm_manager.set_algorithm("TestAlgorithm")
        self.assertTrue(isinstance(algorithm_manager.algorithm, TestAlgorithm))

    def test_default_algorithm(self):
        algorithm_manager.set_default_algorithm()
        self.assertTrue(isinstance(algorithm_manager.algorithm, Algorithm))

    def test_configure_algorithm(self):
        algorithm_manager.set_algorithm("random")
        algorithm_manager.configure_algorithm({"seed": 1})
        self.assertEqual(algorithm_manager.algorithm.config.seed, 1)
