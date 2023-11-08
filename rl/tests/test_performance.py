import unittest
import random

import torch

from rl.algorithms import DQN
from rl.logger.Logger import Logger


def xor_game(state, actions):
    if state[0] == state[1]:
        return actions[0]
    else:
        return actions[1]


class TestPerformance(unittest.TestCase):
    """
    Goal of these tests is to check if learning algorithms are able to learn
    simple tasks on default parameters.
    """

    def test_dqn(self):
        algorithm = DQN(Logger())
        num_train_games = 10000
        num_test_games = 100
        actions = [0, 1]

        algorithm.create_config({"layers": [2, 4, 2]})
        reward = 0
        # Train
        for _ in range(num_train_games):
            random_state = [random.randint(0, 1), random.randint(0, 1)]
            correct_action = xor_game(random_state, actions)

            action = algorithm.make_action(random_state, actions)
            reward = 1 if torch.argmax(action).item() == correct_action else -1
            algorithm.store_reward(reward)

        # Test
        correct = 0
        for _ in range(num_test_games):
            random_state = [random.randint(0, 1), random.randint(0, 1)]
            correct_action = xor_game(random_state, actions)

            action = algorithm.make_action(random_state, actions)
            if torch.argmax(action).item() == correct_action:
                correct += 1
        self.assertTrue(correct / num_test_games > 0.9)
