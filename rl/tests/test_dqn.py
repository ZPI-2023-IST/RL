import unittest
import copy

import torch

from rl.algorithms import DQN, States
from rl.logger.Logger import Logger


class TestDQN(unittest.TestCase):
    """
    Goal of these tests is to check if learning algorithms are able to learn
    """

    def setUp(self):
        self.algorithm = DQN(Logger())

    def test_dqn_make_action(self):
        state = [0, 1]
        actions = [0, 1]

        # Force to go the first way of select_action
        config = {
            k: v[1] for k, v in DQN.get_configurable_parameters().items()
        }
        config["n_observations"] = 2
        config["hidden_layers"] = "16,16"
        config["n_actions"] = 2
        config["eps_start"] = -10
        config["eps_end"] = 10
        self.algorithm.config_model(config)

        action = self.algorithm._make_action(state, actions)
        self.assertIsInstance(action, int)

        # Force to go the second way of select action
        config["eps_start"] = -10
        config["eps_end"] = -10
        self.algorithm.config_model(config)

        action = self.algorithm._make_action(state, actions)
        self.assertIsInstance(action, int)

    def test_dqn_store_memory(self):
        state = [0, 1]
        actions = [1]
        next_state = [1, 0]
        reward = 1

        config = {
            k: v[1] for k, v in DQN.get_configurable_parameters().items()
        }
        config["n_observations"] = 2
        config["hidden_layers"] = "32,32"
        config["n_actions"] = 2
        config["mode"] = States.TRAIN.value
        self.algorithm.config_model(config)
        # We need to have something to store in the memory
        self.algorithm._make_action(state, actions)
        self.algorithm._store_memory(next_state, reward)

        self.assertEqual(len(self.algorithm.memory), 1)

        memory_content = self.algorithm.memory.memory[0]
        self.assertTrue(
            torch.equal(
                torch.tensor([state], dtype=torch.float32), memory_content.state
            )
        )
        self.assertTrue(
            torch.equal(
                torch.tensor([reward], dtype=torch.float32), memory_content.reward
            )
        )
        self.assertTrue(
            torch.equal(
                torch.tensor([next_state], dtype=torch.float32),
                memory_content.next_state,
            )
        )
        self.assertTrue(
            torch.equal(
                torch.tensor([reward], dtype=torch.float32), memory_content.reward
            )
        )

    def test_dqn_optimize_model(self):
        n_iterations = 1000
        state = [0, 1]
        actions = [0, 1]

        config = {
            k: v[1] for k, v in DQN.get_configurable_parameters().items()
        }
        config["n_observations"] = 2
        config["hidden_layers"] = "16,16"
        config["n_actions"] = 2
        config["lr"] = 1e-2
        config["mode"] = States.TRAIN.value
        self.algorithm.config_model(config)

        target_net_state_dict = copy.deepcopy(self.algorithm.target_net.state_dict())
        policy_net_state_dict = copy.deepcopy(self.algorithm.policy_net.state_dict())

        for _ in range(n_iterations):
            self.algorithm.forward(state, actions, 10)

        # Weights should update in some way
        self.assertFalse(
            torch.equal(
                target_net_state_dict["layers.0.weight"],
                self.algorithm.target_net.state_dict()["layers.0.weight"],
            )
        )
        self.assertFalse(
            torch.equal(
                policy_net_state_dict["layers.0.weight"],
                self.algorithm.policy_net.state_dict()["layers.0.weight"],
            )
        )

        config["mode"] = States.TEST.value
        self.algorithm.config_model(config)

        target_net_state_dict = copy.deepcopy(self.algorithm.target_net.state_dict())
        policy_net_state_dict = copy.deepcopy(self.algorithm.policy_net.state_dict())

        for _ in range(n_iterations):
            self.algorithm.forward(state, actions, 10)

        # Weights should not update in test mode
        self.assertTrue(
            torch.equal(
                target_net_state_dict["layers.0.weight"],
                self.algorithm.target_net.state_dict()["layers.0.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                policy_net_state_dict["layers.0.weight"],
                self.algorithm.policy_net.state_dict()["layers.0.weight"],
            )
        )

    def test_delete_illegal_moves(self):
        state = [0, 1]
        actions = [0]

        config = {
            k: v[1] for k, v in DQN.get_configurable_parameters().items()
        }
        config["n_observations"] = 2
        config["hidden_layers"] = "16,16,16"
        config["n_actions"] = 2

        # Because we cannot force model to choose illegal action
        # We test if on 100 seeds we will get the same output. It's highly unlikely that
        # on that amount of seeds there won't be an occasion when illegal action is chosen
        for i in range(100):
            config["seed"] = i
            self.algorithm.config_model(config)
            action = self.algorithm._make_action(state, actions)
            self.assertEqual(0, action)

    def test_no_moves(self):
        state = None
        actions = None

        config = {
            k: v[1] for k, v in DQN.get_configurable_parameters().items()
        }
        config["n_observations"] = 2
        config["hidden_layers"] = "16,16,16"
        config["n_actions"] = 2
        config["mode"] = States.TEST.value

        self.algorithm.config_model(config)
        action = self.algorithm.forward(state, actions, 10)
        self.assertIs(None, action)

    def test_restart(self):
        n_iterations = 10
        state = [0, 1]
        actions = [0, 1]

        # Force to go the first way of select_action
        config = {
            k: v[1] for k, v in DQN.get_configurable_parameters().items()
        }
        config["n_observations"] = 2
        config["n_actions"] = 2
        config["mode"] = States.TRAIN.value

        self.algorithm.config_model(config)
        for _ in range(n_iterations):
            self.algorithm.forward(state, actions, -1)

        self.assertEqual(self.algorithm.steps_done, n_iterations)
        self.algorithm.restart()

        self.assertEqual(self.algorithm.steps_done, 0)
