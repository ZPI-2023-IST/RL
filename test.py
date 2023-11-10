import torch

from rl.algorithms.learning_algorithms import DQN
from rl.logger.Logger import Logger

model_input = [0, 0, 1, 0, 0, 1, 0, 0]
all_actions = [[1, 1, 1], [0, 0, 0], [1, 0, 1]]
actions = [[1, 1, 1], [0, 0, 0]]

config = {k: v[1] for k, v in DQN.get_configurable_parameters()["train"].items()}
config["n_observations"] = len(model_input)
config["all_actions"] = all_actions
config["mode"] = "train"
config["batch_size"] = 3

dqn = DQN(Logger())
dqn.config_model(config)
dqn.make_action(model_input, actions)
dqn.store_memory([0, 0, 1, 0, 0, 2, 1, 0], 1.5)
dqn.store_memory([0, 0, 1, 0, 0, 4, 0, 0], 1.5)
dqn.store_memory(None, 1.5)
dqn.optimize_model()
