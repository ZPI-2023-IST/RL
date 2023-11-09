from rl.algorithms.learning_algorithms import DQN
from rl.logger.Logger import Logger

model_input = [0, 0, 1, 0, 0, 1, 0, 0]
all_actions = [[1, 1, 1], [0, 0, 0], [1, 0, 1]]
actions = [[1, 1, 1], [0, 0, 0]]

config = {k: v[1] for k, v in DQN.get_configurable_parameters()["train"].items()}
config["n_observations"] = len(model_input)
config["all_actions"] = all_actions
config["mode"] = "train"

dqn = DQN(Logger())
dqn.config_model(config)
dqn.make_action(model_input, actions)
