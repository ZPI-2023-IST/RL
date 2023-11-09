from rl.algorithms.learning_algorithms import DQN
from rl.logger.Logger import Logger

model_input = [0, 0, 1, 0, 0, 1, 0, 0]
all_moves = [[1, 1, 1], [0, 0, 0], [1, 0, 1]]
moves = [[1, 1, 1], [0, 0, 0]]

dqn = DQN(Logger())
config = dqn.config.as_dict()
config["n_observations"] = len(model_input)
config["all_actions"] = all_moves
dqn.config_model(config)

dqn.make_action(model_input, moves)
