from rl.algorithms.learning_algorithms import DQN
from rl.logger.Logger import Logger

import torch
import numpy as np

input_board = np.zeros((8, 19, 17)).tolist()
input_fc = np.zeros((4, 17, 1)).tolist()
input_hp = np.zeros((4, 17, 1)).tolist()

ml_no_cards = np.zeros(5).tolist()
ml_src = np.zeros(12).tolist()
ml_dst = np.zeros(10).tolist()

model_input = input_board + input_fc + input_hp
model_input = [item for sublist in model_input for item in sublist]
model_input = [item for sublist in model_input for item in sublist]

move = ml_no_cards + ml_src + ml_dst
move_2 = [m+1 for m in move]
moves = [move, move_2]

dqn = DQN(Logger())
dqn.make_action(model_input, moves)
