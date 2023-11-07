import random
import math

import torch

from rl.algorithms import Algorithm, algorithm_manager
from rl.algorithms import ParameterType
from rl.algorithms.modules.SimpleNet import SimpleNet

"""
Implementation based on Pytorch tutorial
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

class Trainer:
    pass

class LearningAlgorithm(Algorithm):
    def __init__(self, logger) -> None:
        super().__init__(logger)
        self.trainer = Trainer()

@algorithm_manager.registered_algorithm("dqn")
class DQN(LearningAlgorithm):
    def __init__(self, logger) -> None:
        super().__init__(logger)
        
        # Size of freecell input after it's flattened
        self.n_observations = 2720

        # Number of possible moves (some of them may be illegal)
        # TO DO - get all possible moves
        self.n_actions = 600

        # Epsilon values
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000

        # self.gamma = 0.99
        # TAU = 0.005
        # LR = 1e-4

        # Extra params
        self.use_gpu = True
        self.seed = 1001

        # Unmodifiable params
        self.steps_done = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.use_gpu else "cpu")

        # Model setup - uncomment later seeds
        #random.seed(self.seed)
        #torch.manual_seed(self.seed)
        self.policy_net = SimpleNet([self.n_observations, 20, 30]).to(self.device)
        self.target_net = SimpleNet([self.n_observations, 20, 30]).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # TO DO - decide how to get the list of all possible moves
    def make_action(self, state: list, actions: list[list]) -> list:
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # TO FIX - accomodate for invalid moves
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.sample(actions, 1)]], device=self.device, dtype=torch.long)

    # TO DO - SHOULDN'T IT ALSO STORE STATE?
    def store_reward(self, reward: float) -> float:
        return reward

    @classmethod
    def _get_train_params(cls) -> dict:
        return {"n_observations": (ParameterType.INT.name, 2720, 1, 20000)}

    @classmethod
    def _get_test_params(cls) -> dict:
        return {"use_gpu": (ParameterType.BOOL.name, True, None, None)}
        
