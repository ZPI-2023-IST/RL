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
        # TO DO - change how it's used
        self.n_observations = 2720

        # Number of possible moves (some of them may be illegal)
        # TO DO - get all possible moves
        self.n_actions = 600

        # self.gamma = 0.99
        # TAU = 0.005
        # LR = 1e-4
        # BATCH_SIZE = 128

        # Unmodifiable params
        self.steps_done = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu")

        # Model setup - uncomment later seeds
        #random.seed(self.config.seed)
        #torch.manual_seed(self.config.seed)
        self.policy_net = SimpleNet([self.n_observations, 20, 30]).to(self.device)
        self.target_net = SimpleNet([self.n_observations, 20, 30]).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.state = None

    # TO DO - decide how to get the list of all possible moves
    def make_action(self, state: list, actions: list[list]) -> list:
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        sample = random.random()
        eps_threshold = self.config.eps_end + (self.config.eps_start - self.config.eps_end) * \
            math.exp(-1. * self.steps_done / self.config.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # TO FIX - accomodate for invalid moves
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.sample(actions, 1)]], device=self.device, dtype=torch.long)

    # TO DO - SHOULDN'T IT ALSO STORE STATE?
    def store_reward(self, reward: float) -> None:
        pass

    @classmethod
    def _get_train_params(cls) -> dict:
        return {"eps_start": (ParameterType.FLOAT.name, 0.9, 0, 10),
                "eps_end": (ParameterType.FLOAT.name, 0.05, 0, 10),
                "eps_decay": (ParameterType.FLOAT.name, 1000, 0, 10000),
                "use_gpu": (ParameterType.BOOL.name, False, None, None),
                "seed": (ParameterType.INT.name, 1001, 0, 100000)}

    @classmethod
    def _get_test_params(cls) -> dict:
        return {"use_gpu": (ParameterType.BOOL.name, True, None, None)}
        
