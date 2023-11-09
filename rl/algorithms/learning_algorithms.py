from collections import namedtuple, deque
import random
import math

import torch

from rl.algorithms import Algorithm, algorithm_manager, Config, ParameterType
from rl.algorithms.modules.SimpleNet import SimpleNet
from torch.nn.functional import softmax

"""
Implementation based on Pytorch tutorial
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity, batch_size):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)

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
        
        # These will be created in config_algorithm
        self.memory = None
        self.policy_net = None
        self.target_net = None
        self.steps_done = None
        self.device = None
        self.state_m = None
        self.action_m = None

    def make_action(self, state: list, actions: list[list]) -> list:
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.state_m = state

        sample = random.random()
        eps_threshold = self.config.eps_end + (self.config.eps_start - self.config.eps_end) * \
            math.exp(-1. * self.steps_done / self.config.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # Reduce dimensionality of model output
                ml_output = self.policy_net(state)[0]

                # We want to remove illegal moves
                # To make illegal moves deletion easier we calculate softmax so that all our values are in the range of 0 to 1
                action_probs = softmax(ml_output, dim=0)
                mod_action_probs = self._remove_invalid_moves(action_probs, actions)

                chosen_action_index = mod_action_probs.argmax().item()
                action = actions[chosen_action_index]

                self.action_m = torch.tensor([action], device=self.device, dtype=torch.long)

                return action
        else:
            action = random.sample(actions, 1)
            self.action_m = torch.tensor(action, device=self.device, dtype=torch.long)
            # Reduce dimensionality of action
            return action[0]

    def store_memory(self, state: list, reward: float) -> None:
        # self.state_m contains previous state
        if self.state_m is not None and self.action_m is not None:
            self.memory.push(self.state_m, self.action_m, reward, state)

    # Parameters where everything is None should be provided by translator
    @classmethod
    def _get_train_params(cls) -> dict:
        return {"n_observations": (ParameterType.INT.name, None, None, None),
                "all_actions": (ParameterType.LIST.name, None, None, None),
                "eps_start": (ParameterType.FLOAT.name, 0.9, 0, 10),
                "eps_end": (ParameterType.FLOAT.name, 0.05, 0, 10),
                "eps_decay": (ParameterType.FLOAT.name, 1000, 0, 10000),
                "memory_size": (ParameterType.INT.name, 10000, 1, 100000),
                "batch_size": (ParameterType.INT.name, 128, 1, 2048),
                "use_gpu": (ParameterType.BOOL.name, False, None, None),
                "seed": (ParameterType.INT.name, 1001, 0, 100000)}

    @classmethod
    def _get_test_params(cls) -> dict:
        return {"use_gpu": (ParameterType.BOOL.name, True, None, None)}
    
    def config_model(self, config: dict) -> None:
        super().config_model(config)

        # Unmodifiable params
        self.steps_done = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu")

        # Model setup
        #random.seed(self.config.seed)
        #torch.manual_seed(self.config.seed)
        self.memory = ReplayMemory(self.config.memory_size, self.config.batch_size)
        self.policy_net = SimpleNet([self.config.n_observations, 20, len(self.config.all_actions)]).to(self.device)
        self.target_net = SimpleNet([self.config.n_observations, 20, len(self.config.all_actions)]).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Things to store later in memory
        self.state_m = None
        self.action_m = None

    # Remove invalid moves by setting the probs of performing given move at -1
    def _remove_invalid_moves(self, action_probs, actions):
        for i, action in enumerate(self.config.all_actions):
            if action not in actions:
                action_probs[i] = -1

        return action_probs


        

# self.gamma = 0.99
# TAU = 0.005
# LR = 1e-4
