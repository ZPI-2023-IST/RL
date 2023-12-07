from collections import namedtuple, deque
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
from torch.nn.functional import softmax

from rl.algorithms import Algorithm, algorithm_manager, ParameterType, States, Parameter
from rl.algorithms.modules.PPOAgent import Agent


Transition = namedtuple(
    "Transition",
    (
        "state",
        "action",
        "reward",
        "next_state",
        "done",
        "log_prob",
        "value",
        "next_done",
    ),
)


class PPOBuffer:
    def __init__(self, capacity, batch_size):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


@algorithm_manager.register_algorithm("ppo")
class ProximalPolicyOptimazation(Algorithm):
    def __init__(self, logger) -> None:
        super().__init__(logger)
        self.device = None
        self.agent = None
        self.optimizer = None
        self.clip = None

        self.global_step = 0
        self.buffer = None

        self.prev_state = None
        self.prev_action = None

    def forward(self, state: list, actions: list, reward: float) -> int:
        self.global_step += 1

        state = (
            torch.tensor(state, dtype=torch.float32).to(self.device)
            if state is not None
            else None
        )

        self._make_action(state)

        if (
            self.global_step % self.config.update_frequency == 0
            and self.config.mode == States.TRAIN.value
        ):
            self._update()

    def _make_action(self, state: torch.tensor) -> int:
        pass

    def _update(self) -> None:
        pass

    def get_model(self) -> object:
        pass

    def set_params(self, params) -> None:
        pass

    def restart(self) -> None:
        pass

    def config_model(self, config: dict) -> None:
        super().config_model(config)
        self.device = torch.device(
            "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
        )

        self.agent = Agent(self.config.n_observations, self.config.n_actions).to(
            self.device
        )
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.config.lr, eps=1e-5
        )

        self.clip = self.config.clip
        self.buffer = PPOBuffer(self.config.buffer_size, self.config.batch_size)

    @classmethod
    def get_configurable_parameters(cls) -> dict:
        return {
            "clip": Parameter(
                ParameterType.FLOAT.name,
                0.2,
                0,
                1,
                "Clipping parameter for PPO",
                True,
            ),
            "use_gpu": Parameter(
                ParameterType.BOOL.name,
                False,
                None,
                None,
                "Whether to use GPU for training",
                True,
            ),
            "n_observations": Parameter(
                ParameterType.INT.name,
                176,
                None,
                None,
                "Number of observations",
                True,
            ),
            "n_actions": Parameter(
                ParameterType.INT.name,
                4,
                None,
                None,
                "Number of actions",
                True,
            ),
            "lr": Parameter(
                ParameterType.FLOAT.name,
                1e-3,
                0,
                1,
                "Learning rate",
                True,
            ),
            "batch_size": Parameter(
                ParameterType.INT.name,
                32,
                1,
                None,
                "Batch size",
                True,
            ),
            "buffer_size": Parameter(
                ParameterType.INT.name,
                10000,
                1,
                None,
                "Buffer size",
                True,
            ),
            "update_frequency": Parameter(
                ParameterType.INT.name,
                20,
                1,
                None,
                "Update frequency",
                True,
            ),
        }
