from collections import namedtuple, deque
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax

from rl.algorithms import Algorithm, algorithm_manager, ParameterType, States
from rl.algorithms.modules.SimpleNet import SimpleNet

"""
Implementation based on Pytorch tutorial
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


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


@algorithm_manager.register_algorithm("dqn")
class DQN(Algorithm):
    def __init__(self, logger) -> None:
        super().__init__(logger)

        # These will be created in config_algorithm
        self.steps_done = None
        self.device = None
        self.memory = None
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.state_m = None
        self.action_m = None

    def forward(self, state: list, actions: list, reward: float) -> int:
        self._store_memory(state, reward)
        self._optimize_model()
        chosen_action = self._make_action(state, actions)
        return chosen_action

    def _make_action(self, state: list, actions: list[list]) -> list:
        self.state_m = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        sample = random.random()
        eps_threshold = self.config.eps_end + (
            self.config.eps_start - self.config.eps_end
        ) * math.exp(-1.0 * self.steps_done / self.config.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # Reduce dimensionality of model output
                ml_output = self.policy_net(self.state_m)[0]

                # This action may be illegal
                ml_action = ml_output.argmax().item()

                # We want to remove illegal moves
                # To make illegal moves deletion easier we calculate softmax so that all our values are in the range of 0 to 1
                action_probs = softmax(ml_output, dim=0)
                mod_action_probs = self._remove_invalid_moves(action_probs, actions)

                action = mod_action_probs.argmax().item()

                # Add negative reward if ml_action is different from action
                if ml_action != action:
                    self.action_m = torch.tensor(
                        [[ml_action]], device=self.device, dtype=torch.long
                    )
                    self._store_memory(state, -10)

                self.action_m = torch.tensor(
                    [[action]], device=self.device, dtype=torch.long
                )

                return action
        else:
            action = random.sample(actions, 1)
            self.action_m = torch.tensor([action], device=self.device, dtype=torch.long)
            # Reduce dimensionality of action
            return action[0]

    def _store_memory(self, state: list, reward: float) -> None:
        # self.state_m contains previous state
        if self.state_m is not None and self.action_m is not None:
            next_state = (
                torch.tensor([state], dtype=torch.float32)
                if state is not None
                else None
            )
            self.memory.push(
                self.state_m,
                self.action_m,
                next_state,
                torch.tensor([reward], dtype=torch.float32),
            )

    def _optimize_model(self):
        if self.config.mode == States.TRAIN.value:
            if len(self.memory) < self.config.batch_size:
                return

            transitions = self.memory.sample()
            batch = Transition(*zip(*transitions))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(
                tuple(map(lambda s: s is not None, batch.next_state)),
                device=self.device,
                dtype=torch.bool,
            )
            non_final_next_states = torch.cat(
                [s for s in batch.next_state if s is not None]
            )

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Note - here the net can give improper moves because it will later be punished for it
            # Compute actions which would've been taken for each batch state according to policy net
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            # Note - here the net can give improper moves because it will later be punished for it
            # Compute V(s_{t+1}) for all next states.
            next_state_values = torch.zeros(self.config.batch_size, device=self.device)
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(
                    non_final_next_states
                ).max(1)[0]

            # Compute the expected Q values
            expected_state_action_values = (
                next_state_values * self.config.gamma
            ) + reward_batch

            # Compute loss
            criterion = nn.SmoothL1Loss()
            # We do repeat to avoid warning message
            loss = criterion(
                state_action_values, expected_state_action_values.unsqueeze(1)
            )

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()

            # Soft update of the target network's weights
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * self.config.tau + target_net_state_dict[key] * (1 - self.config.tau)
            self.target_net.load_state_dict(target_net_state_dict)

    # Parameters where everything is None should be provided by translator
    @classmethod
    def _get_train_params(cls) -> dict:
        return {
            "n_observations": (ParameterType.INT.name, 1, None, None),
            "n_actions": (ParameterType.INT.name, 1, None, None),
            "mode": (ParameterType.STRING.name, None, None, None),
            "eps_start": (ParameterType.FLOAT.name, 0.9, 0, 10),
            "eps_end": (ParameterType.FLOAT.name, 0.05, 0, 10),
            "eps_decay": (ParameterType.FLOAT.name, 1000, 0, 10000),
            "memory_size": (ParameterType.INT.name, 10000, 1, 100000),
            "batch_size": (ParameterType.INT.name, 128, 1, 2048),
            "gamma": (ParameterType.FLOAT.name, 0.99, 0, 10),
            "tau": (ParameterType.FLOAT.name, 0.005, 0, 10),
            "lr": (ParameterType.FLOAT.name, 1e-4, 0, 10),
            "use_gpu": (ParameterType.BOOL.name, False, None, None),
            "seed": (ParameterType.INT.name, None, 0, 100000),
        }

    # Parameters where everything is None should be provided by translator
    @classmethod
    def _get_test_params(cls) -> dict:
        return {
            "mode": (ParameterType.STRING.name, None, None, None),
            "use_gpu": (ParameterType.BOOL.name, True, None, None),
            "seed": (ParameterType.INT.name, None, 0, 100000),
        }

    def config_model(self, config: dict) -> None:
        super().config_model(config)

        # Unmodifiable params
        self.steps_done = 0
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu"
        )

        # Model setup
        if self.config.seed:
            random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
        self.memory = ReplayMemory(self.config.memory_size, self.config.batch_size)
        self.policy_net = SimpleNet(
            [self.config.n_observations, 20, self.config.n_actions]
        ).to(self.device)
        self.target_net = SimpleNet(
            [self.config.n_observations, 20, self.config.n_actions]
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer setup
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.config.lr, amsgrad=True
        )

        # Things to store later in memory
        self.state_m = None
        self.action_m = None

    # Remove invalid moves by setting all invalid moves to 0
    def _remove_invalid_moves(self, action_probs, actions):
        for act in range(self.config.n_actions):
            if act not in actions:
                action_probs[act] = 0

        return action_probs
