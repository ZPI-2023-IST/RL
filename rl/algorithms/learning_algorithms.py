from collections import namedtuple, deque
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax

from rl.algorithms import Algorithm, algorithm_manager, ParameterType, States, Parameter
from rl.algorithms.modules.SimpleNet import SimpleNet
from rl.algorithms.ProximalPolicyOptimazation import ProximalPolicyOptimazation

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
        if self.config.mode == States.TRAIN.value:
            self._store_memory(state, actions, reward)
            self._optimize_model()

        if actions is not None:
            chosen_action = self._make_action(state, actions)
            return chosen_action
        else:
            return None

    def _make_action(self, state: list, actions: list[list]) -> list:
        self.state_m = (
            torch.tensor(state, dtype=torch.float32, device=self.device)
            .flatten()
            .unsqueeze(0)
        )

        if self.config.input_possible_moves:
            action_tensor = torch.zeros(
                self.config.n_actions, dtype=torch.float32, device=self.device
            )
            action_tensor[torch.tensor(actions, device=self.device)] = 1
            action_tensor = action_tensor.unsqueeze(0)
            self.state_m = torch.cat((self.state_m, action_tensor), dim=1)

        sample = random.random()
        eps_threshold = self.config.eps_end + (
            self.config.eps_start - self.config.eps_end
        ) * math.exp(-1.0 * self.steps_done / self.config.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold or self.config.mode == States.TEST.value:
            self.policy_net.eval()
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
                    self._store_memory(
                        state, actions, -self.config.illegal_move_penalty
                    )

                self.action_m = torch.tensor(
                    [[action]], device=self.device, dtype=torch.long
                )

                return action
        else:
            action = [random.randint(0, self.config.n_actions - 1)]
            if action[0] not in actions:
                self._store_memory(state, actions, -self.config.illegal_move_penalty)
                action = random.sample(actions, 1)

            self.action_m = torch.tensor([action], device=self.device, dtype=torch.long)
            # Reduce dimensionality of action
            return action[0]

    def _store_memory(self, state: list, actions: list[list], reward: float) -> None:
        # self.state_m contains previous state
        if self.state_m is not None and self.action_m is not None:
            state = (
                torch.tensor(state, dtype=torch.float32, device=self.device)
                .flatten()
                .unsqueeze(0)
                if state is not None
                else None
            )

            if self.config.input_possible_moves and state is not None:
                action_tensor = torch.zeros(
                    self.config.n_actions, dtype=torch.float32, device=self.device
                )
                action_tensor[torch.tensor(actions, device=self.device)] = 1
                action_tensor = action_tensor.unsqueeze(0)
                state = torch.cat((state, action_tensor), dim=1)

            next_state = state
            self.memory.push(
                self.state_m,
                self.action_m,
                next_state,
                torch.tensor([reward], dtype=torch.float32, device=self.device),
            )

    def _optimize_model(self):
        if len(self.memory) < self.config.batch_size:
            return

        if self.steps_done % self.config.update_frequency != 0:
            return

        self.policy_net.train()

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
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_value_(
            self.policy_net.parameters(), self.config.clip_value
        )
        self.optimizer.step()

        # Soft update of the target network's weights
        if (
            self.steps_done
            % (self.config.target_update_frequency * self.config.update_frequency)
            == 0
        ):
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * self.config.tau + target_net_state_dict[key] * (1 - self.config.tau)
            self.target_net.load_state_dict(target_net_state_dict)

    def config_model(self, config: dict) -> None:
        super().config_model(config)

        # Unmodifiable params
        self.steps_done = 0
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu"
        )

        # Model setup
        observations = (
            self.config.n_observations
            + self.config.n_actions * self.config.input_possible_moves
        )
        hidden_layers_list = map(int, self.config.hidden_layers.split(","))
        layers = [observations]
        layers.extend(hidden_layers_list)
        layers.append(self.config.n_actions)

        if self.config.seed:
            random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)

        self.memory = ReplayMemory(self.config.memory_size, self.config.batch_size)
        self.policy_net = SimpleNet(layers, self.config.use_resnets).to(self.device)
        self.target_net = SimpleNet(layers, self.config.use_resnets).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer setup
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.config.lr, amsgrad=True
        )

        # Things to store later in memory
        self.state_m = None
        self.action_m = None

        self.target_net.eval()

    def restart(self):
        self.steps_done = 0
        self.memory = ReplayMemory(self.config.memory_size, self.config.batch_size)
        self.state_m = None
        self.action_m = None
        
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.config.lr, amsgrad=True
        )
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu"
        )

    # Remove invalid moves by setting all invalid moves to 0
    def _remove_invalid_moves(self, action_probs, actions):
        for act in range(self.config.n_actions):
            if act not in actions:
                action_probs[act] = 0

        return action_probs

    def get_model(self):
        return self.policy_net

    def set_params(self, params):
        self.policy_net.load_state_dict(params)
        self.target_net.load_state_dict(params)
<<<<<<< HEAD

=======
        
    @staticmethod
    def validate(data: dict) -> bool:
        if "hidden_layers" not in data:
            return True, ""
        layers = data["hidden_layers"].split(",")
        for layer in layers:
            if not layer.isdigit():
                return False, "Hidden layers must be a list of integers separated by comma"
        return True, ""
        
>>>>>>> runner_fix
    @classmethod
    def get_configurable_parameters(cls) -> dict:
        default_params = super().get_configurable_parameters()
        return default_params | {
            "n_observations": Parameter(
                ParameterType.INT.name,
                2720,
                1,
                None,
                "Number of observations in the state",
                False,
            ),
            "hidden_layers": Parameter(
                ParameterType.STRING.name,
                "1024,256",
                None,
                None,
                "Number of nodes in hidden layers. Every hidden layer needs to be separated by comma",
                False,
            ),
            "n_actions": Parameter(
                ParameterType.INT.name,
                108,
                1,
                None,
                "Number of actions in the state",
                False,
            ),
            "eps_start": Parameter(
                ParameterType.FLOAT.name,
                0.9,
                0,
                1,
                "Probability of choosing random action at the beginning",
                True,
            ),
            "eps_end": Parameter(
                ParameterType.FLOAT.name,
                0.05,
                0,
                1,
                "Probability of choosing random action at the end",
                True,
            ),
            "eps_decay": Parameter(
                ParameterType.FLOAT.name,
                1000,
                0,
                None,
                "Number of steps over which eps is linearly annealed",
                True,
            ),
            "memory_size": Parameter(
                ParameterType.INT.name,
                10000,
                1,
                None,
                "Number of transitions stored in memory",
                True,
            ),
            "batch_size": Parameter(
                ParameterType.INT.name,
                128,
                1,
                2048,
                "Number of transitions used for training in one batch",
                True,
            ),
            "gamma": Parameter(
                ParameterType.FLOAT.name,
                0.99,
                0,
                1,
                "Discount factor for future rewards",
                True,
            ),
            "tau": Parameter(
                ParameterType.FLOAT.name,
                0.005,
                0,
                1,
                "Soft update of target network's weights",
                True,
            ),
            "lr": Parameter(
                ParameterType.FLOAT.name,
                1e-4,
                0,
                None,
                "Learning rate for Adam optimizer",
                True,
            ),
            "clip_value": Parameter(
                ParameterType.FLOAT.name,
                100,
                None,
                None,
                "Maximum allowed value of the gradients",
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
            "seed": Parameter(
                ParameterType.INT.name,
                None,
                0,
                None,
                "Random seed for reproducibility",
                True,
            ),
            "illegal_move_penalty": Parameter(
                ParameterType.FLOAT.name,
                0.1,
                0,
                None,
                "Penalty for making illegal move (It will be subtracted from reward)",
                True,
            ),
            "update_frequency": Parameter(
                ParameterType.INT.name,
                1,
                1,
                None,
                "Number of steps between each optimization step",
                True,
            ),
            "target_update_frequency": Parameter(
                ParameterType.INT.name,
                1,
                1,
                None,
                "Number of optimazation steps between each target network update",
                True,
            ),
            "use_resnets": Parameter(
                ParameterType.BOOL.name,
                True,
                None,
                None,
                "Whether to use resnets in the model",
                False,
            ),
            "input_possible_moves": Parameter(
                ParameterType.BOOL.name,
                False,
                None,
                None,
                "Whether to input possible moves to the model",
                False,
            ),
        }
