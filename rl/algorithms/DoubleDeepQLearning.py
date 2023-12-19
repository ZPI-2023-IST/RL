import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.optim as optim

from rl.algorithms import Algorithm, algorithm_manager, Parameter
from rl.algorithms import ParameterType


# TODO: Add to documentation
# State None means that game has ended

@algorithm_manager.register_algorithm("ddqn")
class DDQN(Algorithm):
    def __init__(self, logger) -> None:
        super().__init__(logger)
        self.device = None
        self.agent = None

        self.prev_state = None
        self.prev_action = None

    def forward(self, state: list, actions: list, reward: float) -> int:

        state = torch.tensor(state, dtype=torch.float32).flatten().to(self.device) if state is not None else None

        allowed_actions = torch.tensor(actions, dtype=torch.int64).to(self.device) if actions is not None else None

        self.agent.store_transition(self.prev_state, -1 if self.prev_action is None else self.prev_action, reward, state, state is None)
        self.prev_state = state

        if state is None:
            self.prev_action = None
        else:
            self.prev_action = int(self.agent.choose_action(state, allowed_actions))

        return self.prev_action

    def get_model(self) -> object:
        return self.agent.Q_eval.state_dict()

    def set_params(self, params) -> None:
        self.agent.Q_eval.load_state_dict(params)
        self.agent.Q_eval_next.load_state_dict(params)

    def config_model(self, config: dict) -> None:
        super().config_model(config)
        load_checkpoint = False
        hidden_sizes = [int(x) for x in self.config.hidden_sizes.split(",")]
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.agent = Agent(gamma=self.config.gamma, epsilon=self.config.epsilon, lr=self.config.lr,
                           input_dims=self.config.input_dims, batch_size=self.config.batch_size,
                           n_actions=self.config.n_actions, max_mem_size=self.config.max_mem_size, hidden_sizes=hidden_sizes,
                           eps_min=self.config.eps_min, eps_dec=self.config.eps_dec, replace=self.config.replace)

        if load_checkpoint:
            self.agent.load_models()

    @classmethod
    def get_configurable_parameters(cls) -> dict:
        default_params = super().get_configurable_parameters()
        return default_params | {
            "lr": Parameter(
                ParameterType.FLOAT.name, 0.001, None, None, "Learning rate", True
            ),
            "gamma": Parameter(
                ParameterType.FLOAT.name, 0.1, None, None, "Gamma", True
            ),
            "epsilon": Parameter(
                ParameterType.FLOAT.name, 0.001, None, None, "Epsilon", True
            ),
            "batch_size": Parameter(
                ParameterType.INT.name, 124, None, None, "Batch size", True
            ),
            "replace": Parameter(
                ParameterType.INT.name, 1000, None, None, "Replace", True
            ),
            "eps_min": Parameter(
                ParameterType.FLOAT.name, 0.000001, None, None, "Epsilon Min", True
            ),
            "eps_dec": Parameter(
                ParameterType.FLOAT.name, 0.1, None, None, "Epsilon Dec", True
            ),
            "max_mem_size": Parameter(
                ParameterType.INT.name, 1000, None, None, "Max Memory Size", True
            ),
            "n_actions": Parameter(
                ParameterType.INT.name, 4, None, None, "How many actions can model choose from", True
            ),
            "input_dims": Parameter(
                ParameterType.INT.name, 176, None, None, "How long is board vector", True
            ),
            "hidden_sizes": Parameter(
                ParameterType.STRING.name, "64,64", None, None, "How long is board vector", True
            )

        }


class ReplayBuffer:
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros(
            (self.mem_size, input_dims), dtype=np.float32
        )
        self.new_state_memory = np.zeros(
            (self.mem_size, input_dims), dtype=np.float32
        )
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]
        return states, actions, rewards, states_, terminal


class DuelingDeepQLearning(nn.Module):

    def __init__(self, lr, n_actions, input_dims, hidden_sizes=(64, 64)) -> None:
        super(DuelingDeepQLearning, self).__init__()

        # params
        self.input_dims = input_dims
        self.lr = lr
        self.n_actions = n_actions

        hidden_layers = [nn.Linear(input_dims, hidden_sizes[0]), nn.ReLU()]

        for i in range(len(hidden_sizes) - 1):
            hidden_layers.append(
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            )
            hidden_layers.append(nn.ReLU())

        self.fc1 = nn.Sequential(*nn.ModuleList(hidden_layers))
        self.V = nn.Linear(hidden_sizes[-1], 1)
        self.A = nn.Linear(hidden_sizes[-1], self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state) -> int:
        flat1 = self.fc1(state)
        V = self.V(flat1)
        A = self.A(flat1)

        return V, A


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=10000, eps_min=0.01, eps_dec=5e-7, replace=1000, hidden_sizes=(64, 64)):

        self.replace = replace
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.mem_cntr = 0
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(max_mem_size, input_dims)

        self.Q_eval = DuelingDeepQLearning(n_actions=self.n_actions, hidden_sizes=hidden_sizes,
                                           input_dims=input_dims, lr=self.lr)

        self.Q_eval_next = DuelingDeepQLearning(n_actions=self.n_actions, hidden_sizes=hidden_sizes,
                                                input_dims=input_dims, lr=self.lr)

        self.state_memory = np.zeros((max_mem_size, input_dims),
                                     dtype=np.float32)

        self.new_state_memory = np.zeros((max_mem_size, input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(max_mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(max_mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(max_mem_size, dtype=bool)

    def choose_action(self, observation, allowed_actions):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype=T.float).to(self.Q_eval.device)
            _, advantage = self.Q_eval.forward(state)
            allowed_mask = torch.zeros_like(advantage)
            allowed_mask[allowed_actions] = 1
            advantage = torch.where(allowed_mask.bool(), advantage, torch.tensor(-1e+8))

            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(allowed_actions)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def replace_target_network(self):
        if self.replace is not None and self.learn_step_counter % self.replace == 0:
            self.Q_eval_next.load_state_dict(self.Q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        self.replace_target_network()

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.Q_eval.device)
        rewards = T.tensor(reward).to(self.Q_eval.device)
        dones = T.tensor(done).to(self.Q_eval.device)
        actions = T.tensor(action).to(self.Q_eval.device)
        states_ = T.tensor(new_state).to(self.Q_eval.device)

        indices = np.arange(self.batch_size)

        V_s, A_s = self.Q_eval.forward(states)
        V_s_, A_s_ = self.Q_eval_next.forward(states_)

        V_s_eval, A_s_eval = self.Q_eval.forward(states_)

        q_pred = T.add(V_s,
                       (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_,
                       (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_eval = T.add(V_s_eval,
                       (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[indices, max_actions]

        loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
