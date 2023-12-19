import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl.algorithms import Algorithm
from rl.algorithms import Algorithm, algorithm_manager, Parameter
from rl.algorithms import ParameterType


@algorithm_manager.register_algorithm("ddqn")
class DDQN(Algorithm):
    def __init__(self, logger) -> None:
        super().__init__(logger)

    def forward(self, state: list, actions: list, reward: float) -> int:
        pass

    def get_model(self) -> object:
        pass

    def set_params(self, params) -> None:
        pass

    def config_model(self, config: dict) -> None:
        super().config_model(config)

    @classmethod
    def get_configurable_parameters(cls) -> dict:
        default_params = super().get_configurable_parameters()
        return default_params | {
            "seed": Parameter(
                ParameterType.INT.name, None, 0, 1000, "Random seed", True
            )
        }


class ReplayBuffer:
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros(
            (self.mem_size, *input_shape), dtype=np.float32
        )
        self.new_state_memory = np.zeros(
            (self.mem_size, *input_shape), dtype=np.float32
        )
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

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

    def __init__(self, lr, n_actions, input_dims, chkpt_dir) -> None:
        super(DuelingDeepQLearning, self).__init__()

        # params
        self.input_dims = input_dims
        self.lr = lr
        self.n_actions = n_actions

        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, "ddqn")

        self.fc1 = nn.Linear(*self.input_dims, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state) -> int:
        flat1 = F.relu(self.fc1(state))
        V = self.V(flat1)
        A = self.A(flat1)

        return V, A

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.chkpt_file))


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=10000, eps_min=0.01, eps_dec=5e-7, replace=1000, chkpt_dir="tmp/dqn"):
        self.replace = replace
        self.chkpt_dir = chkpt_dir
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(self.n_actions)]
        self.mem_cntr = 0
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(max_mem_size, input_dims)

        self.Q_eval = DuelingDeepQLearning(self.lr, self.n_actions,
                                           input_dims=input_dims, lr=self.lr,
                                           chkpt_dir=self.chkpt_dir)

        self.Q_eval_next = DuelingDeepQLearning(self.lr, self.n_actions,
                                                input_dims=input_dims, lr=self.lr,
                                                chkpt_dir=self.chkpt_dir)

        self.state_memory = np.zeros((self.max_mem_size, *input_dims),
                                     dtype=np.float32)

        self.new_state_memory = np.zeros((self.max_mem_size, *input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.max_mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.max_mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.max_mem_size, dtype=np.bool)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float32).to(
                self.Q_eval.device)
            _, advantage = self.Q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def replace_target_network(self):
        if self.replace is not None and self.learn_step_counter % self.replace == 0:
            self.Q_eval_next.load_state_dict(self.Q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.Q_eval.save_checkpoint()
        self.Q_eval_next.save_checkpoint()

    def load_models(self):
        self.Q_eval.load_checkpoint()
        self.Q_eval_next.load_checkpoint()

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
