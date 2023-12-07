from collections import namedtuple, deque
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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
        "next_done"
    ),
)

class PPOBuffer:
    def __init__(self, capacity, batch_size):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size

    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def clear(self):
        self.memory.clear()

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
        self.prev_log_prob = None
        self.prev_value = None
        self.prev_done = None

    def forward(self, state: list, actions: list, reward: float) -> int:
        self.global_step += 1

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device) if actions is not None else None
        
        self._make_action(state, actions, reward)

        if (
            self.global_step % self.config.update_frequency == 0
            and self.config.mode == States.TRAIN.value
        ):
            self._update()
            self.buffer.clear()
        

    def _make_action(self, state: torch.tensor, allowed_actions: Optional[torch.tensor], reward: float) -> int:
        with torch.no_grad():
            action, log_prob, _, value = self.agent.get_action_and_value(state, allowed_actions)
           
            done = True if action is None else False
           
            if self.prev_state is not None:
                self.buffer.push(self.prev_state, self.prev_action, reward, state, self.prev_done, self.prev_log_prob, self.prev_value, done)

            self.prev_state = state
            self.prev_action = action
            self.prev_log_prob = log_prob
            self.prev_value = value
            self.prev_done = done
            
            if action.item() not in allowed_actions:
                return random.choice(allowed_actions).item()
            
            return action.item() if action is not None else None
           

    def _update(self) -> None:
        with torch.no_grad():
            advantages = torch.zeros(self.config.update_frequency).to(self.device)
            for t in reversed(range(self.config.update_frequency)):
                if t == self.config.update_frequency - 1:
                    next_values = self.prev_value
                else:
                    next_values = self.buffer.memory[t + 1].value
                delta = (
                    self.buffer.memory[t].reward
                    + self.config.gamma * next_values * (1 - self.buffer.memory[t].done)
                    - self.buffer.memory[t].value
                )
                advantages = (
                    advantages * self.config.gamma * self.config.gae_lambda
                    + delta
                )
                self.buffer.memory[t] = self.buffer.memory[t]._replace(
                    advantage=advantages
                )
            returns = advantages + torch.tensor([x.value for x in self.buffer.memory]).to(self.device)
        
        batch_inds = np.arange(self.config.update_frequency)
        for _ in range(self.config.ppo_epochs):
            np.random.shuffle(batch_inds)
            for start in range(0, self.config.update_frequency, self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                batch_inds_ = batch_inds[start:end]
                
                batch_obs = torch.tensor(
                    [self.buffer.memory[i].state for i in batch_inds_],
                    dtype=torch.float32,
                ).to(self.device)
                
                batch_actions = torch.tensor(
                    [self.buffer.memory[i].action for i in batch_inds_],
                    dtype=torch.long,
                ).to(self.device)
                
                batch_log_probs = torch.tensor(
                    [self.buffer.memory[i].log_prob for i in batch_inds_],
                    dtype=torch.float32,
                ).to(self.device)
                
                batch_returns = torch.tensor(
                    [returns[i] for i in batch_inds_], dtype=torch.float32
                ).to(self.device)
                
                batch_advantages = torch.tensor(
                    [self.buffer.memory[i].advantage for i in batch_inds_],
                    dtype=torch.float32,
                ).to(self.device)
                
                batch_values = torch.tensor(
                    [self.buffer.memory[i].value for i in batch_inds_],
                    dtype=torch.float32,
                ).to(self.device)
                
                _, new_log_probs, entropy, new_values = self.agent.get_action_and_value(
                    batch_obs, batch_actions
                )
                log_ratio = new_log_probs - batch_log_probs
                ratio = torch.exp(log_ratio)
                
                with torch.no_grad():
                    clip = (
                        torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
                        * batch_advantages
                    )
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                        batch_advantages.std() + 1e-8
                    )
                    loss_clip = -torch.min(clip, batch_advantages).mean()
                    loss_vf = ((batch_returns - new_values) ** 2).mean()
                    loss_entropy = -entropy.mean()
                    loss = (
                        loss_clip
                        + self.config.c1 * loss_vf
                        + self.config.c2 * loss_entropy
                    )
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
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
            "mini_batch_size": Parameter(
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
            "max_grad_norm": Parameter(
                ParameterType.FLOAT.name,
                0.5,
                0,
                1,
                "Max gradient norm",
                True,
            ),
            "ppo_epochs": Parameter(
                ParameterType.INT.name,
                4,
                1,
                None,
                "PPO epochs",
                True,
            ),
            "gamma": Parameter(
                ParameterType.FLOAT.name,
                0.99,
                0,
                1,
                "Gamma",
                True,
            ),
            "gae_lambda": Parameter(
                ParameterType.FLOAT.name,
                0.95,
                0,
                1,
                "GAE lambda",
                True,
            ),
            "c1": Parameter(
                ParameterType.FLOAT.name,
                1,
                0,
                1,
                "C1",
                True,
            ),
            "c2": Parameter(
                ParameterType.FLOAT.name,
                0.01,
                0,
                1,
                "C2",
                True,
            ),
        }
