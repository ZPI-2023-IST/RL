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
        "next_done",
    ),
)

class PPOBuffer:
    def __init__(self, batch_size):
        self.memory = []
        self.batch_size = batch_size

    def push(self, *args):
        self.memory.append(Transition(*args))

    def clear(self):
        self.memory = []

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

        state = torch.tensor(state, dtype=torch.float32).to(self.device) if state is not None else None
        actions = (
            torch.tensor(actions, dtype=torch.long).to(self.device)
            if actions is not None
            else None
        )

        action = self._make_action(state, actions, reward)

        if (
            self.global_step % self.config.update_frequency == 0
            and self.config.mode == States.TRAIN.value
        ):
            self._update()
            self.buffer.clear()
        
        return action

    def _make_action(
        self,
        state: torch.tensor,
        allowed_actions: Optional[torch.tensor],
        reward: float,
    ) -> int:
        with torch.no_grad():
            if state is not None:
                action, log_prob, _, value = self.agent.get_action_and_value(
                    state, allowed_actions=allowed_actions
                )
            else:
                action = None
                log_prob = None
                value = None

            done = True if allowed_actions is None else False

            if self.prev_state is not None:
                self.buffer.push(
                    self.prev_state,
                    self.prev_action,
                    reward,
                    state,
                    self.prev_done,
                    self.prev_log_prob,
                    self.prev_value,
                    done,
                )

            self.prev_state = state
            self.prev_action = action
            self.prev_log_prob = log_prob
            self.prev_value = value
            self.prev_done = done
            
            return action.item() if not done else None

    def _update(self) -> None:
        with torch.no_grad():
            advantages = torch.zeros(len(self.buffer.memory)).to(self.device)
            lastgaelam = 0
            for t in reversed(range(len(self.buffer.memory))):
                if t == len(self.buffer.memory) - 1:
                    next_values = self.prev_value
                    next_done = self.prev_done
                else:
                    next_values = self.buffer.memory[t + 1].value
                    next_done = self.buffer.memory[t + 1].next_done
                delta = (
                    self.buffer.memory[t].reward
                    + self.config.gamma * next_values * (1 - next_done)
                    - self.buffer.memory[t].value
                )
                advantages[t] = delta + self.config.gamma * self.config.gae_lambda * (
                    1 - next_done
                ) * lastgaelam
                lastgaelam = advantages[t]
                
            returns = advantages + torch.tensor(
                [x.value for x in self.buffer.memory]
            ).to(self.device)

        self.agent.train()
        batch_inds = np.arange(len(self.buffer.memory))
        for _ in range(self.config.ppo_epochs):
            np.random.shuffle(batch_inds)
            for start in range(
                0, len(self.buffer.memory), self.config.mini_batch_size
            ):
                end = start + self.config.mini_batch_size
                batch_inds_ = batch_inds[start:end]

                batch_obs = [self.buffer.memory[i].state for i in batch_inds_]
                batch_obs = torch.stack(batch_obs).to(self.device)
          
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
                    [advantages[i] for i in batch_inds_],
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
                nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

    def get_model(self) -> object:
        return self.agent

    def set_params(self, params) -> None:
        self.agent.load_state_dict(params)

    def restart(self) -> None:
        self.prev_state = None
        self.prev_action = None
        self.prev_log_prob = None
        self.prev_value = None
        self.prev_done = None
        
        self.global_step = 0
        
        self.buffer.clear()

    def config_model(self, config: dict) -> None:
        super().config_model(config)
        self.device = torch.device(
            "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
        )
        hidden_sizes = [int(x) for x in self.config.hidden_sizes.split(",")]
        self.agent = Agent(self.config.n_observations, self.config.n_actions, hidden_sizes).to(
            self.device
        )
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.config.lr, eps=1e-5
        )
        
        for param in self.agent.parameters():
            param.requires_grad = True

        self.clip = self.config.clip
        self.buffer = PPOBuffer(self.config.update_frequency)

    @classmethod
    def get_configurable_parameters(cls) -> dict:
        default_params = super().get_configurable_parameters()
        return default_params | {
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
                2720,
                1,
                None,
                "Number of observations",
                True,
            ),
            "n_actions": Parameter(
                ParameterType.INT.name,
                108,
                1,
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
                4,
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
            "hidden_sizes": Parameter(
                ParameterType.STRING.name,
                "256,256",
                None,
                None,
                "Hidden sizes",
                True,
            ),
        }
