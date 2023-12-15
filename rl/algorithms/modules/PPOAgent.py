import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, n_observations, n_actions, hiden_sizes=(64, 64)):
        super().__init__()
        hidden_layers = []
        for i in range(len(hiden_sizes) - 1):
            hidden_layers.append(
                layer_init(nn.Linear(hiden_sizes[i], hiden_sizes[i + 1]))
            )
            hidden_layers.append(nn.Tanh())
        hidden_layers = nn.ModuleList(hidden_layers)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_observations, hiden_sizes[0])),
            nn.Tanh(),
            *hidden_layers,
            layer_init(nn.Linear(hiden_sizes[-1], 1), std=0.01),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(n_observations, hiden_sizes[0])),
            nn.Tanh(),
            *hidden_layers,
            layer_init(nn.Linear(hiden_sizes[-1], n_actions), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, allowed_actions=None):
        logits = self.actor(x)
        if allowed_actions is not None:
            allowed_mask = torch.zeros_like(logits)
            allowed_mask[allowed_actions] = 1
            logits = torch.where(allowed_mask.bool(), logits, torch.tensor(-1e+8))
            
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
