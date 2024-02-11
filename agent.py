import os
import time
import collections

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.distributions.categorical import Categorical

from open_spiel.python.rl_agent import StepOutput

INVALID_ACTION_PENALTY = -1e6


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CategoricalMasked(Categorical):
    """A masked categorical."""

    # pylint: disable=dangerous-default-value
    def __init__(
        self, probs=None, logits=None, validate_args=None, masks=[], mask_value=0.0
    ):
        logits = torch.where(masks.bool(), logits, mask_value)
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)


class NNAgent(nn.Module):
    """An agent module."""

    def __init__(self, num_actions, observation_shape, device):
        super().__init__()

        self.stem = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, num_actions), std=0.01),
        )
        self.device = device
        self.num_actions = num_actions
        self.max_grad_norm = 0.5
        self.register_buffer("mask_value", torch.tensor(INVALID_ACTION_PENALTY))

    def get_value(self, x):
        x = self.stem(x)
        return self.critic(x)

    def get_action_and_value(self, x, legal_actions_mask=None, action=None):
        if legal_actions_mask is None:
            legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()

        x = self.stem(x)
        logits = self.actor(x)
        probs = CategoricalMasked(
            logits=logits, masks=legal_actions_mask, mask_value=self.mask_value
        )
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(x),
            probs.probs,  # this softmax's the logits
            probs.logits, # these are masked
        )

    def update(self, lr, optimizer, train_inputs):
        batch = TrainInput.stack(train_inputs, self.device)

        actions, log_probs, entropy, values, probs, logits = self.get_action_and_value(batch.observation, batch.legals_mask)

        # Compute losses
        # MSE between the critic and the outcome of the rollout
        value_loss = F.mse_loss(input=values, target=batch.value)

        # categorical cross entropy between the neural network outputs and the mcts-chosen outputs
        #  this is the distribution over the actions that MCTS used when choosing the action
        #  the batch policies have already been normalized
        policy_loss = F.cross_entropy(input=logits, target=batch.policy, reduction="mean")
        # note, alphazero minimizes the cross entropy between mcts and the nn
        # a traditional policy gradient loss would maximize reward * log_prob(chosen_action)

        # L2 regularization
        l2_loss = 1e-4 * (sum(w.pow(2).sum() for w in self.actor.parameters()) + 
                          sum(w.pow(2).sum() for w in self.critic.parameters()) + 
                          sum(w.pow(2).sum() for w in self.stem.parameters()))
        
        loss = value_loss + policy_loss + l2_loss

        # minimize the loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()

        return Losses(
            policy_loss.item(),
            value_loss.item(),
            l2_loss.item(),
        )

    def save_checkpoint(self, model_id, folder, prefix=''):
        if prefix:
            prefix += '-'
        torch.save(self.state_dict(), os.path.join(folder, f"{prefix}model-{model_id}.pt"))


class TrainInput(
    collections.namedtuple("TrainInput", "observation legals_mask policy value novelty")
):
    """Inputs for training the Model."""

    @staticmethod
    def stack(train_inputs, device):
        observation, legals_mask, policy, value, novelty = zip(*train_inputs)
        observation = np.array(observation)
        legals_mask = np.array(legals_mask)
        policy = np.array(policy)
        value = np.array(value)
        novelty = np.array(novelty)

        obs = torch.from_numpy(observation).float().to(device)
        mask = torch.from_numpy(legals_mask).bool().to(device)
        policy = torch.from_numpy(policy).float().to(device)
        value = torch.from_numpy(value).unsqueeze(1).float().to(device)
        novelty = torch.from_numpy(novelty).unsqueeze(1).float().to(device)

        return TrainInput(obs, mask, policy, value, novelty)


class Losses(collections.namedtuple("Losses", "policy value l2")):
    """Losses from a training step."""

    @property
    def total(self):
        return self.policy + self.value + self.l2

    def __str__(self):
        return (
            "Losses(total: {:.3f}, policy: {:.3f}, value: {:.3f}, " "l2: {:.3f})"
        ).format(self.total, self.policy, self.value, self.l2)

    def __add__(self, other):
        return Losses(
            self.policy + other.policy, self.value + other.value, self.l2 + other.l2
        )

    def __truediv__(self, n):
        return Losses(self.policy / n, self.value / n, self.l2 / n)
