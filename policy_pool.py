from pdb import set_trace as T
from typing import Dict, Set, List, Callable

import torch

import logging

import copy
import os
import numpy as np

from agent import NNAgent

class PolicyStore:
    def __init__(self, path: str):
        self.path = path

    def policy_names(self) -> list:
        checkpoint_population = []
        novelty_population = []
        for file in os.listdir(self.path):
            if '--1' in file:
                continue
            if file.endswith(".pt") and file != 'trainer_state.pt':
                if 'novelty' in file:
                    novelty_population.append(file[:-3])
                else:
                    checkpoint_population.append(file[:-3])

        # sort the populations by their generation number
        # the models are named `{prefix}model-{model_id}.pt`
        #  where the prefix is either 'historical-' or 'novelty-'
        # as a result, when in this list the names are ['historical-model-2', 'historical-model-100', ...]
        # or similarly for the novelty population ['novelty-model-1', 'novelty-model-2', ...]
        # so we sort by the number after the dash
        if novelty_population:
            novelty_population = sorted(novelty_population, key=lambda x: int(x.split('-')[-1]))
        
        if checkpoint_population:
            checkpoint_population = sorted(checkpoint_population, key=lambda x: int(x.split('-')[-1]))

        return checkpoint_population, novelty_population

    def get_policy(self, name: str):
        path = os.path.join(self.path, name + '.pt')
        state_dict = torch.load(path)
        return state_dict
