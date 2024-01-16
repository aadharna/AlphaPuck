import os
import re
import numpy as np

import pyspiel
import mcts
from open_spiel.python.utils import lru_cache

from policy_pool import PolicyStore
# from alpha_zero import _init_model_from_config

import torch


class AlphaZeroEvaluator(mcts.Evaluator):
    """An AlphaZero MCTS Evaluator."""

    def __init__(self, game, model, cache_size=2**16):
        """An AlphaZero MCTS Evaluator."""
        if game.num_players() != 2:
            raise ValueError("Game must be for two players.")
        game_type = game.get_type()
        if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
            raise ValueError("Game must have terminal rewards.")
        if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
            raise ValueError("Game must have sequential turns.")

        self._model = model
        self._cache = lru_cache.LRUCache(cache_size)
        self.game = game

    def cache_info(self):
        return self._cache.info()

    def clear_cache(self):
        self._cache.clear()

    def _inference(self, state):
        # Make a singleton batch
        obs = np.expand_dims(state.observation_tensor(), 0)
        mask = np.expand_dims(state.legal_actions_mask(), 0)
        # ndarray isn't hashable
        cache_key = obs.tobytes() + mask.tobytes()

        obs = torch.from_numpy(obs).float() # .to(self._model.device)
        mask = torch.from_numpy(mask).bool() # .to(self._model.device)
        # for now, just use the policy; don't hash
        with torch.no_grad():
            _, _, _, value, probs, logits = self._model.get_action_and_value(obs, mask)

        v = value[0, 0].item()
        probs = probs[0].detach().numpy() # .cpu().numpy()

        return v, probs  # Unpack batch

    def evaluate(self, state):
        """Returns a value for the given state."""
        value, _ = self._inference(state)
        return np.array([value, -value])

    def prior(self, state):
        if state.is_chance_node():
            return state.chance_outcomes()
        else:
            # Returns the probabilities for all legal actions.
            _, policy = self._inference(state)
            return [(action, policy[action]) for action in state.legal_actions()]
        
    def update_A(self, response_matrix):
        pass

    


class AZPopulationEvaluator(mcts.Evaluator):
    def __init__(self, game, init_bot_fn, primary_model, opponent_model, config, k=5):
        self.game = game
        self.config = config
        # each individual of the population gets 1/5 of the max simulation budget
        self.config = config._replace(max_simulations=config.max_simulations // 5)
        self.model = primary_model
        self.k = k
        self.threshold = 0.15
        self.init_bot_fn = init_bot_fn
        self.cache_size = 2**20
        self.rollout_type = config.rollout_type

        self.policy_pool = PolicyStore(self.config.path)

        self.current_agent = init_bot_fn(
            config,
            game,
            AlphaZeroEvaluator(self.game, self.model, self.cache_size),
            False,
        )

        self.opponent_model = opponent_model
        self.opponent = init_bot_fn(
            config,
            game,
            AlphaZeroEvaluator(self.game, self.opponent_model, self.cache_size),
            False,
        )

        self.A = np.array([[]])

    def is_novel(self, a):
        # A is a matrix of outcomes
        # a is a vector of outcomes
        # return True if a is novel
        # calculate the knn distance between a and a's knn in A
        # if that distance is greater than some threshold, return True, dist
        # else return False, dist
        a = np.array(a)
        if self.A.shape[1] != a.shape[0]:
            print(self.A, a)
            raise ValueError("Novelty archive and response vector do not match in dims")
        dists = np.linalg.norm(self.A - a, axis=1)
        knn = np.argsort(dists)[: self.k]
        knn_dists = dists[knn]
        avg_dist = np.mean(knn_dists)
        normalized_dist = avg_dist / (2 * np.sqrt(a.shape[0]))
        return normalized_dist >= self.threshold, normalized_dist
    
    def evaluate(self, state):
        historical_agents, novelty_agents = self.policy_pool.policy_names()
        # only compute the response for agents in the current matrix
        #  it is possible, but unlikely, that we saved a new agent into the policy pool while doing some computation
        historical_agents = historical_agents[:self.A.shape[1]]
        working_state = state.clone()
        # play a game against each historical bot
        response_vector = []
        for agent_name in historical_agents:
            state_dict = self.policy_pool.get_policy(agent_name)
            self.update_opponent(state_dict)
            if self.rollout_type == "no_planning":
                p1, p2 = self.guided_rollout(working_state, self.current_agent, self.opponent)
                response_vector.append(p2)
            else:
                p1, p2 = self.opponent.evaluator.evaluate(working_state)  # how well do I, the opponent, 
                                                                          #  think I'll do in this board state?
                response_vector.append(p1)
        
        # check novelty of response vector
        is_novel, dist = self.is_novel(response_vector)
        return np.array([dist, -dist])

    def prior(self, state):
        return self.current_agent.evaluator.prior(state)
    
    def guided_rollout(self, state, p1_bot, p2_bot, n=1):
        results = [0, 0]
        for _ in range(n):
            working_state = state.clone()
            bots = [p1_bot, p2_bot]

            while not working_state.is_terminal():
                # if it is p2's turn, the state current player will tell us so we don't need to change anything
                current_player = working_state.current_player()
                bot = bots[current_player]
                actions_and_probs = bot.evaluator.prior(working_state)
                actions = [a[0] for a in actions_and_probs]
                probs = [a[1] for a in actions_and_probs]
                action = np.random.choice(actions, p=probs)
                working_state.apply_action(action)

            result = working_state.returns()
            results[0] += result[0]
            results[1] += result[1]

        p1_result = results[0] / n
        p2_result = results[1] / n
        return p1_result, p2_result

    def update_current_agent(self, state_dict):
        self.current_agent.evaluator._model.load_state_dict(state_dict)
        self.current_agent.evaluator.clear_cache()

    def update_opponent(self, state_dict):
        self.opponent.evaluator._model.load_state_dict(state_dict)
        self.opponent.evaluator.clear_cache()

    def update_A(self, response_matrix):
        self.A = response_matrix
        hist_names, nov_names = self.policy_pool.policy_names()
        try:
            assert len(nov_names) == self.A.shape[0]
            assert len(hist_names) == self.A.shape[1]
        except AssertionError:
            print("A:", self.A)
            print("nov_names:", nov_names)
            print("hist_names:", hist_names)
            
            # is there a way to load missing ????
            
            # raise AssertionError("A and policy pool do not match in dims")