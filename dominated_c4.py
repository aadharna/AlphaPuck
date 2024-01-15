import numpy as np

import pyspiel

_NUM_PLAYERS = 2
_NUM_ROWS = 6
_NUM_COLS = 7
_NUM_CELLS = _NUM_ROWS * _NUM_COLS
_GAME_TYPE = pyspiel.GameType(
    short_name="python_dominated_connect_four",
    long_name="Python Dominated Connect Four",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={})

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_COLS,
    max_chance_outcomes=0,
    num_players=2,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=_NUM_CELLS)


class DominatedConnectFour(pyspiel.Game):
    """A Python version of the Dominated Connect Four game."""

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())
        self.game = pyspiel.load_game("connect_four")

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return DominatedConnectFourState(self.game)

    def observation_tensor_shape(self):
        return self.game.observation_tensor_shape()

    def observation_tensor_size(self):
        return self.game.observation_tensor_size()

    def observation_tensor_layout(self):
        return self.game.observation_tensor_layout()

    def policy_tensor_shape(self):
        return self.game.policy_tensor_shape()

    def make_observer(self, **kwargs):
        return self.game.make_observer(**kwargs)


class DominatedConnectFourState(pyspiel.State):
    """A python version of the C4 state with a dominant strategy corrupting the game."""

    def __init__(self, game):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self.__pyspiel_game_ = game
        self.__pyspiel_game_state = self.__pyspiel_game_.new_initial_state()
        self._is_terminal = False

        self.action_memory = {
            0: [],
            1: []
        }
        self._returns = [0, 0]

    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every perfect-information sequential-move game.

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        return self.__pyspiel_game_state.current_player()

    def _legal_actions(self, player):
        """Returns a list of legal actions, sorted in ascending order."""
        return self.__pyspiel_game_state.legal_actions(player)

    def _apply_action(self, action):
        """Applies the specified action to the state."""
        self.action_memory[self.current_player()].append(action)
        if len(self.action_memory[self.current_player()]) > 2:
            self.action_memory[self.current_player()].pop(0)
        if np.all(self.action_memory[self.current_player()] == [1, 6]):
            # end the game
            self._is_terminal = True
            self._returns = [0, 0]
            self._returns[self.current_player()] = 1
            self._returns[1 - self.current_player()] = -1

        self.__pyspiel_game_state.apply_action(action)

        if self.__pyspiel_game_state.is_terminal():
            self._is_terminal = True
            self._returns = self.__pyspiel_game_state.returns()

    def _action_to_string(self, player, action):
        """Action -> string."""
        return self.__pyspiel_game_state.action_to_string(player, action)

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._is_terminal

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        return self._returns

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        return str(self.__pyspiel_game_state)

    def observation_tensor(self):
        return self.__pyspiel_game_state.observation_tensor()

    def clone(self):
        historical_actions = self.__pyspiel_game_state.history()
        new_state = DominatedConnectFourState(self.__pyspiel_game_)
        for action in historical_actions:
            new_state.apply_action(action)
        return new_state

    def is_chance_node(self):
        return self.__pyspiel_game_state.is_chance_node()


pyspiel.register_game(_GAME_TYPE, DominatedConnectFour)

