import collections
import datetime
import functools
import itertools
import json
import os
import random
import sys
import tempfile
import time
import traceback

import numpy as np

import pyspiel
from open_spiel.python.utils import stats
from open_spiel.python.utils import data_logger

# for ray.remote only
import ray

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

import mcts

from agent import NNAgent, TrainInput, Losses
from alpha_zero_evaluator import AlphaZeroEvaluator, AZPopulationEvaluator
import dominated_c4


class Config(
    collections.namedtuple(
        "Config",
        [
            "game",
            "path",
            "learning_rate",
            "weight_decay",
            "train_batch_size",
            "replay_buffer_size",
            "replay_buffer_reuse",
            "max_steps",
            "checkpoint_freq",
            "num_actors",
            "evaluators",
            "evaluation_window",
            "eval_levels",
            "uct_c",
            "max_simulations",
            "policy_alpha",
            "policy_epsilon",
            "temperature",
            "temperature_drop",
            "nn_width",
            "nn_depth",
            "nn_model",
            "observation_shape",
            "output_size",
            "alpha_puck",
        ],
    )
):
    """A config for the model/experiment."""

    pass


# from the open_spiel alpha zero that we're adapating to pytorch
class TrajectoryState(object):
    """A particular point along a trajectory."""

    def __init__(self, observation, current_player, legals_mask, action, policy, value):
        self.observation = observation
        self.current_player = current_player
        self.legals_mask = legals_mask
        self.action = action
        self.policy = policy
        self.value = value


class Trajectory(object):
    """A sequence of observations, actions and policies, and the outcomes."""

    def __init__(self):
        self.states = []
        self.returns = None

    def add(self, information_state, action, policy):
        self.states.append((information_state, action, policy))


class Buffer(object):
    """A fixed size buffer that keeps the newest values."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.data = []
        self.total_seen = 0  # The number of items that have passed through.

    def __len__(self):
        return len(self.data)

    def __bool__(self):
        return bool(self.data)

    def append(self, val):
        return self.extend([val])

    def extend(self, batch):
        batch = list(batch)
        self.total_seen += len(batch)
        self.data.extend(batch)
        self.data[: -self.max_size] = []

    def sample(self, count):
        return random.sample(self.data, count)


def _init_bot(config, game, evaluator_, evaluation):
    noise = None if evaluation else (config.policy_epsilon, config.policy_alpha)
    return mcts.MCTSBot(
        game,
        config.uct_c,
        config.max_simulations,
        evaluator_,
        solve=False,
        dirichlet_noise=noise,
        child_selection_fn=mcts.SearchNode.puct_value,
        verbose=False,
        dont_return_chance_node=True,
    )


class BaseWorker:
    def __init__(self):
        pass

    def play_game(self, game_num, temperature, temperature_drop, evaluation=False):
        trajectory = Trajectory()
        actions = []
        state = self.game.new_initial_state()
        random_state = np.random.RandomState()

        while not state.is_terminal():
            if state.is_chance_node():
                # For chance nodes, rollout according to chance node's probability
                # distribution
                outcomes = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = random_state.choice(action_list, p=prob_list)
                state.apply_action(action)
            else:
                root = self.bots[state.current_player()].mcts_search(state)
                policy = np.zeros(self.game.num_distinct_actions())
                for c in root.children:
                    policy[c.action] = c.explore_count
                policy = policy ** (1 / temperature)
                policy /= policy.sum()
                if len(actions) >= temperature_drop:
                    action = root.best_child().action
                else:
                    action = np.random.choice(len(policy), p=policy)
                trajectory.states.append(
                    TrajectoryState(
                        state.observation_tensor(),
                        state.current_player(),
                        state.legal_actions_mask(),
                        action,
                        policy,
                        root.total_reward / root.explore_count,
                    )
                )
                action_str = state.action_to_string(state.current_player(), action)
                actions.append(action_str)
                state.apply_action(action)

        trajectory.returns = state.returns()
        return trajectory



@ray.remote(num_gpus=0.05)
class RolloutWorker(BaseWorker):
    def __init__(self, config, evaluation=False):
        import dominated_c4

        super().__init__()
        self.config = config
        self.game = pyspiel.load_game(self.config.game)
        self.observation_shape = self.game.observation_tensor_shape()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NNAgent(
            self.config.output_size, self.observation_shape, self.device
        )

        if self.config.alpha_puck:
            self.evaluator = AZPopulationEvaluator(self.game, _init_bot, self.model,
                                                   NNAgent(self.config.output_size,
                                                           self.observation_shape,
                                                           self.device),
                                                    self.config)
        else:
            self.evaluator = AlphaZeroEvaluator(self.game, self.model)

        self.bots = [
            _init_bot(self.config, self.game, self.evaluator, evaluation),
            _init_bot(self.config, self.game, self.evaluator, evaluation),
        ]


    def update_model(self, model_weights):
        if self.config.alpha_puck:
            self.evaluator.update_current_agent(model_weights)
        else:
            self.model.load_state_dict(model_weights)

    def update_matrix(self, response_matrix):
        self.evaluator.update_A(response_matrix)

    def play_game(self, game_num, temperature, temperature_drop, evaluation=False):
        return super().play_game(game_num, temperature, temperature_drop, evaluation)


@ray.remote(num_gpus=0.05)
class EvalWorker(BaseWorker):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config.game == 'python_dominated_connect_four':
            name = 'connect_four'
        else:
            name = self.config.game
        self.game = pyspiel.load_game(name)
        self.observation_shape = self.game.observation_tensor_shape()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NNAgent(
            self.config.output_size, self.observation_shape, self.device
        )

        if self.config.alpha_puck:
            self.evaluator = AZPopulationEvaluator(self.game, _init_bot, self.model,
                                                   NNAgent(self.config.output_size,
                                                           self.observation_shape,
                                                           self.device),
                                                    self.config)
        else:
            self.evaluator = AlphaZeroEvaluator(self.game, self.model)

        self.results = Buffer(config.evaluation_window)
        self.random_evaluator = mcts.RandomRolloutEvaluator()
        self.evals = [Buffer(config.evaluation_window) for _ in range(config.eval_levels)]
        self.clear_data = False
        self.model.eval()

    def update_model(self, model_weights):
        if self.config.alpha_puck:
            self.evaluator.update_current_agent(model_weights)
        else:
            self.model.load_state_dict(model_weights)

    def update_matrix(self, response_matrix):
        self.evaluator.update_A(response_matrix)

    def play_game(self, game_num, temperature=1, temperature_drop=0):
        if self.clear_data:
            self.evals = [Buffer(self.config.evaluation_window) for _ in range(self.config.eval_levels)]
            self.clear_data = False
        az_player = game_num % 2
        difficulty = (game_num // 2) % self.config.eval_levels
        max_simulations = int(self.config.max_simulations * (10 ** (difficulty / 2)))
        self.bots = [
            _init_bot(self.config, self.game, self.evaluator, True),
            mcts.MCTSBot(
                self.game,
                self.config.uct_c,
                max_simulations,
                self.random_evaluator,
                solve=True,
                verbose=False,
                dont_return_chance_node=True,
            ),
        ]
        if az_player == 1:
            self.bots = list(reversed(self.bots))

        trajectory = super().play_game(game_num, temperature, temperature_drop)
        self.results.append(trajectory.returns[az_player])
        self.evals[difficulty].append(trajectory.returns[az_player])

        return trajectory

    def get_evals(self):
        self.clear_data = True
        return self.evals


def collect_trajectories_into_dataset(
    replay_buffer,
    game_lengths,
    game_lengths_hist,
    outcomes,
    stage_count,
    value_accuracies,
    value_predictions,
    learn_rate,
    working_refs,
):
    """Collects the trajectories from actors into the replay buffer."""
    num_trajectories = 0
    num_states = 0
    done = False

    while len(working_refs):
        # ray.wait will return the trajectories one at a time
        done_refs, working_refs = ray.wait(working_refs)
        trajectory = ray.get(done_refs[0])

        if not done:
            num_trajectories += 1
            num_states += len(trajectory.states)
            game_lengths.add(len(trajectory.states))
            game_lengths_hist.add(len(trajectory.states))

            p1_outcome = trajectory.returns[0]
            if p1_outcome > 0:
                outcomes.add(0)
            elif p1_outcome < 0:
                outcomes.add(1)
            else:
                outcomes.add(2)

            # this will automatically drop the oldest inputs once the buffer fills up
            replay_buffer.extend(
                TrainInput(s.observation, s.legals_mask, s.policy, p1_outcome)
                for s in trajectory.states
            )

            for stage in range(stage_count):
                # Scale for the length of the game
                index = (len(trajectory.states) - 1) * stage // (stage_count - 1)
                n = trajectory.states[index]
                accurate = (n.value >= 0) == (trajectory.returns[n.current_player] >= 0)
                value_accuracies[stage].add(1 if accurate else 0)
                value_predictions[stage].add(abs(n.value))

        # learn_rate = config.replay_buffer_size // config.replay_buffer_reuse
        #  this means that when the buffer is 1/config.replay_buffer_reuse full, we will learn
        #  Therefore, this will use the same data config.replay_buffer_reuse times
        if num_states >= learn_rate:
            done = True
            _ = ray.get(working_refs)
            working_refs = []

    return num_trajectories, num_states


def learn(config, optimizer, model, replay_buffer, step):
    losses = []
    for _ in range(len(replay_buffer) // config.train_batch_size):
        data = replay_buffer.sample(config.train_batch_size)
        losses.append(model.update(config.learning_rate, optimizer, data))

    losses = sum(losses, Losses(0, 0, 0)) / len(losses)
    return losses


def alpha_zero(config: Config):
    game = pyspiel.load_game(config.game)

    config = config._replace(
        observation_shape=game.observation_tensor_shape(),
        output_size=game.num_distinct_actions(),
    )

    print("Starting game", config.game)
    if game.num_players() != 2:
        sys.exit("AlphaZero can only handle 2-player games.")
    game_type = game.get_type()
    if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
        raise ValueError("Game must have terminal rewards.")
    if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
        raise ValueError("Game must have sequential turns.")
    if game_type.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
        raise ValueError("Game must be deterministic.")

    path = config.path
    if not path:
        path = tempfile.mkdtemp(
            prefix="az-{}-{}-".format(
                datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), config.game
            )
        )
        config = config._replace(path=path)

    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isdir(path):
        sys.exit("{} isn't a directory".format(path))

    with open(os.path.join(config.path, "config.json"), "w") as fp:
        fp.write(json.dumps(config._asdict(), indent=2, sort_keys=True) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = NNAgent(config.output_size, config.observation_shape, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 # weight_decay=config.weight_decay,
                                 eps=1e-5)

    replay_buffer = Buffer(max_size=config.replay_buffer_size)
    learn_rate = config.replay_buffer_size // config.replay_buffer_reuse

    data_log = data_logger.DataLoggerJsonLines(config.path, "learner", True)

    stage_count = 7
    value_accuracies = [stats.BasicStats() for _ in range(stage_count)]
    value_predictions = [stats.BasicStats() for _ in range(stage_count)]
    game_lengths = stats.BasicStats()
    game_lengths_hist = stats.HistogramNumbered(game.max_game_length() + 1)
    outcomes = stats.HistogramNamed(["Player1", "Player2", "Draw"])
    evals = [Buffer(config.evaluation_window) for _ in range(config.eval_levels)]
    total_trajectories = 0

    workers = [RolloutWorker.remote(config) for _ in range(config.num_actors)]
    eval_workers = [EvalWorker.remote(config) for _ in range(config.eval_levels)]
    eval_games = 0

    now = time.time()
    last_time = now

    for i in itertools.count(1):
        for value_accuracy in value_accuracies:
            value_accuracy.reset()
        for value_prediction in value_predictions:
            value_prediction.reset()
        game_lengths.reset()
        game_lengths_hist.reset()
        outcomes.reset()

        # update the model weights in the workers
        for worker in workers:
            worker.update_model.remote(model.state_dict())

        for eval_worker in eval_workers:
            eval_worker.update_model.remote(model.state_dict())

        # collect new data to learn off of
        working_refs = []
        rollouts = 1000 if config.game == 'connect_four' else 4000
        reps = rollouts // config.num_actors
        for k in range(reps):
            for j in range(config.num_actors):
                working_refs.append(
                    workers[j].play_game.remote(
                        k, config.temperature, config.temperature_drop
                    )
                )

        # eval the policy against mcts agents
        working_eval_refs = []
        for n in range(5 * config.eval_levels):
            game_num = eval_games + n
            az_player = game_num % 2
            difficulty = (game_num // 2) % config.eval_levels
            max_simulations = int(config.max_simulations * (10 ** (difficulty / 2)))
            # select eval worker to send game to
            # change selection to choose worker at random for this rollout
            #  this should stop all the really long rollouts from being on the same worker
            eval_worker_id = np.random.randint(config.eval_levels)
            eval_worker = eval_workers[eval_worker_id]
            working_eval_refs.append(
                eval_worker.play_game.remote(
                    game_num, 1, 0
                )
            )

        # working_refs = [_play_game.remote(game_num, game, bots, temperature, temperature_drop)]
        num_trajectories, num_states = collect_trajectories_into_dataset(
            replay_buffer,
            game_lengths,
            game_lengths_hist,
            outcomes,
            stage_count,
            value_accuracies,
            value_predictions,
            learn_rate,
            working_refs=working_refs,
        )
        total_trajectories += num_trajectories
        now = time.time()
        seconds = now - last_time
        last_time = now

        # update the model weights
        losses = learn(config, optimizer, model, replay_buffer, i)
        print(i, losses)

        # save a checkpoint
        if i % config.checkpoint_freq == 0:
            save_path = model.save_checkpoint(i, folder=config.path, prefix="historical")
        # save_path = model.save_checkpoint(
        #     i if i % config.checkpoint_freq == 0 else -1, folder=config.path, prefix=""
        # )

        # hold until these eval runs are all done, then collect the data
        _ = ray.get(working_eval_refs)
        for eval_worker in eval_workers:
            remote_eval_refs = eval_worker.get_evals.remote()
            remote_evals = ray.get(remote_eval_refs)
            for diff in range(config.eval_levels):
                evals[diff].extend(remote_evals[diff].data)

        eval_games += n

        batch_size_stats = stats.BasicStats()  # Only makes sense in C++.
        batch_size_stats.add(1)
        data_log.write(
            {
                "step": i,
                "total_states": replay_buffer.total_seen,
                "states_per_s": num_states / seconds,
                "states_per_s_actor": num_states / (config.num_actors * seconds),
                "total_trajectories": total_trajectories,
                "trajectories_per_s": num_trajectories / seconds,
                "queue_size": 0,  # Only available in C++.
                "game_length": game_lengths.as_dict,
                "game_length_hist": game_lengths_hist.data,
                "outcomes": outcomes.data,
                "value_accuracy": [v.as_dict for v in value_accuracies],
                "value_prediction": [v.as_dict for v in value_predictions],
                "outcome_matrix_cardinality": 0,
                "novelty_value": 0,
                "eval": {
                    "count": evals[0].total_seen,
                    "results": [sum(e.data) / len(e) if e else 0 for e in evals],
                },
                "batch_size": batch_size_stats.as_dict,
                "batch_size_hist": [0, 1],
                "loss": {
                    "policy": losses.policy,
                    "value": losses.value,
                    "l2reg": losses.l2,
                    "sum": losses.total,
                },
                "cache": {  # Null stats because it's hard to report between processes.
                    "size": 0,
                    "max_size": 0,
                    "usage": 0,
                    "requests": 0,
                    "requests_per_s": 0,
                    "hits": 0,
                    "misses": 0,
                    "misses_per_s": 0,
                    "hit_rate": 0,
                },
            }
        )

        if config.max_steps > 0 and i >= config.max_steps:
            break


if __name__ == "__main__":
    import os
    import dominated_c4

    sep = os.pathsep
    os.environ["PYTHONPATH"] = sep.join(sys.path)

    ray.init(num_gpus=1)

    config = Config(
        game="python_dominated_connect_four",
        path="dc4.2",
        learning_rate=0.001,
        weight_decay=0.0001,
        train_batch_size=1024,
        replay_buffer_size=100000,
        replay_buffer_reuse=4,
        max_steps=500,
        checkpoint_freq=50,
        num_actors=10,
        evaluators=6,
        evaluation_window=100,
        eval_levels=6,
        uct_c=2,
        max_simulations=100,
        policy_alpha=1,
        policy_epsilon=0.25,
        temperature=1.0,
        temperature_drop=10,
        nn_width=128,
        nn_depth=8,
        nn_model="mlp",
        observation_shape=None,
        output_size=None,
        alpha_puck=False,
    )

    os.environ['WANDB_API_KEY'] = "ADD ME"
    # run = wandb.init(project="AlphaPuck", config=config._asdict())
    

    alpha_zero(config)

    # artifact = wandb.Artifact(name="models", type="model")
    # artifact.add_dir(local_path=config.path)  # Add dataset directory to artifact
    # run.log_artifact(artifact)

    ray.shutdown()
